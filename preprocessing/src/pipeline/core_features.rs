use std::cmp::Ordering;
use std::f64::consts::TAU;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::config::NormalizationConfig;
use crate::domain::{Bar, BarKey, BookLevel, OrderBookSnapshot};
use crate::normalization::{CausalScaler, CausalScalerState};
use crate::utils::math::signed_log1p;
use time::OffsetDateTime;

const DEFAULT_RV_WINDOW: usize = 10;
const LONG_RV_WINDOW_MULTIPLIER: usize = 5;
const LEVEL_FEATURE_COUNT: usize = 3;
const MAX_LEVEL_OFFSET_TICKS: f64 = 256.0;
const RV_MIN_OBSERVATIONS: usize = 2;
const FLOW_REL_CLAMP: f64 = 25.0;
const LIMIT_OF_IMBALANCE_CLAMP: f64 = 10.0;
const TRADE_VOLUME_REL_CLAMP: f64 = 15.0;
const US_SESSION_OPEN_SECONDS: i64 = 9 * 3600 + 30 * 60;
const US_SESSION_CLOSE_SECONDS: i64 = 16 * 3600;
const US_SESSION_LENGTH_SECONDS: f64 = (US_SESSION_CLOSE_SECONDS - US_SESSION_OPEN_SECONDS) as f64;
const DEPTH_LOW_REL: f64 = 1.0;
const DEPTH_HIGH_REL: f64 = 3.0;
const VOLUME_LOW_REL: f64 = 0.5;
const VOLUME_HIGH_REL: f64 = 1.5;
const SPEED_LOW_TPS: f64 = 0.5;
const SPEED_HIGH_TPS: f64 = 1.5;
const REGIME_WINDOW: usize = 60;
const REGIME_MIN_OBSERVATIONS: usize = 2;
const MAX_Z_SCORE: f64 = 8.0;

pub struct CoreFeatureExtractor {
    tick_size: f64,
    epsilon: f64,
    scaler: CausalScaler,
    rv: RollingVariance,
    rv_long: RollingVariance,
    rv_min_window: usize,
    rv_long_min_window: usize,
    z_volume_stats: RollingMeanStd,
    z_spread_stats: RollingMeanStd,
    z_volatility_stats: RollingMeanStd,
}

impl CoreFeatureExtractor {
    pub fn new(tick_size: f64, norm_cfg: &NormalizationConfig) -> Self {
        Self::with_window_and_state(tick_size, norm_cfg, DEFAULT_RV_WINDOW, None)
    }

    pub fn with_window(tick_size: f64, norm_cfg: &NormalizationConfig, rv_window: usize) -> Self {
        Self::with_window_and_state(tick_size, norm_cfg, rv_window, None)
    }

    pub fn with_state(
        tick_size: f64,
        norm_cfg: &NormalizationConfig,
        scaler_state: &CausalScalerState,
    ) -> Self {
        Self::with_window_and_state(tick_size, norm_cfg, DEFAULT_RV_WINDOW, Some(scaler_state))
    }

    pub fn with_window_and_state(
        tick_size: f64,
        norm_cfg: &NormalizationConfig,
        rv_window: usize,
        scaler_state: Option<&CausalScalerState>,
    ) -> Self {
        let short_window = rv_window.max(1);
        let long_window = (rv_window.saturating_mul(LONG_RV_WINDOW_MULTIPLIER)).max(short_window);
        let extractor = Self {
            tick_size: tick_size.max(1e-12),
            epsilon: norm_cfg.log_epsilon.max(1e-12),
            scaler: CausalScaler::with_state(norm_cfg, scaler_state),
            rv: RollingVariance::new(short_window),
            rv_long: RollingVariance::new(long_window),
            rv_min_window: RV_MIN_OBSERVATIONS,
            rv_long_min_window: RV_MIN_OBSERVATIONS,
            z_volume_stats: RollingMeanStd::new(REGIME_WINDOW),
            z_spread_stats: RollingMeanStd::new(REGIME_WINDOW),
            z_volatility_stats: RollingMeanStd::new(REGIME_WINDOW),
        };
        extractor
    }

    pub fn scaler_state(&self) -> CausalScalerState {
        self.scaler.snapshot()
    }

    pub fn compute(&mut self, bars: &[Bar]) -> Vec<CoreFeatureRow> {
        bars.iter()
            .filter_map(|bar| self.compute_row(bar))
            .collect()
    }

    fn compute_row(&mut self, bar: &Bar) -> Option<CoreFeatureRow> {
        let mid_return = bar.mid_return()?;
        let spread_close = bar.spread_close?;
        let spread = spread_close / self.tick_size;
        if !spread.is_finite() {
            return None;
        }
        let spread_open = bar.spread_open?;
        let spread_change_ticks = (spread_close - spread_open) / self.tick_size;

        let mid_close = bar.mid_close?;
        let mid_high = bar.mid_high?;
        let mid_low = bar.mid_low?;
        if mid_close <= 0.0 {
            return None;
        }
        let mid_range = (mid_high - mid_low).max(0.0);
        let mid_range_rel = mid_range / (mid_close.abs() + self.epsilon);

        let imbalance = compute_imbalance(bar, self.epsilon);
        let volume_scaled = self.scaler.normalize_volume(bar.trade_volume_sum);
        let trade_volume_sum_rel = clamp_trade_volume(volume_scaled.relative);
        let trade_count_log = (bar.trade_count as f64).ln_1p();
        let duration_secs = bar.duration().as_seconds_f64().max(1e-6);
        let volume_divisor = volume_scaled.divisor.abs() + self.epsilon;
        let buy_trade_volume_rel = if volume_divisor > 0.0 {
            bar.buy_trade_volume / volume_divisor
        } else {
            0.0
        };
        let sell_trade_volume_rel = if volume_divisor > 0.0 {
            bar.sell_trade_volume / volume_divisor
        } else {
            0.0
        };
        let trade_imbalance_ratio =
            if bar.buy_trade_volume <= self.epsilon && bar.sell_trade_volume <= self.epsilon {
                0.0
            } else {
                compute_depth_imbalance(bar.buy_trade_volume, bar.sell_trade_volume, self.epsilon)
            };
        let has_buy_trade = if bar.buy_trade_count > 0 { 1.0 } else { 0.0 };
        let has_sell_trade = if bar.sell_trade_count > 0 { 1.0 } else { 0.0 };
        let avg_buy_dist_to_ask = average_distance_ticks(
            bar.buy_distance_to_ask_sum,
            bar.buy_trade_count,
            self.tick_size,
        );
        let avg_sell_dist_to_bid = average_distance_ticks(
            bar.sell_distance_to_bid_sum,
            bar.sell_trade_count,
            self.tick_size,
        );

        let rv_sum = self.rv.push(mid_return);
        let rv_count = self.rv.count();
        let rv_var = if rv_count >= self.rv_min_window {
            (rv_sum / rv_count as f64).max(0.0)
        } else {
            0.0
        };
        let rv_log = (rv_var + self.epsilon).ln();
        let vol_est = rv_var.sqrt();

        let rv_long_sum = self.rv_long.push(mid_return);
        let rv_long_count = self.rv_long.count();
        let rv_long_var = if rv_long_count >= self.rv_long_min_window {
            (rv_long_sum / rv_long_count as f64).max(0.0)
        } else {
            0.0
        };
        let rv_long_log = (rv_long_var + self.epsilon).ln();

        let z_volume =
            self.z_volume_stats
                .zscore_and_observe(bar.trade_volume_sum, self.epsilon, REGIME_MIN_OBSERVATIONS);
        let z_spread = self
            .z_spread_stats
            .zscore_and_observe(spread_close, self.epsilon, REGIME_MIN_OBSERVATIONS);
        let z_volatility = self
            .z_volatility_stats
            .zscore_and_observe(vol_est, self.epsilon, REGIME_MIN_OBSERVATIONS);

        let has_volume = bar.trade_volume_sum > self.epsilon && bar.trade_count > 0;
        let (kyle_lambda_log, amihud_log) = if has_volume {
            let trade_imbalance_volume = bar.buy_trade_volume - bar.sell_trade_volume;
            let kyle_lambda_like =
                mid_return.abs() / (trade_imbalance_volume.abs() + self.epsilon);
            let amihud_like = mid_return.abs() / (bar.trade_volume_sum.abs() + self.epsilon);
            (
                kyle_lambda_like.ln_1p(),
                amihud_like.ln_1p(),
            )
        } else {
            (0.0, 0.0)
        };

        let cum_bid = bar.book.cumulative_bid_size();
        let cum_ask = bar.book.cumulative_ask_size();
        let cum_bid_scaled = self.scaler.normalize_depth_bid(cum_bid);
        let cum_ask_scaled = self.scaler.normalize_depth_ask(cum_ask);
        let imbalance_l = compute_depth_imbalance(cum_bid, cum_ask, self.epsilon);
        let level_bundle = compute_level_features(
            &bar.book,
            mid_close,
            self.tick_size,
            cum_bid_scaled.divisor,
            cum_ask_scaled.divisor,
        );
        let add_bid_scaled = clamp_flow(
            self.scaler
                .normalize_flow_bid(bar.limit_add_bid_volume)
                .relative,
        );
        let add_ask_scaled = clamp_flow(
            self.scaler
                .normalize_flow_ask(bar.limit_add_ask_volume)
                .relative,
        );
        let cancel_bid_scaled = clamp_flow(
            self.scaler
                .normalize_flow_bid(bar.limit_cancel_bid_volume)
                .relative,
        );
        let cancel_ask_scaled = clamp_flow(
            self.scaler
                .normalize_flow_ask(bar.limit_cancel_ask_volume)
                .relative,
        );
        let net_bid = bar.limit_add_bid_volume - bar.limit_cancel_bid_volume;
        let net_ask = bar.limit_add_ask_volume - bar.limit_cancel_ask_volume;
        let limit_of_imbalance =
            clamp_limit_imbalance(compute_depth_imbalance(net_bid, net_ask, self.epsilon));

        let ofi_bid_rel = bar.ofi_bid / (cum_bid_scaled.divisor.abs() + self.epsilon);
        let ofi_ask_rel = bar.ofi_ask / (cum_ask_scaled.divisor.abs() + self.epsilon);
        let ofi_bid_log = signed_log1p(ofi_bid_rel);
        let ofi_ask_log = signed_log1p(ofi_ask_rel);
        let avg_depth =
            0.5 * (cum_bid_scaled.divisor.abs() + cum_ask_scaled.divisor.abs()) + self.epsilon;
        let ofi_net = bar.ofi_bid + bar.ofi_ask;
        let ofi_net_rel = ofi_net / avg_depth;
        let ofi_net_log = signed_log1p(ofi_net_rel);
        let spread_regime = 0.0;
        let depth_regime = 0.0;
        let volume_regime = 0.0;
        let volatility_regime = 0.0;
        let speed_regime = 0.0;
        let (tod_sin, tod_cos, is_us_session) = compute_tod_features(&bar.start);

        Some(CoreFeatureRow {
            key: bar.key,
            start_ns: bar.start.unix_timestamp_nanos() as i64,
            end_ns: bar.end.unix_timestamp_nanos() as i64,
            mid_close_price: mid_close,
            mid_high_price: mid_high,
            mid_low_price: mid_low,
            mid_return_bar: mid_return,
            spread_ticks: spread,
            spread_change_ticks,
            mid_range_rel,
            imbalance_best: imbalance,
            cum_bid_size_l_rel: cum_bid_scaled.relative,
            cum_ask_size_l_rel: cum_ask_scaled.relative,
            imbalance_l,
            trade_volume_sum_rel,
            trade_count_log,
            buy_trade_volume_rel,
            sell_trade_volume_rel,
            trade_imbalance_ratio,
            has_buy_trade,
            has_sell_trade,
            avg_buy_dist_to_ask,
            avg_sell_dist_to_bid,
            rv_log,
            rv_long_log,
            kyle_lambda_log,
            amihud_log,
            z_volume,
            z_spread,
            z_volatility,
            tod_sin,
            tod_cos,
            is_us_session,
            spread_regime,
            depth_regime,
            volume_regime,
            volatility_regime,
            speed_regime,
            bid_offset_level_ticks: level_bundle.bid_offsets,
            ask_offset_level_ticks: level_bundle.ask_offsets,
            bid_size_level_rel: level_bundle.bid_sizes_rel,
            ask_size_level_rel: level_bundle.ask_sizes_rel,
            bid_level_present: level_bundle.bid_presence,
            ask_level_present: level_bundle.ask_presence,
            limit_add_bid_volume_rel: add_bid_scaled,
            limit_add_ask_volume_rel: add_ask_scaled,
            limit_cancel_bid_volume_rel: cancel_bid_scaled,
            limit_cancel_ask_volume_rel: cancel_ask_scaled,
            limit_of_imbalance,
            ofi_bid: ofi_bid_log,
            ofi_ask: ofi_ask_log,
            ofi_net_log,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CoreFeatureRow {
    pub key: BarKey,
    pub start_ns: i64,
    pub end_ns: i64,
    pub mid_close_price: f64,
    pub mid_high_price: f64,
    pub mid_low_price: f64,
    pub mid_return_bar: f64,
    pub spread_ticks: f64,
    pub spread_change_ticks: f64,
    pub mid_range_rel: f64,
    pub imbalance_best: f64,
    pub cum_bid_size_l_rel: f64,
    pub cum_ask_size_l_rel: f64,
    pub imbalance_l: f64,
    pub trade_volume_sum_rel: f64,
    pub trade_count_log: f64,
    pub buy_trade_volume_rel: f64,
    pub sell_trade_volume_rel: f64,
    pub trade_imbalance_ratio: f64,
    pub has_buy_trade: f64,
    pub has_sell_trade: f64,
    pub avg_buy_dist_to_ask: f64,
    pub avg_sell_dist_to_bid: f64,
    pub rv_log: f64,
    pub rv_long_log: f64,
    pub kyle_lambda_log: f64,
    pub amihud_log: f64,
    pub z_volume: f64,
    pub z_spread: f64,
    pub z_volatility: f64,
    pub tod_sin: f64,
    pub tod_cos: f64,
    pub is_us_session: f64,
    pub spread_regime: f64,
    pub depth_regime: f64,
    pub volume_regime: f64,
    pub volatility_regime: f64,
    pub speed_regime: f64,
    pub bid_offset_level_ticks: [f64; LEVEL_FEATURE_COUNT],
    pub ask_offset_level_ticks: [f64; LEVEL_FEATURE_COUNT],
    pub bid_size_level_rel: [f64; LEVEL_FEATURE_COUNT],
    pub ask_size_level_rel: [f64; LEVEL_FEATURE_COUNT],
    pub bid_level_present: [f64; LEVEL_FEATURE_COUNT],
    pub ask_level_present: [f64; LEVEL_FEATURE_COUNT],
    pub limit_add_bid_volume_rel: f64,
    pub limit_add_ask_volume_rel: f64,
    pub limit_cancel_bid_volume_rel: f64,
    pub limit_cancel_ask_volume_rel: f64,
    pub limit_of_imbalance: f64,
    pub ofi_bid: f64,
    pub ofi_ask: f64,
    pub ofi_net_log: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct LevelFeatureBundle {
    bid_offsets: [f64; LEVEL_FEATURE_COUNT],
    ask_offsets: [f64; LEVEL_FEATURE_COUNT],
    bid_sizes_rel: [f64; LEVEL_FEATURE_COUNT],
    ask_sizes_rel: [f64; LEVEL_FEATURE_COUNT],
    bid_presence: [f64; LEVEL_FEATURE_COUNT],
    ask_presence: [f64; LEVEL_FEATURE_COUNT],
}

fn compute_level_features(
    book: &OrderBookSnapshot,
    mid_close: f64,
    tick_size: f64,
    bid_depth_divisor: f64,
    ask_depth_divisor: f64,
) -> LevelFeatureBundle {
    let mut bundle = LevelFeatureBundle::default();
    fill_side_features(
        &book.bids,
        mid_close,
        tick_size,
        bid_depth_divisor,
        true,
        &mut bundle.bid_offsets,
        &mut bundle.bid_sizes_rel,
        &mut bundle.bid_presence,
    );
    fill_side_features(
        &book.asks,
        mid_close,
        tick_size,
        ask_depth_divisor,
        false,
        &mut bundle.ask_offsets,
        &mut bundle.ask_sizes_rel,
        &mut bundle.ask_presence,
    );
    bundle
}

fn fill_side_features(
    levels: &[BookLevel],
    mid_close: f64,
    tick_size: f64,
    depth_divisor: f64,
    is_bid: bool,
    offsets: &mut [f64; LEVEL_FEATURE_COUNT],
    sizes_rel: &mut [f64; LEVEL_FEATURE_COUNT],
    presence: &mut [f64; LEVEL_FEATURE_COUNT],
) {
    let divisor = if depth_divisor.is_finite() && depth_divisor > 0.0 {
        depth_divisor
    } else {
        1.0
    };
    for (idx, level) in levels.iter().take(LEVEL_FEATURE_COUNT).enumerate() {
        if level_is_present(level) {
            offsets[idx] = offset_in_ticks(mid_close, level.price, tick_size, is_bid);
            sizes_rel[idx] = normalize_level_size(level.size, divisor);
            presence[idx] = 1.0;
        } else {
            offsets[idx] = 0.0;
            sizes_rel[idx] = 0.0;
            presence[idx] = 0.0;
        }
    }
}

fn offset_in_ticks(mid_close: f64, price: f64, tick_size: f64, is_bid: bool) -> f64 {
    if !mid_close.is_finite() || !price.is_finite() || tick_size <= 0.0 || price <= 0.0 {
        return 0.0;
    }
    let diff = if is_bid {
        mid_close - price
    } else {
        price - mid_close
    };
    let ticks = diff / tick_size;
    if ticks.is_finite() {
        ticks.clamp(-MAX_LEVEL_OFFSET_TICKS, MAX_LEVEL_OFFSET_TICKS)
    } else {
        0.0
    }
}

fn level_is_present(level: &BookLevel) -> bool {
    level.price.is_finite() && level.price > 0.0 && level.size.is_finite() && level.size > 0.0
}

fn normalize_level_size(size: f64, divisor: f64) -> f64 {
    if !size.is_finite() || size <= 0.0 || !divisor.is_finite() || divisor <= 0.0 {
        0.0
    } else {
        size / divisor
    }
}

fn average_distance_ticks(sum: f64, count: usize, tick_size: f64) -> f64 {
    if count == 0 || !sum.is_finite() || tick_size <= 0.0 {
        0.0
    } else {
        (sum / count as f64) / tick_size
    }
}

fn clamp_flow(value: f64) -> f64 {
    value.max(-FLOW_REL_CLAMP).min(FLOW_REL_CLAMP)
}

fn clamp_limit_imbalance(value: f64) -> f64 {
    value
        .max(-LIMIT_OF_IMBALANCE_CLAMP)
        .min(LIMIT_OF_IMBALANCE_CLAMP)
}

fn clamp_trade_volume(value: f64) -> f64 {
    value
        .max(-TRADE_VOLUME_REL_CLAMP)
        .min(TRADE_VOLUME_REL_CLAMP)
}

fn encode_regime(value: f64, low: f64, high: f64) -> f64 {
    if !value.is_finite() {
        0.0
    } else if value < low {
        -1.0
    } else if value > high {
        1.0
    } else {
        0.0
    }
}

fn assign_percentile_regime<F, G>(rows: &mut [CoreFeatureRow], getter: F, setter: G)
where
    F: Fn(&CoreFeatureRow) -> f64,
    G: Fn(&mut CoreFeatureRow, f64),
{
    let mut samples: Vec<f64> = rows
        .iter()
        .map(|row| getter(row))
        .filter(|value| value.is_finite())
        .collect();
    if samples.len() < 3 {
        return;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let low = percentile(&samples, 0.33);
    let high = percentile(&samples, 0.66);
    if !low.is_finite() || !high.is_finite() {
        return;
    }

    rows.iter_mut().for_each(|row| {
        let value = getter(row);
        let regime = encode_regime(value, low, high);
        setter(row, regime);
    });
}

pub fn apply_regimes(rows: &mut [CoreFeatureRow]) {
    if rows.is_empty() {
        return;
    }
    assign_percentile_regime(
        rows,
        |row| row.spread_ticks,
        |row, regime| row.spread_regime = regime,
    );
    assign_percentile_regime(
        rows,
        |row| row.rv_log,
        |row, regime| row.volatility_regime = regime,
    );
    assign_percentile_regime(
        rows,
        |row| row.cum_bid_size_l_rel + row.cum_ask_size_l_rel,
        |row, regime| row.depth_regime = regime,
    );
    assign_percentile_regime(
        rows,
        |row| row.trade_volume_sum_rel,
        |row, regime| row.volume_regime = regime,
    );
    assign_percentile_regime(
        rows,
        |row| row.trade_count_log,
        |row, regime| row.speed_regime = regime,
    );
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let clamped = pct.clamp(0.0, 1.0);
    let idx = ((sorted.len() - 1) as f64 * clamped).round() as usize;
    sorted[idx]
}

fn compute_tod_features(ts: &OffsetDateTime) -> (f64, f64, f64) {
    let time = ts.time();
    let seconds = (time.hour() as i64) * 3600 + (time.minute() as i64) * 60 + time.second() as i64;
    let fractional = (time.nanosecond() as f64) * 1e-9;
    let seconds_since_open = (seconds - US_SESSION_OPEN_SECONDS) as f64 + fractional;
    let session_length = US_SESSION_LENGTH_SECONDS.max(1.0);
    let normalized = (seconds_since_open / session_length).rem_euclid(1.0);
    let angle = normalized * TAU;
    let tod_sin = angle.sin();
    let tod_cos = angle.cos();
    let weekday = ts.weekday().number_days_from_monday();
    let is_weekday = weekday < 5;
    let in_session =
        is_weekday && seconds >= US_SESSION_OPEN_SECONDS && seconds <= US_SESSION_CLOSE_SECONDS;
    let is_us_session = if in_session { 1.0 } else { 0.0 };
    (tod_sin, tod_cos, is_us_session)
}

pub fn write_core_features_parquet(path: &Path, rows: &[CoreFeatureRow]) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let mut rows_owned: Vec<CoreFeatureRow> = rows.to_vec();
    apply_regimes(&mut rows_owned);
    let mut writer = CoreFeatureWriter::create_file(path, rows.len())?;
    writer.append_rows(rows_owned.into_iter())?;
    writer.finish()
}

pub const DEFAULT_FEATURE_FLUSH_ROWS: usize = 2048;

pub struct CoreFeatureWriter<W: Write + Send> {
    writer: ArrowWriter<W>,
    schema: Arc<Schema>,
    buffer: Vec<CoreFeatureRow>,
    flush_every: usize,
}

impl<W: Write + Send> CoreFeatureWriter<W> {
    pub fn try_new(writer: W, flush_every: usize) -> Result<Self> {
        let schema = core_feature_schema();
        let props = WriterProperties::builder().build();
        let arrow_writer = ArrowWriter::try_new(writer, schema.clone(), Some(props))?;
        let capacity = flush_every.max(1);
        Ok(Self {
            writer: arrow_writer,
            schema,
            buffer: Vec::with_capacity(capacity),
            flush_every: capacity,
        })
    }

    pub fn append_rows<I>(&mut self, rows: I) -> Result<()>
    where
        I: IntoIterator<Item = CoreFeatureRow>,
    {
        for row in rows {
            self.buffer.push(row);
            if self.buffer.len() >= self.flush_every {
                self.flush_buffer()?;
            }
        }
        Ok(())
    }

    fn flush_buffer(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let batch = record_batch_from_rows(&self.schema, &self.buffer)?;
        self.writer.write(&batch)?;
        self.buffer.clear();
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.flush_buffer()?;
        self.writer.close()?;
        Ok(())
    }
}

impl CoreFeatureWriter<fs::File> {
    pub fn create_file(path: &Path, flush_every: usize) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create directories for {}", parent.display())
            })?;
        }
        let file = fs::File::create(path)
            .with_context(|| format!("Failed to create output file {}", path.display()))?;
        Self::try_new(file, flush_every.max(1))
    }
}

fn core_feature_schema() -> Arc<Schema> {
    Arc::new(Schema::new(build_core_feature_fields()))
}

fn build_core_feature_fields() -> Vec<Field> {
    let mut fields = vec![
        Field::new("bar_index", DataType::Int64, false),
        Field::new("start_timestamp_ns", DataType::Int64, false),
        Field::new("end_timestamp_ns", DataType::Int64, false),
        Field::new("mid_close_price", DataType::Float64, false),
        Field::new("mid_high_price", DataType::Float64, false),
        Field::new("mid_low_price", DataType::Float64, false),
        Field::new("mid_return_bar", DataType::Float64, false),
        Field::new("spread_ticks", DataType::Float64, false),
        Field::new("spread_change_ticks", DataType::Float64, false),
        Field::new("mid_range_rel", DataType::Float64, false),
        Field::new("imbalance_best", DataType::Float64, false),
        Field::new("cum_bid_size_l_rel", DataType::Float64, false),
        Field::new("cum_ask_size_l_rel", DataType::Float64, false),
        Field::new("imbalance_l", DataType::Float64, false),
        Field::new("trade_volume_sum_rel", DataType::Float64, false),
        Field::new("trade_count_log", DataType::Float64, false),
        Field::new("buy_trade_volume_rel", DataType::Float64, false),
        Field::new("sell_trade_volume_rel", DataType::Float64, false),
        Field::new("trade_imbalance_ratio", DataType::Float64, false),
        Field::new("has_buy_trade", DataType::Float64, false),
        Field::new("has_sell_trade", DataType::Float64, false),
        Field::new("avg_buy_dist_to_ask", DataType::Float64, false),
        Field::new("avg_sell_dist_to_bid", DataType::Float64, false),
        Field::new("rv_log", DataType::Float64, false),
        Field::new("rv_long_log", DataType::Float64, false),
        Field::new("kyle_lambda_log", DataType::Float64, false),
        Field::new("amihud_log", DataType::Float64, false),
        Field::new("z_volume", DataType::Float64, false),
        Field::new("z_spread", DataType::Float64, false),
        Field::new("z_volatility", DataType::Float64, false),
        Field::new("tod_sin", DataType::Float64, false),
        Field::new("tod_cos", DataType::Float64, false),
        Field::new("is_us_session", DataType::Float64, false),
        Field::new("spread_regime", DataType::Float64, false),
        Field::new("depth_regime", DataType::Float64, false),
        Field::new("volume_regime", DataType::Float64, false),
        Field::new("volatility_regime", DataType::Float64, false),
        Field::new("speed_regime", DataType::Float64, false),
        Field::new("limit_add_bid_volume_rel", DataType::Float64, false),
        Field::new("limit_add_ask_volume_rel", DataType::Float64, false),
        Field::new("limit_cancel_bid_volume_rel", DataType::Float64, false),
        Field::new("limit_cancel_ask_volume_rel", DataType::Float64, false),
        Field::new("limit_of_imbalance", DataType::Float64, false),
        Field::new("ofi_bid", DataType::Float64, false),
        Field::new("ofi_ask", DataType::Float64, false),
        Field::new("ofi_net_log", DataType::Float64, false),
    ];
    for level in 1..=LEVEL_FEATURE_COUNT {
        fields.push(Field::new(
            &format!("bid_offset_level_{}_ticks", level),
            DataType::Float64,
            false,
        ));
    }
    for level in 1..=LEVEL_FEATURE_COUNT {
        fields.push(Field::new(
            &format!("ask_offset_level_{}_ticks", level),
            DataType::Float64,
            false,
        ));
    }
    for level in 1..=LEVEL_FEATURE_COUNT {
        fields.push(Field::new(
            &format!("bid_size_level_{}_rel", level),
            DataType::Float64,
            false,
        ));
    }
    for level in 1..=LEVEL_FEATURE_COUNT {
        fields.push(Field::new(
            &format!("ask_size_level_{}_rel", level),
            DataType::Float64,
            false,
        ));
    }
    for level in 1..=LEVEL_FEATURE_COUNT {
        fields.push(Field::new(
            &format!("bid_level_{}_present", level),
            DataType::Float64,
            false,
        ));
    }
    for level in 1..=LEVEL_FEATURE_COUNT {
        fields.push(Field::new(
            &format!("ask_level_{}_present", level),
            DataType::Float64,
            false,
        ));
    }
    fields
}

fn record_batch_from_rows(schema: &Arc<Schema>, rows: &[CoreFeatureRow]) -> Result<RecordBatch> {
    let columns = build_columns(rows);
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    Ok(batch)
}

fn build_columns(rows: &[CoreFeatureRow]) -> Vec<ArrayRef> {
    let bar_index = Int64Array::from_iter_values(rows.iter().map(|r| r.key.index));
    let start_ns = Int64Array::from_iter_values(rows.iter().map(|r| r.start_ns));
    let end_ns = Int64Array::from_iter_values(rows.iter().map(|r| r.end_ns));
    let mid_close_price = Float64Array::from_iter_values(rows.iter().map(|r| r.mid_close_price));
    let mid_high_price = Float64Array::from_iter_values(rows.iter().map(|r| r.mid_high_price));
    let mid_low_price = Float64Array::from_iter_values(rows.iter().map(|r| r.mid_low_price));
    let mid_return = Float64Array::from_iter_values(rows.iter().map(|r| r.mid_return_bar));
    let spread_ticks = Float64Array::from_iter_values(rows.iter().map(|r| r.spread_ticks));
    let spread_change_ticks =
        Float64Array::from_iter_values(rows.iter().map(|r| r.spread_change_ticks));
    let mid_range_rel = Float64Array::from_iter_values(rows.iter().map(|r| r.mid_range_rel));
    let imbalance = Float64Array::from_iter_values(rows.iter().map(|r| r.imbalance_best));
    let cum_bid_rel = Float64Array::from_iter_values(rows.iter().map(|r| r.cum_bid_size_l_rel));
    let cum_ask_rel = Float64Array::from_iter_values(rows.iter().map(|r| r.cum_ask_size_l_rel));
    let imbalance_l = Float64Array::from_iter_values(rows.iter().map(|r| r.imbalance_l));
    let volume_rel = Float64Array::from_iter_values(rows.iter().map(|r| r.trade_volume_sum_rel));
    let trade_count_log = Float64Array::from_iter_values(rows.iter().map(|r| r.trade_count_log));
    let buy_trade_volume_rel =
        Float64Array::from_iter_values(rows.iter().map(|r| r.buy_trade_volume_rel));
    let sell_trade_volume_rel =
        Float64Array::from_iter_values(rows.iter().map(|r| r.sell_trade_volume_rel));
    let trade_imbalance_ratio =
        Float64Array::from_iter_values(rows.iter().map(|r| r.trade_imbalance_ratio));
    let has_buy_trade = Float64Array::from_iter_values(rows.iter().map(|r| r.has_buy_trade));
    let has_sell_trade = Float64Array::from_iter_values(rows.iter().map(|r| r.has_sell_trade));
    let avg_buy_dist_to_ask =
        Float64Array::from_iter_values(rows.iter().map(|r| r.avg_buy_dist_to_ask));
    let avg_sell_dist_to_bid =
        Float64Array::from_iter_values(rows.iter().map(|r| r.avg_sell_dist_to_bid));
    let rv_log = Float64Array::from_iter_values(rows.iter().map(|r| r.rv_log));
    let rv_long_log = Float64Array::from_iter_values(rows.iter().map(|r| r.rv_long_log));
    let kyle_lambda_log =
        Float64Array::from_iter_values(rows.iter().map(|r| r.kyle_lambda_log));
    let amihud_log = Float64Array::from_iter_values(rows.iter().map(|r| r.amihud_log));
    let z_volume = Float64Array::from_iter_values(rows.iter().map(|r| r.z_volume));
    let z_spread = Float64Array::from_iter_values(rows.iter().map(|r| r.z_spread));
    let z_volatility = Float64Array::from_iter_values(rows.iter().map(|r| r.z_volatility));
    let tod_sin = Float64Array::from_iter_values(rows.iter().map(|r| r.tod_sin));
    let tod_cos = Float64Array::from_iter_values(rows.iter().map(|r| r.tod_cos));
    let is_us_session = Float64Array::from_iter_values(rows.iter().map(|r| r.is_us_session));
    let spread_regime = Float64Array::from_iter_values(rows.iter().map(|r| r.spread_regime));
    let depth_regime = Float64Array::from_iter_values(rows.iter().map(|r| r.depth_regime));
    let volume_regime = Float64Array::from_iter_values(rows.iter().map(|r| r.volume_regime));
    let volatility_regime =
        Float64Array::from_iter_values(rows.iter().map(|r| r.volatility_regime));
    let speed_regime = Float64Array::from_iter_values(rows.iter().map(|r| r.speed_regime));
    let limit_add_bid =
        Float64Array::from_iter_values(rows.iter().map(|r| r.limit_add_bid_volume_rel));
    let limit_add_ask =
        Float64Array::from_iter_values(rows.iter().map(|r| r.limit_add_ask_volume_rel));
    let limit_cancel_bid =
        Float64Array::from_iter_values(rows.iter().map(|r| r.limit_cancel_bid_volume_rel));
    let limit_cancel_ask =
        Float64Array::from_iter_values(rows.iter().map(|r| r.limit_cancel_ask_volume_rel));
    let limit_of_imbalance =
        Float64Array::from_iter_values(rows.iter().map(|r| r.limit_of_imbalance));
    let ofi_bid = Float64Array::from_iter_values(rows.iter().map(|r| r.ofi_bid));
    let ofi_ask = Float64Array::from_iter_values(rows.iter().map(|r| r.ofi_ask));
    let ofi_net_log = Float64Array::from_iter_values(rows.iter().map(|r| r.ofi_net_log));

    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(bar_index) as ArrayRef,
        Arc::new(start_ns),
        Arc::new(end_ns),
        Arc::new(mid_close_price),
        Arc::new(mid_high_price),
        Arc::new(mid_low_price),
        Arc::new(mid_return),
        Arc::new(spread_ticks),
        Arc::new(spread_change_ticks),
        Arc::new(mid_range_rel),
        Arc::new(imbalance),
        Arc::new(cum_bid_rel),
        Arc::new(cum_ask_rel),
        Arc::new(imbalance_l),
        Arc::new(volume_rel),
        Arc::new(trade_count_log),
        Arc::new(buy_trade_volume_rel),
        Arc::new(sell_trade_volume_rel),
        Arc::new(trade_imbalance_ratio),
        Arc::new(has_buy_trade),
        Arc::new(has_sell_trade),
        Arc::new(avg_buy_dist_to_ask),
        Arc::new(avg_sell_dist_to_bid),
        Arc::new(rv_log),
        Arc::new(rv_long_log),
        Arc::new(kyle_lambda_log),
        Arc::new(amihud_log),
        Arc::new(z_volume),
        Arc::new(z_spread),
        Arc::new(z_volatility),
        Arc::new(tod_sin),
        Arc::new(tod_cos),
        Arc::new(is_us_session),
        Arc::new(spread_regime),
        Arc::new(depth_regime),
        Arc::new(volume_regime),
        Arc::new(volatility_regime),
        Arc::new(speed_regime),
        Arc::new(limit_add_bid),
        Arc::new(limit_add_ask),
        Arc::new(limit_cancel_bid),
        Arc::new(limit_cancel_ask),
        Arc::new(limit_of_imbalance),
        Arc::new(ofi_bid),
        Arc::new(ofi_ask),
        Arc::new(ofi_net_log),
    ];

    for level in 0..LEVEL_FEATURE_COUNT {
        let arr =
            Float64Array::from_iter_values(rows.iter().map(|r| r.bid_offset_level_ticks[level]));
        columns.push(Arc::new(arr));
    }
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr =
            Float64Array::from_iter_values(rows.iter().map(|r| r.ask_offset_level_ticks[level]));
        columns.push(Arc::new(arr));
    }
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr = Float64Array::from_iter_values(rows.iter().map(|r| r.bid_size_level_rel[level]));
        columns.push(Arc::new(arr));
    }
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr = Float64Array::from_iter_values(rows.iter().map(|r| r.ask_size_level_rel[level]));
        columns.push(Arc::new(arr));
    }
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr = Float64Array::from_iter_values(rows.iter().map(|r| r.bid_level_present[level]));
        columns.push(Arc::new(arr));
    }
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr = Float64Array::from_iter_values(rows.iter().map(|r| r.ask_level_present[level]));
        columns.push(Arc::new(arr));
    }

    columns
}

fn compute_imbalance(bar: &Bar, epsilon: f64) -> f64 {
    match (bar.best_bid_size_close, bar.best_ask_size_close) {
        (Some(bid), Some(ask)) if bid.is_finite() && ask.is_finite() => {
            compute_depth_imbalance(bid, ask, epsilon)
        }
        _ => 0.0,
    }
}

fn compute_depth_imbalance(bid: f64, ask: f64, epsilon: f64) -> f64 {
    let denom = bid + ask;
    if !denom.is_finite() || denom.abs() < epsilon {
        0.0
    } else {
        (bid - ask) / (denom + epsilon)
    }
}

struct RollingVariance {
    window: usize,
    buffer: std::collections::VecDeque<f64>,
    sum: f64,
}

impl RollingVariance {
    fn new(window: usize) -> Self {
        Self {
            window,
            buffer: std::collections::VecDeque::with_capacity(window),
            sum: 0.0,
        }
    }

    fn push(&mut self, value: f64) -> f64 {
        let squared = value * value;
        self.buffer.push_back(squared);
        self.sum += squared;
        if self.buffer.len() > self.window {
            if let Some(front) = self.buffer.pop_front() {
                self.sum -= front;
            }
        }
        self.sum
    }

    fn count(&self) -> usize {
        self.buffer.len()
    }
}

struct RollingMeanStd {
    window: usize,
    buffer: std::collections::VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
}

impl RollingMeanStd {
    fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            buffer: std::collections::VecDeque::with_capacity(window.max(1)),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    fn stats(&self) -> Option<(f64, f64, usize)> {
        let n = self.buffer.len();
        if n == 0 {
            return None;
        }
        let n_f = n as f64;
        let mean = self.sum / n_f;
        let var = (self.sum_sq / n_f) - mean * mean;
        let std = if var > 0.0 { var.sqrt() } else { 0.0 };
        Some((mean, std, n))
    }

    fn observe(&mut self, value: f64) {
        let v = value;
        self.buffer.push_back(v);
        self.sum += v;
        self.sum_sq += v * v;
        if self.buffer.len() > self.window {
            if let Some(front) = self.buffer.pop_front() {
                self.sum -= front;
                self.sum_sq -= front * front;
            }
        }
    }

    fn zscore_and_observe(&mut self, value: f64, epsilon: f64, min_count: usize) -> f64 {
        let (mean, std, count) = match self.stats() {
            Some(stats) => stats,
            None => {
                self.observe(value);
                return 0.0;
            }
        };
        self.observe(value);
        if count < min_count || !std.is_finite() {
            0.0
        } else {
            let denom = std.max(epsilon);
            let mut z = (value - mean) / denom;
            if z > MAX_Z_SCORE {
                z = MAX_Z_SCORE;
            } else if z < -MAX_Z_SCORE {
                z = -MAX_Z_SCORE;
            }
            z
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{NormalizationConfig, RollingWindowConfig};
    use crate::domain::{Bar, BarKey, BookLevel, OrderBookSnapshot, Resolution};
    use time::macros::datetime;

    fn norm_cfg() -> NormalizationConfig {
        NormalizationConfig {
            rolling_depth: RollingWindowConfig {
                window: 1,
                alpha: None,
            },
            rolling_volume: RollingWindowConfig {
                window: 1,
                alpha: None,
            },
            rolling_count: RollingWindowConfig {
                window: 1,
                alpha: None,
            },
            log_epsilon: 1e-12,
        }
    }

    fn sample_bar(
        index: i64,
        mid_open: f64,
        mid_close: f64,
        spread_open: f64,
        spread_close: f64,
        bid_size: f64,
        ask_size: f64,
        trade_count: usize,
        trade_volume: f64,
    ) -> Bar {
        let mut book = OrderBookSnapshot::empty(LEVEL_FEATURE_COUNT.max(2));
        let half_spread = spread_close * 0.5;
        let bid_level = BookLevel {
            price: mid_close - half_spread,
            size: bid_size,
        };
        let ask_level = BookLevel {
            price: mid_close + half_spread,
            size: ask_size,
        };
        if let Some(level) = book.bids.get_mut(0) {
            *level = bid_level;
        }
        if let Some(level) = book.bids.get_mut(1) {
            *level = BookLevel {
                price: bid_level.price - 0.01,
                size: bid_size * 0.25,
            };
        }
        if let Some(level) = book.bids.get_mut(2) {
            *level = BookLevel {
                price: bid_level.price - 0.02,
                size: bid_size * 0.25,
            };
        }
        if let Some(level) = book.asks.get_mut(0) {
            *level = ask_level;
        }
        if let Some(level) = book.asks.get_mut(1) {
            *level = BookLevel {
                price: ask_level.price + 0.01,
                size: ask_size * 0.25,
            };
        }
        if let Some(level) = book.asks.get_mut(2) {
            *level = BookLevel {
                price: ask_level.price + 0.02,
                size: ask_size * 0.25,
            };
        }
        book.best_bid = bid_level;
        book.best_ask = ask_level;

        Bar {
            key: BarKey::new(Resolution::Fast, index),
            start: datetime!(2025-01-01 00:00:00 UTC),
            end: datetime!(2025-01-01 00:00:01 UTC),
            book,
            event_count: 0,
            mid_open: Some(mid_open),
            mid_close: Some(mid_close),
            mid_high: Some(mid_open.max(mid_close)),
            mid_low: Some(mid_open.min(mid_close)),
            spread_open: Some(spread_open),
            spread_close: Some(spread_close),
            best_bid_size_close: Some(bid_size),
            best_ask_size_close: Some(ask_size),
            trade_count,
            trade_volume_sum: trade_volume,
            trade_volume_max: trade_volume,
            trade_volume_weighted_price: 0.0,
            buy_trade_volume: trade_volume * 0.5,
            sell_trade_volume: trade_volume * 0.5,
            buy_trade_count: trade_count / 2,
            sell_trade_count: trade_count - (trade_count / 2),
            buy_distance_to_ask_sum: 0.0,
            sell_distance_to_bid_sum: 0.0,
            limit_add_bid_volume: bid_size * 0.5,
            limit_add_ask_volume: ask_size * 0.5,
            limit_cancel_bid_volume: bid_size * 0.25,
            limit_cancel_ask_volume: ask_size * 0.25,
            ofi_bid: 0.0,
            ofi_ask: 0.0,
        }
    }

    #[test]
    fn computes_group_a_features_with_normalization() {
        let bars = vec![
            sample_bar(0, 100.0, 101.0, 0.6, 0.4, 4.0, 2.0, 3, 10.0),
            sample_bar(1, 101.0, 100.5, 0.4, 0.5, 3.0, 3.0, 1, 5.0),
        ];
        let cfg = norm_cfg();
        let mut extractor = CoreFeatureExtractor::with_window(0.25, &cfg, 2);
        let rows = extractor.compute(&bars);
        assert_eq!(rows.len(), 2);

        let first = &rows[0];
        let expected_return = (101.0_f64 / 100.0_f64).ln();
        assert!((first.mid_return_bar - expected_return).abs() < 1e-12);
        assert!((first.spread_ticks - (0.4 / 0.25)).abs() < 1e-12);
        assert!((first.spread_change_ticks - ((0.4 - 0.6) / 0.25)).abs() < 1e-12);
        let expected_mid_range = (101.0 - 100.0) / (101.0 + cfg.log_epsilon);
        assert!((first.mid_range_rel - expected_mid_range).abs() < 1e-12);
        let expected_imbalance = (4.0 - 2.0) / (4.0 + 2.0 + cfg.log_epsilon);
        assert!((first.imbalance_best - expected_imbalance).abs() < 1e-12);
        let expected_bid_rel_first = 6.0 / (1.0 + cfg.log_epsilon);
        assert!((first.cum_bid_size_l_rel - expected_bid_rel_first).abs() < 1e-9);
        let expected_ask_rel_first = 3.0 / (1.0 + cfg.log_epsilon);
        assert!((first.cum_ask_size_l_rel - expected_ask_rel_first).abs() < 1e-9);
        let expected_depth_imbalance = (6.0 - 3.0) / (9.0 + cfg.log_epsilon);
        assert!((first.imbalance_l - expected_depth_imbalance).abs() < 1e-12);
        assert!((first.trade_volume_sum_rel - 10.0).abs() < 1e-9);
        assert!((first.bid_offset_level_ticks[0] - 0.8).abs() < 1e-12);
        assert!((first.ask_offset_level_ticks[0] - 0.8).abs() < 1e-12);
        let expected_bid_level_rel = 4.0 / (1.0 + cfg.log_epsilon);
        assert!((first.bid_size_level_rel[0] - expected_bid_level_rel).abs() < 1e-9);
        let expected_ask_level_rel = 2.0 / (1.0 + cfg.log_epsilon);
        assert!((first.ask_size_level_rel[0] - expected_ask_level_rel).abs() < 1e-9);
        assert_eq!(first.bid_level_present[0], 1.0);
        assert_eq!(first.ask_level_present[0], 1.0);
        let expected_add_bid_rel = 2.0 / (1.0 + cfg.log_epsilon);
        assert!((first.limit_add_bid_volume_rel - expected_add_bid_rel).abs() < 1e-9);
        let expected_cancel_ask_rel = 0.5 / (1.0 + cfg.log_epsilon);
        assert!((first.limit_cancel_ask_volume_rel - expected_cancel_ask_rel).abs() < 1e-9);
        let net_bid = 4.0 * 0.25;
        let net_ask = 2.0 * 0.25;
        let expected_imbalance = (net_bid - net_ask) / (net_bid + net_ask + cfg.log_epsilon);
        assert!((first.limit_of_imbalance - expected_imbalance).abs() < 1e-12);

        let second = &rows[1];
        assert!((second.trade_volume_sum_rel - 0.5).abs() < 1e-6);
        assert_eq!(second.trade_count_log, (1_f64).ln_1p());
        let expected_spread_change = (0.5 - 0.4) / 0.25;
        assert!((second.spread_change_ticks - expected_spread_change).abs() < 1e-12);
        let expected_mid_range = (101.0 - 100.5) / (100.5 + cfg.log_epsilon);
        assert!((second.mid_range_rel - expected_mid_range).abs() < 1e-12);
        let expected_bid_rel = 4.5 / (6.0 + cfg.log_epsilon);
        assert!((second.cum_bid_size_l_rel - expected_bid_rel).abs() < 1e-12);
        let expected_ask_rel = 4.5 / (3.0 + cfg.log_epsilon);
        assert!((second.cum_ask_size_l_rel - expected_ask_rel).abs() < 1e-12);
        assert!(second.imbalance_l.abs() < 1e-12);
        assert!(second.rv_log > first.rv_log);
    }

    #[test]
    fn write_core_features_serializes_all_columns() -> Result<()> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use tempfile::tempdir;

        let tmp = tempdir()?;
        let path = tmp.path().join("core.parquet");
        let cfg = norm_cfg();
        let mut extractor = CoreFeatureExtractor::with_window(0.25, &cfg, 2);
        let bars = vec![sample_bar(0, 100.0, 101.0, 0.6, 0.4, 4.0, 2.0, 3, 10.0)];
        let rows = extractor.compute(&bars);
        write_core_features_parquet(&path, &rows)?;
        assert!(path.exists());

        let file = std::fs::File::open(&path)?;
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        let batch = reader.next().expect("batch")?;
        assert_eq!(batch.num_rows(), 1);
        let expected_columns = 46 + (6 * LEVEL_FEATURE_COUNT);
        assert_eq!(batch.num_columns(), expected_columns);
        Ok(())
    }
}
