use std::f64::consts::TAU;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::config::NormalizationConfig;
use crate::domain::{Bar, BarKey, BookLevel, OrderBookSnapshot};
use crate::normalization::CausalScaler;
use crate::utils::math::signed_log1p;
use time::OffsetDateTime;

const DEFAULT_RV_WINDOW: usize = 10;
const LEVEL_FEATURE_COUNT: usize = 3;
const MAX_LEVEL_OFFSET_TICKS: f64 = 256.0;
const RV_MIN_OBSERVATIONS: usize = 2;
const FLOW_REL_CLAMP: f64 = 25.0;
const LIMIT_OF_IMBALANCE_CLAMP: f64 = 10.0;
const TRADE_VOLUME_REL_CLAMP: f64 = 15.0;
const US_SESSION_OPEN_SECONDS: i64 = 9 * 3600 + 30 * 60;
const US_SESSION_CLOSE_SECONDS: i64 = 16 * 3600;
const US_SESSION_LENGTH_SECONDS: f64 = (US_SESSION_CLOSE_SECONDS - US_SESSION_OPEN_SECONDS) as f64;
const SPREAD_LOW_TICKS: f64 = 2.0;
const SPREAD_HIGH_TICKS: f64 = 6.0;
const DEPTH_LOW_REL: f64 = 1.0;
const DEPTH_HIGH_REL: f64 = 3.0;
const VOLUME_LOW_REL: f64 = 0.5;
const VOLUME_HIGH_REL: f64 = 1.5;
const VOLATILITY_LOW_ABS: f64 = 0.0005;
const VOLATILITY_HIGH_ABS: f64 = 0.0015;
const SPEED_LOW_TPS: f64 = 0.5;
const SPEED_HIGH_TPS: f64 = 1.5;

pub struct CoreFeatureExtractor {
    tick_size: f64,
    epsilon: f64,
    scaler: CausalScaler,
    rv: RollingVariance,
    rv_min_window: usize,
}

impl CoreFeatureExtractor {
    pub fn new(tick_size: f64, norm_cfg: &NormalizationConfig) -> Self {
        Self::with_window(tick_size, norm_cfg, DEFAULT_RV_WINDOW)
    }

    pub fn with_window(tick_size: f64, norm_cfg: &NormalizationConfig, rv_window: usize) -> Self {
        Self {
            tick_size: tick_size.max(1e-12),
            epsilon: norm_cfg.log_epsilon.max(1e-12),
            scaler: CausalScaler::new(norm_cfg),
            rv: RollingVariance::new(rv_window.max(1)),
            rv_min_window: RV_MIN_OBSERVATIONS,
        }
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
        let speed_tps = bar.trade_count as f64 / duration_secs;
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
        let rv_std = rv_var.sqrt();

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
        let total_depth_rel = cum_bid_scaled.relative + cum_ask_scaled.relative;
        let spread_regime = encode_regime(spread, SPREAD_LOW_TICKS, SPREAD_HIGH_TICKS);
        let depth_regime = encode_regime(total_depth_rel, DEPTH_LOW_REL, DEPTH_HIGH_REL);
        let volume_regime = encode_regime(trade_volume_sum_rel, VOLUME_LOW_REL, VOLUME_HIGH_REL);
        let volatility_regime = encode_regime(rv_std, VOLATILITY_LOW_ABS, VOLATILITY_HIGH_ABS);
        let speed_regime = encode_regime(speed_tps, SPEED_LOW_TPS, SPEED_HIGH_TPS);
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

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directories for {}", parent.display()))?;
    }

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
    let schema = Schema::new(fields);
    let schema = std::sync::Arc::new(schema);

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
        std::sync::Arc::new(bar_index) as ArrayRef,
        std::sync::Arc::new(start_ns),
        std::sync::Arc::new(end_ns),
        std::sync::Arc::new(mid_close_price),
        std::sync::Arc::new(mid_high_price),
        std::sync::Arc::new(mid_low_price),
        std::sync::Arc::new(mid_return),
        std::sync::Arc::new(spread_ticks),
        std::sync::Arc::new(spread_change_ticks),
        std::sync::Arc::new(mid_range_rel),
        std::sync::Arc::new(imbalance),
        std::sync::Arc::new(cum_bid_rel),
        std::sync::Arc::new(cum_ask_rel),
        std::sync::Arc::new(imbalance_l),
        std::sync::Arc::new(volume_rel),
        std::sync::Arc::new(trade_count_log),
        std::sync::Arc::new(buy_trade_volume_rel),
        std::sync::Arc::new(sell_trade_volume_rel),
        std::sync::Arc::new(trade_imbalance_ratio),
        std::sync::Arc::new(has_buy_trade),
        std::sync::Arc::new(has_sell_trade),
        std::sync::Arc::new(avg_buy_dist_to_ask),
        std::sync::Arc::new(avg_sell_dist_to_bid),
        std::sync::Arc::new(rv_log),
        std::sync::Arc::new(tod_sin),
        std::sync::Arc::new(tod_cos),
        std::sync::Arc::new(is_us_session),
        std::sync::Arc::new(spread_regime),
        std::sync::Arc::new(depth_regime),
        std::sync::Arc::new(volume_regime),
        std::sync::Arc::new(volatility_regime),
        std::sync::Arc::new(speed_regime),
        std::sync::Arc::new(limit_add_bid),
        std::sync::Arc::new(limit_add_ask),
        std::sync::Arc::new(limit_cancel_bid),
        std::sync::Arc::new(limit_cancel_ask),
        std::sync::Arc::new(limit_of_imbalance),
        std::sync::Arc::new(ofi_bid),
        std::sync::Arc::new(ofi_ask),
        std::sync::Arc::new(ofi_net_log),
    ];
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr =
            Float64Array::from_iter_values(rows.iter().map(|r| r.bid_offset_level_ticks[level]));
        columns.push(std::sync::Arc::new(arr));
    }
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr =
            Float64Array::from_iter_values(rows.iter().map(|r| r.ask_offset_level_ticks[level]));
        columns.push(std::sync::Arc::new(arr));
    }
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr = Float64Array::from_iter_values(rows.iter().map(|r| r.bid_size_level_rel[level]));
        columns.push(std::sync::Arc::new(arr));
    }
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr = Float64Array::from_iter_values(rows.iter().map(|r| r.ask_size_level_rel[level]));
        columns.push(std::sync::Arc::new(arr));
    }
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr = Float64Array::from_iter_values(rows.iter().map(|r| r.bid_level_present[level]));
        columns.push(std::sync::Arc::new(arr));
    }
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr = Float64Array::from_iter_values(rows.iter().map(|r| r.ask_level_present[level]));
        columns.push(std::sync::Arc::new(arr));
    }

    let batch = RecordBatch::try_new(schema.clone(), columns)?;

    let file = fs::File::create(path)
        .with_context(|| format!("Failed to create output file {}", path.display()))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
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
        let expected_columns = 40 + (6 * LEVEL_FEATURE_COUNT);
        assert_eq!(batch.num_columns(), expected_columns);
        Ok(())
    }
}
