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

const DEFAULT_RV_WINDOW: usize = 10;
const LEVEL_FEATURE_COUNT: usize = 3;

pub struct CoreFeatureExtractor {
    tick_size: f64,
    epsilon: f64,
    scaler: CausalScaler,
    rv: RollingVariance,
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
        let trade_count_log = (bar.trade_count as f64).ln_1p();

        let rv_sum = self.rv.push(mid_return);
        let rv_norm = rv_sum / (mid_close * mid_close + self.epsilon);
        let rv_log = (1.0 + rv_norm).ln();

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

        Some(CoreFeatureRow {
            key: bar.key,
            start_ns: bar.start.unix_timestamp_nanos() as i64,
            end_ns: bar.end.unix_timestamp_nanos() as i64,
            mid_return_bar: mid_return,
            spread_ticks: spread,
            spread_change_ticks,
            mid_range_rel,
            imbalance_best: imbalance,
            cum_bid_size_l_rel: cum_bid_scaled.relative,
            cum_ask_size_l_rel: cum_ask_scaled.relative,
            imbalance_l,
            trade_volume_sum_rel: volume_scaled.relative,
            trade_count_log,
            rv_log,
            bid_offset_level_ticks: level_bundle.bid_offsets,
            ask_offset_level_ticks: level_bundle.ask_offsets,
            bid_size_level_rel: level_bundle.bid_sizes_rel,
            ask_size_level_rel: level_bundle.ask_sizes_rel,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CoreFeatureRow {
    pub key: BarKey,
    pub start_ns: i64,
    pub end_ns: i64,
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
    pub rv_log: f64,
    pub bid_offset_level_ticks: [f64; LEVEL_FEATURE_COUNT],
    pub ask_offset_level_ticks: [f64; LEVEL_FEATURE_COUNT],
    pub bid_size_level_rel: [f64; LEVEL_FEATURE_COUNT],
    pub ask_size_level_rel: [f64; LEVEL_FEATURE_COUNT],
}

#[derive(Debug, Clone, Copy, Default)]
struct LevelFeatureBundle {
    bid_offsets: [f64; LEVEL_FEATURE_COUNT],
    ask_offsets: [f64; LEVEL_FEATURE_COUNT],
    bid_sizes_rel: [f64; LEVEL_FEATURE_COUNT],
    ask_sizes_rel: [f64; LEVEL_FEATURE_COUNT],
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
    );
    fill_side_features(
        &book.asks,
        mid_close,
        tick_size,
        ask_depth_divisor,
        false,
        &mut bundle.ask_offsets,
        &mut bundle.ask_sizes_rel,
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
) {
    let divisor = if depth_divisor.is_finite() && depth_divisor > 0.0 {
        depth_divisor
    } else {
        1.0
    };
    for (idx, level) in levels
        .iter()
        .take(LEVEL_FEATURE_COUNT)
        .enumerate()
    {
        offsets[idx] = offset_in_ticks(mid_close, level.price, tick_size, is_bid);
        sizes_rel[idx] = normalize_level_size(level.size, divisor);
    }
}

fn offset_in_ticks(mid_close: f64, price: f64, tick_size: f64, is_bid: bool) -> f64 {
    if !mid_close.is_finite() || !price.is_finite() || tick_size <= 0.0 {
        return 0.0;
    }
    let diff = if is_bid {
        mid_close - price
    } else {
        price - mid_close
    };
    let ticks = diff / tick_size;
    if ticks.is_finite() {
        ticks
    } else {
        0.0
    }
}

fn normalize_level_size(size: f64, divisor: f64) -> f64 {
    if !size.is_finite() || size <= 0.0 || !divisor.is_finite() || divisor <= 0.0 {
        0.0
    } else {
        size / divisor
    }
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
        Field::new("rv_log", DataType::Float64, false),
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
    let schema = Schema::new(fields);
    let schema = std::sync::Arc::new(schema);

    let bar_index = Int64Array::from_iter_values(rows.iter().map(|r| r.key.index));
    let start_ns = Int64Array::from_iter_values(rows.iter().map(|r| r.start_ns));
    let end_ns = Int64Array::from_iter_values(rows.iter().map(|r| r.end_ns));
    let mid_return = Float64Array::from_iter_values(rows.iter().map(|r| r.mid_return_bar));
    let spread_ticks = Float64Array::from_iter_values(rows.iter().map(|r| r.spread_ticks));
    let spread_change_ticks = Float64Array::from_iter_values(rows.iter().map(|r| r.spread_change_ticks));
    let mid_range_rel = Float64Array::from_iter_values(rows.iter().map(|r| r.mid_range_rel));
    let imbalance = Float64Array::from_iter_values(rows.iter().map(|r| r.imbalance_best));
    let cum_bid_rel = Float64Array::from_iter_values(rows.iter().map(|r| r.cum_bid_size_l_rel));
    let cum_ask_rel = Float64Array::from_iter_values(rows.iter().map(|r| r.cum_ask_size_l_rel));
    let imbalance_l = Float64Array::from_iter_values(rows.iter().map(|r| r.imbalance_l));
    let volume_rel = Float64Array::from_iter_values(rows.iter().map(|r| r.trade_volume_sum_rel));
    let trade_count_log = Float64Array::from_iter_values(rows.iter().map(|r| r.trade_count_log));
    let rv_log = Float64Array::from_iter_values(rows.iter().map(|r| r.rv_log));

    let mut columns: Vec<ArrayRef> = vec![
        std::sync::Arc::new(bar_index) as ArrayRef,
        std::sync::Arc::new(start_ns),
        std::sync::Arc::new(end_ns),
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
        std::sync::Arc::new(rv_log),
    ];
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr = Float64Array::from_iter_values(
            rows.iter().map(|r| r.bid_offset_level_ticks[level]),
        );
        columns.push(std::sync::Arc::new(arr));
    }
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr = Float64Array::from_iter_values(
            rows.iter().map(|r| r.ask_offset_level_ticks[level]),
        );
        columns.push(std::sync::Arc::new(arr));
    }
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr = Float64Array::from_iter_values(
            rows.iter().map(|r| r.bid_size_level_rel[level]),
        );
        columns.push(std::sync::Arc::new(arr));
    }
    for level in 0..LEVEL_FEATURE_COUNT {
        let arr = Float64Array::from_iter_values(
            rows.iter().map(|r| r.ask_size_level_rel[level]),
        );
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
        let expected_columns = 14 + (4 * LEVEL_FEATURE_COUNT);
        assert_eq!(batch.num_columns(), expected_columns);
        Ok(())
    }
}
