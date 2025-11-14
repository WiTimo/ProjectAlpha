use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::config::NormalizationConfig;
use crate::domain::{Bar, BarKey};
use crate::normalization::CausalScaler;

const DEFAULT_RV_WINDOW: usize = 10;

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
        let spread = bar.spread_close? / self.tick_size;
        if !spread.is_finite() {
            return None;
        }

        let imbalance = compute_imbalance(bar, self.epsilon);
        let volume_scaled = self.scaler.normalize_volume(bar.trade_volume_sum);
        let trade_count_log = (bar.trade_count as f64).ln_1p();

        let mid_close = bar.mid_close?;
        if mid_close <= 0.0 {
            return None;
        }
        let rv_sum = self.rv.push(mid_return);
        let rv_norm = rv_sum / (mid_close * mid_close + self.epsilon);
        let rv_log = (1.0 + rv_norm).ln();

        Some(CoreFeatureRow {
            key: bar.key,
            start_ns: bar.start.unix_timestamp_nanos() as i64,
            end_ns: bar.end.unix_timestamp_nanos() as i64,
            mid_return_bar: mid_return,
            spread_ticks: spread,
            imbalance_best: imbalance,
            trade_volume_sum_rel: volume_scaled.relative,
            trade_count_log,
            rv_log,
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
    pub imbalance_best: f64,
    pub trade_volume_sum_rel: f64,
    pub trade_count_log: f64,
    pub rv_log: f64,
}

pub fn write_core_features_parquet(path: &Path, rows: &[CoreFeatureRow]) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directories for {}", parent.display()))?;
    }

    let schema = Schema::new(vec![
        Field::new("bar_index", DataType::Int64, false),
        Field::new("start_timestamp_ns", DataType::Int64, false),
        Field::new("end_timestamp_ns", DataType::Int64, false),
        Field::new("mid_return_bar", DataType::Float64, false),
        Field::new("spread_ticks", DataType::Float64, false),
        Field::new("imbalance_best", DataType::Float64, false),
        Field::new("trade_volume_sum_rel", DataType::Float64, false),
        Field::new("trade_count_log", DataType::Float64, false),
        Field::new("rv_log", DataType::Float64, false),
    ]);
    let schema = std::sync::Arc::new(schema);

    let bar_index = Int64Array::from_iter_values(rows.iter().map(|r| r.key.index));
    let start_ns = Int64Array::from_iter_values(rows.iter().map(|r| r.start_ns));
    let end_ns = Int64Array::from_iter_values(rows.iter().map(|r| r.end_ns));
    let mid_return = Float64Array::from_iter_values(rows.iter().map(|r| r.mid_return_bar));
    let spread_ticks = Float64Array::from_iter_values(rows.iter().map(|r| r.spread_ticks));
    let imbalance = Float64Array::from_iter_values(rows.iter().map(|r| r.imbalance_best));
    let volume_rel = Float64Array::from_iter_values(rows.iter().map(|r| r.trade_volume_sum_rel));
    let trade_count_log = Float64Array::from_iter_values(rows.iter().map(|r| r.trade_count_log));
    let rv_log = Float64Array::from_iter_values(rows.iter().map(|r| r.rv_log));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            std::sync::Arc::new(bar_index) as ArrayRef,
            std::sync::Arc::new(start_ns),
            std::sync::Arc::new(end_ns),
            std::sync::Arc::new(mid_return),
            std::sync::Arc::new(spread_ticks),
            std::sync::Arc::new(imbalance),
            std::sync::Arc::new(volume_rel),
            std::sync::Arc::new(trade_count_log),
            std::sync::Arc::new(rv_log),
        ],
    )?;

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
            let denom = bid + ask;
            if denom.abs() < epsilon {
                0.0
            } else {
                (bid - ask) / (denom + epsilon)
            }
        }
        _ => 0.0,
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
    use crate::domain::{Bar, BarKey, OrderBookSnapshot, Resolution};
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
        spread: f64,
        bid_size: f64,
        ask_size: f64,
        trade_count: usize,
        trade_volume: f64,
    ) -> Bar {
        Bar {
            key: BarKey::new(Resolution::Fast, index),
            start: datetime!(2025-01-01 00:00:00 UTC),
            end: datetime!(2025-01-01 00:00:01 UTC),
            book: OrderBookSnapshot::empty(1),
            event_count: 0,
            mid_open: Some(mid_open),
            mid_close: Some(mid_close),
            mid_high: Some(mid_open.max(mid_close)),
            mid_low: Some(mid_open.min(mid_close)),
            spread_open: Some(spread),
            spread_close: Some(spread),
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
            sample_bar(0, 100.0, 101.0, 0.5, 4.0, 2.0, 3, 10.0),
            sample_bar(1, 101.0, 100.5, 0.4, 3.0, 3.0, 1, 5.0),
        ];
        let cfg = norm_cfg();
        let mut extractor = CoreFeatureExtractor::with_window(0.25, &cfg, 2);
        let rows = extractor.compute(&bars);
        assert_eq!(rows.len(), 2);

        let first = &rows[0];
        let expected_return = (101.0_f64 / 100.0_f64).ln();
        assert!((first.mid_return_bar - expected_return).abs() < 1e-12);
        assert!((first.spread_ticks - (0.5 / 0.25)).abs() < 1e-12);
        let expected_imbalance = (4.0 - 2.0) / (4.0 + 2.0 + cfg.log_epsilon);
        assert!((first.imbalance_best - expected_imbalance).abs() < 1e-12);
        assert!((first.trade_volume_sum_rel - 10.0).abs() < 1e-9);

        let second = &rows[1];
        assert!((second.trade_volume_sum_rel - 0.5).abs() < 1e-6);
        assert_eq!(second.trade_count_log, (1_f64).ln_1p());
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
        let bars = vec![sample_bar(0, 100.0, 101.0, 0.5, 4.0, 2.0, 3, 10.0)];
        let rows = extractor.compute(&bars);
        write_core_features_parquet(&path, &rows)?;
        assert!(path.exists());

        let file = std::fs::File::open(&path)?;
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        let batch = reader.next().expect("batch")?;
        assert_eq!(batch.num_rows(), 1);
        Ok(())
    }
}
