use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::{ArrayRef, Float64Array, Int8Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::config::PipelineConfig;
use crate::domain::{Label, LabelOutcome, LabelStats, MarketEvent, Resolution};
use crate::io::{EventReader, FileEventReader};

use super::bars::build_bars;
use super::context::PipelineContext;
use super::core_features::{CoreFeatureExtractor, write_core_features_parquet};
use super::labeling::{LabelingEngine, TimeSplitAssigner};

/// High-level orchestrator that will wire event sources, bar builders, and sinks.
pub struct PreprocessingPipeline {
    pub config: PipelineConfig,
    pub ctx: PipelineContext,
    feature_extractors: HashMap<Resolution, CoreFeatureExtractor>,
}

impl PreprocessingPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let ctx = PipelineContext::new(&config);
        let mut feature_extractors = HashMap::new();
        for resolution_cfg in &config.resolutions {
            feature_extractors.insert(
                resolution_cfg.resolution,
                CoreFeatureExtractor::new(config.instrument.tick_size, &config.normalization),
            );
        }
        Self {
            config,
            ctx,
            feature_extractors,
        }
    }

    pub fn run(&mut self) -> Result<()> {
        let input_files = collect_input_files(&self.config.io.input_path)?;
        if input_files.is_empty() {
            println!(
                "No input files discovered under {}",
                self.config.io.input_path.display()
            );
            return Ok(());
        }

        if self.config.dry_run {
            println!(
                "[dry-run] {} -> {} file(s) queued",
                self.ctx.instrument.symbol,
                input_files.len()
            );
        } else {
            println!(
                "Starting preprocessing for {} across {} file(s)",
                self.ctx.instrument.symbol,
                input_files.len()
            );
        }

        self.log_resolution_plan();

        let total = input_files.len();
        for (idx, input_file) in input_files.iter().enumerate() {
            println!(
                "[{}/{}] processing {}",
                idx + 1,
                total,
                input_file.display()
            );

            let events = read_events(input_file)?;
            if events.is_empty() {
                println!(
                    "{} yielded no market events; skipping file",
                    input_file.display()
                );
                continue;
            }

            let label_output = derive_output_path(&self.config.io.feature_output_path, input_file);
            self.run_labeling_for(&events, input_file, &label_output)?;
            self.run_core_features_for(&events, input_file)?;
        }

        if self.config.dry_run {
            println!("Dry-run completed; no files were written.");
        }

        Ok(())
    }

    fn run_labeling_for(
        &self,
        events: &[MarketEvent],
        input_file: &Path,
        output_path: &Path,
    ) -> Result<()> {
        if events.is_empty() {
            println!(
                "Labeling: {} yielded no market events (skipping label computation)",
                input_file.display()
            );
            return Ok(());
        }

        let engine = LabelingEngine::new(
            self.config.labeling.clone(),
            self.config.instrument.tick_size,
        );
        let labels = engine.compute_labels(&events);
        if labels.is_empty() {
            println!(
                "Labeling: {} had no mid-price quotes; labels were not produced",
                input_file.display()
            );
            return Ok(());
        }

        let stats = engine.summarize(&labels);
        log_label_stats(&stats, input_file);

        let splitter = TimeSplitAssigner::new(self.config.data_split.clone());
        let assignments = splitter.assign(&labels);
        let summary = splitter.summary(&assignments);
        println!(
            "Labeling: {} splits -> train={} validation={} test={}",
            input_file.display(),
            summary.train,
            summary.validation,
            summary.test
        );

        if !self.config.dry_run {
            write_labels_parquet(output_path, &labels)?;
            println!(
                "Labeling: {} wrote {} labels -> {}",
                input_file.display(),
                labels.len(),
                output_path.display()
            );
        }

        Ok(())
    }

    fn run_core_features_for(&mut self, events: &[MarketEvent], input_file: &Path) -> Result<()> {
        let resolution_plan = self.config.resolutions.clone();
        for resolution_cfg in resolution_plan {
            let bars = build_bars(events, resolution_cfg.resolution, resolution_cfg.levels);
            if bars.is_empty() {
                println!(
                    "Features: {} [{}] produced no bars",
                    input_file.display(),
                    resolution_cfg.resolution
                );
                continue;
            }

            if bars.iter().all(|bar| bar.trade_count == 0) {
                println!(
                    "Features: {} [{}] observed no trade prints; trade_* features remain 0",
                    input_file.display(),
                    resolution_cfg.resolution
                );
            }

            let extractor = self
                .feature_extractors
                .get_mut(&resolution_cfg.resolution)
                .expect("missing feature extractor for resolution");
            let rows = extractor.compute(&bars);
            if rows.is_empty() {
                println!(
                    "Features: {} [{}] had insufficient data for Phase 1 metrics",
                    input_file.display(),
                    resolution_cfg.resolution
                );
                continue;
            }

            if self.config.dry_run {
                println!(
                    "Features[dry-run]: {} [{}] rows={}",
                    input_file.display(),
                    resolution_cfg.resolution,
                    rows.len()
                );
                continue;
            }

            let output_path = derive_feature_output_path(
                &self.config.io.feature_output_path,
                input_file,
                resolution_cfg.resolution,
            );
            write_core_features_parquet(&output_path, &rows)?;
            println!(
                "Features: {} [{}] wrote {} rows -> {}",
                input_file.display(),
                resolution_cfg.resolution,
                rows.len(),
                output_path.display()
            );
        }

        Ok(())
    }

    fn log_resolution_plan(&self) {
        for resolution in &self.config.resolutions {
            log_stage(
                resolution.resolution,
                resolution.bar_seconds,
                resolution.aggregate_from,
            );
        }
    }
}

fn log_stage(resolution: Resolution, bar_seconds: u64, aggregate_from: Option<Resolution>) {
    match aggregate_from {
        Some(parent) => {
            println!(" - {resolution:?} bars: {bar_seconds}s (aggregating from {parent:?})",)
        }
        None => println!(" - {resolution:?} bars: {bar_seconds}s (direct from events)"),
    }
}

fn log_label_stats(stats: &LabelStats, input_file: &Path) {
    let denom = stats.total.max(1) as f64;
    let up_pct = (stats.hit_up as f64 / denom) * 100.0;
    let down_pct = (stats.hit_down as f64 / denom) * 100.0;
    println!(
        "Labeling: {} labels={} (↑ {:.2}% ↓ {:.2}% no-hit={})",
        input_file.display(),
        stats.total,
        up_pct,
        down_pct,
        stats.no_hit
    );
}

fn write_labels_parquet(path: &Path, labels: &[Label]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directories for {}", parent.display()))?;
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("event_index", DataType::Int64, false),
        Field::new("timestamp_ns", DataType::Int64, false),
        Field::new("anchor_price", DataType::Float64, false),
        Field::new("outcome", DataType::Int8, false),
    ]));

    let event_index = Int64Array::from_iter_values(labels.iter().map(|l| l.event_index as i64));
    let timestamp_ns = Int64Array::from_iter_values(
        labels
            .iter()
            .map(|l| l.timestamp.unix_timestamp_nanos() as i64),
    );
    let anchor_price = Float64Array::from_iter_values(labels.iter().map(|l| l.anchor_price));
    let outcome = Int8Array::from_iter_values(labels.iter().map(|l| encode_outcome(l.outcome)));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(event_index) as ArrayRef,
            Arc::new(timestamp_ns),
            Arc::new(anchor_price),
            Arc::new(outcome),
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

const fn encode_outcome(outcome: LabelOutcome) -> i8 {
    match outcome {
        LabelOutcome::HitUp => 1,
        LabelOutcome::HitDown => 0,
        LabelOutcome::NoHit => -1,
    }
}

fn read_events(path: &Path) -> Result<Vec<MarketEvent>> {
    let mut reader = FileEventReader::new(path)?;
    let mut events = Vec::new();
    while let Some(event) = reader.next_event()? {
        events.push(event);
    }
    Ok(events)
}

fn collect_input_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    if !path.exists() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    let entries = fs::read_dir(path)
        .with_context(|| format!("Failed to read directory {}", path.display()))?;
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_file() {
            files.push(entry.path());
        }
    }

    files.sort();
    Ok(files)
}

fn derive_output_path(output_root: &Path, input_file: &Path) -> PathBuf {
    if output_root.exists() && output_root.is_file() {
        return output_root.to_path_buf();
    }

    if !output_root.exists() && output_root.extension().is_some() {
        return output_root.to_path_buf();
    }

    let stem = input_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("features");
    let mut derived = output_root.join(stem);
    derived.set_extension("parquet");
    derived
}

fn derive_feature_output_path(
    output_root: &Path,
    input_file: &Path,
    resolution: Resolution,
) -> PathBuf {
    if output_root.exists() && output_root.is_file() {
        return output_root.to_path_buf();
    }

    if !output_root.exists() && output_root.extension().is_some() {
        return output_root.to_path_buf();
    }

    let stem = input_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("features");
    let mut derived = output_root.join(resolution.as_str());
    derived.push(stem);
    derived.set_extension("parquet");
    derived
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;
    use tempfile::tempdir;

    #[test]
    fn collect_input_files_sorts_and_filters() -> Result<()> {
        let tmp = tempdir()?;
        let path_a = tmp.path().join("b.csv");
        let path_b = tmp.path().join("a.csv");
        File::create(&path_a)?;
        File::create(&path_b)?;
        let subdir = tmp.path().join("nested");
        fs::create_dir(&subdir)?;
        File::create(subdir.join("ignored.csv"))?;

        let files = collect_input_files(tmp.path())?;
        assert_eq!(files, vec![path_b, path_a]);
        Ok(())
    }

    #[test]
    fn derive_output_path_uses_input_stem() {
        let output_root = PathBuf::from("data/preprocessed/training");
        let derived = derive_output_path(&output_root, Path::new("/tmp/foo.csv"));
        assert_eq!(derived, output_root.join("foo.parquet"));
    }

    #[test]
    fn derive_output_path_preserves_explicit_file() {
        let output_file = PathBuf::from("/tmp/custom.arrow");
        let derived = derive_output_path(&output_file, Path::new("/tmp/foo.csv"));
        assert_eq!(derived, output_file);
    }

    #[test]
    fn derive_feature_output_path_places_resolution_folder() {
        let output_root = PathBuf::from("data/preprocessed/training");
        let derived =
            derive_feature_output_path(&output_root, Path::new("/tmp/foo.csv"), Resolution::Fast);
        assert_eq!(derived, output_root.join("fast").join("foo.parquet"));
    }

    #[test]
    fn derive_feature_output_path_allows_explicit_file() {
        let output_file = PathBuf::from("/tmp/custom.parquet");
        let derived =
            derive_feature_output_path(&output_file, Path::new("/tmp/foo.csv"), Resolution::Fast);
        assert_eq!(derived, output_file);
    }

    #[test]
    fn write_labels_parquet_persists_batches() -> Result<()> {
        let tmp = tempdir()?;
        let path = tmp.path().join("labels.parquet");
        let ts = time::macros::datetime!(2025-01-01 00:00:00 UTC);
        let labels = vec![
            Label::new(0, ts, 100.0, LabelOutcome::HitUp),
            Label::new(
                1,
                ts + time::Duration::seconds(1),
                101.25,
                LabelOutcome::HitDown,
            ),
        ];

        write_labels_parquet(&path, &labels)?;
        assert!(path.exists());

        let file = File::open(&path)?;
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        let batch = reader.next().expect("batch")?;
        assert_eq!(batch.num_rows(), labels.len());

        let event_indices = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let timestamps = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let anchor_prices = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let outcomes = batch
            .column(3)
            .as_any()
            .downcast_ref::<Int8Array>()
            .unwrap();

        for (idx, label) in labels.iter().enumerate() {
            assert_eq!(event_indices.value(idx), label.event_index as i64);
            assert_eq!(
                timestamps.value(idx),
                label.timestamp.unix_timestamp_nanos() as i64
            );
            let price = anchor_prices.value(idx);
            assert!(price.is_finite(), "anchor price must be finite");
            assert!((price - label.anchor_price).abs() < f64::EPSILON);
        }

        assert_eq!(outcomes.value(0), 1);
        assert_eq!(outcomes.value(1), 0);
        Ok(())
    }
}
