use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use arrow_array::{ArrayRef, Float64Array, Int8Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use rayon::prelude::*;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::config::{PipelineConfig, ResolutionConfig, TargetSpec};
use crate::domain::{Label, LabelOutcome, LabelStats, MarketEvent, MarketEventKind, Resolution};
use crate::io::{EventReader, FileEventReader};
use crate::normalization::CausalScalerState;

use super::context::PipelineContext;
use super::core_features::{CoreFeatureRow, CoreFeatureWriter, DEFAULT_FEATURE_FLUSH_ROWS};
use super::labeling::{LabelingEngine, MidPriceAnchor, TimeSplitAssigner};
use super::streaming::StreamingFeatureEngine;

/// High-level orchestrator that will wire event sources, bar builders, and sinks.
pub struct PreprocessingPipeline {
    pub config: PipelineConfig,
    pub ctx: PipelineContext,
}

impl PreprocessingPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let ctx = PipelineContext::new(&config);
        Self { config, ctx }
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
        let flush_rows = if self.config.batch_size > 0 {
            self.config.batch_size
        } else {
            DEFAULT_FEATURE_FLUSH_ROWS
        };
        let shared_config = Arc::new(self.config.clone());
        let counter = AtomicUsize::new(0);

        input_files.into_par_iter().for_each(|input_file| {
            let config = shared_config.clone();
            let idx = counter.fetch_add(1, Ordering::Relaxed);

            if config.skip_existing && all_outputs_exist(&config, &input_file) {
                println!(
                    "[{}/{}] skipping {} (all outputs exist)",
                    idx + 1,
                    total,
                    input_file.display()
                );
                return;
            }

            println!(
                "[{}/{}] processing {}",
                idx + 1,
                total,
                input_file.display()
            );

            if let Err(err) = process_file_streaming(config.as_ref(), &input_file, flush_rows) {
                eprintln!(
                    "[{}/{}] ERROR processing {}: {} (skipping)",
                    idx + 1,
                    total,
                    input_file.display(),
                    err
                );
            }
        });

        if self.config.dry_run {
            println!("Dry-run completed; no files were written.");
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

fn process_file_streaming(
    config: &PipelineConfig,
    input_file: &Path,
    flush_rows: usize,
) -> Result<()> {
    let mut reader = FileEventReader::new(input_file, config.instrument.levels)?;
    let mut label_collector = MidSeriesCollector::new();
    let mut streams = build_resolution_streams(config, input_file, flush_rows)?;
    let mut total_events = 0usize;
    let mut event_index = 0usize;

    while let Some(event) = reader.next_event()? {
        total_events += 1;
        label_collector.ingest(event_index, &event);
        for stream in streams.iter_mut() {
            stream.ingest(&event)?;
        }
        event_index += 1;
    }

    if total_events == 0 {
        println!(
            "{} yielded no market events; skipping file",
            input_file.display()
        );
        return Ok(());
    }

    let mut summaries = Vec::with_capacity(streams.len());
    for stream in streams.into_iter() {
        summaries.push(stream.finish()?);
    }
    for summary in &summaries {
        summary.log(input_file);
    }

    let label_engine = LabelingEngine::new(
        config.labeling.clone(),
        config.instrument.tick_size,
    );
    let mid_series = label_collector.into_series();
    if mid_series.is_empty() {
        println!(
            "Labeling: {} had no mid-price quotes; labels were not produced",
            input_file.display()
        );
        return Ok(());
    }
    let labels = label_engine.compute_from_mid_series(&mid_series);
    if labels.is_empty() {
        println!(
            "Labeling: {} had no mid-price quotes; labels were not produced",
            input_file.display()
        );
        return Ok(());
    }

    let stats = label_engine.summarize(&labels);
    log_label_stats(&stats, input_file);

    let splitter = TimeSplitAssigner::new(config.data_split.clone());
    let assignments = splitter.assign(&labels);
    let summary = splitter.summary(&assignments);
    println!(
        "Labeling: {} splits -> train={} validation={} test={}",
        input_file.display(),
        summary.train,
        summary.validation,
        summary.test
    );

    if config.dry_run {
        println!(
            "Labeling[dry-run]: {} labels={}",
            input_file.display(),
            labels.len()
        );
        return Ok(());
    }

    let label_output = derive_output_path(&config.io.feature_output_path, input_file);
    write_labels_parquet(&label_output, &labels, &config.labeling.targets)?;
    println!(
        "Labeling: {} wrote {} labels -> {}",
        input_file.display(),
        labels.len(),
        label_output.display()
    );

    Ok(())
}

fn build_resolution_streams(
    config: &PipelineConfig,
    input_file: &Path,
    flush_rows: usize,
) -> Result<Vec<ResolutionStream>> {
    config
        .resolutions
        .iter()
        .map(|resolution_cfg| ResolutionStream::new(config, resolution_cfg, input_file, flush_rows))
        .collect()
}

fn all_outputs_exist(config: &PipelineConfig, input_file: &Path) -> bool {
    let label_output = derive_output_path(&config.io.feature_output_path, input_file);
    if !label_output.exists() {
        return false;
    }

    for resolution_cfg in &config.resolutions {
        let feature_output = derive_feature_output_path(
            &config.io.feature_output_path,
            input_file,
            resolution_cfg.resolution,
        );
        if !feature_output.exists() {
            return false;
        }
    }

    true
}

struct ResolutionStream {
    resolution: Resolution,
    engine: StreamingFeatureEngine,
    writer: Option<CoreFeatureWriter<fs::File>>,
    output_path: PathBuf,
    rows_emitted: usize,
    dry_run: bool,
    state_writer: Option<NormalizationStateWriter>,
}

struct NormalizationStateWriter {
    per_file_path: PathBuf,
    latest_path: PathBuf,
}

impl ResolutionStream {
    fn new(
        config: &PipelineConfig,
        resolution_cfg: &ResolutionConfig,
        input_file: &Path,
        flush_rows: usize,
    ) -> Result<Self> {
        let output_path = derive_feature_output_path(
            &config.io.feature_output_path,
            input_file,
            resolution_cfg.resolution,
        );
        let writer = if config.dry_run {
            None
        } else {
            Some(CoreFeatureWriter::create_file(&output_path, flush_rows)?)
        };
        let state_writer = if config.dry_run {
            None
        } else {
            NormalizationStateWriter::new(
                &config.io.checkpoint_path,
                resolution_cfg.resolution,
                input_file,
            )
        };
        let engine = StreamingFeatureEngine::new(
            resolution_cfg.resolution,
            resolution_cfg.levels,
            config.instrument.tick_size,
            &config.normalization,
        );
        Ok(Self {
            resolution: resolution_cfg.resolution,
            engine,
            writer,
            output_path,
            rows_emitted: 0,
            dry_run: config.dry_run,
            state_writer,
        })
    }

    fn ingest(&mut self, event: &MarketEvent) -> Result<()> {
        let rows = self.engine.ingest_event(event);
        self.consume_rows(rows)
    }

    fn finish(mut self) -> Result<FeatureSummary> {
        let rows = self.engine.finish();
        self.consume_rows(rows)?;
        if let Some(writer) = self.writer.take() {
            writer.finish()?;
        }
        if let Some(state_writer) = &self.state_writer {
            let state = self.engine.scaler_state();
            state_writer.persist(&state)?;
        }
        Ok(FeatureSummary {
            resolution: self.resolution,
            rows: self.rows_emitted,
            output_path: if self.dry_run {
                None
            } else {
                Some(self.output_path)
            },
            dry_run: self.dry_run,
        })
    }

    fn consume_rows(&mut self, rows: Vec<CoreFeatureRow>) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        self.rows_emitted += rows.len();
        if let Some(writer) = self.writer.as_mut() {
            writer.append_rows(rows.into_iter())?;
        }
        Ok(())
    }
}

impl NormalizationStateWriter {
    fn new(root: &Path, resolution: Resolution, input_file: &Path) -> Option<Self> {
        let stem = input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("features");
        let resolution_folder = root
            .join("normalization")
            .join(resolution.as_str());
        let per_file_path = resolution_folder.join(format!("{stem}.json"));
        let latest_path = root
            .join("normalization")
            .join("latest")
            .join(format!("{}.json", resolution.as_str()));
        Some(Self {
            per_file_path,
            latest_path,
        })
    }

    fn persist(&self, state: &CausalScalerState) -> Result<()> {
        let payload = serde_json::to_vec_pretty(state)?;
        if let Some(parent) = self.per_file_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create normalization checkpoint directory {}",
                    parent.display()
                )
            })?;
        }
        fs::write(&self.per_file_path, &payload).with_context(|| {
            format!(
                "Failed to persist scaler state to {}",
                self.per_file_path.display()
            )
        })?;

        if let Some(parent) = self.latest_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create latest normalization directory {}",
                    parent.display()
                )
            })?;
        }
        fs::write(&self.latest_path, &payload).with_context(|| {
            format!(
                "Failed to persist latest scaler state to {}",
                self.latest_path.display()
            )
        })?;
        Ok(())
    }
}

struct FeatureSummary {
    resolution: Resolution,
    rows: usize,
    output_path: Option<PathBuf>,
    dry_run: bool,
}

impl FeatureSummary {
    fn log(&self, input_file: &Path) {
        if self.rows == 0 {
            println!(
                "Features: {} [{}] produced no bars",
                input_file.display(),
                self.resolution
            );
        } else if self.dry_run {
            println!(
                "Features[dry-run]: {} [{}] rows={}",
                input_file.display(),
                self.resolution,
                self.rows
            );
        } else if let Some(path) = &self.output_path {
            println!(
                "Features: {} [{}] wrote {} rows -> {}",
                input_file.display(),
                self.resolution,
                self.rows,
                path.display()
            );
        }
    }
}

struct MidSeriesCollector {
    anchors: Vec<MidPriceAnchor>,
}

impl MidSeriesCollector {
    fn new() -> Self {
        Self {
            anchors: Vec::new(),
        }
    }

    fn ingest(&mut self, event_index: usize, event: &MarketEvent) {
        if let MarketEventKind::Quote(quote) = &event.kind {
            if quote.mid_price.is_finite() {
                self.anchors
                    .push(MidPriceAnchor::new(event_index, event.timestamp, quote.mid_price));
            }
        }
    }

    fn into_series(self) -> Vec<MidPriceAnchor> {
        self.anchors
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
    if stats.per_target.is_empty() {
        println!(
            "Labeling: {} produced no label targets",
            input_file.display()
        );
        return;
    }

    println!(
        "Labeling: {} targets={}:",
        input_file.display(),
        stats.per_target.len()
    );
    for target_stats in &stats.per_target {
        let denom = target_stats.total.max(1) as f64;
        let up_pct = (target_stats.hit_up as f64 / denom) * 100.0;
        let down_pct = (target_stats.hit_down as f64 / denom) * 100.0;
        println!(
            "  - {} -> labels={} (↑ {:.2}% ↓ {:.2}% no-hit={})",
            target_stats.name, target_stats.total, up_pct, down_pct, target_stats.no_hit
        );
    }
}

fn write_labels_parquet(path: &Path, labels: &[Label], targets: &[TargetSpec]) -> Result<()> {
    if labels.is_empty() {
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directories for {}", parent.display()))?;
    }

    let mut fields = vec![
        Field::new("event_index", DataType::Int64, false),
        Field::new("timestamp_ns", DataType::Int64, false),
        Field::new("anchor_price", DataType::Float64, false),
    ];
    for target in targets {
        fields.push(Field::new(&target.outcome_column(), DataType::Int8, false));
    }
    let schema = Arc::new(Schema::new(fields));

    let event_index = Int64Array::from_iter_values(labels.iter().map(|l| l.event_index as i64));
    let timestamp_ns = Int64Array::from_iter_values(
        labels
            .iter()
            .map(|l| l.timestamp.unix_timestamp_nanos() as i64),
    );
    let anchor_price = Float64Array::from_iter_values(labels.iter().map(|l| l.anchor_price));
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(event_index) as ArrayRef,
        Arc::new(timestamp_ns),
        Arc::new(anchor_price),
    ];
    for (idx, _) in targets.iter().enumerate() {
        let arr = Int8Array::from_iter_values(labels.iter().map(|label| {
            let outcome = label
                .outcomes
                .get(idx)
                .copied()
                .unwrap_or(LabelOutcome::NoHit);
            encode_outcome(outcome)
        }));
        columns.push(Arc::new(arr) as ArrayRef);
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

const fn encode_outcome(outcome: LabelOutcome) -> i8 {
    match outcome {
        LabelOutcome::HitUp => 1,
        LabelOutcome::HitDown => -1,
        LabelOutcome::NoHit => 0,
    }
}

fn collect_input_files(path: &Path) -> Result<Vec<PathBuf>> {
    // If a single file was provided, keep legacy behavior (allow any extension)
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    if !path.exists() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    collect_recursive(path, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_recursive(dir: &Path, acc: &mut Vec<PathBuf>) -> Result<()> {
    let entries = fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory {}", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_recursive(&path, acc)?;
        } else if file_type.is_file() {
            // Only accept .csv files when discovering within directories
            if path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("csv")).unwrap_or(false) {
                acc.push(path);
            }
        }
    }
    Ok(())
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
    fn collect_input_files_recurses_and_filters() -> Result<()> {
        let tmp = tempdir()?;
        let path_a = tmp.path().join("b.csv");
        let path_b = tmp.path().join("a.csv");
        File::create(&path_a)?;
        File::create(&path_b)?;
        let subdir = tmp.path().join("nested");
        fs::create_dir(&subdir)?;
        let nested_file = subdir.join("ignored.csv");
        File::create(&nested_file)?;

        let files = collect_input_files(tmp.path())?;
        assert_eq!(files, vec![path_b.clone(), path_a.clone(), nested_file.clone()]);
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
        let targets = vec![TargetSpec {
            name: "t20".into(),
            up_ticks: 20.0,
            down_ticks: 20.0,
            lookahead_events: 10,
        }];
        let labels = vec![
            Label::new(0, ts, 100.0, vec![LabelOutcome::HitUp]),
            Label::new(
                1,
                ts + time::Duration::seconds(1),
                101.25,
                vec![LabelOutcome::HitDown],
            ),
        ];

        write_labels_parquet(&path, &labels, &targets)?;
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
        assert_eq!(outcomes.value(1), -1);
        Ok(())
    }
}
