use anyhow::Result;

use crate::config::PipelineConfig;
use crate::domain::Resolution;

use super::context::PipelineContext;

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

    /// Placeholder run loop that enumerates configured resolutions.
    /// Future work: plug in event reader -> bar builder -> feature computers -> sink.
    pub fn run(&mut self) -> Result<()> {
        if self.config.dry_run {
            println!(
                "[dry-run] {} -> {} resolutions queued",
                self.ctx.instrument.symbol,
                self.config.resolutions.len()
            );
        } else {
            println!(
                "Starting preprocessing for {} ({} levels)",
                self.ctx.instrument.symbol, self.config.instrument.levels
            );
        }

        for resolution in &self.config.resolutions {
            log_stage(
                resolution.resolution,
                resolution.bar_seconds,
                resolution.aggregate_from,
            );
        }

        if self.config.dry_run {
            println!("Dry-run completed; no files were written.");
        }

        Ok(())
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
