use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use crate::config::{PipelineConfig, PipelineOverrides};
use crate::domain::Resolution;

/// Command-line interface definition for the preprocessing pipeline.
#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Feature engineering pipeline for multi-resolution order-book data."
)]
pub struct Cli {
    /// Optional path to a config file (YAML or JSON). Falls back to a built-in template.
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Override the raw event input path (file or directory) without editing the config file.
    #[arg(long, value_name = "PATH")]
    pub input: Option<PathBuf>,

    /// Override the feature output root (usually a directory) without editing the config file.
    #[arg(long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Comma-separated list of resolutions to process (fast,mid,slow). Default = all.
    #[arg(long, value_delimiter = ',', value_name = "LIST")]
    pub resolutions: Vec<String>,

    /// Run the pipeline without writing files. Useful for smoke-testing configuration.
    #[arg(long)]
    pub dry_run: bool,
}

impl Cli {
    /// Load pipeline configuration and apply CLI overrides.
    pub fn build_config(&self) -> Result<PipelineConfig> {
        let mut cfg = if let Some(path) = &self.config {
            PipelineConfig::from_path(path)
                .with_context(|| format!("Failed to load config from {}", path.display()))?
        } else {
            PipelineConfig::example()
        };

        let overrides = PipelineOverrides {
            input_path: self.input.clone(),
            output_path: self.output.clone(),
            resolutions: self.parse_resolutions()?,
            dry_run: Some(self.dry_run),
        };

        cfg.apply_overrides(overrides);
        cfg.validate()?;
        Ok(cfg)
    }

    fn parse_resolutions(&self) -> Result<Option<Vec<Resolution>>> {
        if self.resolutions.is_empty() {
            return Ok(None);
        }

        let resolutions = self
            .resolutions
            .iter()
            .map(|raw| Resolution::try_from(raw.as_str()))
            .collect::<Result<Vec<_>>>()?;

        Ok(Some(resolutions))
    }
}
