use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use crate::config::PipelineConfig;

/// Command-line interface definition for the preprocessing pipeline.
#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Feature engineering pipeline for multi-resolution order-book data."
)]
pub struct Cli {
    /// Optional path to a config file. Defaults to `preprocessing/config.yaml`.
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,
}

impl Cli {
    /// Load pipeline configuration.
    pub fn build_config(&self) -> Result<PipelineConfig> {
        let config_path = self
            .config
            .clone()
            .unwrap_or_else(|| "config.yaml".into());

        let cfg = PipelineConfig::from_path(&config_path)
            .with_context(|| format!("Failed to load config from {}", config_path.display()))?;

        cfg.validate()?;
        Ok(cfg)
    }
}
