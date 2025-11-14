use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::domain::{Resolution, features::FeatureGroup};

/// Top-level pipeline configuration loaded from disk or synthesized from defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub instrument: InstrumentConfig,
    pub io: IoConfig,
    pub resolutions: Vec<ResolutionConfig>,
    pub normalization: NormalizationConfig,
    #[serde(default)]
    pub dry_run: bool,
}

impl PipelineConfig {
    /// Load configuration from a JSON or YAML file, inferred from the extension.
    pub fn from_path(path: &Path) -> Result<Self> {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file {}", path.display()))?;

        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        match ext {
            "json" => serde_json::from_str(&contents)
                .with_context(|| format!("Invalid JSON in {}", path.display())),
            "yaml" | "yml" => serde_yaml::from_str(&contents)
                .with_context(|| format!("Invalid YAML in {}", path.display())),
            _ => serde_yaml::from_str(&contents)
                .or_else(|_| serde_json::from_str(&contents))
                .with_context(|| format!("Failed to parse {} as YAML or JSON", path.display())),
        }
    }

    /// Built-in configuration that mirrors the documentation tables.
    pub fn example() -> Self {
        Self {
            instrument: InstrumentConfig {
                symbol: "FAKE_INSTRUMENT".into(),
                venue: "SIM".into(),
                tick_size: 0.01,
                levels: 5,
            },
            io: IoConfig {
                input_path: PathBuf::from("data/raw/training/TODO"),
                feature_output_path: PathBuf::from("data/preprocessed/training/TODO"),
                checkpoint_path: PathBuf::from("data/preprocessed/training/.checkpoints"),
            },
            resolutions: vec![
                ResolutionConfig::new(Resolution::Fast, 1, None),
                ResolutionConfig::new(Resolution::Mid, 10, Some(Resolution::Fast)),
                ResolutionConfig::new(Resolution::Slow, 60, Some(Resolution::Mid)),
            ],
            normalization: NormalizationConfig::example(),
            dry_run: false,
        }
    }

    /// Apply runtime overrides without mutating on-disk configuration files.
    pub fn apply_overrides(&mut self, overrides: PipelineOverrides) {
        if let Some(input) = overrides.input_path {
            self.io.input_path = input;
        }

        if let Some(output) = overrides.output_path {
            self.io.feature_output_path = output;
        }

        if let Some(filter) = overrides.resolutions {
            self.resolutions
                .retain(|cfg| filter.contains(&cfg.resolution));
        }

        if let Some(dry_run) = overrides.dry_run {
            self.dry_run = dry_run;
        }
    }
}

/// In-memory overrides derived from CLI flags.
#[derive(Debug, Default, Clone)]
pub struct PipelineOverrides {
    pub input_path: Option<PathBuf>,
    pub output_path: Option<PathBuf>,
    pub resolutions: Option<Vec<Resolution>>,
    pub dry_run: Option<bool>,
}

/// Static instrument metadata that feeds normalization logic (tick size etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentConfig {
    pub symbol: String,
    pub venue: String,
    pub tick_size: f64,
    pub levels: usize,
}

/// IO locations for raw inputs, feature outputs, and checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoConfig {
    pub input_path: PathBuf,
    pub feature_output_path: PathBuf,
    pub checkpoint_path: PathBuf,
}

/// Per-resolution settings including aggregation method and feature coverage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionConfig {
    pub resolution: Resolution,
    pub bar_seconds: u64,
    pub aggregate_from: Option<Resolution>,
    pub levels: usize,
    pub feature_groups: Vec<FeatureGroup>,
    #[serde(default = "default_persist_raw_bars")]
    pub persist_raw_bars: bool,
}

impl ResolutionConfig {
    pub fn new(
        resolution: Resolution,
        bar_seconds: u64,
        aggregate_from: Option<Resolution>,
    ) -> Self {
        Self {
            resolution,
            bar_seconds,
            aggregate_from,
            levels: 5,
            feature_groups: FeatureGroup::default_set(),
            persist_raw_bars: false,
        }
    }
}

const fn default_persist_raw_bars() -> bool {
    false
}

/// Rolling-window hyper-parameters for depth/volume/count normalizers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    pub rolling_depth: RollingWindowConfig,
    pub rolling_volume: RollingWindowConfig,
    pub rolling_count: RollingWindowConfig,
    pub log_epsilon: f64,
}

impl NormalizationConfig {
    fn example() -> Self {
        Self {
            rolling_depth: RollingWindowConfig {
                window: 600,
                alpha: Some(0.02),
            },
            rolling_volume: RollingWindowConfig {
                window: 600,
                alpha: Some(0.02),
            },
            rolling_count: RollingWindowConfig {
                window: 600,
                alpha: Some(0.02),
            },
            log_epsilon: 1e-12,
        }
    }
}

/// Generic description of a rolling statistic used for causal scaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingWindowConfig {
    pub window: usize,
    pub alpha: Option<f64>,
}
