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
    pub labeling: LabelConfig,
    pub data_split: DataSplitConfig,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub skip_existing: bool,
    /// Maximum events to hold in memory before flushing (0 = no limit for backward compatibility)
    #[serde(default)]
    pub batch_size: usize,
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
                symbol: "NQ".into(),
                venue: "SIM".into(),
                tick_size: 0.25,
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
            labeling: LabelConfig::example(),
            data_split: DataSplitConfig::example(),
            dry_run: false,
            skip_existing: false,
            batch_size: 35000, // Optimized for 16GB RAM systems (safe default with headroom)
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.data_split.validate()?;
        self.labeling.validate()?;
        Ok(())
    }
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

/// Parameters that control the event-based label logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelConfig {
    #[serde(default = "LabelConfig::default_targets")]
    pub targets: Vec<TargetSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetSpec {
    pub name: String,
    pub up_ticks: f64,
    pub down_ticks: f64,
    #[serde(default = "LabelConfig::default_lookahead_events")]
    pub lookahead_events: usize,
    #[serde(default)]
    pub horizon_seconds: Option<u64>,
}

impl TargetSpec {
    pub fn outcome_column(&self) -> String {
        format!("outcome_{}", self.name)
    }

    pub fn target_column(&self) -> String {
        format!("target_{}", self.name)
    }
}

impl LabelConfig {
    fn example() -> Self {
        Self {
            targets: Self::default_targets(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.targets.is_empty(),
            "At least one label target must be configured"
        );
        for target in &self.targets {
            anyhow::ensure!(target.up_ticks > 0.0, "up_ticks must be > 0");
            anyhow::ensure!(target.down_ticks > 0.0, "down_ticks must be > 0");
            anyhow::ensure!(
                target.lookahead_events > 0 || target.horizon_seconds.is_some(),
                "either lookahead_events or horizon_seconds must be set and positive"
            );
        }
        Ok(())
    }

    pub fn target_names(&self) -> Vec<String> {
        self.targets.iter().map(|t| t.name.clone()).collect()
    }

    fn default_targets() -> Vec<TargetSpec> {
        vec![
            TargetSpec {
                name: "t20".into(),
                up_ticks: 20.0,
                down_ticks: 20.0,
                lookahead_events: 1200,
                horizon_seconds: None,
            },
            TargetSpec {
                name: "t40".into(),
                up_ticks: 40.0,
                down_ticks: 40.0,
                lookahead_events: 2400,
                horizon_seconds: Some(600),
            },
            TargetSpec {
                name: "t60".into(),
                up_ticks: 60.0,
                down_ticks: 60.0,
                lookahead_events: 3600,
                horizon_seconds: Some(600),
            },
            TargetSpec {
                name: "t100".into(),
                up_ticks: 100.0,
                down_ticks: 100.0,
                lookahead_events: 6000,
                horizon_seconds: Some(600),
            },
        ]
    }

    const fn default_lookahead_events() -> usize {
        600
    }
}

/// Ratios (summing to 1.0) that split the timeline into train/val/test segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSplitConfig {
    pub train_ratio: f64,
    pub validation_ratio: f64,
    pub test_ratio: f64,
}

impl DataSplitConfig {
    fn example() -> Self {
        Self {
            train_ratio: 0.7,
            validation_ratio: 0.15,
            test_ratio: 0.15,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let ratios = [self.train_ratio, self.validation_ratio, self.test_ratio];
        for ratio in ratios {
            anyhow::ensure!(ratio >= 0.0, "split ratios cannot be negative");
        }

        let sum: f64 = ratios.iter().sum();
        anyhow::ensure!(
            (sum - 1.0).abs() < 1e-6,
            "split ratios must sum to 1.0 (currently {sum})"
        );
        Ok(())
    }
}
