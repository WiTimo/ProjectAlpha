use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;

use preprocessing::config::PipelineConfig;
use preprocessing::domain::Resolution;
use preprocessing::normalization::CausalScalerState;
use preprocessing::realtime::{RealtimeConfig, RealtimePreprocessor, ResolutionPlanEntry};

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Tail NinjaTrader L2 logs and emit realtime Phase 5 features"
)]
struct RealtimeArgs {
    /// Optional pipeline config (YAML/JSON) to mirror training parameters.
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// One or more resolutions to emit (comma separated, e.g. fast,mid,slow).
    #[arg(long = "resolutions", value_delimiter = ',', default_value = "fast,mid,slow")]
    resolutions: Vec<Resolution>,

    /// Source log file path (defaults to STDIN when omitted).
    #[arg(long, value_name = "FILE", default_value = "C:/Ninjatrader/L2Log.txt")]
    source: PathBuf,

    /// JSONL output path that the inference watcher will tail.
    #[arg(
        long,
        value_name = "FILE",
        default_value = "runs/realtime/features.jsonl"
    )]
    emit: PathBuf,

    /// Start tailing from the end (true) or from the beginning (false).
    #[arg(long, default_value_t = true)]
    follow: bool,

    /// Poll interval in milliseconds when no new data arrives.
    #[arg(long, default_value_t = 50)]
    poll_ms: u64,

    /// Override tick size without editing config file.
    #[arg(long, value_name = "FLOAT")]
    tick_size: Option<f64>,

    /// Override depth levels without editing config file.
    #[arg(long, value_name = "INT")]
    levels: Option<usize>,

    /// Number of initial base bars to discard for normalization warmup.
    #[arg(long, value_name = "INT", default_value_t = 0)]
    norm_warmup: usize,

    /// Directory containing scaler_state JSON files (e.g. fast.json, mid.json).
    #[arg(long, value_name = "DIR")]
    norm_state_dir: Option<PathBuf>,

    /// Gap in seconds that triggers a new session (flushes multi-resolution caches).
    #[arg(long, value_name = "INT", default_value_t = 900)]
    session_gap_secs: i64,
}

fn main() -> Result<()> {
    let args = RealtimeArgs::parse();
    let pipeline_cfg = load_pipeline_config(&args)?;
    let realtime_cfg = build_realtime_config(&args, &pipeline_cfg)?;
    let mut preprocessor = RealtimePreprocessor::new(&realtime_cfg);
    if let Some(path) = realtime_cfg.source_path.as_deref() {
        preprocessor.run_with_file(&realtime_cfg, path)
    } else {
        preprocessor.run_with_stdin(&realtime_cfg)
    }
}

fn load_pipeline_config(args: &RealtimeArgs) -> Result<PipelineConfig> {
    if let Some(path) = &args.config {
        PipelineConfig::from_path(path)
            .with_context(|| format!("Failed to load config from {}", path.display()))
    } else {
        Ok(PipelineConfig::example())
    }
}

fn build_realtime_config(args: &RealtimeArgs, base: &PipelineConfig) -> Result<RealtimeConfig> {
    let mut cfg = base.clone();
    if let Some(tick) = args.tick_size {
        cfg.instrument.tick_size = tick.max(1e-12);
    }
    if let Some(levels) = args.levels {
        cfg.instrument.levels = levels.max(1);
    }

    let mut requested = Vec::new();
    for res in &args.resolutions {
        if !requested.contains(res) {
            requested.push(*res);
        }
    }
    if requested.is_empty() {
        requested.push(Resolution::Fast);
    }

    let mut plan = Vec::new();
    for res in requested {
        let res_cfg = cfg
            .resolutions
            .iter()
            .find(|r| r.resolution == res)
            .cloned()
            .with_context(|| format!("Resolution {:?} missing from config", res))?;
        let levels = args.levels.unwrap_or(res_cfg.levels).max(1);
        plan.push(ResolutionPlanEntry {
            resolution: res_cfg.resolution,
            levels,
        });
    }

    let normalization_state = if let Some(dir) = args.norm_state_dir.as_deref() {
        load_normalization_states(dir, &plan)?
    } else {
        HashMap::new()
    };
    let session_reset_gap_ns = (args.session_gap_secs.max(0) as i128) * 1_000_000_000i128;

    Ok(RealtimeConfig {
        source_path: Some(args.source.clone()),
        start_from_end: args.follow,
        poll_interval: Duration::from_millis(args.poll_ms),
        emit_path: Some(args.emit.clone()),
        plan,
        tick_size: cfg.instrument.tick_size,
        normalization: cfg.normalization.clone(),
        warmup_bars: args.norm_warmup,
        normalization_state,
        session_reset_gap_ns,
    })
}

fn load_normalization_states(
    dir: &Path,
    plan: &[ResolutionPlanEntry],
) -> Result<HashMap<Resolution, CausalScalerState>> {
    let mut states = HashMap::new();
    for entry in plan {
        let file = dir.join(format!("{}.json", entry.resolution.as_str()));
        if !file.exists() {
            continue;
        }
        let contents = fs::read_to_string(&file)
            .with_context(|| format!("Failed to read scaler state from {}", file.display()))?;
        let state: CausalScalerState = serde_json::from_str(&contents)
            .with_context(|| format!("Invalid scaler state JSON in {}", file.display()))?;
        states.insert(entry.resolution, state);
    }
    Ok(states)
}
