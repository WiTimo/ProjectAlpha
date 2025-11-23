use crate::config::{NormalizationConfig, RollingWindowConfig};
use crate::utils::math::signed_log1p;
use serde::{Deserialize, Serialize};

use super::rolling::{Ewma, RollingStatistic, WindowedMean};

/// Helper struct that keeps depth/volume/count normalizers causal.
#[derive(Debug, Clone)]
pub struct CausalScaler {
    depth_bid: Normalizer,
    depth_ask: Normalizer,
    flow_bid: Normalizer,
    flow_ask: Normalizer,
    volume: Normalizer,
    count: Normalizer,
    epsilon: f64,
}

impl CausalScaler {
    pub fn new(cfg: &NormalizationConfig) -> Self {
        Self {
            depth_bid: Normalizer::from_cfg(&cfg.rolling_depth),
            depth_ask: Normalizer::from_cfg(&cfg.rolling_depth),
            flow_bid: Normalizer::from_cfg(&cfg.rolling_depth),
            flow_ask: Normalizer::from_cfg(&cfg.rolling_depth),
            volume: Normalizer::from_cfg(&cfg.rolling_volume),
            count: Normalizer::from_cfg(&cfg.rolling_count),
            epsilon: cfg.log_epsilon,
        }
    }

    pub fn with_state(cfg: &NormalizationConfig, state: Option<&CausalScalerState>) -> Self {
        let mut scaler = Self::new(cfg);
        if let Some(snapshot) = state {
            scaler.apply_state(snapshot);
        }
        scaler
    }

    pub fn snapshot(&self) -> CausalScalerState {
        CausalScalerState {
            depth_bid: self.depth_bid.snapshot(),
            depth_ask: self.depth_ask.snapshot(),
            flow_bid: self.flow_bid.snapshot(),
            flow_ask: self.flow_ask.snapshot(),
            volume: self.volume.snapshot(),
            count: self.count.snapshot(),
        }
    }

    pub fn apply_state(&mut self, state: &CausalScalerState) {
        self.depth_bid.apply_state(&state.depth_bid);
        self.depth_ask.apply_state(&state.depth_ask);
        self.flow_bid.apply_state(&state.flow_bid);
        self.flow_ask.apply_state(&state.flow_ask);
        self.volume.apply_state(&state.volume);
        self.count.apply_state(&state.count);
    }

    pub fn normalize_depth(&mut self, value: f64) -> ScaledValue {
        self.depth_bid.normalize(value, self.epsilon)
    }

    pub fn normalize_depth_bid(&mut self, value: f64) -> ScaledValue {
        self.depth_bid.normalize(value, self.epsilon)
    }

    pub fn normalize_depth_ask(&mut self, value: f64) -> ScaledValue {
        self.depth_ask.normalize(value, self.epsilon)
    }

    pub fn normalize_flow_bid(&mut self, value: f64) -> ScaledValue {
        self.flow_bid.normalize(value, self.epsilon)
    }

    pub fn normalize_flow_ask(&mut self, value: f64) -> ScaledValue {
        self.flow_ask.normalize(value, self.epsilon)
    }

    pub fn normalize_volume(&mut self, value: f64) -> ScaledValue {
        self.volume.normalize(value, self.epsilon)
    }

    pub fn normalize_count(&mut self, value: f64) -> ScaledValue {
        self.count.normalize(value, self.epsilon)
    }
}

/// Serializable snapshot of all causal normalizers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalScalerState {
    pub depth_bid: NormalizerState,
    pub depth_ask: NormalizerState,
    pub flow_bid: NormalizerState,
    pub flow_ask: NormalizerState,
    pub volume: NormalizerState,
    pub count: NormalizerState,
}

#[derive(Debug, Clone)]
struct Normalizer {
    ewma: Option<Ewma>,
    mean: Option<WindowedMean>,
    latest: Option<f64>,
}

impl Normalizer {
    fn from_cfg(cfg: &RollingWindowConfig) -> Self {
        Self {
            ewma: cfg.alpha.map(Ewma::new),
            mean: if cfg.window > 0 {
                Some(WindowedMean::new(cfg.window))
            } else {
                None
            },
            latest: None,
        }
    }

    fn observe(&mut self, value: f64) {
        if let Some(ewma) = &mut self.ewma {
            ewma.observe(value);
        }
        if let Some(mean) = &mut self.mean {
            mean.observe(value);
        }
        self.latest = Some(value);
    }

    fn divisor(&self) -> f64 {
        self.ewma
            .as_ref()
            .and_then(|ewma| ewma.value())
            .or_else(|| self.mean.as_ref().and_then(|m| m.value()))
            .or(self.latest)
            .unwrap_or(1.0)
    }

    fn normalize(&mut self, value: f64, epsilon: f64) -> ScaledValue {
        let divisor = self.divisor().abs() + epsilon;
        self.observe(value);
        let rel = value / divisor;
        let log = signed_log1p(rel);
        ScaledValue {
            raw: value,
            relative: rel,
            log,
            divisor,
        }
    }

    fn snapshot(&self) -> NormalizerState {
        NormalizerState {
            ewma_value: self.ewma.as_ref().and_then(|ewma| ewma.state()),
            mean_values: self.mean.as_ref().map(|mean| mean.values()),
            latest: self.latest,
        }
    }

    fn apply_state(&mut self, state: &NormalizerState) {
        if let Some(ewma) = self.ewma.as_mut() {
            ewma.set_state(state.ewma_value);
        }
        if let Some(mean) = self.mean.as_mut() {
            if let Some(values) = &state.mean_values {
                mean.seed(values);
            } else {
                mean.seed(&[]);
            }
        }
        self.latest = state.latest;
    }
}

/// Serializable representation of a single Normalizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizerState {
    pub ewma_value: Option<f64>,
    pub mean_values: Option<Vec<f64>>,
    pub latest: Option<f64>,
}

/// Common scaled representation that downstream feature builders can pick from.
#[derive(Debug, Clone, Copy)]
pub struct ScaledValue {
    pub raw: f64,
    pub relative: f64,
    pub log: f64,
    pub divisor: f64,
}
