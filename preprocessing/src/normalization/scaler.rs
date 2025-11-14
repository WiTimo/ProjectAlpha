use crate::config::{NormalizationConfig, RollingWindowConfig};
use crate::utils::math::signed_log1p;

use super::rolling::{Ewma, RollingStatistic, WindowedMean};

/// Helper struct that keeps depth/volume/count normalizers causal.
#[derive(Debug, Clone)]
pub struct CausalScaler {
    depth_bid: Normalizer,
    depth_ask: Normalizer,
    volume: Normalizer,
    count: Normalizer,
    epsilon: f64,
}

impl CausalScaler {
    pub fn new(cfg: &NormalizationConfig) -> Self {
        Self {
            depth_bid: Normalizer::from_cfg(&cfg.rolling_depth),
            depth_ask: Normalizer::from_cfg(&cfg.rolling_depth),
            volume: Normalizer::from_cfg(&cfg.rolling_volume),
            count: Normalizer::from_cfg(&cfg.rolling_count),
            epsilon: cfg.log_epsilon,
        }
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

    pub fn normalize_volume(&mut self, value: f64) -> ScaledValue {
        self.volume.normalize(value, self.epsilon)
    }

    pub fn normalize_count(&mut self, value: f64) -> ScaledValue {
        self.count.normalize(value, self.epsilon)
    }
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
}

/// Common scaled representation that downstream feature builders can pick from.
#[derive(Debug, Clone, Copy)]
pub struct ScaledValue {
    pub raw: f64,
    pub relative: f64,
    pub log: f64,
    pub divisor: f64,
}
