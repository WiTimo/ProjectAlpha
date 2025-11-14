use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::Resolution;
use super::bar::{Bar, BarKey};

/// Logical grouping that mirrors the README sections.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FeatureGroup {
    TopOfBook,
    Depth,
    OrderFlow,
    Trades,
    Volatility,
    Liquidity,
    TimeOfDay,
    Regime,
    CrossResolution,
    Meta,
}

impl FeatureGroup {
    pub fn default_set() -> Vec<Self> {
        use FeatureGroup::*;
        vec![
            TopOfBook,
            Depth,
            OrderFlow,
            Trades,
            Volatility,
            Liquidity,
            TimeOfDay,
            Regime,
            CrossResolution,
            Meta,
        ]
    }
}

/// Sparse feature vector keyed by snake_case names.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub group: FeatureGroup,
    pub bar_key: BarKey,
    pub resolution: Resolution,
    pub values: BTreeMap<String, f64>,
}

impl FeatureVector {
    pub fn new(group: FeatureGroup, bar: &Bar) -> Self {
        Self {
            group,
            bar_key: bar.key,
            resolution: bar.key.resolution,
            values: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, name: impl Into<String>, value: f64) {
        self.values.insert(name.into(), value);
    }
}

/// Trait implemented by every feature computer.
pub trait FeatureComputer {
    fn group(&self) -> FeatureGroup;
    fn compute(&mut self, bar: &Bar) -> FeatureVector;
}
