use anyhow::Result;

use crate::domain::{Bar, FeatureVector, MarketEvent};

/// Identifier trait for debugging/logging.
pub trait StageName {
    fn name(&self) -> &'static str;
}

/// Consumes raw events (from disk, stream, etc.).
pub trait EventSourceStage: StageName {
    fn next_event(&mut self) -> Result<Option<MarketEvent>>;
}

/// Converts events into fully formed bars.
pub trait BarStage: StageName {
    fn on_event(&mut self, event: &MarketEvent) -> Result<Option<Bar>>;
}

/// Computes feature vectors from a completed bar.
pub trait FeatureStage: StageName {
    fn on_bar(&mut self, bar: &Bar) -> Result<Vec<FeatureVector>>;
}

/// Persists feature vectors to disk or elsewhere.
pub trait SinkStage: StageName {
    fn on_feature(&mut self, feature: FeatureVector) -> Result<()>;
}
