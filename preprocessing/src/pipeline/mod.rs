mod context;
mod executor;
pub mod stages;

pub use context::PipelineContext;
pub use executor::PreprocessingPipeline;
pub use stages::{BarStage, EventSourceStage, FeatureStage, SinkStage, StageName};
