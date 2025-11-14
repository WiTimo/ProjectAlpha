use crate::config::PipelineConfig;
use crate::domain::{InstrumentId, Session, SessionClock};

/// Shared objects (instrument metadata, clocks, caches) passed to stages.
#[derive(Debug, Clone)]
pub struct PipelineContext {
    pub instrument: InstrumentId,
    pub session_clock: SessionClock,
}

impl PipelineContext {
    pub fn new(config: &PipelineConfig) -> Self {
        let instrument = InstrumentId::new(&config.instrument.symbol, &config.instrument.venue);
        let session_clock = SessionClock::new(Session::default());
        Self {
            instrument,
            session_clock,
        }
    }
}
