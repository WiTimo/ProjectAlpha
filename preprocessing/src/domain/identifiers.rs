use serde::{Deserialize, Serialize};
use time::{Duration, OffsetDateTime, Time, macros::time};

/// Unique identifier for a tradable instrument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentId {
    pub symbol: String,
    pub venue: String,
}

impl InstrumentId {
    pub fn new(symbol: impl Into<String>, venue: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            venue: venue.into(),
        }
    }
}

/// Trading session descriptor used for time-of-day features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub open_time: Time,
    pub close_time: Time,
    pub timezone_offset: Duration,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            open_time: time!(09:30:00),
            close_time: time!(16:00:00),
            timezone_offset: Duration::ZERO,
        }
    }
}

/// Utility helper for deriving intra-day timestamps.
#[derive(Debug, Clone)]
pub struct SessionClock {
    pub session: Session,
}

impl SessionClock {
    pub fn new(session: Session) -> Self {
        Self { session }
    }

    pub fn seconds_since_open(&self, ts: OffsetDateTime) -> i64 {
        let open =
            ts.date().with_time(self.session.open_time).assume_utc() + self.session.timezone_offset;
        (ts - open).whole_seconds().max(0)
    }

    pub fn seconds_until_close(&self, ts: OffsetDateTime) -> i64 {
        let close = ts.date().with_time(self.session.close_time).assume_utc()
            + self.session.timezone_offset;
        (close - ts).whole_seconds().max(0)
    }
}
