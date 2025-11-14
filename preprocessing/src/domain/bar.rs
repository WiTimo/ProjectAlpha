use serde::{Deserialize, Serialize};
use time::{Duration, OffsetDateTime};

use super::Resolution;
use super::events::{MarketEvent, MarketEventKind};
use super::order_book::OrderBookSnapshot;

/// Unique identifier for a bar based on resolution + sequential index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BarKey {
    pub resolution: Resolution,
    pub index: i64,
}

impl BarKey {
    pub const fn new(resolution: Resolution, index: i64) -> Self {
        Self { resolution, index }
    }
}

/// Aggregated state we keep per bar (book snapshot + placeholders for stats).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub key: BarKey,
    pub start: OffsetDateTime,
    pub end: OffsetDateTime,
    pub book: OrderBookSnapshot,
    pub event_count: usize,
}

impl Bar {
    pub fn duration(&self) -> Duration {
        self.end - self.start
    }
}

/// Mutable helper that ingests events and turns into a finalized `Bar` later.
#[derive(Debug)]
pub struct BarAccumulator {
    pub key: BarKey,
    pub start: OffsetDateTime,
    pub end: OffsetDateTime,
    pub book: OrderBookSnapshot,
    pub event_count: usize,
}

impl BarAccumulator {
    pub fn new(resolution: Resolution, start: OffsetDateTime, levels: usize) -> Self {
        let span = resolution.bar_duration();
        Self {
            key: BarKey::new(resolution, 0),
            start,
            end: start + span,
            book: OrderBookSnapshot::empty(levels),
            event_count: 0,
        }
    }

    pub fn ingest(&mut self, event: &MarketEvent) {
        self.event_count += 1;
        self.book.touch(event.timestamp);
        match &event.kind {
            MarketEventKind::Quote(_quote) => {
                // TODO: accumulate top-of-book stats.
            }
            MarketEventKind::Trade(_trade) => {
                // TODO: accumulate trade stats.
            }
            MarketEventKind::OrderFlow(_flow) => {
                // TODO: accumulate OFI metrics.
            }
        }
    }

    pub fn finalize(self) -> Bar {
        Bar {
            key: self.key,
            start: self.start,
            end: self.end,
            book: self.book,
            event_count: self.event_count,
        }
    }
}
