use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

/// Bid/ask side marker used for book- and flow-related structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BookSide {
    Bid,
    Ask,
}

impl BookSide {
    pub const fn opposite(self) -> Self {
        match self {
            BookSide::Bid => BookSide::Ask,
            BookSide::Ask => BookSide::Bid,
        }
    }
}

/// Single price level on either side of the limit order book.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct BookLevel {
    pub price: f64,
    pub size: f64,
}

/// Snapshot of the top L levels per side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub last_update: Option<OffsetDateTime>,
    pub best_bid: BookLevel,
    pub best_ask: BookLevel,
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
}

impl OrderBookSnapshot {
    pub fn empty(levels: usize) -> Self {
        Self {
            last_update: None,
            best_bid: BookLevel::default(),
            best_ask: BookLevel::default(),
            bids: vec![BookLevel::default(); levels],
            asks: vec![BookLevel::default(); levels],
        }
    }

    /// Placeholder update method to be implemented with actual book logic later.
    pub fn touch(&mut self, timestamp: OffsetDateTime) {
        self.last_update = Some(timestamp);
    }
}
