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

    pub fn apply_quote(&mut self, bids: &[BookLevel], asks: &[BookLevel]) {
        for (dst, src) in self.bids.iter_mut().zip(bids.iter().chain(std::iter::repeat(&BookLevel::default()))) {
            *dst = *src;
        }
        for (dst, src) in self.asks.iter_mut().zip(asks.iter().chain(std::iter::repeat(&BookLevel::default()))) {
            *dst = *src;
        }
        if let Some(first_bid) = bids.first() {
            self.best_bid = *first_bid;
        }
        if let Some(first_ask) = asks.first() {
            self.best_ask = *first_ask;
        }
    }

    pub fn cumulative_bid_size(&self) -> f64 {
        Self::sum_depth(&self.bids)
    }

    pub fn cumulative_ask_size(&self) -> f64 {
        Self::sum_depth(&self.asks)
    }

    fn sum_depth(levels: &[BookLevel]) -> f64 {
        levels
            .iter()
            .map(|lvl| if lvl.size.is_finite() && lvl.size > 0.0 { lvl.size } else { 0.0 })
            .sum()
    }
}
