use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use super::order_book::{BookLevel, BookSide};

/// Canonical market data event processed by the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketEvent {
    pub timestamp: OffsetDateTime,
    pub kind: MarketEventKind,
}

/// Different payloads we care about (quotes, trades, L2 operations).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MarketEventKind {
    Quote(QuoteEvent),
    Trade(TradeEvent),
    OrderFlow(OrderFlowEvent),
}

/// Quote/book-top change recorded at an event timestamp.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuoteEvent {
    pub best_bid_price: f64,
    pub best_bid_size: f64,
    pub best_ask_price: f64,
    pub best_ask_size: f64,
    pub mid_price: f64,
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
}

/// Trade print enriched with simple aggressor tagging.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TradeEvent {
    pub price: f64,
    pub size: f64,
    pub aggressor: Option<BookSide>,
}

/// L2 add/cancel/remove event aggregated per bar later.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowEvent {
    pub side: BookSide,
    pub operation: OrderFlowOperation,
    pub size: f64,
    pub price_level_index: Option<usize>,
}

/// Operation types we treat specially for OFI computations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderFlowOperation {
    Add,
    Cancel,
    Execute,
}
