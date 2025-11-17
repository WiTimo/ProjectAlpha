use serde::{Deserialize, Serialize};
use time::{Duration, OffsetDateTime};

use super::Resolution;
use super::events::{
    MarketEvent, MarketEventKind, OrderFlowEvent, OrderFlowOperation, QuoteEvent, TradeEvent,
};
use super::order_book::{BookSide, OrderBookSnapshot};

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
    pub mid_open: Option<f64>,
    pub mid_close: Option<f64>,
    pub mid_high: Option<f64>,
    pub mid_low: Option<f64>,
    pub spread_open: Option<f64>,
    pub spread_close: Option<f64>,
    pub best_bid_size_close: Option<f64>,
    pub best_ask_size_close: Option<f64>,
    pub trade_count: usize,
    pub trade_volume_sum: f64,
    pub trade_volume_max: f64,
    pub trade_volume_weighted_price: f64,
    pub limit_add_bid_volume: f64,
    pub limit_add_ask_volume: f64,
    pub limit_cancel_bid_volume: f64,
    pub limit_cancel_ask_volume: f64,
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
    mid_open: Option<f64>,
    mid_close: Option<f64>,
    mid_high: Option<f64>,
    mid_low: Option<f64>,
    spread_open: Option<f64>,
    spread_close: Option<f64>,
    best_bid_size_close: Option<f64>,
    best_ask_size_close: Option<f64>,
    trade_count: usize,
    trade_volume_sum: f64,
    trade_volume_max: f64,
    trade_volume_weighted_price: f64,
    limit_add_bid_volume: f64,
    limit_add_ask_volume: f64,
    limit_cancel_bid_volume: f64,
    limit_cancel_ask_volume: f64,
}

impl BarAccumulator {
    pub fn new(key: BarKey, start: OffsetDateTime, end: OffsetDateTime, levels: usize) -> Self {
        Self {
            key,
            start,
            end,
            book: OrderBookSnapshot::empty(levels),
            event_count: 0,
            mid_open: None,
            mid_close: None,
            mid_high: None,
            mid_low: None,
            spread_open: None,
            spread_close: None,
            best_bid_size_close: None,
            best_ask_size_close: None,
            trade_count: 0,
            trade_volume_sum: 0.0,
            trade_volume_max: 0.0,
            trade_volume_weighted_price: 0.0,
            limit_add_bid_volume: 0.0,
            limit_add_ask_volume: 0.0,
            limit_cancel_bid_volume: 0.0,
            limit_cancel_ask_volume: 0.0,
        }
    }

    pub fn ingest(&mut self, event: &MarketEvent) {
        self.event_count += 1;
        self.book.touch(event.timestamp);
        match &event.kind {
            MarketEventKind::Quote(quote) => {
                self.update_quote(quote);
            }
            MarketEventKind::Trade(trade) => {
                self.update_trade(trade);
            }
            MarketEventKind::OrderFlow(flow) => {
                self.update_order_flow(flow);
            }
        }
    }

    pub fn finalize(self) -> Option<Bar> {
        if self.event_count == 0 && self.trade_count == 0 && self.mid_open.is_none() {
            return None;
        }

        Some(Bar {
            key: self.key,
            start: self.start,
            end: self.end,
            book: self.book,
            event_count: self.event_count,
            mid_open: self.mid_open,
            mid_close: self.mid_close,
            mid_high: self.mid_high,
            mid_low: self.mid_low,
            spread_open: self.spread_open,
            spread_close: self.spread_close,
            best_bid_size_close: self.best_bid_size_close,
            best_ask_size_close: self.best_ask_size_close,
            trade_count: self.trade_count,
            trade_volume_sum: self.trade_volume_sum,
            trade_volume_max: self.trade_volume_max,
            trade_volume_weighted_price: self.trade_volume_weighted_price,
            limit_add_bid_volume: self.limit_add_bid_volume,
            limit_add_ask_volume: self.limit_add_ask_volume,
            limit_cancel_bid_volume: self.limit_cancel_bid_volume,
            limit_cancel_ask_volume: self.limit_cancel_ask_volume,
        })
    }

    fn update_quote(&mut self, quote: &QuoteEvent) {
        let mid = quote.mid_price;
        if mid.is_finite() && mid > 0.0 {
            if self.mid_open.is_none() {
                self.mid_open = Some(mid);
                self.mid_high = Some(mid);
                self.mid_low = Some(mid);
            } else {
                self.mid_high = Some(self.mid_high.unwrap_or(mid).max(mid));
                self.mid_low = Some(self.mid_low.unwrap_or(mid).min(mid));
            }
            self.mid_close = Some(mid);
        }

        let spread = quote.best_ask_price - quote.best_bid_price;
        if spread.is_finite() && spread >= 0.0 {
            if self.spread_open.is_none() {
                self.spread_open = Some(spread);
            }
            self.spread_close = Some(spread);
        }

        if quote.best_bid_size.is_finite() {
            self.best_bid_size_close = Some(quote.best_bid_size);
        }
        if quote.best_ask_size.is_finite() {
            self.best_ask_size_close = Some(quote.best_ask_size);
        }

        self.book.apply_quote(&quote.bids, &quote.asks);
    }

    fn update_trade(&mut self, trade: &TradeEvent) {
        self.trade_count += 1;
        if trade.size.is_finite() && trade.size >= 0.0 {
            self.trade_volume_sum += trade.size;
            self.trade_volume_max = self.trade_volume_max.max(trade.size);
            if trade.price.is_finite() {
                self.trade_volume_weighted_price += trade.price * trade.size;
            }
        }
    }

    fn update_order_flow(&mut self, flow: &OrderFlowEvent) {
        if !flow.size.is_finite() || flow.size <= 0.0 {
            return;
        }
        let size = flow.size;
        match (flow.operation, flow.side) {
            (OrderFlowOperation::Add, BookSide::Bid) => {
                self.limit_add_bid_volume += size;
            }
            (OrderFlowOperation::Add, BookSide::Ask) => {
                self.limit_add_ask_volume += size;
            }
            (OrderFlowOperation::Cancel, BookSide::Bid)
            | (OrderFlowOperation::Execute, BookSide::Bid) => {
                self.limit_cancel_bid_volume += size;
            }
            (OrderFlowOperation::Cancel, BookSide::Ask)
            | (OrderFlowOperation::Execute, BookSide::Ask) => {
                self.limit_cancel_ask_volume += size;
            }
        }
    }
}

impl Bar {
    pub fn mid_return(&self) -> Option<f64> {
        let open = self.mid_open?;
        let close = self.mid_close?;
        if open > 0.0 && close > 0.0 {
            Some((close / open).ln())
        } else {
            None
        }
    }
}
