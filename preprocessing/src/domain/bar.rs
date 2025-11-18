use serde::{Deserialize, Serialize};
use time::{Duration, OffsetDateTime};

use super::Resolution;
use super::events::{
    MarketEvent, MarketEventKind, OrderFlowEvent, OrderFlowOperation, QuoteEvent, TradeEvent,
};
use super::order_book::{BookLevel, BookSide, OrderBookSnapshot};

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
    pub buy_trade_volume: f64,
    pub sell_trade_volume: f64,
    pub buy_trade_count: usize,
    pub sell_trade_count: usize,
    pub buy_distance_to_ask_sum: f64,
    pub sell_distance_to_bid_sum: f64,
    pub limit_add_bid_volume: f64,
    pub limit_add_ask_volume: f64,
    pub limit_cancel_bid_volume: f64,
    pub limit_cancel_ask_volume: f64,
    pub ofi_bid: f64,
    pub ofi_ask: f64,
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
    buy_trade_volume: f64,
    sell_trade_volume: f64,
    buy_trade_count: usize,
    sell_trade_count: usize,
    buy_distance_to_ask_sum: f64,
    sell_distance_to_bid_sum: f64,
    limit_add_bid_volume: f64,
    limit_add_ask_volume: f64,
    limit_cancel_bid_volume: f64,
    limit_cancel_ask_volume: f64,
    ofi_bid: f64,
    ofi_ask: f64,
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
            buy_trade_volume: 0.0,
            sell_trade_volume: 0.0,
            buy_trade_count: 0,
            sell_trade_count: 0,
            buy_distance_to_ask_sum: 0.0,
            sell_distance_to_bid_sum: 0.0,
            limit_add_bid_volume: 0.0,
            limit_add_ask_volume: 0.0,
            limit_cancel_bid_volume: 0.0,
            limit_cancel_ask_volume: 0.0,
            ofi_bid: 0.0,
            ofi_ask: 0.0,
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
            buy_trade_volume: self.buy_trade_volume,
            sell_trade_volume: self.sell_trade_volume,
            buy_trade_count: self.buy_trade_count,
            sell_trade_count: self.sell_trade_count,
            buy_distance_to_ask_sum: self.buy_distance_to_ask_sum,
            sell_distance_to_bid_sum: self.sell_distance_to_bid_sum,
            limit_add_bid_volume: self.limit_add_bid_volume,
            limit_add_ask_volume: self.limit_add_ask_volume,
            limit_cancel_bid_volume: self.limit_cancel_bid_volume,
            limit_cancel_ask_volume: self.limit_cancel_ask_volume,
            ofi_bid: self.ofi_bid,
            ofi_ask: self.ofi_ask,
        })
    }

    fn update_quote(&mut self, quote: &QuoteEvent) {
        let mid = quote.mid_price;
        self.accumulate_ofi(quote);
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
            self.update_aggressor_stats(trade);
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

    fn accumulate_ofi(&mut self, quote: &QuoteEvent) {
        let prev_bid = self.book.best_bid;
        let prev_ask = self.book.best_ask;
        self.ofi_bid += ofi_bid_contribution(prev_bid, quote.best_bid_price, quote.best_bid_size);
        self.ofi_ask += ofi_ask_contribution(prev_ask, quote.best_ask_price, quote.best_ask_size);
    }

    fn update_aggressor_stats(&mut self, trade: &TradeEvent) {
        let size = trade.size;
        if !size.is_finite() || size <= 0.0 {
            return;
        }
        match trade.aggressor {
            Some(BookSide::Bid) => {
                self.buy_trade_count += 1;
                self.buy_trade_volume += size;
                if let Some(dist) = self.distance_to_ask(trade.price) {
                    self.buy_distance_to_ask_sum += dist;
                }
            }
            Some(BookSide::Ask) => {
                self.sell_trade_count += 1;
                self.sell_trade_volume += size;
                if let Some(dist) = self.distance_to_bid(trade.price) {
                    self.sell_distance_to_bid_sum += dist;
                }
            }
            None => {}
        }
    }

    fn distance_to_ask(&self, trade_price: f64) -> Option<f64> {
        let ask = self.book.best_ask.price;
        if !ask.is_finite() || ask <= 0.0 || !trade_price.is_finite() || trade_price <= 0.0 {
            return None;
        }
        Some((ask - trade_price).max(0.0))
    }

    fn distance_to_bid(&self, trade_price: f64) -> Option<f64> {
        let bid = self.book.best_bid.price;
        if !bid.is_finite() || bid <= 0.0 || !trade_price.is_finite() || trade_price <= 0.0 {
            return None;
        }
        Some((trade_price - bid).max(0.0))
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

fn ofi_bid_contribution(prev: BookLevel, new_price: f64, new_size: f64) -> f64 {
    if !new_price.is_finite() || new_price <= 0.0 || !new_size.is_finite() || new_size <= 0.0 {
        return 0.0;
    }
    if !prev.price.is_finite() || prev.price <= 0.0 || !prev.size.is_finite() || prev.size <= 0.0 {
        return 0.0;
    }
    if new_price > prev.price {
        new_size
    } else if new_price < prev.price {
        -prev.size
    } else {
        new_size - prev.size
    }
}

fn ofi_ask_contribution(prev: BookLevel, new_price: f64, new_size: f64) -> f64 {
    if !new_price.is_finite() || new_price <= 0.0 || !new_size.is_finite() || new_size <= 0.0 {
        return 0.0;
    }
    if !prev.price.is_finite() || prev.price <= 0.0 || !prev.size.is_finite() || prev.size <= 0.0 {
        return 0.0;
    }
    if new_price < prev.price {
        -new_size
    } else if new_price > prev.price {
        prev.size
    } else {
        prev.size - new_size
    }
}
