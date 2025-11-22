use anyhow::{Context, Result};
use time::{
    Duration, OffsetDateTime, PrimitiveDateTime, format_description::FormatItem,
    macros::format_description,
};

use crate::domain::events::{OrderFlowEvent, OrderFlowOperation};
use crate::domain::order_book::{BookLevel, BookSide};

const TIMESTAMP_FORMAT: &[FormatItem<'static>] =
    format_description!("[year][month][day][hour][minute][second]");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2Operation {
    Add,
    Update,
    Remove,
}

#[derive(Debug, Clone, Copy)]
pub struct ParsedRow {
    pub timestamp: OffsetDateTime,
    pub side: BookSide,
    pub operation: L2Operation,
    pub level: usize,
    pub price: f64,
    pub size: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct ParsedTrade {
    pub timestamp: OffsetDateTime,
    pub price: f64,
    pub size: f64,
    pub aggressor: Option<BookSide>,
}

#[derive(Debug, Clone)]
pub struct QuoteSnapshot {
    pub timestamp: OffsetDateTime,
    pub bid: BookLevel,
    pub ask: BookLevel,
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
}

pub struct QuoteUpdateResult {
    pub flows: Vec<OrderFlowEvent>,
    pub snapshot: Option<QuoteSnapshot>,
}

#[derive(Debug, Clone)]
pub struct QuoteState {
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
}

impl QuoteState {
    pub fn new(levels: usize) -> Self {
        Self {
            bids: vec![BookLevel::default(); levels],
            asks: vec![BookLevel::default(); levels],
        }
    }

    pub fn update(&mut self, row: ParsedRow) -> QuoteUpdateResult {
        let flows = self.diff_flow(&row);
        self.apply(row);
        let snapshot = self.snapshot(row.timestamp);
        QuoteUpdateResult { flows, snapshot }
    }

    fn apply(&mut self, row: ParsedRow) {
        let levels = match row.side {
            BookSide::Bid => &mut self.bids,
            BookSide::Ask => &mut self.asks,
        };
        if row.level >= levels.len() {
            return;
        }
        match row.operation {
            L2Operation::Add | L2Operation::Update => {
                levels[row.level] = BookLevel {
                    price: row.price,
                    size: row.size,
                };
            }
            L2Operation::Remove => {
                levels[row.level] = BookLevel::default();
            }
        }
    }

    fn snapshot(&self, timestamp: OffsetDateTime) -> Option<QuoteSnapshot> {
        let bid = self.bids.get(0).copied();
        let ask = self.asks.get(0).copied();
        match (bid, ask) {
            (Some(b), Some(a)) if is_valid_level(b) && is_valid_level(a) => Some(QuoteSnapshot {
                timestamp,
                bid: b,
                ask: a,
                bids: self.bids.clone(),
                asks: self.asks.clone(),
            }),
            _ => None,
        }
    }

    fn diff_flow(&self, row: &ParsedRow) -> Vec<OrderFlowEvent> {
        let mut events = Vec::new();
        let levels = match row.side {
            BookSide::Bid => &self.bids,
            BookSide::Ask => &self.asks,
        };
        if row.level >= levels.len() {
            return events;
        }

        let prev = levels[row.level];
        match row.operation {
            L2Operation::Add => push_flow_event(
                &mut events,
                row.side,
                OrderFlowOperation::Add,
                row.size,
                row.level,
            ),
            L2Operation::Update => {
                if price_changed(prev.price, row.price) {
                    push_flow_event(
                        &mut events,
                        row.side,
                        OrderFlowOperation::Cancel,
                        prev.size,
                        row.level,
                    );
                    push_flow_event(
                        &mut events,
                        row.side,
                        OrderFlowOperation::Add,
                        row.size,
                        row.level,
                    );
                } else {
                    let delta = row.size - prev.size;
                    if delta > 0.0 {
                        push_flow_event(
                            &mut events,
                            row.side,
                            OrderFlowOperation::Add,
                            delta,
                            row.level,
                        );
                    } else if delta < 0.0 {
                        push_flow_event(
                            &mut events,
                            row.side,
                            OrderFlowOperation::Cancel,
                            -delta,
                            row.level,
                        );
                    }
                }
            }
            L2Operation::Remove => {
                push_flow_event(
                    &mut events,
                    row.side,
                    OrderFlowOperation::Cancel,
                    prev.size,
                    row.level,
                );
            }
        }
        events
    }
}

pub fn parse_l2_line(line: &str) -> Result<Option<ParsedRow>> {
    let mut fields = line.split(';');
    let kind = fields.next().unwrap_or_default().trim();
    if kind != "L2" {
        return Ok(None);
    }

    let side = match fields.next().unwrap_or_default().trim() {
        "0" => BookSide::Ask,
        "1" => BookSide::Bid,
        _ => return Ok(None),
    };

    let timestamp_raw = fields.next().unwrap_or_default().trim();
    let offset_raw = fields.next().unwrap_or_default().trim();
    let timestamp = parse_timestamp(timestamp_raw, offset_raw)
        .with_context(|| format!("Invalid timestamp '{timestamp_raw}' / '{offset_raw}'"))?;

    let operation = match fields.next().unwrap_or_default().trim() {
        "0" => L2Operation::Add,
        "1" => L2Operation::Update,
        "2" => L2Operation::Remove,
        _ => L2Operation::Update,
    };

    let level: usize = fields
        .next()
        .unwrap_or_default()
        .trim()
        .parse()
        .unwrap_or(usize::MAX);

    // Market maker ID (ignored)
    let _ = fields.next();

    let price_str = fields.next().unwrap_or_default();
    let size_str = fields.next().unwrap_or_default();
    let price = parse_number(price_str)?;
    let size = parse_number(size_str)?;

    Ok(Some(ParsedRow {
        timestamp,
        side,
        operation,
        level,
        price,
        size,
    }))
}

pub fn parse_l1_line(line: &str) -> Result<Option<ParsedTrade>> {
    let mut fields = line.split(';');
    let kind = fields.next().unwrap_or_default().trim();
    if kind != "L1" {
        return Ok(None);
    }

    let aggressor = match fields.next().unwrap_or_default().trim() {
        "0" => Some(BookSide::Ask),
        "1" => Some(BookSide::Bid),
        _ => None,
    };

    let timestamp_raw = fields.next().unwrap_or_default().trim();
    let offset_raw = fields.next().unwrap_or_default().trim();
    let timestamp = parse_timestamp(timestamp_raw, offset_raw)
        .with_context(|| format!("Invalid timestamp '{timestamp_raw}' / '{offset_raw}'"))?;

    let price_str = fields.next().unwrap_or_default();
    let size_str = fields.next().unwrap_or_default();
    let price = parse_number(price_str)?;
    let size = parse_number(size_str)?;

    Ok(Some(ParsedTrade {
        timestamp,
        price,
        size,
        aggressor,
    }))
}

fn parse_timestamp(ts: &str, offset: &str) -> Result<OffsetDateTime> {
    let base = PrimitiveDateTime::parse(ts, TIMESTAMP_FORMAT)?;
    let mut datetime = base.assume_utc();
    if !offset.is_empty() {
        let off: i64 = offset.parse::<i64>()?;
        let nanos = off * 100;
        datetime += Duration::nanoseconds(nanos);
    }
    Ok(datetime)
}

fn parse_number(raw: &str) -> Result<f64> {
    let normalized = raw.trim().replace(',', ".");
    let value = if normalized.is_empty() {
        0.0
    } else {
        normalized.parse::<f64>()?
    };
    Ok(value)
}

fn is_valid_level(level: BookLevel) -> bool {
    level.price.is_finite() && level.size.is_finite() && level.price > 0.0 && level.size > 0.0
}

fn price_changed(a: f64, b: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return true;
    }
    (a - b).abs() > 1e-9
}

fn push_flow_event(
    events: &mut Vec<OrderFlowEvent>,
    side: BookSide,
    operation: OrderFlowOperation,
    size: f64,
    level: usize,
) {
    if !size.is_finite() || size <= 0.0 {
        return;
    }
    events.push(OrderFlowEvent {
        side,
        operation,
        size,
        price_level_index: Some(level),
    });
}
