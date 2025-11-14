use std::fs::File;
use std::io::{BufRead, BufReader, Lines};
use std::path::PathBuf;

use anyhow::{Context, Result};
use time::{
    Duration, OffsetDateTime, PrimitiveDateTime, format_description::FormatItem,
    macros::format_description,
};

use crate::domain::events::{QuoteEvent, TradeEvent};
use crate::domain::order_book::{BookLevel, BookSide};
use crate::domain::{MarketEvent, MarketEventKind};

/// Abstract event reader so we can swap CSV, Parquet, Arrow, etc.
pub trait EventReader {
    fn next_event(&mut self) -> Result<Option<MarketEvent>>;
}

/// File-backed reader for the NinjaTrader-style semi-colon separated dumps.
pub struct FileEventReader {
    path: PathBuf,
    lines: Lines<BufReader<File>>,
    quote_state: QuoteState,
}

impl FileEventReader {
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let file = File::open(&path)
            .with_context(|| format!("Failed to open input file {}", path.display()))?;
        let reader = BufReader::new(file);
        Ok(Self {
            path,
            lines: reader.lines(),
            quote_state: QuoteState::default(),
        })
    }
}

impl EventReader for FileEventReader {
    fn next_event(&mut self) -> Result<Option<MarketEvent>> {
        while let Some(line) = self.lines.next() {
            let line =
                line.with_context(|| format!("Failed to read line from {}", self.path.display()))?;
            if line.trim().is_empty() {
                continue;
            }

            if let Some(row) = parse_l2_line(&line)? {
                if let Some((bid, ask)) = self.quote_state.update(row) {
                    let quote = QuoteEvent {
                        best_bid_price: bid.price,
                        best_bid_size: bid.size,
                        best_ask_price: ask.price,
                        best_ask_size: ask.size,
                        mid_price: 0.5 * (bid.price + ask.price),
                    };
                    return Ok(Some(MarketEvent {
                        timestamp: row.timestamp,
                        kind: MarketEventKind::Quote(quote),
                    }));
                }
            }

            if let Some(trade) = parse_l1_line(&line)? {
                let event = MarketEvent {
                    timestamp: trade.timestamp,
                    kind: MarketEventKind::Trade(TradeEvent {
                        price: trade.price,
                        size: trade.size,
                        aggressor: trade.aggressor,
                    }),
                };
                return Ok(Some(event));
            }
        }

        Ok(None)
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct QuoteState {
    best_bid: Option<BookLevel>,
    best_ask: Option<BookLevel>,
}

impl QuoteState {
    fn update(&mut self, row: ParsedRow) -> Option<(BookLevel, BookLevel)> {
        match row.side {
            BookSide::Bid => match row.operation {
                L2Operation::Add | L2Operation::Update => {
                    self.best_bid = Some(BookLevel {
                        price: row.price,
                        size: row.size,
                    })
                }
                L2Operation::Remove => self.best_bid = None,
            },
            BookSide::Ask => match row.operation {
                L2Operation::Add | L2Operation::Update => {
                    self.best_ask = Some(BookLevel {
                        price: row.price,
                        size: row.size,
                    })
                }
                L2Operation::Remove => self.best_ask = None,
            },
        }

        match (self.best_bid, self.best_ask) {
            (Some(bid), Some(ask)) => Some((bid, ask)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum L2Operation {
    Add,
    Update,
    Remove,
}

#[derive(Debug, Clone, Copy)]
struct ParsedRow {
    timestamp: OffsetDateTime,
    side: BookSide,
    operation: L2Operation,
    price: f64,
    size: f64,
}

#[derive(Debug, Clone, Copy)]
struct ParsedTrade {
    timestamp: OffsetDateTime,
    price: f64,
    size: f64,
    aggressor: Option<BookSide>,
}

const TIMESTAMP_FORMAT: &[FormatItem<'static>] =
    format_description!("[year][month][day][hour][minute][second]");

fn parse_l2_line(line: &str) -> Result<Option<ParsedRow>> {
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
        .with_context(|| format!("Invalid timestamp '{}' / '{}'", timestamp_raw, offset_raw))?;

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
    if level != 0 {
        return Ok(None);
    }

    // Market maker ID (ignored but consume field)
    let _ = fields.next();

    let price_str = fields.next().unwrap_or_default();
    let size_str = fields.next().unwrap_or_default();
    let price = parse_number(price_str)?;
    let size = parse_number(size_str)?;

    Ok(Some(ParsedRow {
        timestamp,
        side,
        operation,
        price,
        size,
    }))
}

fn parse_l1_line(line: &str) -> Result<Option<ParsedTrade>> {
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
        .with_context(|| format!("Invalid timestamp '{}' / '{}'", timestamp_raw, offset_raw))?;

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
        let nanos = off * 100; // offsets are in 100-nanosecond units
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn parse_line_skips_non_l2_and_levels() -> Result<()> {
        assert!(parse_l2_line("L1;0;20230101;0;0;0;;1;1")?.is_none());
        assert!(parse_l2_line("L2;0;20230101000000;0;0;1;;1;1")?.is_none());
        Ok(())
    }

    #[test]
    fn parse_line_handles_decimal_comma() -> Result<()> {
        let row = parse_l2_line("L2;1;20231221060001;2720000;0;0;;17044,75;1")?.expect("row");
        assert_eq!(row.side, BookSide::Bid);
        assert!((row.price - 17044.75).abs() < 1e-6);
        assert!((row.size - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn reader_emits_quote_once_both_sides_present() -> Result<()> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "L2;0;20231221060001;2720000;0;0;;17048;1")?;
        writeln!(file, "L2;1;20231221060001;2720000;0;0;;17044;1")?;

        let mut reader = FileEventReader::new(file.path())?;
        let event = reader.next_event()?.expect("expected quote event");
        match event.kind {
            MarketEventKind::Quote(quote) => {
                assert_eq!(quote.best_ask_price, 17048.0);
                assert_eq!(quote.best_bid_price, 17044.0);
                assert!((quote.mid_price - 17046.0).abs() < 1e-6);
            }
            _ => panic!("expected quote"),
        }
        Ok(())
    }

    #[test]
    fn reader_emits_trades_from_l1_lines() -> Result<()> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "L1;1;20231221060001;2720000;17045,5;3")?;

        let mut reader = FileEventReader::new(file.path())?;
        let event = reader.next_event()?.expect("expected trade event");
        match event.kind {
            MarketEventKind::Trade(trade) => {
                assert!((trade.price - 17045.5).abs() < 1e-6);
                assert!((trade.size - 3.0).abs() < 1e-6);
                assert_eq!(trade.aggressor, Some(BookSide::Bid));
            }
            _ => panic!("expected trade"),
        }
        Ok(())
    }
}
