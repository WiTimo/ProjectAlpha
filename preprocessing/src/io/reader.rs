use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Lines};
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::domain::events::{QuoteEvent, TradeEvent};
use crate::domain::{MarketEvent, MarketEventKind};
use crate::io::parser::{QuoteState, parse_l1_line, parse_l2_line};

/// Abstract event reader so we can swap CSV, Parquet, Arrow, etc.
pub trait EventReader {
    fn next_event(&mut self) -> Result<Option<MarketEvent>>;
}

/// File-backed reader for the NinjaTrader-style semi-colon separated dumps.
pub struct FileEventReader {
    path: PathBuf,
    lines: Lines<BufReader<File>>,
    quote_state: QuoteState,
    pending: VecDeque<MarketEvent>,
}

impl FileEventReader {
    pub fn new(path: impl Into<PathBuf>, levels: usize) -> Result<Self> {
        let path = path.into();
        let file = File::open(&path)
            .with_context(|| format!("Failed to open input file {}", path.display()))?;
        let reader = BufReader::new(file);
        Ok(Self {
            path,
            lines: reader.lines(),
            quote_state: QuoteState::new(levels.max(1)),
            pending: VecDeque::new(),
        })
    }
}

impl EventReader for FileEventReader {
    fn next_event(&mut self) -> Result<Option<MarketEvent>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }

            let Some(line) = self.lines.next() else {
                return Ok(None);
            };

            let line =
                line.with_context(|| format!("Failed to read line from {}", self.path.display()))?;
            if line.trim().is_empty() {
                continue;
            }

            if let Some(row) = parse_l2_line(&line)? {
                let row_copy = row;
                let result = self.quote_state.update(row);
                for flow in result.flows {
                    self.pending.push_back(MarketEvent {
                        timestamp: row_copy.timestamp,
                        kind: MarketEventKind::OrderFlow(flow),
                    });
                }
                if let Some(snapshot) = result.snapshot {
                    let quote = QuoteEvent {
                        best_bid_price: snapshot.bid.price,
                        best_bid_size: snapshot.bid.size,
                        best_ask_price: snapshot.ask.price,
                        best_ask_size: snapshot.ask.size,
                        mid_price: 0.5 * (snapshot.bid.price + snapshot.ask.price),
                        bids: snapshot.bids,
                        asks: snapshot.asks,
                    };
                    self.pending.push_back(MarketEvent {
                        timestamp: snapshot.timestamp,
                        kind: MarketEventKind::Quote(quote),
                    });
                }
                continue;
            }

            if let Some(trade) = parse_l1_line(&line)? {
                self.pending.push_back(MarketEvent {
                    timestamp: trade.timestamp,
                    kind: MarketEventKind::Trade(TradeEvent {
                        price: trade.price,
                        size: trade.size,
                        aggressor: trade.aggressor,
                    }),
                });
            }
        }
    }
}

// parser implementations moved to crate::io::parser for reuse across batch and realtime pipelines

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::events::OrderFlowOperation;
    use crate::domain::order_book::BookSide;
    use crate::io::parser::parse_l2_line;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn parse_line_skips_non_l2_and_levels() -> Result<()> {
        assert!(parse_l2_line("L1;0;20230101;0;0;0;;1;1")?.is_none());
        let row = parse_l2_line("L2;0;20230101000000;0;0;1;;1;1")?.expect("row");
        assert_eq!(row.level, 1);
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

        let mut reader = FileEventReader::new(file.path(), 1)?;
        let first = reader.next_event()?.expect("expected order flow");
        match first.kind {
            MarketEventKind::OrderFlow(flow) => {
                assert_eq!(flow.side, BookSide::Ask);
                assert_eq!(flow.operation, OrderFlowOperation::Add);
                assert!((flow.size - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected order flow add"),
        }

        let second = reader.next_event()?.expect("expected order flow");
        match second.kind {
            MarketEventKind::OrderFlow(flow) => {
                assert_eq!(flow.side, BookSide::Bid);
                assert_eq!(flow.operation, OrderFlowOperation::Add);
                assert!((flow.size - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected order flow add"),
        }

        let quote_event = reader.next_event()?.expect("expected quote event");
        match quote_event.kind {
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

        let mut reader = FileEventReader::new(file.path(), 1)?;
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
