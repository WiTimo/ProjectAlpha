use crate::domain::{Bar, BarAccumulator, BarKey, MarketEvent, Resolution};
use crate::utils::time::align_to_resolution;

/// Build chronological bars for a given resolution using the raw market events.
pub fn build_bars(events: &[MarketEvent], resolution: Resolution, levels: usize) -> Vec<Bar> {
    if events.is_empty() {
        return Vec::new();
    }

    let mut bars = Vec::new();
    let mut period_start = align_to_resolution(events[0].timestamp, resolution);
    let mut period_end = period_start + resolution.bar_duration();
    let mut index: i64 = 0;
    let mut accumulator = BarAccumulator::new(
        BarKey::new(resolution, index),
        period_start,
        period_end,
        levels,
    );

    for event in events {
        while event.timestamp >= accumulator.end {
            let next_start = accumulator.end;
            if let Some(bar) = accumulator.finalize() {
                bars.push(bar);
            }
            index += 1;
            period_start = next_start;
            period_end = period_start + resolution.bar_duration();
            accumulator = BarAccumulator::new(
                BarKey::new(resolution, index),
                period_start,
                period_end,
                levels,
            );
            if event.timestamp < accumulator.end {
                break;
            }
        }

        accumulator.ingest(event);
    }

    if let Some(bar) = accumulator.finalize() {
        bars.push(bar);
    }

    bars
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::events::{MarketEvent, MarketEventKind, QuoteEvent, TradeEvent};
    use crate::domain::order_book::BookSide;
    use time::macros::datetime;

    fn quote_event(
        ts: time::OffsetDateTime,
        bid: f64,
        ask: f64,
        bid_size: f64,
        ask_size: f64,
    ) -> MarketEvent {
        MarketEvent {
            timestamp: ts,
            kind: MarketEventKind::Quote(QuoteEvent {
                best_bid_price: bid,
                best_bid_size: bid_size,
                best_ask_price: ask,
                best_ask_size: ask_size,
                mid_price: 0.5 * (bid + ask),
            }),
        }
    }

    fn trade_event(ts: time::OffsetDateTime, price: f64, size: f64) -> MarketEvent {
        MarketEvent {
            timestamp: ts,
            kind: MarketEventKind::Trade(TradeEvent {
                price,
                size,
                aggressor: Some(BookSide::Bid),
            }),
        }
    }

    #[test]
    fn build_bars_tracks_mid_and_trades() {
        let events = vec![
            quote_event(datetime!(2025-01-01 00:00:00 UTC), 100.0, 100.5, 2.0, 3.0),
            trade_event(datetime!(2025-01-01 00:00:00 UTC), 100.2, 1.0),
            quote_event(
                datetime!(2025-01-01 00:00:00.500 UTC),
                100.1,
                100.6,
                1.5,
                2.5,
            ),
            quote_event(datetime!(2025-01-01 00:00:01 UTC), 100.2, 100.7, 1.0, 2.0),
            trade_event(datetime!(2025-01-01 00:00:01.100 UTC), 100.3, 2.0),
        ];

        let bars = build_bars(&events, Resolution::Fast, 1);
        assert_eq!(bars.len(), 2);
        let first = &bars[0];
        assert!(first.mid_open.is_some());
        assert!(first.mid_close.is_some());
        assert_eq!(first.trade_count, 1);
        assert!((first.trade_volume_sum - 1.0).abs() < 1e-6);
        assert!(first.spread_close.is_some());

        let second = &bars[1];
        assert_eq!(second.trade_count, 1);
        assert!(second.best_bid_size_close.is_some());
        assert!(second.best_ask_size_close.is_some());
    }
}
