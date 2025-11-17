use std::collections::BTreeMap;

use ordered_float::OrderedFloat;
use time::OffsetDateTime;

use crate::config::{DataSplitConfig, LabelConfig};
use crate::domain::{Label, LabelOutcome, LabelStats, MarketEvent, MarketEventKind};

/// Event-based label calculator that produces hit-up / hit-down outcomes and splits.
pub struct LabelingEngine {
    params: LabelConfig,
    tick_size: f64,
}

impl LabelingEngine {
    pub fn new(params: LabelConfig, tick_size: f64) -> Self {
        Self { params, tick_size }
    }

    /// Convert a chronologically ordered stream of market events into labels.
    pub fn compute_labels(&self, events: &[MarketEvent]) -> Vec<Label> {
        let mut mid_series: Vec<(usize, LabelAnchor)> = Vec::with_capacity(events.len());
        let mut last_mid = None;

        for (idx, event) in events.iter().enumerate() {
            match &event.kind {
                MarketEventKind::Quote(quote) if quote.mid_price.is_finite() => {
                    last_mid = Some(quote.mid_price);
                }
                _ => {}
            }

            if let Some(mid) = last_mid {
                mid_series.push((
                    idx,
                    LabelAnchor {
                        timestamp: event.timestamp,
                        price: mid,
                    },
                ));
            }
        }

        if mid_series.is_empty() {
            return Vec::new();
        }

        let up_delta = self.params.up_ticks * self.tick_size;
        let down_delta = self.params.down_ticks * self.tick_size;

        let mut next_up_hit: Vec<Option<usize>> = vec![None; mid_series.len()];
        let mut next_down_hit: Vec<Option<usize>> = vec![None; mid_series.len()];
        let mut up_waiters: BTreeMap<OrderedFloat<f64>, Vec<usize>> = BTreeMap::new();
        let mut down_waiters: BTreeMap<OrderedFloat<f64>, Vec<usize>> = BTreeMap::new();

        for (idx, (_, anchor)) in mid_series.iter().enumerate() {
            let price = anchor.price;

            while let Some((&OrderedFloat(threshold), _)) = up_waiters.first_key_value() {
                if threshold > price {
                    break;
                }
                let (_, indices) = up_waiters.pop_first().expect("checked via peek");
                for anchor_idx in indices {
                    if next_up_hit[anchor_idx].is_none() {
                        next_up_hit[anchor_idx] = Some(idx);
                    }
                }
            }

            while let Some((&OrderedFloat(threshold), _)) = down_waiters.last_key_value() {
                if threshold < price {
                    break;
                }
                let (_, indices) = down_waiters.pop_last().expect("checked via peek");
                for anchor_idx in indices {
                    if next_down_hit[anchor_idx].is_none() {
                        next_down_hit[anchor_idx] = Some(idx);
                    }
                }
            }

            let up_threshold = OrderedFloat(price + up_delta);
            up_waiters.entry(up_threshold).or_default().push(idx);

            let down_threshold = OrderedFloat(price - down_delta);
            down_waiters.entry(down_threshold).or_default().push(idx);
        }

        mid_series
            .iter()
            .enumerate()
            .map(|(anchor_idx, (event_index, anchor))| {
                let up_hit = next_up_hit[anchor_idx];
                let down_hit = next_down_hit[anchor_idx];
                let outcome = match (up_hit, down_hit) {
                    (Some(up_idx), Some(down_idx)) => {
                        if up_idx <= down_idx {
                            LabelOutcome::HitUp
                        } else {
                            LabelOutcome::HitDown
                        }
                    }
                    (Some(_), None) => LabelOutcome::HitUp,
                    (None, Some(_)) => LabelOutcome::HitDown,
                    (None, None) => LabelOutcome::NoHit,
                };

                Label::new(*event_index, anchor.timestamp, anchor.price, outcome)
            })
            .collect()
    }

    pub fn summarize(&self, labels: &[Label]) -> LabelStats {
        LabelStats::from_labels(labels)
    }
}

#[derive(Debug, Clone, Copy)]
struct LabelAnchor {
    timestamp: OffsetDateTime,
    price: f64,
}

/// Train/validation/test designation assigned strictly in chronological order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetSplit {
    Train,
    Validation,
    Test,
}

pub struct TimeSplitAssigner {
    cfg: DataSplitConfig,
}

impl TimeSplitAssigner {
    pub fn new(cfg: DataSplitConfig) -> Self {
        Self { cfg }
    }

    pub fn assign(&self, labels: &[Label]) -> Vec<DatasetSplit> {
        let total = labels.len();
        if total == 0 {
            return Vec::new();
        }

        let (train_end, val_end) = self.boundaries(total);
        (0..total)
            .map(|idx| {
                if idx < train_end {
                    DatasetSplit::Train
                } else if idx < val_end {
                    DatasetSplit::Validation
                } else {
                    DatasetSplit::Test
                }
            })
            .collect()
    }

    pub fn summary(&self, assignments: &[DatasetSplit]) -> SplitSummary {
        let mut stats = SplitSummary::default();
        for split in assignments {
            match split {
                DatasetSplit::Train => stats.train += 1,
                DatasetSplit::Validation => stats.validation += 1,
                DatasetSplit::Test => stats.test += 1,
            }
        }
        stats
    }

    fn boundaries(&self, total: usize) -> (usize, usize) {
        let total_f = total as f64;
        let train_end = (self.cfg.train_ratio * total_f).floor() as usize;
        let val_len = (self.cfg.validation_ratio * total_f).floor() as usize;
        let val_end = (train_end + val_len).min(total);
        (train_end.min(total), val_end)
    }
}

/// Convenient counts for logging when only labeling has been run.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SplitSummary {
    pub train: usize,
    pub validation: usize,
    pub test: usize,
}

impl SplitSummary {
    #[cfg(test)]
    pub fn total(&self) -> usize {
        self.train + self.validation + self.test
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::events::{MarketEvent, MarketEventKind, QuoteEvent};
    use crate::domain::order_book::BookLevel;
    use time::macros::datetime;

    fn quote_event(ts: time::OffsetDateTime, mid: f64) -> MarketEvent {
        MarketEvent {
            timestamp: ts,
            kind: MarketEventKind::Quote(QuoteEvent {
                best_bid_price: mid - 0.25,
                best_bid_size: 1.0,
                best_ask_price: mid + 0.25,
                best_ask_size: 1.0,
                mid_price: mid,
                bids: vec![BookLevel {
                    price: mid - 0.25,
                    size: 1.0,
                }],
                asks: vec![BookLevel {
                    price: mid + 0.25,
                    size: 1.0,
                }],
            }),
        }
    }

    #[test]
    fn labeler_detects_up_and_down_hits() {
        let params = LabelConfig {
            up_ticks: 2.0,
            down_ticks: 2.0,
        };
        let engine = LabelingEngine::new(params, 0.25);
        let events = vec![
            quote_event(datetime!(2025-01-01 00:00:00 UTC), 100.0),
            quote_event(datetime!(2025-01-01 00:00:01 UTC), 100.5),
            quote_event(datetime!(2025-01-01 00:00:02 UTC), 99.0),
            quote_event(datetime!(2025-01-01 00:00:03 UTC), 100.0),
        ];

        let labels = engine.compute_labels(&events);
        assert_eq!(labels.len(), events.len());
        assert_eq!(labels[0].outcome, LabelOutcome::HitUp);
        assert_eq!(labels[1].outcome, LabelOutcome::HitDown);
    }

    #[test]
    fn labeler_marks_no_hit_when_threshold_missing() {
        let params = LabelConfig {
            up_ticks: 10.0,
            down_ticks: 10.0,
        };
        let engine = LabelingEngine::new(params, 0.25);
        let events = vec![
            quote_event(datetime!(2025-01-01 00:00:00 UTC), 100.0),
            quote_event(datetime!(2025-01-01 00:00:01 UTC), 100.2),
            quote_event(datetime!(2025-01-01 00:00:02 UTC), 100.1),
        ];

        let labels = engine.compute_labels(&events);
        assert!(
            labels
                .iter()
                .all(|label| label.outcome == LabelOutcome::NoHit)
        );
    }

    #[test]
    fn labeler_outputs_finite_values() {
        let params = LabelConfig {
            up_ticks: 1.0,
            down_ticks: 1.0,
        };
        let engine = LabelingEngine::new(params, 0.25);
        let events = vec![
            quote_event(datetime!(2025-01-01 00:00:00 UTC), 100.0),
            quote_event(datetime!(2025-01-01 00:00:01 UTC), 100.5),
        ];

        let labels = engine.compute_labels(&events);
        assert!(!labels.is_empty());
        for label in labels {
            assert!(
                label.anchor_price.is_finite(),
                "anchor price must be finite"
            );
            let ts_ns = label.timestamp.unix_timestamp_nanos();
            assert!(ts_ns >= 0, "timestamp must be non-negative");
        }
    }

    #[test]
    fn splitter_preserves_chronological_order() {
        let cfg = DataSplitConfig {
            train_ratio: 0.5,
            validation_ratio: 0.25,
            test_ratio: 0.25,
        };
        let splitter = TimeSplitAssigner::new(cfg);
        let dummy_labels = vec![
            Label::new(
                0,
                datetime!(2025-01-01 00:00:00 UTC),
                0.0,
                LabelOutcome::NoHit
            );
            8
        ];

        let assignments = splitter.assign(&dummy_labels);
        assert_eq!(assignments.len(), 8);
        assert!(
            assignments[..4]
                .iter()
                .all(|split| matches!(split, DatasetSplit::Train))
        );
        assert!(
            assignments[4..6]
                .iter()
                .all(|split| matches!(split, DatasetSplit::Validation))
        );
        assert!(
            assignments[6..]
                .iter()
                .all(|split| matches!(split, DatasetSplit::Test))
        );

        let summary = splitter.summary(&assignments);
        assert_eq!(summary.train, 4);
        assert_eq!(summary.validation, 2);
        assert_eq!(summary.test, 2);
        assert_eq!(summary.total(), 8);
    }
}
