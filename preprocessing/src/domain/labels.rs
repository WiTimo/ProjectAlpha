use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

/// Binary classification outcome for the event-based labeling scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LabelOutcome {
    /// Price hit the positive target before the negative one.
    HitUp,
    /// Price hit the negative target before the positive one.
    HitDown,
    /// Neither target was reached within the allowed lookahead horizon.
    NoHit,
}

impl LabelOutcome {
    /// Map the outcome to a binary value where up = 1 and down = 0. Returns `None` for `NoHit`.
    pub const fn as_binary(self) -> Option<u8> {
        match self {
            LabelOutcome::HitUp => Some(1),
            LabelOutcome::HitDown => Some(0),
            LabelOutcome::NoHit => None,
        }
    }
}

/// Label tied to a particular event index and timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label {
    pub event_index: usize,
    pub timestamp: OffsetDateTime,
    pub outcome: LabelOutcome,
    pub anchor_price: f64,
}

impl Label {
    pub fn new(
        event_index: usize,
        timestamp: OffsetDateTime,
        anchor_price: f64,
        outcome: LabelOutcome,
    ) -> Self {
        Self {
            event_index,
            timestamp,
            outcome,
            anchor_price,
        }
    }
}

/// Summary statistics for sanity-checking the label distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelStats {
    pub total: usize,
    pub hit_up: usize,
    pub hit_down: usize,
    pub no_hit: usize,
}

impl LabelStats {
    pub fn from_labels(labels: &[Label]) -> Self {
        let mut hit_up = 0;
        let mut hit_down = 0;
        let mut no_hit = 0;

        for label in labels {
            match label.outcome {
                LabelOutcome::HitUp => hit_up += 1,
                LabelOutcome::HitDown => hit_down += 1,
                LabelOutcome::NoHit => no_hit += 1,
            }
        }

        Self {
            total: labels.len(),
            hit_up,
            hit_down,
            no_hit,
        }
    }

    pub fn positive_ratio(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.hit_up as f64 / self.total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::macros::datetime;

    #[test]
    fn stats_track_distribution() {
        let ts = datetime!(2025-01-01 00:00:00 UTC);
        let labels = vec![
            Label::new(0, ts, 100.0, LabelOutcome::HitUp),
            Label::new(1, ts, 100.0, LabelOutcome::HitDown),
            Label::new(2, ts, 100.0, LabelOutcome::NoHit),
        ];

        let stats = LabelStats::from_labels(&labels);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.hit_up, 1);
        assert_eq!(stats.hit_down, 1);
        assert_eq!(stats.no_hit, 1);
        assert_eq!(stats.positive_ratio(), 1.0 / 3.0);
    }
}
