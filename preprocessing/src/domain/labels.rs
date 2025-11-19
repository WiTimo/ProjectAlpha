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

/// Label tied to a particular event index and timestamp with multi-horizon outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label {
    pub event_index: usize,
    pub timestamp: OffsetDateTime,
    pub anchor_price: f64,
    pub outcomes: Vec<LabelOutcome>,
}

impl Label {
    pub fn new(
        event_index: usize,
        timestamp: OffsetDateTime,
        anchor_price: f64,
        outcomes: Vec<LabelOutcome>,
    ) -> Self {
        Self {
            event_index,
            timestamp,
            anchor_price,
            outcomes,
        }
    }
}

/// Per-target distribution statistics for sanity checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetLabelStats {
    pub name: String,
    pub total: usize,
    pub hit_up: usize,
    pub hit_down: usize,
    pub no_hit: usize,
}

impl TargetLabelStats {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            total: 0,
            hit_up: 0,
            hit_down: 0,
            no_hit: 0,
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

/// Summary statistics for all configured targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelStats {
    pub per_target: Vec<TargetLabelStats>,
}

impl LabelStats {
    pub fn from_labels(labels: &[Label], target_names: &[String]) -> Self {
        let mut per_target: Vec<TargetLabelStats> = target_names
            .iter()
            .map(|name| TargetLabelStats::new(name))
            .collect();

        for label in labels {
            for (idx, outcome) in label.outcomes.iter().enumerate() {
                if let Some(stats) = per_target.get_mut(idx) {
                    stats.total += 1;
                    match outcome {
                        LabelOutcome::HitUp => stats.hit_up += 1,
                        LabelOutcome::HitDown => stats.hit_down += 1,
                        LabelOutcome::NoHit => stats.no_hit += 1,
                    }
                }
            }
        }

        Self { per_target }
    }

    pub fn positive_ratio(&self, target_name: &str) -> Option<f64> {
        self.per_target
            .iter()
            .find(|stats| stats.name == target_name)
            .map(TargetLabelStats::positive_ratio)
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
            Label::new(0, ts, 100.0, vec![LabelOutcome::HitUp]),
            Label::new(1, ts, 100.0, vec![LabelOutcome::HitDown]),
            Label::new(2, ts, 100.0, vec![LabelOutcome::NoHit]),
        ];

        let stats = LabelStats::from_labels(&labels, &["t20".to_string()]);
        assert_eq!(stats.per_target.len(), 1);
        let target_stats = &stats.per_target[0];
        assert_eq!(target_stats.total, 3);
        assert_eq!(target_stats.hit_up, 1);
        assert_eq!(target_stats.hit_down, 1);
        assert_eq!(target_stats.no_hit, 1);
        assert_eq!(target_stats.positive_ratio(), 1.0 / 3.0);
    }
}
