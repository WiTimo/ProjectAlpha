use crate::config::NormalizationConfig;
use crate::domain::{Bar, BarAccumulator, BarKey, MarketEvent, Resolution};
use crate::normalization::CausalScalerState;
use crate::pipeline::core_features::{CoreFeatureExtractor, CoreFeatureRow};
use crate::utils::time::align_to_resolution;
use time::OffsetDateTime;

/// Stateful feature engine that incrementally ingests events and emits feature rows
/// once bars close. Shared between offline preprocessing and realtime streaming.
pub struct StreamingFeatureEngine {
    resolution: Resolution,
    levels: usize,
    extractor: CoreFeatureExtractor,
    accumulator: Option<BarAccumulator>,
}

impl StreamingFeatureEngine {
    pub fn new(
        resolution: Resolution,
        levels: usize,
        tick_size: f64,
        normalization: &NormalizationConfig,
    ) -> Self {
        Self::with_scaler_state(resolution, levels, tick_size, normalization, None)
    }

    pub fn with_scaler_state(
        resolution: Resolution,
        levels: usize,
        tick_size: f64,
        normalization: &NormalizationConfig,
        scaler_state: Option<&CausalScalerState>,
    ) -> Self {
        let extractor = if let Some(state) = scaler_state {
            CoreFeatureExtractor::with_state(tick_size, normalization, state)
        } else {
            CoreFeatureExtractor::new(tick_size, normalization)
        };
        Self {
            resolution,
            levels,
            extractor,
            accumulator: None,
        }
    }

    /// Ingest a single market event and return any completed feature rows.
    pub fn ingest_event(&mut self, event: &MarketEvent) -> Vec<CoreFeatureRow> {
        self.ensure_accumulator(event.timestamp);
        let mut rows = Vec::new();

        while let Some(acc) = self.accumulator.as_ref() {
            if event.timestamp < acc.end {
                break;
            }
            if let Some(bar) = self.advance_window() {
                rows.extend(self.extractor.compute(std::slice::from_ref(&bar)));
            }
        }

        if let Some(acc) = self.accumulator.as_mut() {
            acc.ingest(event);
        }

        rows
    }

    /// Flush any partially completed bar and return the remaining feature rows.
    pub fn finish(&mut self) -> Vec<CoreFeatureRow> {
        let mut rows = Vec::new();
        if let Some(acc) = self.accumulator.take() {
            if let Some(bar) = acc.finalize() {
                rows.extend(self.extractor.compute(std::slice::from_ref(&bar)));
            }
        }
        rows
    }

    pub fn scaler_state(&self) -> CausalScalerState {
        self.extractor.scaler_state()
    }

    fn ensure_accumulator(&mut self, timestamp: OffsetDateTime) {
        if self.accumulator.is_some() {
            return;
        }
        let start = align_to_resolution(timestamp, self.resolution);
        let end = start + self.resolution.bar_duration();
        let key = BarKey::new(self.resolution, 0);
        self.accumulator = Some(BarAccumulator::new(key, start, end, self.levels));
    }

    fn advance_window(&mut self) -> Option<Bar> {
        let acc = self.accumulator.take()?;
        let next_start = acc.end;
        let next_index = acc.key.index + 1;
        let next_end = next_start + self.resolution.bar_duration();
        self.accumulator = Some(BarAccumulator::new(
            BarKey::new(self.resolution, next_index),
            next_start,
            next_end,
            self.levels,
        ));
        acc.finalize()
    }
}
