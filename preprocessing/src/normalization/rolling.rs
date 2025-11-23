use std::collections::VecDeque;

/// Generic interface for causal rolling statistics.
pub trait RollingStatistic {
    fn observe(&mut self, value: f64);
    fn value(&self) -> Option<f64>;
}

/// Exponentially weighted moving average.
#[derive(Debug, Clone)]
pub struct Ewma {
    alpha: f64,
    state: Option<f64>,
}

impl Ewma {
    pub fn new(alpha: f64) -> Self {
        Self { alpha, state: None }
    }

    /// Snapshot the internal state so it can be serialized or inspected.
    pub fn state(&self) -> Option<f64> {
        self.state
    }

    /// Restore the EWMA state, replacing any previously accumulated value.
    pub fn set_state(&mut self, state: Option<f64>) {
        self.state = state;
    }
}

impl RollingStatistic for Ewma {
    fn observe(&mut self, value: f64) {
        self.state = Some(match self.state {
            Some(prev) => (1.0 - self.alpha) * prev + self.alpha * value,
            None => value,
        });
    }

    fn value(&self) -> Option<f64> {
        self.state
    }
}

/// Simple moving average over a fixed-size window.
#[derive(Debug, Clone)]
pub struct WindowedMean {
    window: usize,
    buffer: VecDeque<f64>,
    sum: f64,
}

impl WindowedMean {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            buffer: VecDeque::with_capacity(window),
            sum: 0.0,
        }
    }

    /// Return the buffered samples in chronological order for serialization.
    pub fn values(&self) -> Vec<f64> {
        self.buffer.iter().copied().collect()
    }

    /// Replace the rolling window contents with pre-recorded samples.
    pub fn seed(&mut self, samples: &[f64]) {
        self.buffer.clear();
        self.sum = 0.0;
        if self.window == 0 || samples.is_empty() {
            return;
        }
        let keep = samples.len().min(self.window);
        let start = samples.len() - keep;
        for value in &samples[start..] {
            self.buffer.push_back(*value);
            self.sum += *value;
        }
    }
}

impl RollingStatistic for WindowedMean {
    fn observe(&mut self, value: f64) {
        self.buffer.push_back(value);
        self.sum += value;
        if self.buffer.len() > self.window {
            if let Some(front) = self.buffer.pop_front() {
                self.sum -= front;
            }
        }
    }

    fn value(&self) -> Option<f64> {
        if self.buffer.is_empty() {
            None
        } else {
            Some(self.sum / self.buffer.len() as f64)
        }
    }
}
