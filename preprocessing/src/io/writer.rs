use std::path::PathBuf;

use anyhow::Result;

use crate::domain::FeatureVector;

/// Sink abstraction for materializing features (CSV, Arrow, DB, etc.).
pub trait FeatureWriter {
    fn write(&mut self, feature: &FeatureVector) -> Result<()>;
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Simple placeholder writer that just tracks where files would go.
pub struct FileFeatureWriter {
    pub output_path: PathBuf,
    pub written: usize,
}

impl FileFeatureWriter {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            output_path: path.into(),
            written: 0,
        }
    }
}

impl FeatureWriter for FileFeatureWriter {
    fn write(&mut self, feature: &FeatureVector) -> Result<()> {
        let _ = feature;
        self.written += 1;
        Ok(())
    }
}
