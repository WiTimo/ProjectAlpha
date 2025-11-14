use std::path::PathBuf;

use anyhow::Result;

use crate::domain::MarketEvent;

/// Abstract event reader so we can swap CSV, Parquet, Arrow, etc.
pub trait EventReader {
    fn next_event(&mut self) -> Result<Option<MarketEvent>>;
}

/// Placeholder file-backed reader. Real implementation will parse exchange data.
pub struct FileEventReader {
    pub path: PathBuf,
}

impl FileEventReader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl EventReader for FileEventReader {
    fn next_event(&mut self) -> Result<Option<MarketEvent>> {
        let _ = &self.path;
        Ok(None)
    }
}
