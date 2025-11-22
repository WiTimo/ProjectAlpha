use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result};

/// Minimal file tailer that reads new lines appended to a file (like `tail -F`).
pub struct FileTail {
    reader: BufReader<File>,
    buffer: String,
}

impl FileTail {
    pub fn open(path: &Path, start_from_end: bool) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open realtime source {}", path.display()))?;
        if start_from_end {
            file.seek(SeekFrom::End(0))?;
        }
        Ok(Self {
            reader: BufReader::new(file),
            buffer: String::new(),
        })
    }

    pub fn read_line(&mut self) -> Result<Option<String>> {
        self.buffer.clear();
        let bytes = self.reader.read_line(&mut self.buffer)?;
        if bytes == 0 {
            return Ok(None);
        }
        if self.buffer.ends_with('\n') {
            self.buffer.pop();
            if self.buffer.ends_with('\r') {
                self.buffer.pop();
            }
        }
        Ok(Some(self.buffer.clone()))
    }
}
