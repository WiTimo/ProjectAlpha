use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use time::Duration;

/// Supported bar resolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Resolution {
    Fast,
    Mid,
    Slow,
}

impl Resolution {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Resolution::Fast => "fast",
            Resolution::Mid => "mid",
            Resolution::Slow => "slow",
        }
    }

    pub const fn bar_duration(&self) -> Duration {
        match self {
            Resolution::Fast => Duration::seconds(1),
            Resolution::Mid => Duration::seconds(10),
            Resolution::Slow => Duration::seconds(60),
        }
    }

    pub const fn all() -> [Resolution; 3] {
        [Resolution::Fast, Resolution::Mid, Resolution::Slow]
    }
}

impl fmt::Display for Resolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Resolution {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Resolution::try_from(s)
    }
}

impl TryFrom<&str> for Resolution {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "fast" => Ok(Resolution::Fast),
            "mid" | "medium" => Ok(Resolution::Mid),
            "slow" => Ok(Resolution::Slow),
            other => bail!(
                "Unknown resolution '{}'. Expected fast, mid, or slow",
                other
            ),
        }
    }
}
