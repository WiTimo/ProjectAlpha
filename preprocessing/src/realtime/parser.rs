use anyhow::{Context, Result};
use serde::Serialize;
use time::{
    OffsetDateTime,
    format_description::{FormatItem, well_known::Rfc3339},
    macros::format_description,
};

/// Trait used to parse log lines into structured events.
pub trait LineParser {
    fn parse(&self, line: &str) -> Result<Option<ParsedEvent>>;
}

#[derive(Debug, Clone, Serialize)]
pub struct ParsedEvent {
    pub timestamp_ns: i128,
    pub price: Option<f64>,
    pub size: Option<f64>,
    pub side: Option<String>,
    pub event_type: String,
}

#[derive(Debug, Clone, Default)]
pub struct NinjaTraderL2Parser;

impl LineParser for NinjaTraderL2Parser {
    fn parse(&self, line: &str) -> Result<Option<ParsedEvent>> {
        if line.is_empty() {
            return Ok(None);
        }

        let tokens: Vec<&str> = line.split(',').collect();
        if tokens.len() < 8 {
            return Ok(None);
        }

        let timestamp_ns = parse_timestamp(tokens[0])?;
        let event_type = tokens[1].trim().to_string();
        let price = tokens[2].trim().parse::<f64>().ok();
        let size = tokens[3].trim().parse::<f64>().ok();
        let side = tokens.get(4).map(|s| s.trim().to_ascii_lowercase());

        Ok(Some(ParsedEvent {
            timestamp_ns,
            price,
            size,
            side,
            event_type,
        }))
    }
}

fn parse_timestamp(raw: &str) -> Result<i128> {
    if let Ok(ts) = OffsetDateTime::parse(raw, &Rfc3339) {
        return Ok(ts.unix_timestamp_nanos());
    }
    const FALLBACK_FMT: &[FormatItem<'static>] =
        format_description!("[year]/[month]/[day] [hour]:[minute]:[second]");
    let ts = OffsetDateTime::parse(raw, FALLBACK_FMT)
        .with_context(|| format!("Unsupported timestamp format: {raw}"))?;
    Ok(ts.unix_timestamp_nanos())
}
