mod tail;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Serialize;
use time::OffsetDateTime;

use crate::config::NormalizationConfig;
use crate::domain::events::{QuoteEvent, TradeEvent};
use crate::domain::{Bar, BarAccumulator, BarKey, MarketEvent, MarketEventKind, Resolution};
use crate::io::parser::{QuoteState, parse_l1_line, parse_l2_line};
use crate::pipeline::{CoreFeatureExtractor, CoreFeatureRow};
use crate::realtime::tail::FileTail;
use crate::utils::time::align_to_resolution;

/// Plan entry describing a resolution that should be emitted in realtime.
#[derive(Debug, Clone)]
pub struct ResolutionPlanEntry {
    pub resolution: Resolution,
    pub levels: usize,
}

/// Configuration for the realtime preprocessing binary.
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    pub source_path: Option<PathBuf>,
    pub start_from_end: bool,
    pub poll_interval: Duration,
    pub emit_path: Option<PathBuf>,
    pub plan: Vec<ResolutionPlanEntry>,
    pub tick_size: f64,
    pub normalization: NormalizationConfig,
}

/// Streaming feature payload emitted by the realtime preprocessor.
#[derive(Debug, Serialize)]
pub struct RealtimeFeaturePayload {
    pub resolution: Resolution,
    pub bar_index: i64,
    pub start_timestamp_ns: i64,
    pub end_timestamp_ns: i64,
    pub features: BTreeMap<String, f64>,
}

/// Streaming processor that keeps reading lines and emitting JSON payloads.
pub struct RealtimePreprocessor {
    quote_state: QuoteState,
    pending_events: VecDeque<MarketEvent>,
    engines: HashMap<Resolution, StreamingFeatureEngine>,
    resolution_order: Vec<Resolution>,
    base_resolution: Resolution,
    extra_resolutions: Vec<Resolution>,
    extra_feature_cache: HashMap<Resolution, BTreeMap<String, f64>>,
}

impl RealtimePreprocessor {
    pub fn new(cfg: &RealtimeConfig) -> Self {
        let mut plan = cfg.plan.clone();
        if plan.is_empty() {
            plan.push(ResolutionPlanEntry {
                resolution: Resolution::Fast,
                levels: 5,
            });
        }
        let max_levels = plan.iter().map(|entry| entry.levels).max().unwrap_or(1);
        let mut engines = HashMap::new();
        for entry in &plan {
            engines.insert(
                entry.resolution,
                StreamingFeatureEngine::new(
                    entry.resolution,
                    entry.levels,
                    cfg.tick_size,
                    &cfg.normalization,
                ),
            );
        }
        let resolution_order: Vec<Resolution> =
            plan.iter().map(|entry| entry.resolution).collect();
        let base_resolution = resolution_order.first().copied().unwrap_or(Resolution::Fast);
        let extra_resolutions = if resolution_order.len() > 1 {
            resolution_order[1..].to_vec()
        } else {
            Vec::new()
        };
        Self {
            quote_state: QuoteState::new(max_levels),
            pending_events: VecDeque::new(),
            engines,
            resolution_order,
            base_resolution,
            extra_resolutions,
            extra_feature_cache: HashMap::new(),
        }
    }

    /// Run using stdin as the input source.
    pub fn run_with_stdin(&mut self, cfg: &RealtimeConfig) -> Result<()> {
        let stdin = io::stdin();
        let lock = stdin.lock();
        self.process_reader(lock, cfg)
    }

    /// Run by tailing a log file continuously.
    pub fn run_with_file(&mut self, cfg: &RealtimeConfig, path: &Path) -> Result<()> {
        let mut tail = FileTail::open(path, cfg.start_from_end)?;
        let mut writer = output_writer(cfg)?;
        loop {
            match tail.read_line()? {
                Some(line) => {
                    let payloads = self.process_line(&line)?;
                    for payload in payloads {
                        let json = serde_json::to_string(&payload)?;
                        writeln!(writer, "{}", json)?;
                    }
                    writer.flush().ok();
                }
                None => thread::sleep(cfg.poll_interval),
            }
        }
    }

    fn process_reader<R: BufRead>(&mut self, reader: R, cfg: &RealtimeConfig) -> Result<()> {
        let mut writer = output_writer(cfg)?;
        for line in reader.lines() {
            let line = line?;
            let payloads = self.process_line(&line)?;
            for payload in payloads {
                let json = serde_json::to_string(&payload)?;
                writeln!(writer, "{}", json)?;
            }
        }
        Ok(())
    }

    fn process_line(&mut self, raw_line: &str) -> Result<Vec<RealtimeFeaturePayload>> {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(row) = parse_l2_line(trimmed)? {
            let row_copy = row;
            let update = self.quote_state.update(row_copy);
            self.enqueue_order_flow(row_copy.timestamp, update.flows);
            if let Some(snapshot) = update.snapshot {
                self.pending_events.push_back(MarketEvent {
                    timestamp: snapshot.timestamp,
                    kind: MarketEventKind::Quote(QuoteEvent {
                        best_bid_price: snapshot.bid.price,
                        best_bid_size: snapshot.bid.size,
                        best_ask_price: snapshot.ask.price,
                        best_ask_size: snapshot.ask.size,
                        mid_price: 0.5 * (snapshot.bid.price + snapshot.ask.price),
                        bids: snapshot.bids,
                        asks: snapshot.asks,
                    }),
                });
            }
        } else if let Some(trade) = parse_l1_line(trimmed)? {
            self.pending_events.push_back(MarketEvent {
                timestamp: trade.timestamp,
                kind: MarketEventKind::Trade(TradeEvent {
                    price: trade.price,
                    size: trade.size,
                    aggressor: trade.aggressor,
                }),
            });
        }

        let mut payloads = Vec::new();
        let order = self.resolution_order.clone();
        while let Some(event) = self.pending_events.pop_front() {
            for resolution in order.iter().copied() {
                if let Some(engine) = self.engines.get_mut(&resolution) {
                    let rows = engine.ingest_event(&event);
                    for row in rows {
                        if resolution == self.base_resolution {
                            if let Some(payload) = self.build_payload(row) {
                                payloads.push(payload);
                            }
                        } else {
                            let aliased = build_feature_map_with_suffix(&row, resolution);
                            self.extra_feature_cache.insert(resolution, aliased);
                        }
                    }
                }
            }
        }

        Ok(payloads)
    }

    fn enqueue_order_flow(
        &mut self,
        timestamp: OffsetDateTime,
        flows: Vec<crate::domain::OrderFlowEvent>,
    ) {
        for flow in flows {
            self.pending_events.push_back(MarketEvent {
                timestamp,
                kind: MarketEventKind::OrderFlow(flow),
            });
        }
    }
    fn build_payload(&mut self, row: CoreFeatureRow) -> Option<RealtimeFeaturePayload> {
        if !self.extra_resolutions_ready() {
            return None;
        }
        let mut features = build_feature_map(&row);
        for resolution in &self.extra_resolutions {
            if let Some(extra) = self.extra_feature_cache.get(resolution) {
                features.extend(extra.iter().map(|(k, v)| (k.clone(), *v)));
            }
        }
        Some(RealtimeFeaturePayload::from_row_with_features(row, features))
    }

    fn extra_resolutions_ready(&self) -> bool {
        self.extra_resolutions.is_empty()
            || self
                .extra_resolutions
                .iter()
                .all(|res| self.extra_feature_cache.contains_key(res))
    }
}

impl RealtimeFeaturePayload {
    fn from_row_with_features(row: CoreFeatureRow, features: BTreeMap<String, f64>) -> Self {
        Self {
            resolution: row.key.resolution,
            bar_index: row.key.index,
            start_timestamp_ns: row.start_ns,
            end_timestamp_ns: row.end_ns,
            features,
        }
    }
}

struct StreamingFeatureEngine {
    resolution: Resolution,
    levels: usize,
    extractor: CoreFeatureExtractor,
    accumulator: Option<BarAccumulator>,
}

impl StreamingFeatureEngine {
    fn new(
        resolution: Resolution,
        levels: usize,
        tick_size: f64,
        normalization: &NormalizationConfig,
    ) -> Self {
        let extractor = CoreFeatureExtractor::new(tick_size, normalization);
        Self {
            resolution,
            levels,
            extractor,
            accumulator: None,
        }
    }

    fn ingest_event(&mut self, event: &MarketEvent) -> Vec<CoreFeatureRow> {
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

fn build_feature_map(row: &CoreFeatureRow) -> BTreeMap<String, f64> {
    let mut map = BTreeMap::new();
    map.insert("mid_close_price".into(), row.mid_close_price);
    map.insert("mid_high_price".into(), row.mid_high_price);
    map.insert("mid_low_price".into(), row.mid_low_price);
    map.insert("mid_return_bar".into(), row.mid_return_bar);
    map.insert("spread_ticks".into(), row.spread_ticks);
    map.insert("spread_change_ticks".into(), row.spread_change_ticks);
    map.insert("mid_range_rel".into(), row.mid_range_rel);
    map.insert("imbalance_best".into(), row.imbalance_best);
    map.insert("cum_bid_size_l_rel".into(), row.cum_bid_size_l_rel);
    map.insert("cum_ask_size_l_rel".into(), row.cum_ask_size_l_rel);
    map.insert("imbalance_l".into(), row.imbalance_l);
    map.insert("trade_volume_sum_rel".into(), row.trade_volume_sum_rel);
    map.insert("trade_count_log".into(), row.trade_count_log);
    map.insert("buy_trade_volume_rel".into(), row.buy_trade_volume_rel);
    map.insert("sell_trade_volume_rel".into(), row.sell_trade_volume_rel);
    map.insert("trade_imbalance_ratio".into(), row.trade_imbalance_ratio);
    map.insert("has_buy_trade".into(), row.has_buy_trade);
    map.insert("has_sell_trade".into(), row.has_sell_trade);
    map.insert("avg_buy_dist_to_ask".into(), row.avg_buy_dist_to_ask);
    map.insert("avg_sell_dist_to_bid".into(), row.avg_sell_dist_to_bid);
    map.insert("rv_log".into(), row.rv_log);
    map.insert("tod_sin".into(), row.tod_sin);
    map.insert("tod_cos".into(), row.tod_cos);
    map.insert("is_us_session".into(), row.is_us_session);
    map.insert("spread_regime".into(), row.spread_regime);
    map.insert("depth_regime".into(), row.depth_regime);
    map.insert("volume_regime".into(), row.volume_regime);
    map.insert("volatility_regime".into(), row.volatility_regime);
    map.insert("speed_regime".into(), row.speed_regime);
    map.insert(
        "limit_add_bid_volume_rel".into(),
        row.limit_add_bid_volume_rel,
    );
    map.insert(
        "limit_add_ask_volume_rel".into(),
        row.limit_add_ask_volume_rel,
    );
    map.insert(
        "limit_cancel_bid_volume_rel".into(),
        row.limit_cancel_bid_volume_rel,
    );
    map.insert(
        "limit_cancel_ask_volume_rel".into(),
        row.limit_cancel_ask_volume_rel,
    );
    map.insert("limit_of_imbalance".into(), row.limit_of_imbalance);
    map.insert("ofi_bid".into(), row.ofi_bid);
    map.insert("ofi_ask".into(), row.ofi_ask);
    map.insert("ofi_net_log".into(), row.ofi_net_log);

    for (idx, value) in row.bid_offset_level_ticks.iter().enumerate() {
        map.insert(format!("bid_offset_level_{}_ticks", idx + 1), *value);
    }
    for (idx, value) in row.ask_offset_level_ticks.iter().enumerate() {
        map.insert(format!("ask_offset_level_{}_ticks", idx + 1), *value);
    }
    for (idx, value) in row.bid_size_level_rel.iter().enumerate() {
        map.insert(format!("bid_size_level_{}_rel", idx + 1), *value);
    }
    for (idx, value) in row.ask_size_level_rel.iter().enumerate() {
        map.insert(format!("ask_size_level_{}_rel", idx + 1), *value);
    }
    for (idx, value) in row.bid_level_present.iter().enumerate() {
        map.insert(format!("bid_level_{}_present", idx + 1), *value);
    }
    for (idx, value) in row.ask_level_present.iter().enumerate() {
        map.insert(format!("ask_level_{}_present", idx + 1), *value);
    }

    map
}

fn build_feature_map_with_suffix(
    row: &CoreFeatureRow,
    resolution: Resolution,
) -> BTreeMap<String, f64> {
    let suffix = resolution.as_str();
    build_feature_map(row)
        .into_iter()
        .map(|(key, value)| (format!("{key}@{suffix}"), value))
        .collect()
}

fn output_writer(cfg: &RealtimeConfig) -> Result<Box<dyn Write + Send>> {
    if let Some(path) = &cfg.emit_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create realtime output directory {}",
                    parent.display()
                )
            })?;
        }
        let file = File::options()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("Failed to open emitter file {}", path.display()))?;
        Ok(Box::new(file))
    } else {
        Ok(Box::new(io::stdout()))
    }
}
