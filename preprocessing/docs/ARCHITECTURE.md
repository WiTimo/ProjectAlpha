# Preprocessing Architecture Overview

This crate is split into a few focused modules so that the large feature matrix described in `README.md` and `NORMALIZATION.md` can be implemented incrementally.

- `cli`: Defines the `Cli` struct (via `clap`) and produces a `PipelineConfig` with command-line overrides. Use `cargo run -- --help` to inspect all options.
- `config`: Strongly-typed configuration model with helpers to load JSON/YAML, synthesize defaults, and apply overrides at runtime. The `NormalizationConfig` mirrors the causal scaling guide in `NORMALIZATION.md`.
- `domain`: Contains the core data types (resolutions, identifiers, events, book snapshots, bars, and feature vectors). This is where schema-level changes should live so every pipeline stage shares the same vocabulary.
- `normalization`: Houses causal rolling statistics (`Ewma`, `WindowedMean`) and the `CausalScaler` helper that turns raw volumes/depths/counts into relative/log-transformed values.
- `pipeline`: Orchestrator (`PreprocessingPipeline`) plus stage traits (`EventSourceStage`, `BarStage`, `FeatureStage`, `SinkStage`). The idea is to wire together an event reader, bar builder, feature computers, and sink per resolution.
- `io`: Placeholder reader/writer abstractions so you can later swap in CSV/Parquet/Arrow without touching pipeline logic.
- `utils`: Small math/time helpers shared by multiple modules.

## Typical control flow

1. `main.rs` parses CLI arguments and calls `PreprocessingPipeline::run`.
2. The pipeline builds a context (instrument metadata, session clock) and iterates through configured `ResolutionConfig` entries.
3. For each resolution you will eventually:
   - Create an `EventReader` (direct feed for `fast`, aggregate from lower resolutions otherwise).
   - Feed events into a `BarStage` that manages `BarAccumulator`s and produces finalized `Bar` instances.
   - Pass completed bars through a set of `FeatureStage`s (one per `FeatureGroup` section in the README). Each stage can use `CausalScaler` instances and the normalization helpers.
   - Emit `FeatureVector`s into one or more `FeatureWriter`s (per split or destination).

With this structure in place you can focus on implementing each feature group one module at a time while keeping the normalization rules causal and testable.
