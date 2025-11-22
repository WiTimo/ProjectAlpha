//! Core library exports for the preprocessing crate.
//! The goal is to keep `main.rs` small and host all orchestrated logic here.

pub mod cli;
pub mod config;
pub mod domain;
pub mod io;
pub mod normalization;
pub mod pipeline;
pub mod realtime;
pub mod utils;

pub use config::PipelineConfig;
pub use pipeline::PreprocessingPipeline;
