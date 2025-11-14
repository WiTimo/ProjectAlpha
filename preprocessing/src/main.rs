use anyhow::Result;
use clap::Parser;

use preprocessing::{PreprocessingPipeline, cli::Cli};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = cli.build_config()?;
    let mut pipeline = PreprocessingPipeline::new(config);
    pipeline.run()
}
