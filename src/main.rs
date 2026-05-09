mod batching;
mod cli;
mod daemon;
mod execution;
mod executors;
mod fingerprint;
mod queue;
mod scheduler;
mod telemetry;
mod vram_safety;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set global tracing subscriber");

    let command = cli::GpuRunCli::parse();
    command.execute().await
}
