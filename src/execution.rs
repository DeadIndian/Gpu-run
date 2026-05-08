use anyhow::{anyhow, Result};
use tracing::info;

pub async fn run_job(
    args: Vec<String>,
    no_batch: bool,
    priority: Option<String>,
    watch: bool,
) -> Result<()> {
    if args.is_empty() {
        return Err(anyhow!("No command specified. Use `gpu-run run <command>`."));
    }

    info!(
        ?args,
        no_batch,
        ?priority,
        watch,
        "Submitting a new job"
    );

    println!("Job submission is not yet implemented.");
    Ok(())
}
