use anyhow::{anyhow, Result};
use tokio::process::Command;
use tracing::info;

use crate::fingerprint;
use crate::queue::{self, Job};

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

    let env_vars: std::collections::HashMap<String, String> = std::env::vars().collect();
    let fingerprint = fingerprint::WorkloadFingerprint::from_command(&args, &env_vars)?;
    let job = Job::new(args, no_batch, priority, watch, fingerprint)?;
    let job_id = job.id.clone();
    queue::append_job(job)?;

    println!("Job queued with id: {}", job_id);
    println!("Start the daemon with `gpu-run daemon start` to process pending jobs.");
    Ok(())
}

pub async fn execute_job(job: &Job) -> Result<i32> {
    if job.args.is_empty() {
        return Err(anyhow!("Job has no command to execute."));
    }

    let mut command = Command::new(&job.args[0]);
    if job.args.len() > 1 {
        command.args(&job.args[1..]);
    }
    command
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit());

    let mut child = command.spawn()?;
    let status = child.wait().await?;
    Ok(status.code().unwrap_or(-1))
}
