use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::process::Command;
use tracing::{debug, info, warn};

use crate::fingerprint::{BackendKind, WorkloadFingerprint};
use crate::queue::Job;

pub trait BackendExecutor {
    fn can_execute(&self, fingerprint: &WorkloadFingerprint) -> bool;
    async fn execute_batch(&self, jobs: Vec<Job>) -> Result<Vec<i32>>; // Returns exit codes for each job
    async fn execute_single(&self, job: &Job) -> Result<i32>;
}

pub struct LlamaCppExecutor;

impl BackendExecutor for LlamaCppExecutor {
    fn can_execute(&self, fingerprint: &WorkloadFingerprint) -> bool {
        matches!(fingerprint.backend, BackendKind::LlamaCpp)
    }

    async fn execute_batch(&self, jobs: Vec<Job>) -> Result<Vec<i32>> {
        if jobs.is_empty() {
            return Ok(vec![]);
        }

        if jobs.len() == 1 {
            let exit_code = self.execute_single(&jobs[0]).await?;
            return Ok(vec![exit_code]);
        }

        info!("Executing batch of {} llama.cpp jobs", jobs.len());

        // Create temporary prompt file
        let prompt_file = create_prompt_file(&jobs)?;
        
        // Build batch command using the first job's parameters as base
        let base_job = &jobs[0];
        let mut cmd_args = base_job.args.clone();
        
        // Remove existing prompt-related flags
        cmd_args.retain(|arg| !arg.starts_with("--prompt") && !arg.starts_with("-p"));
        
        // Add prompt file flag
        cmd_args.push("--prompt-file".to_string());
        cmd_args.push(prompt_file.to_string_lossy().to_string());
        
        // Set batch size to number of jobs if not already set
        if !cmd_args.iter().any(|arg| arg == "-b" || arg == "--batch-size") {
            cmd_args.push("-b".to_string());
            cmd_args.push(jobs.len().to_string());
        }

        debug!("Executing batch command: {}", cmd_args.join(" "));

        let mut command = Command::new(&cmd_args[0]);
        if cmd_args.len() > 1 {
            command.args(&cmd_args[1..]);
        }
        command
            .stdin(std::process::Stdio::inherit())
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit());

        let mut child = command.spawn()?;
        let status = child.wait().await?;
        let exit_code = status.code().unwrap_or(-1);

        // Clean up prompt file
        if let Err(e) = fs::remove_file(&prompt_file) {
            warn!("Failed to remove temporary prompt file {:?}: {}", prompt_file, e);
        }

        // For now, return the same exit code for all jobs in the batch
        // In a more sophisticated implementation, we might track individual job success/failure
        Ok(vec![exit_code; jobs.len()])
    }

    async fn execute_single(&self, job: &Job) -> Result<i32> {
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
}

pub struct ComfyUIExecutor;

impl BackendExecutor for ComfyUIExecutor {
    fn can_execute(&self, fingerprint: &WorkloadFingerprint) -> bool {
        matches!(fingerprint.backend, BackendKind::ComfyUI)
    }

    async fn execute_batch(&self, jobs: Vec<Job>) -> Result<Vec<i32>> {
        if jobs.is_empty() {
            return Ok(vec![]);
        }

        info!("Executing batch of {} ComfyUI jobs via queue API", jobs.len());

        // For ComfyUI, we submit all jobs to the queue API
        let mut exit_codes = Vec::new();
        
        for job in jobs {
            let exit_code = self.execute_single(&job).await?;
            exit_codes.push(exit_code);
        }

        Ok(exit_codes)
    }

    async fn execute_single(&self, job: &Job) -> Result<i32> {
        // Check if this is a CLI invocation or API submission
        if job.args.iter().any(|arg| arg == "--workflow") {
            // CLI invocation - execute directly
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
        } else {
            // API submission - submit to ComfyUI queue
            submit_to_comfyui_queue(job).await
        }
    }
}

pub struct Automatic1111Executor;

impl BackendExecutor for Automatic1111Executor {
    fn can_execute(&self, fingerprint: &WorkloadFingerprint) -> bool {
        matches!(fingerprint.backend, BackendKind::Automatic1111)
    }

    async fn execute_batch(&self, jobs: Vec<Job>) -> Result<Vec<i32>> {
        if jobs.is_empty() {
            return Ok(vec![]);
        }

        info!("Executing batch of {} Automatic1111 jobs", jobs.len());

        // For A1111, we can use --batch-size flag for similar jobs
        let mut exit_codes = Vec::new();
        
        // Group jobs by similar parameters
        let mut job_groups = HashMap::new();
        for job in jobs {
            let key = extract_batch_key(&job);
            job_groups.entry(key).or_insert_with(Vec::new).push(job);
        }

        for (_batch_key, group_jobs) in job_groups {
            if group_jobs.len() == 1 {
                let exit_code = self.execute_single(&group_jobs[0]).await?;
                exit_codes.push(exit_code);
            } else {
                // Create batch command
                let exit_codes_batch = execute_a1111_batch(&group_jobs).await?;
                exit_codes.extend(exit_codes_batch);
            }
        }

        Ok(exit_codes)
    }

    async fn execute_single(&self, job: &Job) -> Result<i32> {
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
}

pub struct GenericExecutor;

impl BackendExecutor for GenericExecutor {
    fn can_execute(&self, _fingerprint: &WorkloadFingerprint) -> bool {
        true // Can execute any backend
    }

    async fn execute_batch(&self, jobs: Vec<Job>) -> Result<Vec<i32>> {
        // Generic executor doesn't support batching - execute sequentially
        let mut exit_codes = Vec::new();
        for job in jobs {
            let exit_code = self.execute_single(&job).await?;
            exit_codes.push(exit_code);
        }
        Ok(exit_codes)
    }

    async fn execute_single(&self, job: &Job) -> Result<i32> {
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
}

// Helper functions

fn create_prompt_file(jobs: &[Job]) -> Result<PathBuf> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs();
    
    let filename = format!("gpu-run-batch-{}.txt", timestamp);
    let mut path = std::env::temp_dir();
    path.push(filename);

    let mut file = File::create(&path)?;
    
    for job in jobs {
        // Extract prompt from job arguments
        if let Some(prompt) = extract_prompt_from_args(&job.args) {
            writeln!(file, "{}", prompt)?;
        } else {
            writeln!(file, "# Job: {}", job.args.join(" "))?;
        }
    }

    Ok(path)
}

fn extract_prompt_from_args(args: &[String]) -> Option<String> {
    for (i, arg) in args.iter().enumerate() {
        if arg == "--prompt" || arg == "-p" {
            if i + 1 < args.len() {
                return Some(args[i + 1].clone());
            }
        }
    }
    None
}

fn extract_batch_key(job: &Job) -> String {
    format!(
        "{:?}|{:?}|{:?}|{:?}",
        job.fingerprint.model_key,
        job.fingerprint.resolution,
        job.fingerprint.sampler_key,
        job.fingerprint.adapter_hashes.join(",")
    )
}

async fn execute_a1111_batch(jobs: &[Job]) -> Result<Vec<i32>> {
    let base_job = &jobs[0];
    let mut cmd_args = base_job.args.clone();
    
    // Remove existing batch size flag
    let mut batch_size_idx = None;
    for (i, arg) in cmd_args.iter().enumerate() {
        if arg == "--batch-size" && i + 1 < cmd_args.len() {
            batch_size_idx = Some(i);
            break;
        }
    }
    
    if let Some(idx) = batch_size_idx {
        cmd_args.remove(idx + 1);
        cmd_args.remove(idx);
    }
    
    // Add batch size
    cmd_args.push("--batch-size".to_string());
    cmd_args.push(jobs.len().to_string());

    debug!("Executing A1111 batch: {}", cmd_args.join(" "));

    let mut command = Command::new(&cmd_args[0]);
    if cmd_args.len() > 1 {
        command.args(&cmd_args[1..]);
    }
    command
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit());

    let mut child = command.spawn()?;
    let status = child.wait().await?;
    let exit_code = status.code().unwrap_or(-1);

    Ok(vec![exit_code; jobs.len()])
}

async fn submit_to_comfyui_queue(job: &Job) -> Result<i32> {
    // This would implement ComfyUI API queue submission
    // For now, fall back to CLI execution
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

pub enum ExecutorType {
    LlamaCpp(LlamaCppExecutor),
    ComfyUI(ComfyUIExecutor),
    Automatic1111(Automatic1111Executor),
    Generic(GenericExecutor),
}

impl ExecutorType {
    pub fn for_backend(backend: BackendKind) -> Self {
        match backend {
            BackendKind::LlamaCpp => ExecutorType::LlamaCpp(LlamaCppExecutor),
            BackendKind::ComfyUI => ExecutorType::ComfyUI(ComfyUIExecutor),
            BackendKind::Automatic1111 => ExecutorType::Automatic1111(Automatic1111Executor),
            _ => ExecutorType::Generic(GenericExecutor),
        }
    }
}

// Implement the trait methods for the enum
impl ExecutorType {
    pub async fn execute_batch(&self, jobs: Vec<Job>) -> Result<Vec<i32>> {
        match self {
            ExecutorType::LlamaCpp(executor) => executor.execute_batch(jobs).await,
            ExecutorType::ComfyUI(executor) => executor.execute_batch(jobs).await,
            ExecutorType::Automatic1111(executor) => executor.execute_batch(jobs).await,
            ExecutorType::Generic(executor) => executor.execute_batch(jobs).await,
        }
    }

    pub async fn execute_single(&self, job: &Job) -> Result<i32> {
        match self {
            ExecutorType::LlamaCpp(executor) => executor.execute_single(job).await,
            ExecutorType::ComfyUI(executor) => executor.execute_single(job).await,
            ExecutorType::Automatic1111(executor) => executor.execute_single(job).await,
            ExecutorType::Generic(executor) => executor.execute_single(job).await,
        }
    }
}

pub fn get_executor_for_backend(backend: BackendKind) -> ExecutorType {
    ExecutorType::for_backend(backend)
}
