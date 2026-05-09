use anyhow::Result;
use std::fs::{remove_file, File};
use std::io::{Read, Write};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, Duration};
use tracing::{debug, error, info};

use crate::batching::BatchAssembler;
use crate::executors::get_executor_for_backend;
use crate::queue;
use crate::scheduler;
use crate::telemetry;
use crate::vram_safety::{VramSafetyConfig, VramSafetyLayer};
use crate::telemetry::GpuTelemetrySnapshot;

pub async fn show_queue() -> Result<()> {
    let state = queue::load_queue()?;
    let pending = scheduler::pending_count(&state);
    let running = scheduler::running_count(&state);
    println!("Queued jobs: {} pending, {} running", pending, running);
    for job in &state.jobs {
        println!(
            "{}  {:8}  {}  {}",
            job.id,
            job.status,
            job.created_at,
            job.command_line()
        );
    }
    Ok(())
}

pub async fn show_status() -> Result<()> {
    let snapshot = telemetry::current_snapshot();
    let state = queue::load_queue()?;
    println!("Telemetry source: {}", snapshot.source);
    println!("GPU utilization: {:.1}%", snapshot.utilization);
    println!("VRAM used: {} MiB / {} MiB", snapshot.vram_used_mb, snapshot.vram_total_mb);
    println!("GPU temperature: {:.1} °C", snapshot.temperature_celsius);
    println!("");
    println!("Queue summary:");
    println!("  pending: {}", scheduler::pending_count(&state));
    println!("  running: {}", scheduler::running_count(&state));
    println!("  completed: {}", scheduler::completed_count(&state));
    println!("  failed: {}", scheduler::failed_count(&state));
    println!("  total jobs: {}", state.jobs.len());
    Ok(())
}

pub async fn cancel_job(job_id: String) -> Result<()> {
    let mut state = queue::load_queue()?;
    if let Some(job) = state.jobs.iter_mut().find(|job| job.id == job_id) {
        if job.status == queue::JobStatus::Pending {
            job.status = queue::JobStatus::Cancelled;
            queue::save_queue(&state)?;
            println!("Cancelled job {}", job_id);
            return Ok(());
        }
        println!("Cannot cancel job {} because it is {}.", job_id, job.status);
        return Ok(());
    }

    println!("Job {} not found.", job_id);
    Ok(())
}

pub async fn clear_queue() -> Result<()> {
    let mut state = queue::load_queue()?;
    state.jobs.retain(|job| job.status == queue::JobStatus::Running);
    queue::save_queue(&state)?;
    println!("Cleared pending jobs. Running jobs remain unchanged.");
    Ok(())
}

pub async fn install_daemon() -> Result<()> {
    let exe = std::env::current_exe()?;
    let unit_content = format!(
        "[Unit]\nDescription=gpu-run daemon\nAfter=network.target\n\n[Service]\nType=simple\nExecStart={}\nRestart=always\n\n[Install]\nWantedBy=default.target\n",
        exe.display()
    );
    let config_dir = queue::data_dir()?.join("systemd/user");
    std::fs::create_dir_all(&config_dir)?;
    let unit_file = config_dir.join("gpu-run.service");
    let mut file = File::create(&unit_file)?;
    file.write_all(unit_content.as_bytes())?;
    println!("Installed systemd user unit at {}", unit_file.display());
    println!("Run `systemctl --user daemon-reload` and `systemctl --user enable --now gpu-run.service`");
    Ok(())
}

pub async fn start_daemon() -> Result<()> {
    if let Some(pid) = read_pid()? {
        if is_process_running(pid) {
            println!("Daemon already running with pid {}", pid);
            return Ok(());
        }
    }

    let exe = std::env::current_exe()?;
    let mut command = Command::new(exe);
    command.arg("daemon").arg("run");
    command.stdin(std::process::Stdio::null());
    command.stdout(std::process::Stdio::null());
    command.stderr(std::process::Stdio::null());
    let child = command.spawn()?;
    write_pid(child.id())?;
    println!("Started daemon with pid {}", child.id());
    Ok(())
}

pub async fn stop_daemon() -> Result<()> {
    if let Some(pid) = read_pid()? {
        if is_process_running(pid) {
            let status = Command::new("kill").arg(pid.to_string()).status()?;
            if status.success() {
                remove_pid_file()?;
                println!("Stopped daemon {}", pid);
                return Ok(());
            }
            println!("Failed to stop daemon {}", pid);
            return Ok(());
        }
    }
    println!("No running daemon found.");
    Ok(())
}

pub async fn restart_daemon() -> Result<()> {
    stop_daemon().await?;
    start_daemon().await
}

pub async fn daemon_status() -> Result<()> {
    if let Some(pid) = read_pid()? {
        if is_process_running(pid) {
            println!("Daemon is running with pid {}", pid);
        } else {
            println!("Daemon pid file exists but process {} is not running.", pid);
        }
    } else {
        println!("Daemon is not running.");
    }
    show_status().await
}

pub async fn doctor() -> Result<()> {
    println!("gpu-run doctor\n");
    if telemetry::has_nvidia_smi() {
        println!("nvidia-smi is available.");
    } else {
        println!("nvidia-smi is not available.");
    }
    if let Some(pid) = read_pid()? {
        println!("Daemon pid file: {}", pid);
        println!("Daemon running: {}", is_process_running(pid));
    } else {
        println!("Daemon is not running.");
    }
    println!("Queue path: {}", queue::queue_file_path()?.display());
    Ok(())
}

pub async fn daemon_run_loop() -> Result<()> {
    let pid = std::process::id();
    write_pid(pid)?;
    println!("GPU-run daemon is running with pid {}", pid);

    // Initialize components
    let mut batch_assembler = BatchAssembler::new(2000);
    let vram_config = VramSafetyConfig::default();
    let mut vram_safety = VramSafetyLayer::new(vram_config);
    
    loop {
        let mut state = queue::load_queue()?;
        let gpu_snapshot: GpuTelemetrySnapshot = telemetry::current_snapshot();
        
        // Add new pending jobs to batch assembler
        let pending_jobs: Vec<_> = state.jobs
            .iter()
            .filter(|job| job.status == queue::JobStatus::Pending)
            .cloned()
            .collect();
            
        for job in pending_jobs {
            let immediate_jobs = batch_assembler.add_job(job);
            
            // Execute jobs that shouldn't be batched immediately
            for job in immediate_jobs {
                if can_execute_job_vram_safe(&mut vram_safety, &job, &gpu_snapshot).await {
                    execute_single_job(&mut state, &job).await?;
                }
            }
        }
        
        // Check for ready batches
        let ready_batches = batch_assembler.get_ready_batches();
        
        for batch in ready_batches {
            if batch.len() == 1 {
                // Single job - execute normally
                if can_execute_job_vram_safe(&mut vram_safety, &batch[0], &gpu_snapshot).await {
                    execute_single_job(&mut state, &batch[0]).await?;
                }
            } else {
                // Multiple jobs - check VRAM and potentially split batch
                match vram_safety.can_execute_batch(&batch, &gpu_snapshot) {
                    Ok(crate::vram_safety::VramDecision::Proceed) => {
                        execute_batch_jobs(&mut state, &batch).await?;
                    }
                    Ok(crate::vram_safety::VramDecision::Wait(reason)) => {
                        debug!("Batch waiting due to VRAM constraints: {}", reason.message());
                        // Split batch into smaller chunks
                        let suggested_size = vram_safety.suggest_batch_size(&batch, &gpu_snapshot)?;
                        if suggested_size > 0 && suggested_size < batch.len() {
                            let smaller_batch: Vec<_> = batch.iter().take(suggested_size).cloned().collect();
                            info!("Splitting batch into smaller chunk of {} jobs", smaller_batch.len());
                            if can_execute_batch_vram_safe(&mut vram_safety, &smaller_batch, &gpu_snapshot).await {
                                execute_batch_jobs(&mut state, &smaller_batch).await?;
                            }
                        }
                    }
                    Err(e) => {
                        error!("VRAM safety check failed: {}", e);
                        // Proceed with caution for single jobs
                        if batch.len() == 1 {
                            execute_single_job(&mut state, &batch[0]).await?;
                        }
                    }
                }
            }
        }
        
        // Also check for individual jobs that should run immediately (no-batch or high priority)
        if let Some(index) = scheduler::pick_next_job(&state) {
            let job = &state.jobs[index];
            if job.no_batch || job.priority_rank >= 3 {
                if can_execute_job_vram_safe(&mut vram_safety, job, &gpu_snapshot).await {
                    let job_clone = job.clone();
                    execute_single_job(&mut state, &job_clone).await?;
                }
            }
        }

        sleep(Duration::from_millis(100)).await; // Check more frequently for batching
    }
}

async fn execute_single_job(state: &mut queue::QueueState, job: &queue::Job) -> Result<()> {
    let job_index = state.jobs.iter()
        .position(|j| j.id == job.id)
        .ok_or_else(|| anyhow::anyhow!("Job {} not found in state", job.id))?;
    
    info!("Executing single job: {}", job.command_line());
    
    state.jobs[job_index].status = queue::JobStatus::Running;
    state.jobs[job_index].started_at = Some(current_time_micros());
    queue::save_queue(state)?;
    
    let executor = get_executor_for_backend(job.fingerprint.backend.clone());
    let result = executor.execute_single(job).await;
    
    finalize_job_execution(state, job_index, result).await
}

async fn execute_batch_jobs(state: &mut queue::QueueState, jobs: &[queue::Job]) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    
    info!("Executing batch of {} jobs", jobs.len());
    
    // Mark all jobs as running
    for job in jobs {
        if let Some(index) = state.jobs.iter().position(|j| j.id == job.id) {
            state.jobs[index].status = queue::JobStatus::Running;
            state.jobs[index].started_at = Some(current_time_micros());
        }
    }
    queue::save_queue(state)?;
    
    // Get executor for the first job's backend (all should be same backend)
    let executor = get_executor_for_backend(jobs[0].fingerprint.backend.clone());
    let result = executor.execute_batch(jobs.to_vec()).await;
    
    // Update all jobs with results
    for (i, job) in jobs.iter().enumerate() {
        if let Some(index) = state.jobs.iter().position(|j| j.id == job.id) {
            let exit_code = result.as_ref().map(|codes| codes.get(i).copied().unwrap_or(-1)).unwrap_or(-1);
            finalize_job_execution(state, index, Ok(exit_code)).await?;
        }
    }
    
    Ok(())
}

async fn finalize_job_execution(
    state: &mut queue::QueueState,
    job_index: usize,
    result: Result<i32>,
) -> Result<()> {
    state.jobs[job_index].finished_at = Some(current_time_micros());
    match result {
        Ok(code) => {
            state.jobs[job_index].exit_code = Some(code);
            state.jobs[job_index].status = if code == 0 {
                queue::JobStatus::Completed
            } else {
                queue::JobStatus::Failed
            };
            debug!("Job {} completed with exit code: {}", state.jobs[job_index].id, code);
        }
        Err(err) => {
            state.jobs[job_index].exit_code = Some(-1);
            state.jobs[job_index].status = queue::JobStatus::Failed;
            state.jobs[job_index].error_message = Some(err.to_string());
            error!("Job {} failed: {}", state.jobs[job_index].id, err);
        }
    }
    queue::save_queue(state)?;
    Ok(())
}

async fn can_execute_job_vram_safe(
    vram_safety: &mut VramSafetyLayer,
    job: &queue::Job,
    gpu_snapshot: &GpuTelemetrySnapshot,
) -> bool {
    match vram_safety.can_execute_batch(&[job.clone()], gpu_snapshot) {
        Ok(crate::vram_safety::VramDecision::Proceed) => true,
        Ok(crate::vram_safety::VramDecision::Wait(reason)) => {
            debug!("Job {} waiting due to VRAM: {}", job.id, reason.message());
            false
        }
        Err(e) => {
            error!("VRAM check failed for job {}: {}", job.id, e);
            true // Proceed with caution if check fails
        }
    }
}

async fn can_execute_batch_vram_safe(
    vram_safety: &mut VramSafetyLayer,
    batch: &[queue::Job],
    gpu_snapshot: &GpuTelemetrySnapshot,
) -> bool {
    match vram_safety.can_execute_batch(batch, gpu_snapshot) {
        Ok(crate::vram_safety::VramDecision::Proceed) => true,
        Ok(crate::vram_safety::VramDecision::Wait(reason)) => {
            debug!("Batch waiting due to VRAM: {}", reason.message());
            false
        }
        Err(e) => {
            error!("VRAM check failed for batch: {}", e);
            batch.len() == 1 // Only proceed if it's a single job
        }
    }
}

fn current_time_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_micros() as u64)
        .unwrap_or_default()
}

fn read_pid() -> Result<Option<u32>> {
    let path = queue::pid_file_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let mut file = File::open(&path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let pid = contents.trim().parse().ok();
    Ok(pid)
}

fn write_pid(pid: u32) -> Result<()> {
    let path = queue::pid_file_path()?;
    let mut file = File::create(&path)?;
    write!(file, "{}", pid)?;
    Ok(())
}

fn remove_pid_file() -> Result<()> {
    let path = queue::pid_file_path()?;
    if path.exists() {
        remove_file(path)?;
    }
    Ok(())
}

fn is_process_running(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}
