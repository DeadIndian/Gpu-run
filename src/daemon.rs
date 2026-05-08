use anyhow::Result;
use std::fs::{remove_file, File};
use std::io::{Read, Write};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, Duration};

use crate::execution;
use crate::queue;
use crate::scheduler;
use crate::telemetry;

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

    loop {
        let mut state = queue::load_queue()?;
        if let Some(index) = scheduler::pick_next_job(&state) {
            state.jobs[index].status = queue::JobStatus::Running;
            state.jobs[index].started_at = Some(current_time_micros());
            queue::save_queue(&state)?;
            let job = state.jobs[index].clone();

            let result = execution::execute_job(&job).await;
            state.jobs[index].finished_at = Some(current_time_micros());
            match result {
                Ok(code) => {
                    state.jobs[index].exit_code = Some(code);
                    state.jobs[index].status = if code == 0 {
                        queue::JobStatus::Completed
                    } else {
                        queue::JobStatus::Failed
                    };
                }
                Err(err) => {
                    state.jobs[index].exit_code = Some(-1);
                    state.jobs[index].status = queue::JobStatus::Failed;
                    state.jobs[index].error_message = Some(err.to_string());
                }
            }
            queue::save_queue(&state)?;
            continue;
        }

        sleep(Duration::from_secs(1)).await;
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
