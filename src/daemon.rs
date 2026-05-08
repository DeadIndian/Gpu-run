use anyhow::Result;
use tracing::info;

pub async fn show_queue() -> Result<()> {
    info!("show_queue called");
    println!("Queue is not yet implemented.");
    Ok(())
}

pub async fn show_status() -> Result<()> {
    info!("show_status called");
    println!("Status is not yet implemented.");
    Ok(())
}

pub async fn cancel_job(job_id: String) -> Result<()> {
    info!(job_id = %job_id, "cancel_job called");
    println!("Cancel job {} is not yet implemented.", job_id);
    Ok(())
}

pub async fn clear_queue() -> Result<()> {
    info!("clear_queue called");
    println!("Clear queue is not yet implemented.");
    Ok(())
}

pub async fn install_daemon() -> Result<()> {
    info!("install_daemon called");
    println!("Daemon installation is not yet implemented.");
    Ok(())
}

pub async fn start_daemon() -> Result<()> {
    info!("start_daemon called");
    println!("Daemon start is not yet implemented.");
    Ok(())
}

pub async fn stop_daemon() -> Result<()> {
    info!("stop_daemon called");
    println!("Daemon stop is not yet implemented.");
    Ok(())
}

pub async fn restart_daemon() -> Result<()> {
    info!("restart_daemon called");
    println!("Daemon restart is not yet implemented.");
    Ok(())
}

pub async fn daemon_status() -> Result<()> {
    info!("daemon_status called");
    println!("Daemon status is not yet implemented.");
    Ok(())
}

pub async fn doctor() -> Result<()> {
    info!("doctor called");
    println!("Doctor checks are not yet implemented.");
    Ok(())
}
