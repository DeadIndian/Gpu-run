use anyhow::Result;
use tracing::info;

pub async fn schedule_job() -> Result<()> {
    info!("schedule_job called");
    println!("Scheduler is not yet implemented.");
    Ok(())
}
