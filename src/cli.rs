use crate::{daemon, execution};
use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "gpu-run", version, about = "Intelligent GPU execution wrapper", long_about = None)]
pub struct GpuRunCli {
    #[command(subcommand)]
    pub command: GpuRunCommand,
}

#[derive(Subcommand, Debug)]
pub enum GpuRunCommand {
    Run {
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
        #[arg(long)]
        no_batch: bool,
        #[arg(long)]
        priority: Option<String>,
        #[arg(long)]
        watch: bool,
    },
    Queue,
    Status,
    Cancel {
        job_id: String,
    },
    Clear,
    InstallDaemon,
    Daemon {
        #[command(subcommand)]
        command: DaemonCommand,
    },
    Doctor,
}

#[derive(Subcommand, Debug)]
pub enum DaemonCommand {
    Start,
    Stop,
    Restart,
    Status,
}

impl GpuRunCli {
    pub async fn execute(self) -> Result<()> {
        match self.command {
            GpuRunCommand::Run {
                args,
                no_batch,
                priority,
                watch,
            } => execution::run_job(args, no_batch, priority, watch).await,
            GpuRunCommand::Queue => daemon::show_queue().await,
            GpuRunCommand::Status => daemon::show_status().await,
            GpuRunCommand::Cancel { job_id } => daemon::cancel_job(job_id).await,
            GpuRunCommand::Clear => daemon::clear_queue().await,
            GpuRunCommand::InstallDaemon => daemon::install_daemon().await,
            GpuRunCommand::Daemon { command } => match command {
                DaemonCommand::Start => daemon::start_daemon().await,
                DaemonCommand::Stop => daemon::stop_daemon().await,
                DaemonCommand::Restart => daemon::restart_daemon().await,
                DaemonCommand::Status => daemon::daemon_status().await,
            },
            GpuRunCommand::Doctor => daemon::doctor().await,
        }
    }
}
