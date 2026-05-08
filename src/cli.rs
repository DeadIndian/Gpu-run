use crate::{daemon, execution};
use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "gpu-run", version, about = "Intelligent GPU execution wrapper", long_about = "gpu-run is a lightweight GPU execution wrapper that queues heavy commands and reduces idle GPU time.")]
pub struct GpuRunCli {
    #[command(subcommand)]
    pub command: GpuRunCommand,
}

#[derive(Subcommand, Debug)]
pub enum GpuRunCommand {
    /// Submit a command for GPU-aware execution
    Run {
        #[arg(trailing_var_arg = true, required = true)]
        args: Vec<String>,
        /// Skip batching and run the job immediately when the daemon is free
        #[arg(long)]
        no_batch: bool,
        /// Priority for the submitted job: high, medium, or low
        #[arg(long)]
        priority: Option<String>,
        /// Keep the job active as a watch target for future similar workloads
        #[arg(long)]
        watch: bool,
    },
    /// Show queued and running jobs
    Queue,
    /// Show current GPU and queue status
    Status,
    /// Cancel a pending job by id
    Cancel {
        job_id: String,
    },
    /// Cancel all pending jobs
    Clear,
    /// Setup daemon installation or service integration
    InstallDaemon,
    /// Control the gpu-run daemon lifecycle
    Daemon {
        #[command(subcommand)]
        command: DaemonCommand,
    },
    /// Run diagnostics and environment checks
    Doctor,
    /// Show help for gpu-run or a specific subcommand
    Help {
        #[arg(value_name = "COMMAND")]
        command: Option<String>,
    },
    #[command(hide = true)]
    DaemonRun,
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
            GpuRunCommand::Help { command } => {
                Self::show_help(command);
                Ok(())
            }
            GpuRunCommand::DaemonRun => daemon::daemon_run_loop().await,
            GpuRunCommand::Doctor => daemon::doctor().await,
        }
    }

    fn show_help(command: Option<String>) {
        let mut cmd = GpuRunCli::command();
        if let Some(subcommand) = command {
            if let Some(sub) = cmd.find_subcommand_mut(&subcommand) {
                let _ = sub.print_help();
                println!();
                return;
            }
            println!("Unknown command: {}\n", subcommand);
        }
        let _ = cmd.print_help();
        println!();
    }
}
