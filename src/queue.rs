use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::fingerprint::WorkloadFingerprint;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobStatus::Pending => write!(f, "pending"),
            JobStatus::Running => write!(f, "running"),
            JobStatus::Completed => write!(f, "completed"),
            JobStatus::Failed => write!(f, "failed"),
            JobStatus::Cancelled => write!(f, "cancelled"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub args: Vec<String>,
    pub no_batch: bool,
    pub priority: Option<String>,
    pub priority_rank: u8,
    pub watch: bool,
    pub status: JobStatus,
    pub created_at: u64,
    pub started_at: Option<u64>,
    pub finished_at: Option<u64>,
    pub exit_code: Option<i32>,
    pub error_message: Option<String>,
    pub fingerprint: WorkloadFingerprint,
}

impl Job {
    pub fn new(
        args: Vec<String>,
        no_batch: bool,
        priority: Option<String>,
        watch: bool,
        fingerprint: WorkloadFingerprint,
    ) -> Result<Self> {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_micros() as u64;
        let id = format!("job-{}", created_at);

        Ok(Job {
            id,
            args,
            no_batch,
            priority_rank: parse_priority(&priority),
            priority,
            watch,
            status: JobStatus::Pending,
            created_at,
            started_at: None,
            finished_at: None,
            exit_code: None,
            error_message: None,
            fingerprint,
        })
    }

    pub fn command_line(&self) -> String {
        self.args.join(" ")
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueueState {
    pub jobs: Vec<Job>,
}

impl Default for QueueState {
    fn default() -> Self {
        QueueState { jobs: Vec::new() }
    }
}

pub fn data_dir() -> Result<PathBuf> {
    let base = env::var("XDG_DATA_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| env::var("HOME").ok().map(|home| PathBuf::from(home).join(".local/share")))
        .ok_or_else(|| anyhow!("Unable to determine data directory"))?;

    let path = base.join("gpu-run");
    create_dir_all(&path)?;
    Ok(path)
}

pub fn queue_file_path() -> Result<PathBuf> {
    Ok(data_dir()?.join("queue.json"))
}

pub fn pid_file_path() -> Result<PathBuf> {
    Ok(data_dir()?.join("daemon.pid"))
}

pub fn load_queue() -> Result<QueueState> {
    let path = queue_file_path()?;
    if !path.exists() {
        return Ok(QueueState::default());
    }
    let mut file = File::open(&path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let state = serde_json::from_str(&contents)?;
    Ok(state)
}

pub fn save_queue(state: &QueueState) -> Result<()> {
    let path = queue_file_path()?;
    let mut file = File::create(&path)?;
    let contents = serde_json::to_string_pretty(state)?;
    file.write_all(contents.as_bytes())?;
    Ok(())
}

pub fn append_job(job: Job) -> Result<()> {
    let mut state = load_queue()?;
    state.jobs.push(job);
    save_queue(&state)
}

fn parse_priority(priority: &Option<String>) -> u8 {
    match priority.as_deref().map(str::to_lowercase).as_deref() {
        Some("high") => 3,
        Some("medium") | Some("normal") => 2,
        Some("low") => 1,
        _ => 2,
    }
}
