use crate::queue::{JobStatus, QueueState};

pub fn pick_next_job(state: &QueueState) -> Option<usize> {
    state
        .jobs
        .iter()
        .enumerate()
        .filter(|(_, job)| job.status == JobStatus::Pending)
        .min_by_key(|(_, job)| (std::cmp::Reverse(job.priority_rank), job.created_at))
        .map(|(index, _)| index)
}

pub fn pending_count(state: &QueueState) -> usize {
    state.jobs.iter().filter(|job| job.status == JobStatus::Pending).count()
}

pub fn running_count(state: &QueueState) -> usize {
    state.jobs.iter().filter(|job| job.status == JobStatus::Running).count()
}

pub fn completed_count(state: &QueueState) -> usize {
    state.jobs.iter().filter(|job| job.status == JobStatus::Completed).count()
}

pub fn failed_count(state: &QueueState) -> usize {
    state.jobs.iter().filter(|job| job.status == JobStatus::Failed).count()
}
