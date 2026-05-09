use crate::batching::SimilarityLevel;
use crate::fingerprint::WorkloadFingerprint;
use crate::queue::{Job, JobStatus, QueueState};

pub fn pick_next_job(state: &QueueState) -> Option<usize> {
    state
        .jobs
        .iter()
        .enumerate()
        .filter(|(_, job)| job.status == JobStatus::Pending)
        .min_by_key(|(_, job)| (std::cmp::Reverse(job.priority_rank), job.created_at))
        .map(|(index, _)| index)
}

pub fn find_similar_jobs(state: &QueueState, target_job: &Job) -> Vec<usize> {
    state
        .jobs
        .iter()
        .enumerate()
        .filter(|(_, job)| {
            job.status == JobStatus::Pending
                && !job.no_batch
                && job.id != target_job.id
        })
        .filter_map(|(index, job)| {
            let similarity = compute_similarity(&target_job.fingerprint, &job.fingerprint);
            if similarity.can_batch() {
                Some(index)
            } else {
                None
            }
        })
        .collect()
}

pub fn compute_similarity(fp1: &WorkloadFingerprint, fp2: &WorkloadFingerprint) -> SimilarityLevel {
    // Different backends - no similarity
    if fp1.backend != fp2.backend {
        return SimilarityLevel::Level0;
    }

    // Level 1: Exact match for all key parameters
    if fp1.model_key == fp2.model_key
        && fp1.resolution == fp2.resolution
        && fp1.context_length == fp2.context_length
        && fp1.sampler_key == fp2.sampler_key
        && fp1.adapter_hashes == fp2.adapter_hashes
    {
        return SimilarityLevel::Level1;
    }

    // Level 2: Same model family (same model_key)
    if fp1.model_key == fp2.model_key {
        return SimilarityLevel::Level2;
    }

    // Level 3: Same backend only
    SimilarityLevel::Level3
}

pub fn create_batch(state: &QueueState, job_indices: Vec<usize>) -> Vec<Job> {
    job_indices
        .into_iter()
        .filter_map(|i| state.jobs.get(i).cloned())
        .collect()
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
