use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::fingerprint::WorkloadFingerprint;
use crate::queue::{Job, JobStatus};

#[derive(Debug, Clone, PartialEq)]
pub enum SimilarityLevel {
    Level1, // Exact batch candidate: same model + same resolution/sampler params
    Level2, // Same model family: same model_key, different params
    Level3, // Same backend: same BackendKind only
    Level0, // No similarity: different backends or unrecognized
}

impl SimilarityLevel {
    pub fn can_batch(&self) -> bool {
        matches!(self, SimilarityLevel::Level1)
    }
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

#[derive(Debug)]
pub struct BatchWindow {
    pub fingerprint: WorkloadFingerprint,
    pub jobs: Vec<Job>,
    pub created_at: Instant,
    pub expires_at: Instant,
}

impl BatchWindow {
    pub fn new(fingerprint: WorkloadFingerprint, job: Job, timeout_ms: u64) -> Self {
        let now = Instant::now();
        Self {
            fingerprint,
            jobs: vec![job],
            created_at: now,
            expires_at: now + Duration::from_millis(timeout_ms),
        }
    }

    pub fn add_job(&mut self, job: Job) {
        self.jobs.push(job);
        // Reset expiration timer when new job is added
        self.expires_at = Instant::now() + Duration::from_millis(2000); // Default timeout
    }

    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }

    pub fn can_add_job(&self, job: &Job) -> bool {
        if job.no_batch {
            return false;
        }
        if job.status != JobStatus::Pending {
            return false;
        }
        compute_similarity(&self.fingerprint, &job.fingerprint).can_batch()
    }
}

#[derive(Debug)]
pub struct BatchAssembler {
    windows: HashMap<String, BatchWindow>, // Key: fingerprint hash
    default_timeout_ms: u64,
}

impl BatchAssembler {
    pub fn new(default_timeout_ms: u64) -> Self {
        Self {
            windows: HashMap::new(),
            default_timeout_ms,
        }
    }

    pub fn add_job(&mut self, job: Job) -> Vec<Job> {
        if job.no_batch {
            return vec![job]; // Return immediately for no-batch jobs
        }

        let fingerprint_key = self.fingerprint_key(&job.fingerprint);
        
        if let Some(window) = self.windows.get_mut(&fingerprint_key) {
            if window.can_add_job(&job) {
                window.add_job(job);
                return vec![]; // Job added to batch window, don't execute yet
            }
        }

        // Create new batch window for this job
        let window = BatchWindow::new(job.fingerprint.clone(), job, self.default_timeout_ms);
        self.windows.insert(fingerprint_key, window);
        vec![]
    }

    pub fn get_ready_batches(&mut self) -> Vec<Vec<Job>> {
        let mut ready_batches = Vec::new();
        let mut expired_keys = Vec::new();

        for (key, window) in &self.windows {
            if window.is_expired() && !window.jobs.is_empty() {
                ready_batches.push(window.jobs.clone());
                expired_keys.push(key.clone());
            }
        }

        // Remove expired windows
        for key in expired_keys {
            self.windows.remove(&key);
        }

        ready_batches
    }

    pub fn flush_all(&mut self) -> Vec<Vec<Job>> {
        let mut all_batches = Vec::new();
        
        for (_, window) in std::mem::take(&mut self.windows) {
            if !window.jobs.is_empty() {
                all_batches.push(window.jobs);
            }
        }

        all_batches
    }

    fn fingerprint_key(&self, fp: &WorkloadFingerprint) -> String {
        // Create a hash key for Level 1 similarity (exact match)
        format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
            fp.backend,
            fp.model_key,
            fp.resolution,
            fp.context_length,
            fp.sampler_key,
            fp.adapter_hashes.join(",")
        )
    }

    pub fn pending_job_count(&self) -> usize {
        self.windows.values().map(|w| w.jobs.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::BackendKind;

    fn create_test_fingerprint(backend: BackendKind, model: Option<&str>) -> WorkloadFingerprint {
        WorkloadFingerprint {
            backend,
            model_key: model.map(|s| s.to_string()),
            resolution: None,
            context_length: None,
            sampler_key: None,
            adapter_hashes: Vec::new(),
            extra: std::collections::BTreeMap::new(),
        }
    }

    #[test]
    fn test_similarity_levels() {
        let fp1 = create_test_fingerprint(BackendKind::LlamaCpp, Some("llama3:8b"));
        let fp2 = create_test_fingerprint(BackendKind::LlamaCpp, Some("llama3:8b"));
        let fp3 = create_test_fingerprint(BackendKind::LlamaCpp, Some("llama3:7b"));
        let fp4 = create_test_fingerprint(BackendKind::Ollama, Some("llama3:8b"));

        assert_eq!(compute_similarity(&fp1, &fp2), SimilarityLevel::Level1);
        assert_eq!(compute_similarity(&fp1, &fp3), SimilarityLevel::Level2);
        assert_eq!(compute_similarity(&fp1, &fp4), SimilarityLevel::Level0);
    }

    #[test]
    fn test_batch_window() {
        let fp = create_test_fingerprint(BackendKind::LlamaCpp, Some("llama3:8b"));
        let job = Job::new(
            vec!["llama.cpp".to_string(), "--model".to_string(), "model.gguf".to_string()],
            false,
            None,
            false,
            fp.clone(),
        ).unwrap();

        let mut window = BatchWindow::new(fp, job, 1000);
        assert!(!window.is_expired());
        assert_eq!(window.jobs.len(), 1);
    }
}
