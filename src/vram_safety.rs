use anyhow::Result;
use std::collections::HashMap;
use tracing::debug;

use crate::fingerprint::{BackendKind, WorkloadFingerprint};
use crate::queue::Job;
use crate::telemetry::GpuTelemetrySnapshot;

#[derive(Debug, Clone)]
pub struct VramEstimate {
    pub required_mb: u64,
    pub confidence: f32, // 0.0 to 1.0, how confident we are in this estimate
    pub notes: String,
}

#[derive(Debug)]
pub struct VramSafetyConfig {
    pub headroom_mb: u64,
    pub conservative_threshold: f32, // Below this confidence, use conservative estimates
    pub max_batch_size: usize,
}

impl Default for VramSafetyConfig {
    fn default() -> Self {
        Self {
            headroom_mb: 512, // 512MB headroom by default
            conservative_threshold: 0.5,
            max_batch_size: 16,
        }
    }
}

pub struct VramSafetyLayer {
    config: VramSafetyConfig,
    historical_data: HashMap<String, VramEstimate>, // Cache of past estimates
}

impl VramSafetyLayer {
    pub fn new(config: VramSafetyConfig) -> Self {
        Self {
            config,
            historical_data: HashMap::new(),
        }
    }

    pub fn can_execute_batch(&mut self, jobs: &[Job], snapshot: &GpuTelemetrySnapshot) -> Result<VramDecision> {
        if jobs.is_empty() {
            return Ok(VramDecision::Proceed);
        }

        let total_estimate = self.estimate_batch_vram(jobs)?;
        let available_mb = snapshot.vram_total_mb.saturating_sub(snapshot.vram_used_mb);

        debug!(
            "VRAM Check: {}MB required, {}MB available, {}MB headroom",
            total_estimate.required_mb,
            available_mb,
            self.config.headroom_mb
        );

        if total_estimate.required_mb + self.config.headroom_mb <= available_mb {
            Ok(VramDecision::Proceed)
        } else if total_estimate.confidence < self.config.conservative_threshold {
            // Low confidence - be conservative
            let conservative_estimate = (total_estimate.required_mb as f32 * 1.5) as u64;
            if conservative_estimate + self.config.headroom_mb <= available_mb {
                Ok(VramDecision::Proceed)
            } else {
                Ok(VramDecision::Wait(VramWaitReason::InsufficientVram {
                    required_mb: conservative_estimate,
                    available_mb,
                    headroom_mb: self.config.headroom_mb,
                }))
            }
        } else {
            // High confidence but not enough VRAM
            Ok(VramDecision::Wait(VramWaitReason::InsufficientVram {
                required_mb: total_estimate.required_mb,
                available_mb,
                headroom_mb: self.config.headroom_mb,
            }))
        }
    }

    pub fn suggest_batch_size(&mut self, jobs: &[Job], snapshot: &GpuTelemetrySnapshot) -> Result<usize> {
        if jobs.is_empty() {
            return Ok(0);
        }

        let single_job_estimate = self.estimate_single_job_vram(&jobs[0])?;
        let available_mb = snapshot.vram_total_mb.saturating_sub(snapshot.vram_used_mb);
        let safe_available = available_mb.saturating_sub(self.config.headroom_mb);

        if single_job_estimate.required_mb == 0 {
            return Ok(1); // Can't estimate, assume single job
        }

        let max_by_vram = (safe_available / single_job_estimate.required_mb) as usize;
        let suggested_size = max_by_vram.min(self.config.max_batch_size).min(jobs.len());

        debug!(
            "Batch size suggestion: {} jobs (VRAM limit: {}, config limit: {}, total jobs: {})",
            suggested_size, max_by_vram, self.config.max_batch_size, jobs.len()
        );

        Ok(suggested_size.max(1))
    }

    fn estimate_batch_vram(&mut self, jobs: &[Job]) -> Result<VramEstimate> {
        if jobs.len() == 1 {
            return self.estimate_single_job_vram(&jobs[0]);
        }

        // For batches, estimate based on the first job and multiply
        let base_estimate = self.estimate_single_job_vram(&jobs[0])?;
        
        // Batching efficiency: multiple jobs often share model loading overhead
        let efficiency_factor = match jobs[0].fingerprint.backend {
            BackendKind::LlamaCpp => 0.7, // 30% efficiency gain for batching
            BackendKind::ComfyUI => 0.8, // 20% efficiency gain
            BackendKind::Automatic1111 => 0.75, // 25% efficiency gain
            _ => 0.9, // 10% efficiency gain for unknown backends
        };

        let batch_required = (base_estimate.required_mb as f32 * 
            (1.0 + (jobs.len() - 1) as f32 * efficiency_factor)) as u64;

        Ok(VramEstimate {
            required_mb: batch_required,
            confidence: base_estimate.confidence,
            notes: format!(
                "Batch of {} jobs, efficiency factor: {:.2}",
                jobs.len(), efficiency_factor
            ),
        })
    }

    fn estimate_single_job_vram(&mut self, job: &Job) -> Result<VramEstimate> {
        let cache_key = self.create_cache_key(job);
        
        // Check cache first
        if let Some(cached) = self.historical_data.get(&cache_key) {
            return Ok(cached.clone());
        }

        let estimate = self.calculate_vram_estimate(job)?;
        
        // Cache the estimate
        self.historical_data.insert(cache_key, estimate.clone());
        
        Ok(estimate)
    }

    fn calculate_vram_estimate(&self, job: &Job) -> Result<VramEstimate> {
        match &job.fingerprint.backend {
            BackendKind::LlamaCpp => self.estimate_llamacpp_vram(&job.fingerprint),
            BackendKind::ComfyUI => self.estimate_comfyui_vram(&job.fingerprint),
            BackendKind::Automatic1111 => self.estimate_a1111_vram(&job.fingerprint),
            BackendKind::Ollama => self.estimate_ollama_vram(&job.fingerprint),
            BackendKind::Python => self.estimate_python_vram(&job.fingerprint),
            BackendKind::Unknown => Ok(VramEstimate {
                required_mb: 2048, // Conservative 2GB estimate
                confidence: 0.1,
                notes: "Unknown backend - conservative estimate".to_string(),
            }),
        }
    }

    fn estimate_llamacpp_vram(&self, fp: &WorkloadFingerprint) -> Result<VramEstimate> {
        let base_mb = match fp.model_key.as_deref() {
            Some(model) if model.contains("7b") => 5000,
            Some(model) if model.contains("13b") => 8000,
            Some(model) if model.contains("34b") => 20000,
            Some(model) if model.contains("70b") => 40000,
            _ => 6000, // Conservative default
        };

        let context_multiplier = match fp.context_length {
            Some(_ctx) if _ctx <= 2048 => 1.0,
            Some(_ctx) if _ctx <= 4096 => 1.2,
            Some(_ctx) if _ctx <= 8192 => 1.5,
            Some(_ctx) => 2.0,
            None => 1.0,
        };

        let required_mb = (base_mb as f32 * context_multiplier) as u64;

        Ok(VramEstimate {
            required_mb,
            confidence: 0.8,
            notes: format!("llama.cpp model: {:?}, context: {:?}", fp.model_key, fp.context_length),
        })
    }

    fn estimate_comfyui_vram(&self, fp: &WorkloadFingerprint) -> Result<VramEstimate> {
        let base_mb = match fp.model_key.as_deref() {
            Some(model) if model.contains("sd1.5") => 2000,
            Some(model) if model.contains("sd2.1") => 4000,
            Some(model) if model.contains("sdxl") => 6000,
            Some(model) if model.contains("flux") => 12000,
            _ => 4000,
        };

        let resolution_multiplier = match fp.resolution {
            Some((w, h)) => {
                let pixels = w * h;
                if pixels <= 512 * 512 { 1.0 }
                else if pixels <= 768 * 768 { 1.5 }
                else if pixels <= 1024 * 1024 { 2.0 }
                else { 3.0 }
            }
            None => 1.5, // Assume medium resolution
        };

        let required_mb = (base_mb as f32 * resolution_multiplier) as u64;

        Ok(VramEstimate {
            required_mb,
            confidence: 0.7,
            notes: format!("ComfyUI model: {:?}, resolution: {:?}", fp.model_key, fp.resolution),
        })
    }

    fn estimate_a1111_vram(&self, fp: &WorkloadFingerprint) -> Result<VramEstimate> {
        // Similar to ComfyUI but with different model naming
        let base_mb = match fp.model_key.as_deref() {
            Some(model) if model.contains("1.5") => 2000,
            Some(model) if model.contains("2.1") || model.contains("2.0") => 4000,
            Some(model) if model.contains("xl") || model.contains("sdxl") => 6000,
            _ => 3000,
        };

        let resolution_multiplier = match fp.resolution {
            Some((w, h)) => {
                let pixels = w * h;
                if pixels <= 512 * 512 { 1.0 }
                else if pixels <= 768 * 768 { 1.4 }
                else if pixels <= 1024 * 1024 { 1.8 }
                else { 2.5 }
            }
            None => 1.4,
        };

        let required_mb = (base_mb as f32 * resolution_multiplier) as u64;

        Ok(VramEstimate {
            required_mb,
            confidence: 0.75,
            notes: format!("A1111 model: {:?}, resolution: {:?}", fp.model_key, fp.resolution),
        })
    }

    fn estimate_ollama_vram(&self, fp: &WorkloadFingerprint) -> Result<VramEstimate> {
        let base_mb = match fp.model_key.as_deref() {
            Some(model) if model.contains("7b") => 5000,
            Some(model) if model.contains("13b") => 8000,
            Some(model) if model.contains("34b") => 20000,
            Some(model) if model.contains("70b") => 40000,
            _ => 6000,
        };

        Ok(VramEstimate {
            required_mb: base_mb,
            confidence: 0.6,
            notes: format!("Ollama model: {:?}", fp.model_key),
        })
    }

    fn estimate_python_vram(&self, fp: &WorkloadFingerprint) -> Result<VramEstimate> {
        // Very conservative estimate for generic Python scripts
        let base_mb = match fp.model_key.as_deref() {
            Some(model) if model.contains("7b") => 5000,
            Some(model) if model.contains("13b") => 8000,
            Some(model) if model.contains("34b") => 20000,
            Some(model) if model.contains("70b") => 40000,
            _ => 4000,
        };

        Ok(VramEstimate {
            required_mb: base_mb,
            confidence: 0.3,
            notes: format!("Python script with model: {:?}", fp.model_key),
        })
    }

    fn create_cache_key(&self, job: &Job) -> String {
        format!(
            "{:?}|{:?}|{:?}|{:?}",
            job.fingerprint.backend,
            job.fingerprint.model_key,
            job.fingerprint.resolution,
            job.fingerprint.context_length
        )
    }

    pub fn record_actual_usage(&mut self, job: &Job, actual_mb: u64) {
        let cache_key = self.create_cache_key(job);
        let estimate = VramEstimate {
            required_mb: actual_mb,
            confidence: 0.9, // High confidence for actual measurements
            notes: "Actual measured usage".to_string(),
        };
        self.historical_data.insert(cache_key, estimate);
    }
}

#[derive(Debug)]
pub enum VramDecision {
    Proceed,
    Wait(VramWaitReason),
}

#[derive(Debug)]
pub enum VramWaitReason {
    InsufficientVram {
        required_mb: u64,
        available_mb: u64,
        headroom_mb: u64,
    },
}

impl VramWaitReason {
    pub fn message(&self) -> String {
        match self {
            VramWaitReason::InsufficientVram { required_mb, available_mb, headroom_mb } => {
                format!(
                    "Insufficient VRAM: {}MB required, {}MB available, {}MB headroom needed",
                    required_mb, available_mb, headroom_mb
                )
            }
        }
    }
}
