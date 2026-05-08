use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkloadFingerprint {
    pub backend: BackendKind,
    pub model_key: Option<String>,
    pub resolution: Option<(u32, u32)>,
    pub context_length: Option<u32>,
    pub sampler_key: Option<String>,
    pub adapter_hashes: Vec<String>,
    pub extra: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BackendKind {
    Ollama,
    LlamaCpp,
    ComfyUI,
    Automatic1111,
    Python,
    Unknown,
}

pub trait FingerprintExtractor {
    fn extract(&self, args: &[String], env: &std::collections::HashMap<String, String>) -> anyhow::Result<WorkloadFingerprint>;
}
