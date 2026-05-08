use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::collections::HashMap;

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

impl WorkloadFingerprint {
    pub fn from_command(args: &[String], _env: &HashMap<String, String>) -> Result<Self> {
        let backend = detect_backend(args);
        let mut model_key = None;
        let mut resolution = None;
        let mut context_length = None;
        let mut sampler_key = None;
        let mut adapter_hashes = Vec::new();
        let mut extra = BTreeMap::new();

        match backend {
            BackendKind::Ollama => {
                model_key = args.get(2).cloned();
                context_length = parse_flag_value(args, "--ctx-size").and_then(|v| v.parse().ok());
                sampler_key = parse_sampler_args(args);
            }
            BackendKind::LlamaCpp => {
                model_key = parse_flag_value(args, "--model").or_else(|| parse_model_positional(args));
                context_length = parse_flag_value(args, "-c").and_then(|v| v.parse().ok());
                sampler_key = parse_sampler_args(args);
            }
            BackendKind::Automatic1111 => {
                model_key = parse_flag_value(args, "--ckpt");
                resolution = parse_two_values(args, "--width", "--height");
                sampler_key = parse_sampler_args(args);
                adapter_hashes = parse_multiple_values(args, "--lora");
            }
            BackendKind::ComfyUI => {
                if let Some(workflow) = parse_flag_value(args, "--workflow") {
                    extra.insert("workflow".to_string(), workflow);
                }
            }
            BackendKind::Python => {
                model_key = parse_multiple_values(args, "--model")
                    .into_iter()
                    .next()
                    .or_else(|| parse_multiple_values(args, "--checkpoint").into_iter().next())
                    .or_else(|| parse_multiple_values(args, "--ckpt").into_iter().next());
            }
            BackendKind::Unknown => {}
        }

        Ok(WorkloadFingerprint {
            backend,
            model_key,
            resolution,
            context_length,
            sampler_key,
            adapter_hashes,
            extra,
        })
    }
}

fn detect_backend(args: &[String]) -> BackendKind {
    if args.is_empty() {
        return BackendKind::Unknown;
    }

    let program = args[0].to_lowercase();

    if program.contains("ollama") {
        BackendKind::Ollama
    } else if program.contains("llama") || program.contains("llama.cpp") {
        BackendKind::LlamaCpp
    } else if program.contains("comfy") {
        BackendKind::ComfyUI
    } else if program.contains("automatic1111") || program.contains("a1111") || program.contains("sd") {
        BackendKind::Automatic1111
    } else if program.contains("python") || program.ends_with(".py") {
        BackendKind::Python
    } else {
        BackendKind::Unknown
    }
}

fn parse_flag_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find_map(|window| if window[0] == flag { Some(window[1].clone()) } else { None })
}

fn parse_two_values(args: &[String], width_flag: &str, height_flag: &str) -> Option<(u32, u32)> {
    let width = parse_flag_value(args, width_flag).and_then(|v| v.parse().ok());
    let height = parse_flag_value(args, height_flag).and_then(|v| v.parse().ok());
    match (width, height) {
        (Some(w), Some(h)) => Some((w, h)),
        _ => None,
    }
}

fn parse_sampler_args(args: &[String]) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(temp) = parse_flag_value(args, "--temp") {
        parts.push(format!("temp={}", temp));
    }
    if let Some(top_p) = parse_flag_value(args, "--top-p") {
        parts.push(format!("top_p={}", top_p));
    }
    if let Some(top_k) = parse_flag_value(args, "--top-k") {
        parts.push(format!("top_k={}", top_k));
    }
    if let Some(cfg) = parse_flag_value(args, "--cfg_scale") {
        parts.push(format!("cfg={}", cfg));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

fn parse_multiple_values(args: &[String], flag: &str) -> Vec<String> {
    args.windows(2)
        .filter_map(|window| if window[0] == flag { Some(window[1].clone()) } else { None })
        .collect()
}

fn parse_model_positional(args: &[String]) -> Option<String> {
    args.iter()
        .skip(1)
        .find(|arg| arg.ends_with(".bin") || arg.ends_with(".pt") || arg.ends_with(".safetensors") || arg.contains("model"))
        .cloned()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum BackendKind {
    Ollama,
    LlamaCpp,
    ComfyUI,
    Automatic1111,
    Python,
    Unknown,
}

#[allow(dead_code)]
pub trait FingerprintExtractor {
    fn extract(&self, args: &[String], env: &HashMap<String, String>) -> Result<WorkloadFingerprint>;
}

#[allow(dead_code)]
pub struct BasicFingerprintExtractor;

impl FingerprintExtractor for BasicFingerprintExtractor {
    fn extract(&self, args: &[String], env: &HashMap<String, String>) -> Result<WorkloadFingerprint> {
        WorkloadFingerprint::from_command(args, env)
    }
}
