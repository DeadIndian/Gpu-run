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
                model_key = parse_flag_value(args, "--ckpt")
                    .or_else(|| parse_flag_value(args, "--model"))
                    .or_else(|| parse_a1111_model_from_args(args));
                resolution = parse_two_values(args, "--width", "--height")
                    .or_else(|| parse_a1111_resolution_from_args(args));
                sampler_key = parse_a1111_sampler_args(args);
                adapter_hashes = parse_multiple_values(args, "--lora");
            }
            BackendKind::ComfyUI => {
                if let Some(workflow) = parse_flag_value(args, "--workflow") {
                    extra.insert("workflow".to_string(), workflow.clone());
                    // Try to parse workflow JSON for richer fingerprinting
                    if let Ok(parsed) = parse_comfyui_workflow(&workflow) {
                        model_key = parsed.model_key;
                        resolution = parsed.resolution;
                        sampler_key = parsed.sampler_key;
                        adapter_hashes = parsed.adapter_hashes;
                        extra.extend(parsed.extra);
                    }
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

fn parse_a1111_model_from_args(args: &[String]) -> Option<String> {
    // Look for model files in arguments (common patterns)
    args.iter()
        .find(|arg| {
            arg.ends_with(".safetensors") || 
            arg.ends_with(".ckpt") || 
            arg.ends_with(".pth") ||
            arg.contains("models/") ||
            arg.contains("Stable-diffusion")
        })
        .cloned()
}

fn parse_a1111_resolution_from_args(args: &[String]) -> Option<(u32, u32)> {
    // Look for resolution in various formats: --size 512x512, --size 512,512
    if let Some(size_arg) = parse_flag_value(args, "--size") {
        if size_arg.contains('x') {
            let parts: Vec<&str> = size_arg.split('x').collect();
            if parts.len() == 2 {
                if let (Ok(w), Ok(h)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                    return Some((w, h));
                }
            }
        } else if size_arg.contains(',') {
            let parts: Vec<&str> = size_arg.split(',').collect();
            if parts.len() == 2 {
                if let (Ok(w), Ok(h)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                    return Some((w, h));
                }
            }
        }
    }
    None
}

fn parse_a1111_sampler_args(args: &[String]) -> Option<String> {
    let mut parts = Vec::new();
    
    // Standard A1111 sampler parameters
    let sampler_flags = [
        ("--sampler", "sampler"),
        ("--sampler_name", "sampler"),
        ("--steps", "steps"),
        ("--cfg_scale", "cfg"),
        ("--cfg", "cfg"),
        ("--seed", "seed"),
        ("--sampling_method", "sampler"),
    ];
    
    for (flag, key) in sampler_flags {
        if let Some(value) = parse_flag_value(args, flag) {
            parts.push(format!("{}={}", key, value));
        }
    }
    
    // Handle negative prompt
    if let Some(_) = parse_flag_value(args, "--negative_prompt") {
        parts.push("has_negative".to_string());
    }
    
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

fn parse_model_positional(args: &[String]) -> Option<String> {
    args.iter()
        .skip(1)
        .find(|arg| arg.ends_with(".bin") || arg.ends_with(".pt") || arg.ends_with(".safetensors") || arg.contains("model"))
        .cloned()
}

// ComfyUI workflow parsing
#[derive(Debug)]
struct ParsedComfyUIWorkflow {
    model_key: Option<String>,
    resolution: Option<(u32, u32)>,
    sampler_key: Option<String>,
    adapter_hashes: Vec<String>,
    extra: BTreeMap<String, String>,
}

fn parse_comfyui_workflow(workflow_path: &str) -> Result<ParsedComfyUIWorkflow> {
    // Try to read and parse the workflow JSON file
    match std::fs::read_to_string(workflow_path) {
        Ok(content) => {
            match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(json) => {
                    let mut result = ParsedComfyUIWorkflow {
                        model_key: None,
                        resolution: None,
                        sampler_key: None,
                        adapter_hashes: Vec::new(),
                        extra: BTreeMap::new(),
                    };

                    // Look for CheckpointLoaderSimple nodes (model loaders)
                    if let Some(nodes) = json.get("nodes").and_then(|n| n.as_object()) {
                        for (_node_id, node) in nodes {
                            if let Some(node_obj) = node.as_object() {
                                let class_type = node_obj.get("class_type").and_then(|c| c.as_str()).unwrap_or("");
                                
                                match class_type {
                                    "CheckpointLoaderSimple" => {
                                        if let Some(ckpt_name) = node_obj.get("inputs")
                                            .and_then(|i| i.as_array())
                                            .and_then(|a| a.get(0))
                                            .and_then(|v| v.as_str()) {
                                            result.model_key = Some(ckpt_name.to_string());
                                        }
                                    }
                                    "KSampler" => {
                                        // Extract sampler parameters
                                        if let Some(inputs) = node_obj.get("inputs").and_then(|i| i.as_array()) {
                                            let mut sampler_parts = Vec::new();
                                            
                                            // sampler_name
                                            if let Some(sampler) = inputs.get(1).and_then(|v| v.as_str()) {
                                                sampler_parts.push(format!("sampler={}", sampler));
                                            }
                                            
                                            // scheduler
                                            if let Some(scheduler) = inputs.get(2).and_then(|v| v.as_str()) {
                                                sampler_parts.push(format!("scheduler={}", scheduler));
                                            }
                                            
                                            // steps
                                            if let Some(steps) = inputs.get(3).and_then(|v| v.as_u64()) {
                                                sampler_parts.push(format!("steps={}", steps));
                                            }
                                            
                                            // cfg
                                            if let Some(cfg) = inputs.get(4).and_then(|v| v.as_f64()) {
                                                sampler_parts.push(format!("cfg={}", cfg));
                                            }
                                            
                                            if !sampler_parts.is_empty() {
                                                result.sampler_key = Some(sampler_parts.join(" "));
                                            }
                                        }
                                    }
                                    "EmptyLatentImage" => {
                                        // Extract resolution
                                        if let Some(inputs) = node_obj.get("inputs").and_then(|i| i.as_array()) {
                                            if let (Some(width), Some(height)) = (
                                                inputs.get(0).and_then(|v| v.as_u64()),
                                                inputs.get(1).and_then(|v| v.as_u64())
                                            ) {
                                                result.resolution = Some((width as u32, height as u32));
                                            }
                                        }
                                    }
                                    "LoraLoader" => {
                                        // Extract LoRA names
                                        if let Some(inputs) = node_obj.get("inputs").and_then(|i| i.as_array()) {
                                            if let Some(lora_name) = inputs.get(0).and_then(|v| v.as_str()) {
                                                result.adapter_hashes.push(lora_name.to_string());
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }

                    Ok(result)
                }
                Err(_) => {
                    // JSON parsing failed, return basic fingerprint
                    Ok(ParsedComfyUIWorkflow {
                        model_key: None,
                        resolution: None,
                        sampler_key: None,
                        adapter_hashes: Vec::new(),
                        extra: BTreeMap::new(),
                    })
                }
            }
        }
        Err(_) => {
            // File reading failed, return basic fingerprint
            Ok(ParsedComfyUIWorkflow {
                model_key: None,
                resolution: None,
                sampler_key: None,
                adapter_hashes: Vec::new(),
                extra: BTreeMap::new(),
            })
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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
