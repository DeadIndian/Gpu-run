use anyhow::Result;
use std::process::Command;

#[allow(dead_code)]
pub trait GpuTelemetry: Send + Sync {
    fn utilization(&self) -> Result<f32>;
    fn vram_used_mb(&self) -> Result<u64>;
    fn vram_total_mb(&self) -> Result<u64>;
    fn temperature_celsius(&self) -> Result<f32>;
}

#[allow(dead_code)]
pub struct StubTelemetry;

impl GpuTelemetry for StubTelemetry {
    fn utilization(&self) -> Result<f32> {
        Ok(0.0)
    }

    fn vram_used_mb(&self) -> Result<u64> {
        Ok(0)
    }

    fn vram_total_mb(&self) -> Result<u64> {
        Ok(0)
    }

    fn temperature_celsius(&self) -> Result<f32> {
        Ok(0.0)
    }
}

#[derive(Debug)]
pub struct GpuTelemetrySnapshot {
    pub utilization: f32,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub temperature_celsius: f32,
    pub source: String,
}

impl Default for GpuTelemetrySnapshot {
    fn default() -> Self {
        GpuTelemetrySnapshot {
            utilization: 0.0,
            vram_used_mb: 0,
            vram_total_mb: 0,
            temperature_celsius: 0.0,
            source: "stub".to_string(),
        }
    }
}

pub fn current_snapshot() -> GpuTelemetrySnapshot {
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        if output.status.success() {
            if let Ok(stdout) = String::from_utf8(output.stdout) {
                if let Some(line) = stdout.lines().next() {
                    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
                    if parts.len() == 4 {
                        if let (Ok(util), Ok(used), Ok(total), Ok(temp)) = (
                            parts[0].parse::<f32>(),
                            parts[1].parse::<u64>(),
                            parts[2].parse::<u64>(),
                            parts[3].parse::<f32>(),
                        ) {
                            return GpuTelemetrySnapshot {
                                utilization: util,
                                vram_used_mb: used,
                                vram_total_mb: total,
                                temperature_celsius: temp,
                                source: "nvidia-smi".to_string(),
                            };
                        }
                    }
                }
            }
        }
    }

    GpuTelemetrySnapshot::default()
}

pub fn has_nvidia_smi() -> bool {
    Command::new("nvidia-smi")
        .arg("--help")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}
