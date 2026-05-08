use anyhow::Result;

pub trait GpuTelemetry: Send + Sync {
    fn utilization(&self) -> Result<f32>;
    fn vram_used_mb(&self) -> Result<u64>;
    fn vram_total_mb(&self) -> Result<u64>;
    fn temperature_celsius(&self) -> Result<f32>;
}

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
