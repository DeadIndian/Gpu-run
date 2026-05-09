use std::future::Future;

#[cfg(feature = "python")]
use anyhow::Result;
#[cfg(feature = "python")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
mod batching;
#[cfg(feature = "python")]
mod cli;
#[cfg(feature = "python")]
mod daemon;
#[cfg(feature = "python")]
mod execution;
#[cfg(feature = "python")]
mod executors;
#[cfg(feature = "python")]
mod fingerprint;
#[cfg(feature = "python")]
mod queue;
#[cfg(feature = "python")]
mod scheduler;
#[cfg(feature = "python")]
mod telemetry;
#[cfg(feature = "python")]
mod vram_safety;

#[cfg(feature = "python")]
fn block_on<F>(future: F) -> PyResult<()>
where
    F: Future<Output = Result<()>>,
{
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(2)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    runtime
        .block_on(async { future.await })
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[cfg(feature = "python")]
#[cfg(feature = "python")]
#[pyfunction]
fn start_daemon() -> PyResult<()> {
    block_on(daemon::start_daemon())
}

#[cfg(feature = "python")]
#[pyfunction]
fn restart_daemon() -> PyResult<()> {
    block_on(daemon::restart_daemon())
}

#[cfg(feature = "python")]
#[pyfunction]
fn stop_daemon() -> PyResult<()> {
    block_on(daemon::stop_daemon())
}

#[cfg(feature = "python")]
#[pyfunction]
fn show_queue() -> PyResult<()> {
    block_on(daemon::show_queue())
}

#[cfg(feature = "python")]
#[pyfunction]
fn show_status() -> PyResult<()> {
    block_on(daemon::show_status())
}

#[cfg(feature = "python")]
#[pyfunction]
fn cancel_job(job_id: String) -> PyResult<()> {
    block_on(daemon::cancel_job(job_id))
}

#[cfg(feature = "python")]
#[pyfunction]
fn clear_queue() -> PyResult<()> {
    block_on(daemon::clear_queue())
}

#[cfg(feature = "python")]
#[pyfunction]
fn install_daemon() -> PyResult<()> {
    block_on(daemon::install_daemon())
}

#[cfg(feature = "python")]
#[pyfunction]
fn doctor() -> PyResult<()> {
    block_on(daemon::doctor())
}

#[cfg(feature = "python")]
#[pyfunction(signature=(args, no_batch=false, priority=None, watch=false))]
fn run_gpu_run(
    args: Vec<String>,
    no_batch: bool,
    priority: Option<String>,
    watch: bool,
) -> PyResult<()> {
    block_on(execution::run_job(args, no_batch, priority, watch))
}

#[cfg(feature = "python")]
#[allow(deprecated)]
#[pymodule]
fn gpu_run(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_daemon, m)?)?;
    m.add_function(wrap_pyfunction!(restart_daemon, m)?)?;
    m.add_function(wrap_pyfunction!(stop_daemon, m)?)?;
    m.add_function(wrap_pyfunction!(show_queue, m)?)?;
    m.add_function(wrap_pyfunction!(show_status, m)?)?;
    m.add_function(wrap_pyfunction!(cancel_job, m)?)?;
    m.add_function(wrap_pyfunction!(clear_queue, m)?)?;
    m.add_function(wrap_pyfunction!(install_daemon, m)?)?;
    m.add_function(wrap_pyfunction!(doctor, m)?)?;
    m.add_function(wrap_pyfunction!(run_gpu_run, m)?)?;
    Ok(())
}

#[cfg(not(feature = "python"))]
fn __dummy_lib_root() {}
