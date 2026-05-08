# gpu-run: Intelligent GPU Execution Wrapper

## 1. Project Overview

**gpu-run** is a lightweight, zero-integration CLI wrapper that sits between the user and their existing local AI tools. It intercepts GPU-heavy commands, queues them intelligently, detects similar workloads, dynamically batches compatible jobs, and schedules them to maximize GPU utilization.

Instead of the common pattern of **(GPU spike → idle → spike)**, `gpu-run` turns sequential, bursty execution into smooth, high-throughput pipelines.

**Tagline**: "Make your GPU actually work like a GPU."

---

## 2. Problem Statement

Modern local AI workflows (Stable Diffusion, Flux, Ollama, llama.cpp, custom Python scripts, etc.) suffer from severe GPU underutilization because:

- Tools execute tasks immediately and sequentially.
- Users run jobs one-by-one (manually or via simple scripts).
- No awareness of upcoming similar workloads.
- Result: Frequent idle periods between short inference bursts.

---

## 3. Goals

### Primary Goals

- **Dramatically increase effective GPU throughput** for batchable workloads.
- **Zero code changes** — drop-in replacement for existing commands.
- **Stay extremely lightweight** and easy to install/use.
- **Work across different AI tools** (image gen, LLMs, custom scripts).

### Secondary Goals

- Reduce total wall-clock time for multi-job sessions.
- Minimize thermal cycling and power waste.
- Provide visibility into GPU utilization and queue status.
- Be extensible for new backends and advanced scheduling policies.

---

## 4. Platform Support

**Supported platforms: Linux and Windows only.** macOS is explicitly out of scope — Metal does not expose the batching or telemetry primitives necessary for the core value proposition, and supporting it would add significant complexity for limited gain.

| Platform | GPU Telemetry         | Tool Compatibility                | Priority | Notes                         |
| -------- | --------------------- | --------------------------------- | -------- | ----------------------------- |
| Linux    | Excellent (NVML/ROCm) | Best (all tools)                  | Tier 0   | Primary development platform  |
| Windows  | Very Good (NVML)      | Very Good (Ollama, ComfyUI, etc.) | Tier 0   | Native only — no WSL required |

### Implementation Approach

#### 1. Telemetry Abstraction Layer

A `GpuTelemetry` trait abstracts over platform-specific implementations:

```rust
pub trait GpuTelemetry: Send + Sync {
    fn utilization(&self) -> Result<f32>;       // 0.0–100.0
    fn vram_used_mb(&self) -> Result<u64>;
    fn vram_total_mb(&self) -> Result<u64>;
    fn temperature_celsius(&self) -> Result<f32>;
}
```

Backends:

- **Linux NVIDIA**: `nvml-wrapper` crate
- **Linux AMD**: `rocm-smi` subprocess + JSON parsing
- **Windows NVIDIA**: `nvml-wrapper` (NVML ships with drivers)
- **Windows AMD**: `rocm-smi` subprocess or WMI fallback
- **Fallback**: Poll `nvidia-smi --query-gpu` / `rocm-smi` subprocesses if libraries unavailable

Feature flags: `nvml`, `rocm`

#### 2. Process Execution

Abstract platform differences behind a `PlatformExecutor` trait:

```rust
pub trait PlatformExecutor {
    fn spawn(&self, cmd: &ResolvedCommand) -> Result<Child>;
    fn resolve_path(&self, name: &str) -> Result<PathBuf>;
    fn inject_env(&self, cmd: &mut Command, vars: &EnvMap);
}
```

- **Linux**: standard `std::process::Command`
- **Windows**: `.exe` extension handling, `CUDA_VISIBLE_DEVICES` injection, PowerShell/CMD detection
- Environment variable passthrough: `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `TRANSFORMERS_CACHE`, etc.

#### 3. IPC (Daemon ↔ CLI)

- **Linux**: Unix domain sockets (`/run/user/<uid>/gpu-run.sock`)
- **Windows**: Named pipes (`\\.\pipe\gpu-run-<username>`)
- Local TCP (`127.0.0.1:PORT`) as fallback for both platforms
- Protocol: length-prefixed JSON or MessagePack frames

#### 4. Installation & Distribution

- **Linux**: Single binary + systemd user service unit file; optional `.deb`/`.rpm` packages
- **Windows**: Single `.exe` + optional MSI installer; Scoop/Winget support; auto-detects NVIDIA/AMD drivers; adds to `PATH`; optional "Run at startup" via Task Scheduler
- Both: `gpu-run install-daemon` command sets up the background service automatically

#### 5. Configuration

```toml
[telemetry]
provider = "auto"        # auto | nvml | rocm | none
poll_interval_ms = 500

[execution]
shell = "auto"           # auto | cmd | powershell | pwsh (Windows only)
job_timeout_secs = 3600

[scheduler]
default_batch_window_ms = 2000   # how long to wait for similar jobs before dispatching
max_queue_depth = 64
vram_headroom_mb = 512           # reserved VRAM buffer before refusing new jobs
```

---

## 5. Workload Fingerprinting Design

This is the most complex component and deserves explicit design. Fingerprinting works differently per backend because tools expose completely different interfaces.

### Fingerprint Structure

```rust
pub struct WorkloadFingerprint {
    pub backend: BackendKind,
    pub model_key: Option<String>,      // normalized model name/hash
    pub resolution: Option<(u32, u32)>, // image gen only
    pub context_length: Option<u32>,    // LLM only
    pub sampler_key: Option<String>,    // normalized sampler + params
    pub adapter_hashes: Vec<String>,    // LoRA/ControlNet hashes
    pub extra: BTreeMap<String, String>,// backend-specific extras
}
```

### Per-Backend Fingerprint Extraction

Each backend implements a `FingerprintExtractor` trait:

```rust
pub trait FingerprintExtractor {
    /// Extract fingerprint from raw CLI args + environment
    fn extract(&self, args: &[String], env: &EnvMap) -> Result<WorkloadFingerprint>;
}
```

**Ollama** (`ollama run <model> <prompt>`):

- `model_key`: model name from args (normalized, e.g. `llama3:8b`)
- `context_length`: from `--ctx-size` flag or `~/.ollama/config`
- Sampler params: `--temperature`, `--top-p`, `--top-k` if present
- Note: Ollama manages its own batching internally; `gpu-run` queues at the _job_ level, not the token level

**llama.cpp / llama-cli**:

- `model_key`: SHA-256 of the model file path (not content, for speed) or `--model` arg
- `context_length`: `-c` / `--ctx-size`
- `sampler_key`: hash of `--temp`, `--top-p`, `--top-k`, `--repeat-penalty`
- Batch size: `-b` flag

**ComfyUI** (via `--workflow <file>`):

- Parse workflow JSON to extract: checkpoint name, VAE, resolution, sampler, steps, CFG
- `model_key`: checkpoint filename (normalized)
- `resolution`: `(width, height)` from the KSampler or EmptyLatentImage node
- `adapter_hashes`: LoRA node hashes if present
- Note: ComfyUI has its own queue API; `gpu-run` wraps the CLI invocation, not the API

**Automatic1111 / Forge** (via CLI flags):

- `model_key`: `--ckpt` arg filename
- `resolution`: `--width` × `--height`
- `sampler_key`: `--sampler_name` + `--steps` + `--cfg_scale`
- `adapter_hashes`: `--lora` args

**Generic Python scripts** (`python script.py ...`):

- Best-effort: extract `--model`, `--checkpoint`, `--ckpt` flags if present
- `model_key`: value of first recognized model flag, else `None`
- Fingerprint degrades gracefully to backend-only (Level 3 similarity)
- Users can annotate scripts with a comment `# gpu-run: model=<name> resolution=512x512` for richer fingerprinting

### Similarity Levels

```
Level 1 — Exact batch candidate:  same model + same resolution/sampler params
Level 2 — Same model family:      same model_key, different params
Level 3 — Same backend:           same BackendKind only
Level 0 — No similarity:          different backends or unrecognized
```

Only Level 1 jobs are candidates for dynamic batching. Levels 2–3 are used for scheduling order (run similar work together to avoid model reloads).

---

## 6. Batching Design

### What Batching Means Per Backend

Batching is **not** a single universal operation — it means different things per tool, and `gpu-run` must be explicit about this:

| Backend        | Batching Mechanism                                                           | gpu-run Role                                                |
| -------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------- |
| llama.cpp      | `-b` flag controls token batch size; multiple prompts via `--prompt-file`    | Merge Level-1 jobs into a single `--prompt-file` invocation |
| Ollama         | Manages its own batching via `num_parallel`; cannot be externally batched    | Queue jobs sequentially; avoid model reload between them    |
| ComfyUI        | Supports queue via REST API (`/prompt`); can submit multiple workflows       | Submit all Level-1 jobs to ComfyUI queue in one session     |
| A1111 / Forge  | `--batch_size` flag for same-prompt repeats; no native multi-prompt batching | Group same-params jobs; use `--batch_size N` where possible |
| Python scripts | Tool-defined; `gpu-run` cannot safely batch generically                      | Queue sequentially; minimize idle gaps between jobs         |

### Batch Window

When a job is submitted, the scheduler waits up to `batch_window_ms` (configurable, default 2000ms) for additional Level-1-similar jobs before dispatching. This allows scripts that submit many jobs in a loop to be collected into a single batch.

```
t=0ms    Job A submitted (llama.cpp, llama3:8b, ctx=2048)
t=200ms  Job B submitted (same fingerprint) → reset window
t=400ms  Job C submitted (same fingerprint) → reset window
t=2400ms Window expires → dispatch A+B+C as merged --prompt-file invocation
```

### VRAM Safety

Before dispatching any batch:

1. Query current VRAM usage via telemetry.
2. Estimate batch VRAM requirement (per-backend heuristics + historical data).
3. If `vram_used + estimated_batch > vram_total - vram_headroom_mb`, split the batch or wait.
4. VRAM estimation degrades to conservative defaults for unknown tools.

---

## 7. Architecture

```
User → gpu-run CLI → IPC Client → Daemon (gpu-rund)
                                        │
                              ┌─────────▼──────────┐
                              │   Job Receiver      │
                              └─────────┬──────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  Fingerprint Engine │
                              │  (per-backend)      │
                              └─────────┬──────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  Scheduler /        │
                              │  Batch Assembler    │
                              └─────────┬──────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  VRAM Safety Check  │
                              └─────────┬──────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  Executor           │
                              │  (per-backend)      │
                              └─────────┬──────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  Actual Tools       │
                              │  (Ollama, llama.cpp,│
                              │   ComfyUI, etc.)    │
                              └────────────────────┘

              [Telemetry Monitor runs continuously in background]
```

**Components:**

1. **CLI** (`gpu-run`) — thin client, forwards commands to daemon via IPC
2. **Daemon** (`gpu-rund`) — persistent background service, owns the queue and scheduler
3. **Fingerprint Engine** — per-backend extractors, produces `WorkloadFingerprint`
4. **Scheduler / Batch Assembler** — FIFO base with similarity grouping + batch window logic
5. **VRAM Safety Layer** — checks telemetry before dispatch, splits or defers jobs
6. **Telemetry Monitor** — continuous GPU polling, feeds VRAM/util data to scheduler
7. **Backend Executors** — pluggable runners, one per supported tool

---

## 8. Security Model

Routing arbitrary shell commands through a persistent background daemon is a real attack surface. The following design decisions are non-negotiable:

1. **User-scoped only**: The daemon runs as the invoking user, never root/SYSTEM. It inherits no elevated privileges.
2. **No remote access**: IPC socket/pipe is user-scoped and not exposed on any network interface. Local TCP fallback binds only to `127.0.0.1`.
3. **Command allowlist (optional)**: Users can configure an explicit allowlist of permitted command prefixes. Unlisted commands are rejected with an error, not silently executed.
4. **No credential passthrough**: `gpu-run` does not read, store, or forward API keys, tokens, or secrets from environment variables beyond what the child process needs.
5. **Audit log**: Every executed command, its arguments, submitting PID, and timestamp are written to `~/.local/share/gpu-run/audit.log` (Linux) or `%APPDATA%\gpu-run\audit.log` (Windows). Log rotation at 10MB.
6. **Job isolation**: Jobs run as separate child processes; they do not share memory or file descriptors with the daemon or each other.
7. **Socket permissions**: Unix socket is `chmod 0700`. Named pipe ACL restricts to current user SID.

---

## 9. CLI Interface

```bash
# Basic usage
gpu-run ollama run llama3 "Explain quantum computing"
gpu-run python generate_images.py
gpu-run comfy --workflow workflow.json

# Queue management
gpu-run queue                      # show queued + running jobs
gpu-run status                     # GPU utilization dashboard
gpu-run cancel <job-id>
gpu-run clear                      # cancel all pending jobs

# Scheduler hints
gpu-run --priority high <command>
gpu-run --no-batch <command>       # skip batching, run immediately when GPU is free
gpu-run --watch <command>          # re-run when similar jobs arrive

# Daemon management
gpu-run install-daemon             # install and start gpu-rund as a service
gpu-run daemon stop | start | restart | status

# Diagnostics
gpu-run doctor                     # check GPU drivers, telemetry, daemon health
```

---

## 10. Key Features by Phase

### MVP (v0.1)

- `gpu-run` CLI wrapper
- Background daemon (`gpu-rund`) with FIFO queue
- Basic fingerprinting for Ollama and llama.cpp
- Sequential execution with minimal idle gaps (Level 2/3 similarity scheduling)
- Real-time status dashboard (`gpu-run status`, `gpu-run queue`)
- GPU telemetry (utilization, VRAM, temperature)
- Graceful interruption and job cancellation
- Audit logging
- Linux-first; Windows support in parallel from day one in core logic

### Phase 2 — Batching (v0.5)

- Level-1 dynamic batching for llama.cpp (prompt-file merging)
- ComfyUI queue integration
- A1111/Forge batch-size grouping
- Batch window + VRAM safety layer
- Fingerprinting for ComfyUI and A1111/Forge
- Priority queuing and job dependencies

### Phase 3 — Advanced (v1.0+)

- Predictive scheduling using historical job patterns
- Multi-GPU support (round-robin / load balancing)
- Plugin system for user-defined backend runners
- Web UI (optional)
- Persistent queue across reboots (SQLite-backed)
- Energy & cost tracking
- Prometheus metrics endpoint

---

## 11. Supported Tools

### Phase 1

- Ollama (`ollama run`, `ollama serve`)
- llama.cpp / llama-cli
- Generic Python scripts (best-effort fingerprinting)

### Phase 2

- ComfyUI (via CLI + REST API queue)
- Automatic1111 / Forge / SD.Next
- Hugging Face `diffusers` scripts

### Later

- KoboldCPP, LM Studio (CLI mode)
- InvokeAI, Fooocus
- Video generation tools (SVD, Wan, etc.)
- User-defined plugin runners

---

## 12. Non-Functional Requirements

- **Overhead**: < 5% CPU/RAM when idle
- **Latency**: Job submission < 50ms end-to-end
- **Compatibility**: Linux (kernel ≥ 5.10) and Windows 10/11, NVIDIA and AMD GPUs
- **Security**: User-scoped only, no root, audit log, optional allowlist
- **Observability**: Structured logs (JSON) + optional Prometheus metrics
- **Reliability**: Daemon crash does not lose queued jobs (SQLite WAL persistence)
- **Graceful degradation**: If telemetry is unavailable, falls back to sequential FIFO without VRAM checks

---

## 13. Technical Stack

| Concern           | Choice                                       |
| ----------------- | -------------------------------------------- |
| Language          | Rust                                         |
| GPU Telemetry     | `nvml-wrapper`, `rocm-smi` subprocess        |
| Queue/Persistence | SQLite via `rusqlite` + in-memory cache      |
| IPC               | Unix sockets (Linux) / Named pipes (Windows) |
| Serialization     | `serde` + MessagePack (`rmp-serde`)          |
| CLI               | `clap` v4                                    |
| Async runtime     | `tokio`                                      |
| Configuration     | TOML via `toml` + `serde`                    |
| Logging           | `tracing` + `tracing-subscriber`             |

---

## 14. Success Metrics

These are **targets to validate post-MVP**, not pre-release commitments:

- GPU utilization > 80% during active queued sessions on batchable workloads (llama.cpp prompt-file batching is the primary test case)
- Measurable throughput improvement on ≥ 3 representative benchmark workflows before claiming any speedup number publicly
- Job submission latency < 50ms at p99
- < 2% of users report compatibility issues after v1.0
- Zero privilege-escalation issues in security audit

---

## 15. Risks & Mitigations

| Risk                                   | Mitigation                                                                                                                         |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Fingerprinting fails for unknown tools | Graceful degradation to Level-3 (backend-only) similarity; user annotation comments                                                |
| VRAM estimation is wrong → OOM         | Conservative headroom default (512MB); per-backend heuristics refined over time; hard fallback to sequential on estimation failure |
| Batching breaks tool-specific caching  | Per-backend executor is aware of tool internals; opt-out with `--no-batch`                                                         |
| Daemon crash loses queue               | SQLite WAL persistence; jobs requeued on daemon restart                                                                            |
| User distrust of command routing       | Audit log, optional allowlist, source-available code, explicit security model documentation                                        |
| Windows named pipe complexity          | Local TCP fallback; named pipes well-supported in Rust via `tokio`                                                                 |

---

## 16. Roadmap

| Milestone | Scope                                                        | Target   |
| --------- | ------------------------------------------------------------ | -------- |
| v0.1 MVP  | FIFO queue, daemon, Ollama + llama.cpp, telemetry, audit log | 3 weeks  |
| v0.5      | Level-1 batching, ComfyUI + A1111 support, VRAM safety       | +4 weeks |
| v1.0      | Stable multi-backend, Windows-first-class, web dashboard     | +6 weeks |
| v2.0      | Multi-GPU, plugins, predictive scheduling                    | TBD      |

---

## 17. License

MIT
