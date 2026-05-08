# Gpu-run

gpu-run is a lightweight GPU execution wrapper that queues and schedules local commands for better GPU utilization.

## MVP Features

- CLI job submission with persistent job queue
- Background daemon to process queued jobs sequentially
- Queue and status inspection commands
- Basic GPU telemetry via `nvidia-smi`
- Job cancellation and queue clearing
- Simple workload fingerprinting for Ollama, llama.cpp, Automatic1111, ComfyUI, and Python scripts

## Usage

```bash
# Queue a command
gpu-run run ollama run llama3 "Explain quantum computing"

# Start the daemon to process jobs
gpu-run daemon start

# Show queued jobs
gpu-run queue

# View current status and GPU telemetry
gpu-run status

# Cancel a pending job
gpu-run cancel job-<timestamp>

# Clear pending jobs
gpu-run clear

# Stop the background daemon
gpu-run daemon stop

# Show CLI help
gpu-run help

gpu-run help run

# Run diagnostics
gpu-run doctor
```

## Packaging

### Flatpak

A Flatpak manifest is available at `flatpak/manifest.json`. Build and install locally with:

```bash
flatpak-builder --force-clean build-dir flatpak/manifest.json --install --user
```

This package uses `org.freedesktop.Platform` and installs `gpu-run` as a CLI command.

### Winget

A Winget manifest template is included at `winget/manifest.yaml`.

To publish to Winget, build a Windows release archive containing `gpu-run.exe`, upload it to GitHub Releases, and replace the placeholder `InstallerUrl` / `InstallerSha256` values.
