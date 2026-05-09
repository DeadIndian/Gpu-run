# GPU-Run Python Package

Intelligent GPU execution wrapper - Python package that wraps the Rust-based gpu-run CLI tool.

## Installation

```bash
pip install gpu-run
```

## Usage

```python
import gpu_run

# Start the daemon
gpu_run.start_daemon()

# Show queue status  
gpu_run.show_queue()

# Run a command
exit_code = gpu_run.run_gpu_run(["ollama", "run", "llama3", "Hello world"])
```

## Requirements

- Python 3.8+
- gpu-run binary (automatically installed with this package)

## How it Works

This Python package provides a convenient wrapper around the Rust-based `gpu-run` CLI tool. It includes:

- The compiled Rust binary for your platform
- Python bindings for easy integration
- Cross-platform compatibility (Linux, Windows, macOS)

## Development

For development, see the main repository at https://github.com/gpu-run/gpu-run

## License

MIT License - see LICENSE file for details.
