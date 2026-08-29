# vLLM Workspace

Serves an LLM using [vLLM](https://docs.vllm.ai/) with an OpenAI-compatible API.

## Installation

> Reference: [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#requirements)

```bash
python3 -m venv .venv
source .venv/bin/activate

uv pip install vllm --torch-backend=auto
```

## Usage

```bash
python run_vllm.py --model <HuggingFace model ID> [options]
```

### `run_vllm.py`

Wrapper script that launches `vllm serve` with configurable parameters. It builds the CLI command and runs the server as a subprocess.

**Generated command:**
```
vllm serve <model> --host <host> --port <port> --gpu-memory-utilization <fraction> --max-model-len <length>
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *(required)* | HuggingFace model identifier (e.g. `Qwen/Qwen3-32B-FP8`) |
| `--host` | `0.0.0.0` | Host address to bind the server to |
| `--port` | `8000` | Port number for the API server |
| `--gpu-memory-utilization` | `0.90` | Fraction of GPU memory to allocate for KV cache and model weights |
| `--max-model-len` | `8192` | Maximum sequence length (input + output tokens) the server will accept |

### Examples

```bash
# Serve Qwen3-32B FP8 with defaults
python run_vllm.py --model Qwen/Qwen3-32B-FP8

# Serve with a shorter context to save memory
python run_vllm.py --model Qwen/Qwen3-32B-FP8 --max-model-len 4096

# Serve an AWQ-quantized model
python run_vllm.py --model Qwen/Qwen3-32B-AWQ
```