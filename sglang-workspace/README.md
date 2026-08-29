# SGLang Workspace

Serves an LLM using [SGLang](https://github.com/sgl-project/sglang) with an OpenAI-compatible API.

## Installation

> Reference: [SGLang installation](https://sgl-project-sglang-93.mintlify.app/installation)

```bash
python3 -m venv .venv
source .venv/bin/activate

uv pip install sglang
```

## Usage

```bash
python run_sglang.py --model <HuggingFace model ID> [options]
```

### `run_sglang.py`

Wrapper script that launches SGLang's built-in server (`sglang.launch_server`) with configurable parameters.

**Generated command:**
```
python -m sglang.launch_server --model-path <model> --host <host> --port <port> --mem-fraction-static <fraction> --context-length <length>
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *(required)* | HuggingFace model identifier (e.g. `Qwen/Qwen3-32B-FP8`) |
| `--host` | `0.0.0.0` | Host address to bind the server to |
| `--port` | `8000` | Port number for the API server |
| `--mem-fraction-static` | `0.90` | Fraction of GPU memory reserved for static model weights and KV cache |
| `--context-length` | `8192` | Maximum context length the server will handle |

### Examples

```bash
# Serve Qwen3-32B FP8 with defaults
python run_sglang.py --model Qwen/Qwen3-32B-FP8

# Serve an MoE model
python run_sglang.py --model Qwen/Qwen3-30B-A3B-FP8
```