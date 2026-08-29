# TensorRT-LLM Workspace

End-to-end pipeline for quantizing, building a TensorRT engine, and serving an LLM using [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) with an OpenAI-compatible API.

## Installation

> Reference: [TensorRT-LLM Linux installation](https://nvidia.github.io/TensorRT-LLM/installation/linux.html)

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu130
sudo apt-get -y install libopenmpi-dev
pip install --ignore-installed pip setuptools wheel
pip install tensorrt_llm
```

The script also requires `huggingface-cli` (from the `huggingface_hub` package) to download models.

## Usage

```bash
python run_trtllm.py --model <HuggingFace model ID> [options]
```

### `run_trtllm.py`

Automated pipeline that performs four stages:

1. **Clone TensorRT-LLM repo** (v1.2.1) &mdash; needed for the `quantize.py` calibration script
2. **Download the source model** via `huggingface-cli` into `models/<model_name>/`
3. **Quantize** the model using TensorRT-LLM's `quantize.py` (FP8, INT4_AWQ, etc.) &mdash; produces a TRT-LLM checkpoint in `<model_name>-checkpoint/`
4. **Build the TensorRT engine** via `trtllm-build` &mdash; outputs to `<model_name>-engine/`
5. **Serve** the engine via `trtllm-serve` on an OpenAI-compatible endpoint

Each stage is idempotent: if the output directory already exists, the stage is skipped. Use `--force-quantize` or `--force-build` to re-run specific stages.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *(required)* | HuggingFace model ID (e.g. `Qwen/Qwen3-32B-FP8`) |
| `--source-model` | *(auto)* | Base model for quantization; inferred by stripping `-FP8`/`-AWQ`/`-GPTQ` suffixes |
| `--quant` | *(auto)* | Quantization format: `fp8`, `int4_awq`, `w4a8_awq`, `int4_gptq`, `full_prec`. Inferred from model name if omitted |
| `--tp-size` | `1` | Tensor parallelism degree |
| `--host` | `0.0.0.0` | Server bind address |
| `--port` | `8000` | Server port |
| `--max-batch-size` | `64` | Maximum batch size for the engine and server |
| `--max-seq-len` | `2048` | Maximum sequence length for the engine |
| `--max-num-tokens` | `8192` | Maximum number of tokens in a batch |
| `--opt-num-tokens` | `4096` | Optimal number of tokens (performance tuning) |
| `--calib-size` | `128` | Number of calibration samples for quantization |
| `--calib-max-seq-length` | `512` | Maximum sequence length during calibration |
| `--batch-size` | `1` | Batch size during calibration |
| `--force-quantize` | `false` | Delete existing checkpoint and re-quantize |
| `--force-build` | `false` | Delete existing engine and rebuild |

### Examples

```bash
# Full pipeline: download, quantize to FP8, build engine, and serve
python run_trtllm.py --model Qwen/Qwen3-32B-FP8

# Force rebuild of the TensorRT engine
python run_trtllm.py --model Qwen/Qwen3-32B-FP8 --force-build

# Explicit quantization format
python run_trtllm.py --model Qwen/Qwen3-32B --quant fp8 --source-model Qwen/Qwen3-32B
```

### Directory Layout

After running the script, the workspace will contain:

```
tensorrt-llm-workspace/
├── models/
│   └── Qwen3-32B/              # Downloaded base model weights
├── Qwen3-32B-FP8-checkpoint/   # TRT-LLM quantized checkpoint
├── Qwen3-32B-FP8-engine/       # Compiled TensorRT engine
└── run_trtllm.py
```