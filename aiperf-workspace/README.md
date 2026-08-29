# AI-Perf Workspace

Load-testing and profiling of LLM inference servers using [AI-Perf](https://github.com/ai-dynamo/aiperf).

## Installation

> Reference: [AI-Perf GitHub](https://github.com/ai-dynamo/aiperf)

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install aiperf
```

## Usage

```bash
python run_aiperf.py --framework <name> --model <HuggingFace model ID> --concurrency <N> [options]
```

### `run_aiperf.py`

Wrapper script that runs `aiperf profile` against a locally running inference server. It:

1. Constructs the `aiperf profile` command from the provided arguments
2. Creates a timestamped output directory under `results/`
3. Saves the exact command to `command.txt` for reproducibility
4. Streams output to both the terminal and `stdout.log`

**Generated command:**
```
aiperf profile \
    --model <model> \
    --url <url> \
    --endpoint-type chat \
    --tokenizer <model> \
    --streaming \
    --synthetic-input-tokens-mean <isl> \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean <osl> \
    --output-tokens-stddev 0 \
    --concurrency <N> \
    --request-count <count> \
    --warmup-request-count <warmup> \
    --artifact-dir <output_dir>
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--framework` | *(required)* | Framework label (e.g. `vllm`, `sglang`, `tensorrtllm`). Used in the output directory name |
| `--model` | *(required)* | HuggingFace model ID. Also used as the `--tokenizer` for aiperf |
| `--concurrency` | *(required)* | Number of concurrent requests during profiling |
| `--url` | `http://localhost:8000` | Base URL of the inference server |
| `--isl` | `1024` | Input sequence length (synthetic prompt token count) |
| `--osl` | `256` | Output sequence length (requested generation token count) |
| `--request-count` | `300` | Number of profiling requests (after warmup) |
| `--warmup-request-count` | `30` | Number of warmup requests before profiling starts |
| `--results-dir` | `results` | Base directory for experiment outputs |
| `--use-server-token-count` | `false` | Use the token count reported by the server instead of local tokenizer counting |
| `--ignore_eos` | `false` | Pass `--extra-inputs ignore_eos:true` to force the server to generate exactly `osl` tokens |

### Examples

```bash
# Standard benchmark: Qwen3-32B-FP8 on vLLM, concurrency 32
python run_aiperf.py --framework vllm --model Qwen/Qwen3-32B-FP8 --concurrency 32

# Higher concurrency
python run_aiperf.py --framework sglang --model Qwen/Qwen3-32B-FP8 --concurrency 64

# Long context benchmark
python run_aiperf.py --framework vllm --model Qwen/Qwen3-32B-FP8 --concurrency 8 --isl 8000 --osl 512

# TensorRT-LLM (requires --use-server-token-count and --ignore_eos)
python run_aiperf.py --framework tensorrtllm --model Qwen/Qwen3-32B-FP8 --concurrency 32 \
    --use-server-token-count --ignore_eos
```

## Results

Each experiment is saved to `results/<framework>_<model>_c<concurrency>_isl<isl>_osl<osl>_<timestamp>/` containing:

| File | Description |
|------|-------------|
| `command.txt` | Exact `aiperf` command that was executed |
| `stdout.log` | Full terminal output from the benchmark run |
| `profile_export_aiperf.json` | Aggregated metrics (throughput, latency percentiles, TTFT, ITL, etc.) |
| `profile_export_aiperf.csv` | Same metrics in CSV format |
| `profile_export.jsonl` | Per-request raw data |
| `inputs.json` | Generated synthetic input prompts |
| `server_metrics_export.*` | Server-side Prometheus metrics (when available, e.g. vLLM) |
| `logs/` | AI-Perf internal logs |
