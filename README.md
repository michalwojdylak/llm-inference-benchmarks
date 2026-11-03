# LLM Inference Benchmarks

Performance benchmarks comparing **vLLM**, **SGLang**, and **TensorRT-LLM** inference engines on a single NVIDIA GPU using [AI-Perf](https://github.com/ai-dynamo/aiperf) as the load-testing tool.

## Experiment Setup

- **GPU:** 1x NVIDIA GPU (single-GPU setup)
- **Benchmark tool:** [AI-Perf](https://github.com/ai-dynamo/aiperf) v0.11.0
- **Protocol:** OpenAI-compatible Chat API (streaming)
- **Workload:** synthetic prompts with fixed input/output token lengths, zero stddev
- **Warmup:** 30 requests before each profiling run
- **Profiling:** 300 requests per run

### Models Tested

| Model | Quantization | Frameworks |
|-------|-------------|------------|
| Qwen/Qwen3-32B-FP8 | FP8 | vLLM, SGLang, TensorRT-LLM |
| Qwen/Qwen3-32B-AWQ | INT4 AWQ | vLLM |
| JunHowie/Qwen3-32B-GPTQ-Int4 (first round: AlphaGaO/Qwen3-32B-GPTQ) | INT4 GPTQ | vLLM |
| Qwen/Qwen3-30B-A3B-FP8 | FP8 (MoE) | SGLang |

### Concurrency & Sequence Length Configurations

| Config | Input Tokens | Output Tokens | Concurrency |
|--------|-------------|---------------|-------------|
| Standard low | 1024 | 256 | 8 |
| Standard mid | 1024 | 256 | 32 |
| Standard high | 1024 | 256 | 64 |
| Long context | 8000 | 512 | 8 |

## Results Summary

Experiments 1-3 below come from the November 2026 rerun in `aiperf-workspace/results_matched/`: FP8 KV cache on every engine, capped at the same 35,328-token KV budget, with `--max-model-len`/`--context-length`/`--max_seq_len` 9216, 8192 tokens per scheduler step, `ignore_eos` and server-side token counts. See [`matched-kv-rerun/`](matched-kv-rerun/) for the scripts and software versions. The first round (each engine on its own defaults, with BF16 KV on vLLM and SGLang) is kept in `aiperf-workspace/results/`.

### Qwen3-32B-FP8 &mdash; Concurrency Scaling (ISL=1024, OSL=256)

| Framework | Concurrency | Throughput (req/s) | Output (tok/s) | TTFT avg (ms) | TTFT p99 (ms) | ITL avg (ms) |
|-----------|-------------|-------------------|----------------|---------------|---------------|-------------|
| vLLM | 8 | 0.47 | 119.2 | 4,202 | 6,724 | 50.1 |
| vLLM | 32 | 0.90 | 230.3 | 9,181 | 16,569 | 98.5 |
| vLLM | 64 | 0.89 | 228.9 | 40,408 | 63,466 | 99.2 |
| SGLang | 8 | 0.46 | 118.0 | 2,365 | 2,640 | 58.0 |
| SGLang | 32 | 1.10 | 282.1 | 9,080 | 28,291 | 76.1 |
| SGLang | 64 | 1.11 | 283.3 | 34,020 | 53,361 | 77.5 |
| TensorRT-LLM | 8 | 0.47 | 119.4 | 1,122 | 1,824 | 54.7 |
| TensorRT-LLM | 32 | 1.15 | 295.1 | 6,336 | 23,384 | 70.2 |
| TensorRT-LLM | 64 | 1.17 | 298.7 | 30,137 | 46,226 | 71.5 |

At concurrency 8, TTFT depends on whether requests arrive in synchronized waves (every request has the same length, so they finish and restart together). vLLM and SGLang ran in waves of 8, TensorRT-LLM in waves of about 4; `matched-kv-rerun/cluster.py` measures this.

### Qwen3-32B-FP8 &mdash; Long Context (ISL=8000, OSL=512, Concurrency=8)

| Framework | KV budget (tokens) | Throughput (req/s) | Output (tok/s) | TTFT avg (ms) | TTFT p99 (ms) | ITL avg (ms) |
|-----------|-------------------|-------------------|----------------|---------------|---------------|-------------|
| vLLM | 35,328 | 0.09 | 46.0 | 51,602 | 54,478 | 72.3 |
| SGLang | 35,328 | 0.10 | 48.9 | 48,263 | 53,816 | 68.4 |
| TensorRT-LLM | 35,328 | 0.05 | 25.1 | 82,437 | 85,486 | 155.6 |
| TensorRT-LLM | 71,552 (natural) | 0.16 | 81.8 | 5,050 | 14,138 | 87.9 |

The last row is a 24-request diagnostic run in `aiperf-workspace/results_diag/`.

### Quantization Comparison &mdash; vLLM, Concurrency=32 (ISL=1024, OSL=256)

| Model | Throughput (req/s) | Output (tok/s) | TTFT avg (ms) | TTFT p99 (ms) | ITL avg (ms) |
|-------|-------------------|----------------|---------------|---------------|-------------|
| Qwen/Qwen3-32B-FP8 | 0.90 | 230.3 | 9,181 | 16,569 | 98.5 |
| Qwen/Qwen3-32B-AWQ | 1.30 | 332.7 | 6,864 | 12,334 | 66.8 |
| JunHowie/Qwen3-32B-GPTQ-Int4 | 1.34 | 342.5 | 6,520 | 11,358 | 65.4 |

The first round used `AlphaGaO/Qwen3-32B-GPTQ`, which vLLM 0.32 rejects (`desc_act=True`).

### SGLang &mdash; MoE Model (ISL=1024, OSL=256, Concurrency=32)

First round only (BF16 KV cache, no budget cap); not rerun.

| Model | Throughput (req/s) | Output (tok/s) | TTFT avg (ms) | TTFT p99 (ms) | ITL avg (ms) |
|-------|-------------------|----------------|---------------|---------------|-------------|
| Qwen3-32B-FP8 | 0.97 | 249.1 | 12,332 | 27,517 | 76.6 |
| Qwen3-30B-A3B-FP8 (MoE) | 3.51 | 897.4 | 779 | 1,303 | 31.1 |

## Key Metrics

- **Throughput (req/s)** &mdash; requests completed per second
- **Output tok/s** &mdash; total output tokens generated per second across all concurrent users
- **TTFT (Time To First Token)** &mdash; latency until the first token is streamed back
- **ITL (Inter-Token Latency)** &mdash; average time between consecutive output tokens

## Repository Structure

```
llm-inference-benchmarks/
├── README.md                    # This file
├── vllm-workspace/              # vLLM server setup & launch script
│   ├── README.md
│   └── run_vllm.py
├── sglang-workspace/            # SGLang server setup & launch script
│   ├── README.md
│   └── run_sglang.py
├── tensorrt-llm-workspace/      # TensorRT-LLM build & serve pipeline
│   ├── README.md
│   └── run_trtllm.py
├── matched-kv-rerun/            # Equal FP8 KV budget rerun: driver, setup and analysis scripts
└── aiperf-workspace/            # AI-Perf benchmark runner & results
    ├── README.md
    ├── run_aiperf.py
    ├── results_matched/         # Equal-budget rerun (Experiments 1-3)
    ├── results_diag/            # Diagnostic runs referenced in the blog post
    └── results/                 # First round: raw experiment outputs (JSON, CSV, logs)
        ├── vllm_Qwen3-32B-FP8_c32_isl1024_osl256_*/
        ├── sglang_Qwen3-32B-FP8_c32_isl1024_osl256_*/
        ├── tensorrtllm_Qwen3-32B-FP8_c32_isl1024_osl256_*/
        └── ...
```

Each framework has its own workspace directory with an isolated Python virtual environment (`.venv`). The benchmark client (AI-Perf) runs from a separate workspace and connects to whichever server is running on `localhost:8000`.

## How to Reproduce

### 1. Start an inference server

Pick one framework and follow its workspace README:

```bash
# Example: vLLM
cd vllm-workspace
python3 -m venv .venv && source .venv/bin/activate
uv pip install vllm --torch-backend=auto
python run_vllm.py --model Qwen/Qwen3-32B-FP8
```

### 2. Run the benchmark

In a separate terminal:

```bash
cd aiperf-workspace
python3 -m venv .venv && source .venv/bin/activate
pip install aiperf

python run_aiperf.py \
    --framework vllm \
    --model Qwen/Qwen3-32B-FP8 \
    --concurrency 32
```

Results are saved to `aiperf-workspace/results/<framework>_<model>_<config>_<timestamp>/`.