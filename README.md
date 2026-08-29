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
| Qwen/Qwen3-32B-GPTQ | INT4 GPTQ | vLLM |
| Qwen/Qwen3-30B-A3B-FP8 | FP8 (MoE) | SGLang |

### Concurrency & Sequence Length Configurations

| Config | Input Tokens | Output Tokens | Concurrency |
|--------|-------------|---------------|-------------|
| Standard low | 1024 | 256 | 8 |
| Standard mid | 1024 | 256 | 32 |
| Standard high | 1024 | 256 | 64 |
| Long context | 8000 | 512 | 8 |

## Results Summary

### Qwen3-32B-FP8 &mdash; Concurrency Scaling (ISL=1024, OSL=256)

| Framework | Concurrency | Throughput (req/s) | Output (tok/s) | TTFT avg (ms) | TTFT p99 (ms) | ITL avg (ms) |
|-----------|-------------|-------------------|----------------|---------------|---------------|-------------|
| vLLM | 8 | 0.50 | 127.1 | 834 | 1,838 | 59.2 |
| vLLM | 32 | 0.95 | 243.9 | 11,125 | 25,768 | 84.7 |
| vLLM | 64 | 0.96 | 244.5 | 40,565 | 49,068 | 84.8 |
| SGLang | 8 | 0.45 | 115.7 | 2,029 | 2,609 | 60.6 |
| SGLang | 32 | 0.97 | 249.1 | 12,332 | 27,517 | 76.6 |
| SGLang | 64 | 0.98 | 250.1 | 40,601 | 53,767 | 77.8 |
| TensorRT-LLM | 8 | 0.46 | 117.9 | 2,541 | 2,891 | 54.0 |
| TensorRT-LLM | 32 | 1.13 | 289.2 | 4,554 | 10,325 | 83.0 |
| TensorRT-LLM | 64 | 1.34 | 342.1 | 11,738 | 42,304 | 124.4 |

### Qwen3-32B-FP8 &mdash; Long Context (ISL=8000, OSL=512, Concurrency=8)

| Framework | Throughput (req/s) | Output (tok/s) | TTFT avg (ms) | TTFT p99 (ms) | ITL avg (ms) |
|-----------|-------------------|----------------|---------------|---------------|-------------|
| vLLM | 0.08 | 41.0 | 63,824 | 75,340 | 68.0 |
| SGLang | 0.08 | 38.2 | 68,593 | 82,825 | 72.6 |
| TensorRT-LLM | 0.15 | 75.9 | 8,560 | 38,053 | 88.4 |

### Quantization Comparison &mdash; vLLM, Concurrency=32 (ISL=1024, OSL=256)

| Model | Throughput (req/s) | Output (tok/s) | TTFT avg (ms) | TTFT p99 (ms) | ITL avg (ms) |
|-------|-------------------|----------------|---------------|---------------|-------------|
| Qwen3-32B-FP8 | 0.95 | 243.9 | 11,125 | 25,768 | 84.7 |
| Qwen3-32B-AWQ | 1.36 | 349.3 | 2,770 | 11,617 | 78.9 |
| Qwen3-32B-GPTQ | 1.36 | 347.8 | 2,732 | 11,522 | 79.3 |

### SGLang &mdash; MoE Model (ISL=1024, OSL=256, Concurrency=32)

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
└── aiperf-workspace/            # AI-Perf benchmark runner & results
    ├── README.md
    ├── run_aiperf.py
    └── results/                 # Raw experiment outputs (JSON, CSV, logs)
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