# Matched KV Budget Rerun

Scripts for the November 2026 rerun of Experiments 1-3, where every engine uses an FP8 KV cache and the same KV budget. Results are in `aiperf-workspace/results_matched/` (main runs) and `aiperf-workspace/results_diag/` (diagnostics).

The scripts expect the layout used on the benchmark machine: engine workspaces under `~/projects/hosting/{vllm-workspace,sglang-workspace,tensorrt-llm,aiperf-workspace}`, each with its own `.venv`, and these scripts in `~/rerun/` with a `logs/` folder.

| Script | Purpose |
|--------|---------|
| `setup_engines.sh` | Create the vLLM and SGLang virtual environments |
| `setup_trt.sh` | Create the TensorRT-LLM virtual environment |
| `download_bf16.sh` | Download the BF16 `Qwen/Qwen3-32B` for TensorRT-LLM quantization |
| `build_trt.sh` | Quantize to FP8 weights and FP8 KV cache, then build the engine (`max_seq_len` 9216, `max_num_tokens` 8192) |
| `orchestrate.py` | Start one engine, run its AIPerf configurations, save server command, server log, Prometheus snapshots and engine statistics with each result, stop the engine |
| `poll_stats.py` | Record GPU memory every 2 s, and TensorRT-LLM iteration statistics from `/metrics` |
| `analyze.py` | Summarize a results folder: throughput, latency, running requests, and queue wait from Prometheus |
| `cluster.py` | Count how many requests started within 200 ms of each other (detects runs arriving in waves) |

## Running

```bash
# Natural FP8 KV capacity of an engine (prints the log lines, no benchmark)
python3 orchestrate.py sglang --probe

# Full run at the shared budget
for job in vllm sglang tensorrtllm vllm-awq vllm-gptq; do
    python3 orchestrate.py $job --budget 35328
done
python3 analyze.py ~/projects/hosting/aiperf-workspace/results_matched
```

35,328 tokens is vLLM's natural FP8 KV capacity at `--gpu-memory-utilization 0.90`. SGLang reserves 60,954 tokens and TensorRT-LLM 71,552 at their 0.90 settings.

## Software

vLLM 0.32.0, SGLang 0.5.9, TensorRT-LLM 1.2.1 (TensorRT 10.14.1, `pyarrow<20`), AIPerf 0.11.0, NVIDIA driver 595.71.05, CUDA 13.0, on an AWS g6e.2xlarge (L40S 48 GB).
