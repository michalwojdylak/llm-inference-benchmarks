#!/usr/bin/env python3
"""Rerun driver: start one engine, run AIPerf configs against it, stop it.

Usage: orchestrate.py <job> [--budget TOKENS] [--probe] [diagnostic options]
  --probe    start the server with its natural KV capacity, print the log lines
             that report it, then stop (no benchmark).
Diagnostic options (used for results_diag): --attn, --only, --results,
--request-count, --trt-extra.
"""

import argparse
import datetime
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

HOME = pathlib.Path.home()
HOSTING = HOME / "projects" / "hosting"
RERUN = HOME / "rerun"
RESULTS = HOSTING / "aiperf-workspace" / "results_matched"
TRT_ENGINE = HOSTING / "tensorrt-llm" / "Qwen3-32B-FP8-engine"
URL = "http://localhost:8000"

# Same scheduler token limit on every engine (TRT-LLM's is baked into the engine).
MAX_BATCHED_TOKENS = 8192
MAX_MODEL_LEN = 9216

ENGINE_RUNS = [(8, 1024, 256), (32, 1024, 256), (64, 1024, 256), (8, 8000, 512)]
QUANT_RUNS = [(32, 1024, 256)]

JOBS = {
    "vllm": ("vllm", "Qwen/Qwen3-32B-FP8", ENGINE_RUNS),
    "sglang": ("sglang", "Qwen/Qwen3-32B-FP8", ENGINE_RUNS),
    "tensorrtllm": ("tensorrtllm", "Qwen/Qwen3-32B-FP8", ENGINE_RUNS),
    "vllm-awq": ("vllm", "Qwen/Qwen3-32B-AWQ", QUANT_RUNS),
    "vllm-gptq": ("vllm", "JunHowie/Qwen3-32B-GPTQ-Int4", QUANT_RUNS),  # AlphaGaO uses desc_act=True, unsupported in vLLM 0.32
}


def server_cmd(engine, model, budget, attn=None, trt_extra=None):
    if engine == "vllm":
        cmd = [
            str(HOSTING / "vllm-workspace/.venv/bin/vllm"), "serve", model,
            "--host", "0.0.0.0", "--port", "8000",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--max-num-batched-tokens", str(MAX_BATCHED_TOKENS),
            "--kv-cache-dtype", "fp8",
        ]
        if budget:
            cmd += ["--num-gpu-blocks-override", str(budget // 16)]
        if attn:
            cmd += ["--attention-backend", attn]
        return cmd

    if engine == "sglang":
        cmd = [
            str(HOSTING / "sglang-workspace/.venv/bin/python"), "-m", "sglang.launch_server",
            "--model-path", model,
            "--host", "0.0.0.0", "--port", "8000",
            "--mem-fraction-static", "0.90",
            "--context-length", str(MAX_MODEL_LEN),
            "--chunked-prefill-size", str(MAX_BATCHED_TOKENS),
            "--kv-cache-dtype", "fp8_e4m3",
            "--enable-metrics",
        ]
        if budget:
            cmd += ["--max-total-tokens", str(budget)]
        return cmd

    if engine == "tensorrtllm":
        options = {"kv_cache_config": {"free_gpu_memory_fraction": 0.9}}
        if budget:
            options["kv_cache_config"]["max_tokens"] = budget
        options.update(json.loads(trt_extra or "{}"))
        options_path = RERUN / "trtllm_options.yaml"
        options_path.write_text(json.dumps(options))  # JSON is valid YAML
        return [
            str(HOSTING / "tensorrt-llm/.venv/bin/trtllm-serve"), "serve", str(TRT_ENGINE),
            "--backend", "tensorrt",
            "--tokenizer", model,
            "--max_batch_size", "64",
            "--host", "0.0.0.0", "--port", "8000",
            "--extra_llm_api_options", str(options_path),
        ]

    raise ValueError(engine)


def wait_ready(proc, timeout=3600):
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{URL}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(10)
    raise TimeoutError("server not ready")


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    # wait for the GPU to be released before the next server starts
    for _ in range(60):
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout.strip()
        if used and int(used) < 1000:
            break
        time.sleep(5)
    # and for port 8000 to be free (SGLang's HTTP process can outlive the GPU worker)
    for _ in range(60):
        with socket.socket() as s:
            try:
                s.bind(("0.0.0.0", 8000))
                return
            except OSError:
                pass
        time.sleep(5)


# TRT-LLM 1.2.1's tensorrt backend cannot enable Prometheus (return_perf_metrics fails),
# so it has no snapshot; poll_stats.py records its /metrics iteration stats instead.
PROM_PATH = {"vllm": "/metrics", "sglang": "/metrics", "tensorrtllm": None}


def snapshot(engine, path):
    if PROM_PATH[engine] is None:
        path.write_text("# no Prometheus endpoint for this engine\n")
        return
    try:
        with urllib.request.urlopen(URL + PROM_PATH[engine], timeout=10) as r:
            path.write_bytes(r.read())
    except Exception as e:
        path.write_text(f"# snapshot failed: {e}\n")


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("job", choices=JOBS)
    p.add_argument("--budget", type=int, default=0)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--attn", help="vLLM attention backend override (diagnostic)")
    p.add_argument("--only", help="run only these concurrency:isl pairs, e.g. 8:1024")
    p.add_argument("--results", type=pathlib.Path, default=RESULTS)
    p.add_argument("--trt-extra", help="JSON merged into TRT-LLM extra_llm_api_options (diagnostic)")
    p.add_argument("--request-count", type=int, help="override AIPerf request count (diagnostic)")
    args = p.parse_args()

    engine, model, runs = JOBS[args.job]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    server_log = RERUN / "logs" / f"server_{args.job}_{'probe' if args.probe else args.budget}_{stamp}.log"
    if args.only:
        keep = set(args.only.split(","))
        runs = [r for r in runs if f"{r[0]}:{r[1]}" in keep]
    cmd = server_cmd(engine, model, args.budget, args.attn, args.trt_extra)
    log("server: " + " ".join(cmd))

    # FlashInfer JIT needs ninja (venv bin) and nvcc on PATH
    env = dict(os.environ)
    env["PATH"] = f"{pathlib.Path(cmd[0]).parent}:/usr/local/cuda/bin:{env['PATH']}"
    env["CUDA_HOME"] = "/usr/local/cuda"
    with open(server_log, "w") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    try:
        wait_ready(proc)
        log(f"ready, log {server_log}")
        if args.probe:
            return

        run_dirs = []
        for conc, isl, osl in runs:
            name = f"{engine}_{model.split('/')[-1]}_c{conc}_isl{isl}_osl{osl}_kv{args.budget}"
            log(f"run {name}")
            snap_before = RERUN / "logs" / f"prom_before_{name}_{stamp}.txt"
            snapshot(engine, snap_before)
            poll_out = RERUN / "logs" / f"stats_{name}_{stamp}.jsonl"
            poller = subprocess.Popen(
                [sys.executable, str(RERUN / "poll_stats.py"), engine, str(poll_out)],
                start_new_session=True,
            )
            rc = subprocess.run(
                [
                    str(HOSTING / "aiperf-workspace/.venv/bin/python"), "run_aiperf.py",
                    "--framework", engine, "--model", model,
                    "--concurrency", str(conc), "--isl", str(isl), "--osl", str(osl),
                    "--results-dir", str(args.results),
                    "--use-server-token-count", "--ignore_eos",
                ] + (["--request-count", str(args.request_count)] if args.request_count else []),
                cwd=HOSTING / "aiperf-workspace",
                env={**os.environ, "PATH": f"{HOSTING / 'aiperf-workspace/.venv/bin'}:{os.environ['PATH']}"},
                stdout=subprocess.DEVNULL,
            ).returncode
            os.killpg(poller.pid, signal.SIGTERM)
            time.sleep(3)
            snap_after = RERUN / "logs" / f"prom_after_{name}_{stamp}.txt"
            snapshot(engine, snap_after)
            run_dir = sorted(args.results.glob(f"{engine}_{model.split('/')[-1]}_c{conc}_isl{isl}_osl{osl}_*"))[-1]
            shutil.copy(poll_out, run_dir / "engine_stats.jsonl")
            run_dirs.append(run_dir)
            shutil.copy(snap_before, run_dir / "prometheus_before.txt")
            shutil.copy(snap_after, run_dir / "prometheus_after.txt")
            (run_dir / "server_command.txt").write_text(" ".join(cmd) + "\n")
            log(f"done rc={rc} -> {run_dir.name}")
            shutil.copy(server_log, run_dir / "server.log")
    finally:
        stop(proc)
        log("server stopped")


if __name__ == "__main__":
    main()
