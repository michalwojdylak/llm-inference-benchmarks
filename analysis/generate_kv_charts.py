"""Charts for the KV cache blog post.

Numbers come from the equal-budget runs (aiperf-workspace/results_matched, summarized by
equal-kv-budget/analyze.py), the diagnostic runs (aiperf-workspace/results_diag) and the
engines' server logs saved with each run.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from generate_charts import (
    BG_COLOR, COLORS, TEXT_COLOR, add_bar_labels, apply_style, save, style_ax,
)

ENGINES = ["vLLM", "SGLang", "TensorRT-LLM"]
ENGINE_COLORS = [COLORS["vllm"], COLORS["sglang"], COLORS["trtllm"]]
SHARED_BUDGET = 35328

# FP8 KV capacity each engine reserves at its 0.90 memory setting (server logs, --probe runs)
NATURAL_KV_TOKENS = [35328, 60954, 71552]
SHORT_REQUEST_TOKENS = 1024 + 256
LONG_REQUEST_TOKENS = 8008 + 512

# vLLM 0.32 at --gpu-memory-utilization 0.90 with FP8 KV (server logs of the c32 runs)
CHECKPOINTS = {
    "models": ["FP8", "AWQ", "GPTQ"],
    "weights_gib": [32.52, 18.24, 18.13],
    "kv_gib": [4.45, 18.63, 18.79],
    "kv_tokens": [2278 * 16, 9540 * 16, 9621 * 16],
}

# Average running requests, middle 70% of each run (analyze.py)
RUNNING = {
    "concurrency": [8, 32, 64],
    "vLLM": [6.0, 24.0, 24.0],
    "SGLang": [6.9, 21.3, 21.7],
    "TensorRT-LLM": [7.5, 24.5, 24.4],
}

# TTFT split from Prometheus: queue wait vs the rest, seconds
TTFT_SPLIT = {
    "labels": ["vLLM\nc8", "vLLM\nc32", "vLLM\nc64", "SGLang\nc8", "SGLang\nc32", "SGLang\nc64"],
    "ttft": [4.20, 9.18, 40.41, 2.37, 9.08, 34.02],
    "queue": [0.00, 1.35, 27.76, 0.00, 4.64, 27.20],
    "colors": [COLORS["vllm"]] * 3 + [COLORS["sglang"]] * 3,
}

# TensorRT-LLM long context (ISL 8000, OSL 512, c8): shared budget vs its full KV cache
TRT_LONG = {
    "labels": ["35,328 tokens\n(shared budget)", "71,552 tokens\n(full cache)"],
    "throughput": [25.1, 81.8],
    "ttft": [82.4, 5.05],
    "running": [3.9, 7.5],
}


def chart_kv_capacity():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(ENGINES))
    bars = ax.bar(x, [t / 1000 for t in NATURAL_KV_TOKENS], 0.55, color=ENGINE_COLORS, edgecolor="none", zorder=3)
    for bar, tokens in zip(bars, NATURAL_KV_TOKENS):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{tokens:,} tokens\n{tokens // SHORT_REQUEST_TOKENS} short / {tokens // LONG_REQUEST_TOKENS} long requests",
            ha="center", va="bottom", fontsize=10, color=TEXT_COLOR,
        )
    ax.axhline(SHARED_BUDGET / 1000, color=TEXT_COLOR, linestyle="--", linewidth=1, zorder=4)
    ax.text(0.5, SHARED_BUDGET / 1000 - 1, "shared budget", ha="center", va="top", fontsize=10, color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(ENGINES, fontsize=12)
    style_ax(ax, "FP8 KV cache (thousand tokens)", "KV Cache Each Engine Reserves at a 0.90 Memory Setting")
    ax.set_ylim(0, max(NATURAL_KV_TOKENS) / 1000 * 1.35)
    save(fig, "kv_capacity")


def chart_weights_vs_kv():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(CHECKPOINTS["models"]))
    w = 0.36
    bars_w = ax.bar(x - w / 2, CHECKPOINTS["weights_gib"], w, label="Weights", color=COLORS["fp8"], edgecolor="none", zorder=3)
    bars_k = ax.bar(x + w / 2, CHECKPOINTS["kv_gib"], w, label="Left for KV cache", color=COLORS["awq"], edgecolor="none", zorder=3)
    add_bar_labels(ax, bars_w, fmt="{:.1f} GiB", padding=0.4)
    add_bar_labels(ax, bars_k, fmt="{:.1f} GiB", padding=0.4)
    for bar, tokens in zip(bars_k, CHECKPOINTS["kv_tokens"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{tokens / 1000:.0f}K\ntokens",
                ha="center", va="center", fontsize=9, color=BG_COLOR, fontweight="bold", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Qwen3-32B-{m}" for m in CHECKPOINTS["models"]], fontsize=12)
    style_ax(ax, "GiB", "vLLM Memory Split by Checkpoint (FP8 KV, 0.90 utilization)")
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    ax.set_ylim(0, max(CHECKPOINTS["weights_gib"]) * 1.25)
    save(fig, "kv_weights_vs_kv")


def chart_running_requests():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(RUNNING["concurrency"]))
    w = 0.24
    for i, (engine, color) in enumerate(zip(ENGINES, ENGINE_COLORS)):
        bars = ax.bar(x + (i - 1) * w, RUNNING[engine], w, label=engine, color=color, edgecolor="none", zorder=3)
        add_bar_labels(ax, bars, padding=0.4)
    fit = SHARED_BUDGET / SHORT_REQUEST_TOKENS
    ax.axhline(fit, color=TEXT_COLOR, linestyle="--", linewidth=1, zorder=4)
    ax.text(2.4, fit + 0.6, f"{fit:.1f} requests fit in 35,328 tokens", ha="right", va="bottom", fontsize=10, color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{c}" for c in RUNNING["concurrency"]], fontsize=13)
    style_ax(ax, "Requests generating at once (avg)", "Running Requests vs Concurrency - Equal KV Budget")
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    ax.set_ylim(0, 40)
    save(fig, "kv_running_requests")


def chart_ttft_split():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(TTFT_SPLIT["labels"]))
    rest = [t - q for t, q in zip(TTFT_SPLIT["ttft"], TTFT_SPLIT["queue"])]
    ax.bar(x, TTFT_SPLIT["queue"], 0.55, color=TTFT_SPLIT["colors"], edgecolor="none", zorder=3, alpha=0.45)
    bars = ax.bar(x, rest, 0.55, bottom=TTFT_SPLIT["queue"], color=TTFT_SPLIT["colors"], edgecolor="none", zorder=3)
    for bar, total in zip(bars, TTFT_SPLIT["ttft"]):
        ax.text(bar.get_x() + bar.get_width() / 2, total + 0.6, f"{total:.1f}s", ha="center", va="bottom", fontsize=10, color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(TTFT_SPLIT["labels"], fontsize=11)
    style_ax(ax, "TTFT (sec)", "Where TTFT Goes - ISL 1024, OSL 256")
    ax.legend(
        handles=[Patch(color=TEXT_COLOR, alpha=0.45), Patch(color=TEXT_COLOR)],
        labels=["Waiting for a free slot", "Scheduled to first token"],
        frameon=False, loc="upper left", fontsize=11,
    )
    ax.set_ylim(0, max(TTFT_SPLIT["ttft"]) * 1.2)
    save(fig, "kv_ttft_split")


def chart_trt_long_context():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    x = np.arange(2)
    colors = [COLORS["trtllm"], COLORS["trtllm"]]
    panels = [
        ("throughput", "Output Tokens / sec", "Throughput", "{:.1f}", 1.5),
        ("ttft", "TTFT (sec)", "Time To First Token", "{:.1f}s", 1.5),
        ("running", "Requests", "Running Requests (avg)", "{:.1f}", 0.15),
    ]
    for ax, (key, ylabel, title, fmt, pad) in zip(axes, panels):
        bars = ax.bar(x, TRT_LONG[key], 0.5, color=colors, edgecolor="none", zorder=3)
        bars[1].set(hatch="//", edgecolor=BG_COLOR)
        add_bar_labels(ax, bars, fmt=fmt, padding=pad)
        ax.set_xticks(x)
        ax.set_xticklabels(TRT_LONG["labels"], fontsize=10)
        style_ax(ax, ylabel, title)
        ax.set_ylim(0, max(TRT_LONG[key]) * 1.3)
    fig.suptitle("TensorRT-LLM, Long Context (ISL 8000, OSL 512, c8) - Only the KV Budget Changes",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    save(fig, "kv_trt_long_context")


if __name__ == "__main__":
    apply_style()
    chart_kv_capacity()
    chart_weights_vs_kv()
    chart_running_requests()
    chart_ttft_split()
    chart_trt_long_context()
