import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Modern color palette (Tailwind-inspired)
# ---------------------------------------------------------------------------
COLORS = {
    "vllm": "#6366f1",       # indigo-500
    "sglang": "#06b6d4",     # cyan-500
    "trtllm": "#f59e0b",     # amber-500
    "fp8": "#6366f1",        # indigo-500
    "awq": "#10b981",        # emerald-500
    "gptq": "#f43f5e",       # rose-500
    "dense": "#6366f1",      # indigo-500
    "moe": "#06b6d4",        # cyan-500
}

BG_COLOR = "#0f172a"         # slate-900
CARD_COLOR = "#1e293b"       # slate-800
TEXT_COLOR = "#e2e8f0"       # slate-200
GRID_COLOR = "#334155"       # slate-700
ACCENT_GRID = "#475569"      # slate-600


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": CARD_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlepad": 16,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.5,
        "grid.linewidth": 0.6,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


def style_ax(ax, ylabel, title=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.tick_params(axis="both", length=0)


def add_bar_labels(ax, bars, fmt="{:.1f}", fontsize=9, padding=4):
    for bar in bars:
        val = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + padding,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=TEXT_COLOR,
            fontweight="medium",
        )


def save(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {name}.png")


# ===================================================================
# DATA
# ===================================================================
# Concurrency, long-context and quantization numbers come from the equal-budget runs
# (aiperf-workspace/results_matched): FP8 KV cache and a 35,328-token KV budget on
# every engine. MoE uses SGLang's default KV settings (aiperf-workspace/results).
CONCURRENCY_SCALING = {
    "concurrency": [8, 32, 64],
    "vllm":   {"throughput": [119.2, 230.3, 228.9], "ttft": [4202, 9181, 40408], "itl": [50.1, 98.5, 99.2]},
    "sglang": {"throughput": [118.0, 282.1, 283.3], "ttft": [2365, 9080, 34020], "itl": [58.0, 76.1, 77.5]},
    "trtllm": {"throughput": [119.4, 295.1, 298.7], "ttft": [1122, 6336, 30137], "itl": [54.7, 70.2, 71.5]},
}

# The last bar is TensorRT-LLM at its natural KV capacity (71,552 tokens), a 24-request
# run from aiperf-workspace/results_diag.
LONG_CONTEXT = {
    "engines": ["vLLM", "SGLang", "TensorRT-LLM", "TensorRT-LLM\n(full KV cache)"],
    "throughput": [46.0, 48.9, 25.1, 81.8],
    "ttft": [51602, 48263, 82437, 5050],
    "itl": [72.3, 68.4, 155.6, 87.9],
    "colors": [COLORS["vllm"], COLORS["sglang"], COLORS["trtllm"], COLORS["trtllm"]],
}

QUANTIZATION = {
    "models": ["FP8", "AWQ", "GPTQ"],
    "throughput": [230.3, 332.7, 342.5],
    "ttft": [9181, 6864, 6520],
    "itl": [98.5, 66.8, 65.4],
    "colors": [COLORS["fp8"], COLORS["awq"], COLORS["gptq"]],
}

MOE = {
    "models": ["Qwen3-32B\n(Dense)", "Qwen3-30B-A3B\n(MoE)"],
    "throughput": [249.1, 897.4],
    "ttft": [12332, 779],
    "itl": [76.6, 31.1],
    "colors": [COLORS["dense"], COLORS["moe"]],
}


# ===================================================================
# CHART 1 - Concurrency Scaling: Output Token Throughput
# ===================================================================
def chart_concurrency_throughput():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(CONCURRENCY_SCALING["concurrency"]))
    w = 0.24

    bars_v = ax.bar(x - w, CONCURRENCY_SCALING["vllm"]["throughput"], w, label="vLLM", color=COLORS["vllm"], edgecolor="none", zorder=3)
    bars_s = ax.bar(x,     CONCURRENCY_SCALING["sglang"]["throughput"], w, label="SGLang", color=COLORS["sglang"], edgecolor="none", zorder=3)
    bars_t = ax.bar(x + w, CONCURRENCY_SCALING["trtllm"]["throughput"], w, label="TensorRT-LLM", color=COLORS["trtllm"], edgecolor="none", zorder=3)

    add_bar_labels(ax, bars_v, padding=3)
    add_bar_labels(ax, bars_s, padding=3)
    add_bar_labels(ax, bars_t, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"c{c}" for c in CONCURRENCY_SCALING["concurrency"]], fontsize=13)
    style_ax(ax, "Output Tokens / sec", "Concurrency Scaling - Output Token Throughput (equal KV budget)")
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    ax.set_ylim(0, max(CONCURRENCY_SCALING["trtllm"]["throughput"]) * 1.25)

    save(fig, "concurrency_throughput")


# ===================================================================
# CHART 2 - Concurrency Scaling: TTFT
# ===================================================================
def chart_concurrency_ttft():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(CONCURRENCY_SCALING["concurrency"]))
    w = 0.24

    bars_v = ax.bar(x - w, [v / 1000 for v in CONCURRENCY_SCALING["vllm"]["ttft"]], w, label="vLLM", color=COLORS["vllm"], edgecolor="none", zorder=3)
    bars_s = ax.bar(x,     [v / 1000 for v in CONCURRENCY_SCALING["sglang"]["ttft"]], w, label="SGLang", color=COLORS["sglang"], edgecolor="none", zorder=3)
    bars_t = ax.bar(x + w, [v / 1000 for v in CONCURRENCY_SCALING["trtllm"]["ttft"]], w, label="TensorRT-LLM", color=COLORS["trtllm"], edgecolor="none", zorder=3)

    add_bar_labels(ax, bars_v, fmt="{:.1f}s", padding=0.3)
    add_bar_labels(ax, bars_s, fmt="{:.1f}s", padding=0.3)
    add_bar_labels(ax, bars_t, fmt="{:.1f}s", padding=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"c{c}" for c in CONCURRENCY_SCALING["concurrency"]], fontsize=13)
    style_ax(ax, "Time To First Token (sec)", "Concurrency Scaling - TTFT, equal KV budget (lower is better)")
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    ax.set_ylim(0, max(v / 1000 for v in CONCURRENCY_SCALING["vllm"]["ttft"]) * 1.25)

    save(fig, "concurrency_ttft")


# ===================================================================
# CHART 3 - Concurrency Scaling: ITL
# ===================================================================
def chart_concurrency_itl():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(CONCURRENCY_SCALING["concurrency"]))
    w = 0.24

    bars_v = ax.bar(x - w, CONCURRENCY_SCALING["vllm"]["itl"], w, label="vLLM", color=COLORS["vllm"], edgecolor="none", zorder=3)
    bars_s = ax.bar(x,     CONCURRENCY_SCALING["sglang"]["itl"], w, label="SGLang", color=COLORS["sglang"], edgecolor="none", zorder=3)
    bars_t = ax.bar(x + w, CONCURRENCY_SCALING["trtllm"]["itl"], w, label="TensorRT-LLM", color=COLORS["trtllm"], edgecolor="none", zorder=3)

    add_bar_labels(ax, bars_v, padding=1)
    add_bar_labels(ax, bars_s, padding=1)
    add_bar_labels(ax, bars_t, padding=1)

    ax.set_xticks(x)
    ax.set_xticklabels([f"c{c}" for c in CONCURRENCY_SCALING["concurrency"]], fontsize=13)
    style_ax(ax, "Inter-Token Latency (ms)", "Concurrency Scaling - ITL, equal KV budget (lower is better)")
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    ax.set_ylim(0, max(max(CONCURRENCY_SCALING[e]["itl"]) for e in ("vllm", "sglang", "trtllm")) * 1.35)

    save(fig, "concurrency_itl")


# ===================================================================
# CHART 4 - Long Context: Combined 2-panel
# ===================================================================
def chart_long_context():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(len(LONG_CONTEXT["engines"]))
    w = 0.55

    # Throughput
    bars1 = ax1.bar(x, LONG_CONTEXT["throughput"], w, color=LONG_CONTEXT["colors"], edgecolor="none", zorder=3)
    bars1[-1].set(hatch="//", edgecolor=BG_COLOR)
    add_bar_labels(ax1, bars1, padding=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(LONG_CONTEXT["engines"], fontsize=10)
    style_ax(ax1, "Output Tokens / sec", "Throughput")
    ax1.set_ylim(0, max(LONG_CONTEXT["throughput"]) * 1.3)

    # TTFT
    bars2 = ax2.bar(x, [v / 1000 for v in LONG_CONTEXT["ttft"]], w, color=LONG_CONTEXT["colors"], edgecolor="none", zorder=3)
    bars2[-1].set(hatch="//", edgecolor=BG_COLOR)
    add_bar_labels(ax2, bars2, fmt="{:.1f}s", padding=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(LONG_CONTEXT["engines"], fontsize=10)
    style_ax(ax2, "TTFT (sec)", "Time To First Token")
    ax2.set_ylim(0, max(v / 1000 for v in LONG_CONTEXT["ttft"]) * 1.25)

    fig.suptitle("Long Context - ISL 8000, OSL 512, Concurrency 8, FP8 KV", fontsize=16, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    save(fig, "long_context")


# ===================================================================
# CHART 5 - Quantization: Combined 2-panel
# ===================================================================
def chart_quantization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(len(QUANTIZATION["models"]))
    w = 0.55

    # Throughput
    bars1 = ax1.bar(x, QUANTIZATION["throughput"], w, color=QUANTIZATION["colors"], edgecolor="none", zorder=3)
    add_bar_labels(ax1, bars1, padding=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(QUANTIZATION["models"], fontsize=12)
    style_ax(ax1, "Output Tokens / sec", "Throughput")
    ax1.set_ylim(0, max(QUANTIZATION["throughput"]) * 1.25)

    # TTFT
    bars2 = ax2.bar(x, [v / 1000 for v in QUANTIZATION["ttft"]], w, color=QUANTIZATION["colors"], edgecolor="none", zorder=3)
    add_bar_labels(ax2, bars2, fmt="{:.1f}s", padding=0.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(QUANTIZATION["models"], fontsize=12)
    style_ax(ax2, "TTFT (sec)", "Time To First Token")
    ax2.set_ylim(0, max(v / 1000 for v in QUANTIZATION["ttft"]) * 1.25)

    fig.suptitle("Quantization Comparison - vLLM, Concurrency 32, equal KV budget", fontsize=16, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    save(fig, "quantization")


# ===================================================================
# CHART 6 - MoE: Combined 3-panel
# ===================================================================
def chart_moe():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5.5))
    x = np.arange(len(MOE["models"]))
    w = 0.5

    # Throughput
    bars1 = ax1.bar(x, MOE["throughput"], w, color=MOE["colors"], edgecolor="none", zorder=3)
    add_bar_labels(ax1, bars1, padding=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(MOE["models"], fontsize=10)
    style_ax(ax1, "Output Tokens / sec", "Throughput")
    ax1.set_ylim(0, max(MOE["throughput"]) * 1.25)

    # TTFT
    bars2 = ax2.bar(x, [v / 1000 for v in MOE["ttft"]], w, color=MOE["colors"], edgecolor="none", zorder=3)
    add_bar_labels(ax2, bars2, fmt="{:.1f}s", padding=0.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(MOE["models"], fontsize=10)
    style_ax(ax2, "TTFT (sec)", "Time To First Token")
    ax2.set_ylim(0, max(v / 1000 for v in MOE["ttft"]) * 1.25)

    # ITL
    bars3 = ax3.bar(x, MOE["itl"], w, color=MOE["colors"], edgecolor="none", zorder=3)
    add_bar_labels(ax3, bars3, padding=1)
    ax3.set_xticks(x)
    ax3.set_xticklabels(MOE["models"], fontsize=10)
    style_ax(ax3, "ITL (ms)", "Inter-Token Latency")
    ax3.set_ylim(0, max(MOE["itl"]) * 1.35)

    fig.suptitle("MoE vs Dense - SGLang, Concurrency 32", fontsize=16, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    save(fig, "moe_comparison")


# ===================================================================
# CHART 7 - Summary radar / heatmap-style comparison
# ===================================================================
def chart_engine_summary():
    fig, ax = plt.subplots(figsize=(10, 5))

    categories = ["Throughput\nc64", "TTFT\nc8", "TTFT\nc32", "ITL\nc32", "Long-Ctx\nThroughput", "Long-Ctx\nTTFT"]

    # Normalized scores (higher = better, 0-1 scale)
    # "Higher is better" metrics: score = val / max_val
    # "Lower is better" metrics:  score = min_val / val  (best gets 1.0, worst still visible)
    # Equal-budget runs only, so TensorRT-LLM's long-context bars are at the 35,328-token budget
    vllm_scores   = [228.9/298.7, 1122/4202, 6336/9181, 70.2/98.5, 46.0/48.9, 48263/51602]
    sglang_scores = [283.3/298.7, 1122/2365, 6336/9080, 70.2/76.1, 48.9/48.9, 48263/48263]
    trtllm_scores = [298.7/298.7, 1122/1122, 6336/6336, 70.2/70.2, 25.1/48.9, 48263/82437]

    x = np.arange(len(categories))
    w = 0.24

    bars_v = ax.bar(x - w, vllm_scores, w, label="vLLM", color=COLORS["vllm"], edgecolor="none", zorder=3)
    bars_s = ax.bar(x,     sglang_scores, w, label="SGLang", color=COLORS["sglang"], edgecolor="none", zorder=3)
    bars_t = ax.bar(x + w, trtllm_scores, w, label="TensorRT-LLM", color=COLORS["trtllm"], edgecolor="none", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    style_ax(ax, "Relative Score", "Engine Comparison - Equal KV Budget (higher is better)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=11)

    save(fig, "engine_summary")


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    apply_style()
    print(f"Generating charts into {OUTPUT_DIR}/\n")
    chart_concurrency_throughput()
    chart_concurrency_ttft()
    chart_concurrency_itl()
    chart_long_context()
    chart_quantization()
    chart_moe()
    chart_engine_summary()
    print(f"\nDone - {len(list(OUTPUT_DIR.glob('*.png')))} charts generated.")
