#!/usr/bin/env python3
"""Summarise matched-KV rerun results: client metrics, decode occupancy, server queue/prefill split."""

import glob
import json
import os
import re
import sys

results = sys.argv[1]

# Histogram names (without _sum/_count) for queue wait and prefill per engine.
QUEUE = re.compile(r"(request_queue_time_seconds|queue_time_seconds)$")
PREFILL = re.compile(r"(request_prefill_time_seconds|prefill_time_seconds)$")


def prom(path):
    vals = {}
    if not os.path.exists(path):
        return vals
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"([^{\s]+)(\{[^}]*\})?\s+([-0-9.eE+naNinf]+)", line)
        if m:
            vals[m.group(1)] = vals.get(m.group(1), 0.0) + float(m.group(3))
    return vals


def hist_mean(before, after, pattern):
    for name in after:
        if name.endswith("_sum") and pattern.search(name[:-4]):
            base = name[:-4]
            n = after.get(base + "_count", 0) - before.get(base + "_count", 0)
            s = after[name] - before.get(name, 0)
            if n > 0:
                return s / n, base
    return None, None


def occupancy(path):
    reqs = []
    for line in open(path):
        r = json.loads(line)
        md, mt = r["metadata"], r["metrics"]
        if md.get("benchmark_phase") != "profiling" or "time_to_first_token" not in mt:
            continue
        s = md["request_start_ns"]
        reqs.append((s + mt["time_to_first_token"]["value"] * 1e6, md["request_end_ns"]))
    t0 = min(a for a, _ in reqs)
    t1 = max(b for _, b in reqs)
    lo, hi = t0 + (t1 - t0) * 0.15, t0 + (t1 - t0) * 0.85
    ev = sorted([(a, 1) for a, _ in reqs] + [(b, -1) for _, b in reqs])
    cur = area = mx = 0
    prev = lo
    for t, d in ev:
        if t > lo:
            tt = min(t, hi)
            if tt > prev:
                area += cur * (tt - prev)
                prev = tt
        cur += d
        if lo <= t <= hi:
            mx = max(mx, cur)
    if prev < hi:
        area += cur * (hi - prev)
    return area / (hi - lo), mx


rows = []
for d in sorted(glob.glob(os.path.join(results, "*/"))):
    if not os.path.exists(d + "profile_export_aiperf.json"):
        continue  # run still in progress or failed
    j = json.load(open(d + "profile_export_aiperf.json"))
    g = lambda k, s="avg": (j.get(k) or {}).get(s)
    occ, occ_max = occupancy(d + "profile_export.jsonl")
    before, after = prom(d + "prometheus_before.txt"), prom(d + "prometheus_after.txt")
    q, qn = hist_mean(before, after, QUEUE)
    p, pn = hist_mean(before, after, PREFILL)
    rows.append({
        "run": os.path.basename(d.rstrip("/")),
        "req_s": g("request_throughput"), "tok_s": g("output_token_throughput"),
        "ttft": g("time_to_first_token"), "ttft_p50": g("time_to_first_token", "p50"),
        "ttft_p99": g("time_to_first_token", "p99"), "itl": g("inter_token_latency"),
        "osl": g("output_sequence_length"), "osl_min": g("output_sequence_length", "min"),
        "isl": g("input_sequence_length"), "errors": len(j.get("error_summary") or []),
        "occ": occ, "occ_max": occ_max, "queue_s": q, "prefill_s": p, "queue_metric": qn, "prefill_metric": pn,
    })

fmt = lambda v, f: "-" if v is None else format(v, f)
for r in rows:
    print(f"{r['run'][:-16]:44s} req/s={fmt(r['req_s'],'.2f')} tok/s={fmt(r['tok_s'],'.1f')} "
          f"TTFT={fmt(r['ttft'],'.0f')} p50={fmt(r['ttft_p50'],'.0f')} p99={fmt(r['ttft_p99'],'.0f')} "
          f"ITL={fmt(r['itl'],'.1f')} OSL={fmt(r['osl'],'.1f')}/{fmt(r['osl_min'],'.0f')} ISL={fmt(r['isl'],'.0f')} "
          f"err={r['errors']} gen={r['occ']:.1f}({r['occ_max']}) queue={fmt(r['queue_s'],'.2f')}s prefill={fmt(r['prefill_s'],'.2f')}s")
json.dump(rows, open(os.path.join(results, "summary.json"), "w"), indent=1)
