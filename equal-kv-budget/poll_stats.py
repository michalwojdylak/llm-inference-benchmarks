#!/usr/bin/env python3
"""Poll GPU memory (all engines) and TRT-LLM iteration stats every 2 s into a JSONL file."""

import json
import subprocess
import sys
import time
import urllib.request

engine, out_path = sys.argv[1], sys.argv[2]

with open(out_path, "a") as out:
    while True:
        rec = {"t": time.time()}
        rec["gpu_mem_used_mib"] = int(subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout.strip() or -1)
        if engine == "tensorrtllm":
            try:
                with urllib.request.urlopen("http://localhost:8000/metrics", timeout=5) as r:
                    rec["iter_stats"] = json.loads(r.read())
            except Exception as e:
                rec["error"] = str(e)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        time.sleep(2)
