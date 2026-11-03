"""How many requests start within 200 ms of each other (lockstep check)."""
import json, sys
for path in sys.argv[1:]:
    starts, osl = [], []
    for line in open(path):
        r = json.loads(line)
        md, mt = r["metadata"], r["metrics"]
        if md.get("benchmark_phase") != "profiling":
            continue
        starts.append(md["request_start_ns"] / 1e9)
        if "output_sequence_length" in mt:
            osl.append(mt["output_sequence_length"]["value"])
    starts.sort()
    groups, cur = [], 1
    for a, b in zip(starts, starts[1:]):
        if b - a < 0.2:
            cur += 1
        else:
            groups.append(cur); cur = 1
    groups.append(cur)
    print(path.split("/")[-2][:60], "n=", len(starts), "mean group size=%.2f" % (sum(groups) / len(groups)),
          "OSL min/max=", min(osl), max(osl))
