#!/usr/bin/env python3

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
    )

    args = parser.parse_args()

    cmd = [
        "vllm",
        "serve",
        args.model,

        "--host",
        args.host,

        "--port",
        str(args.port),

        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),

        "--max-model-len",
        str(args.max_model_len),
    ]

    print("Running:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()