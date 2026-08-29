#!/usr/bin/env python3

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=8192,
    )

    args = parser.parse_args()

    cmd = [
        "python",
        "-m",
        "sglang.launch_server",

        "--model-path",
        args.model,

        "--host",
        args.host,

        "--port",
        str(args.port),

        "--mem-fraction-static",
        str(args.mem_fraction_static),

        "--context-length",
        str(args.context_length),
    ]

    print("Running:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()