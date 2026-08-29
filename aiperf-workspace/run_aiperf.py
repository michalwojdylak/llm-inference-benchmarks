#!/usr/bin/env python3

import argparse
import datetime
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--framework",
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--url",
        default="http://localhost:8000",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--isl",
        type=int,
        default=1024,
        help="Input sequence length",
    )

    parser.add_argument(
        "--osl",
        type=int,
        default=256,
        help="Output sequence length",
    )

    parser.add_argument(
        "--request-count",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--warmup-request-count",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--results-dir",
        default="results",
    )

    parser.add_argument(
        "--use-server-token-count",
        action="store_true",
        help="Use token count returned by the server",
    )

    parser.add_argument(
        "--ignore_eos",
        action="store_true",
        help="Add --extra-inputs ignore_eos:true",
    )

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    model_name = args.model.split("/")[-1]

    experiment_name = (
        f"{args.framework}"
        f"_{model_name}"
        f"_c{args.concurrency}"
        f"_isl{args.isl}"
        f"_osl{args.osl}"
        f"_{timestamp}"
    )

    output_dir = os.path.join(
        args.results_dir,
        experiment_name,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    cmd = [
        "aiperf",
        "profile",

        "--model",
        args.model,

        "--url",
        args.url,

        "--endpoint-type",
        "chat",

        "--tokenizer",
        args.model,

        "--streaming",

        "--synthetic-input-tokens-mean",
        str(args.isl),

        "--synthetic-input-tokens-stddev",
        "0",

        "--output-tokens-mean",
        str(args.osl),

        "--output-tokens-stddev",
        "0",

        "--concurrency",
        str(args.concurrency),

        "--request-count",
        str(args.request_count),

        "--warmup-request-count",
        str(args.warmup_request_count),

        "--artifact-dir",
        output_dir,
    ]

    #
    # Optional:
    # --use-server-token-count
    #

    if args.use_server_token_count:
        cmd.append(
            "--use-server-token-count"
        )

    #
    # Optional:
    # --extra-inputs ignore_eos:true
    #

    if args.ignore_eos:
        cmd.extend([
            "--extra-inputs",
            "ignore_eos:true",
        ])

    print("=" * 80)
    print("Running benchmark")
    print("=" * 80)
    print(" ".join(cmd))
    print()

    with open(
        os.path.join(output_dir, "command.txt"),
        "w",
    ) as f:
        f.write(" ".join(cmd))

    with open(
        os.path.join(output_dir, "stdout.log"),
        "w",
    ) as out:

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        for line in process.stdout:
            print(line, end="")
            out.write(line)

        process.wait()

        if process.returncode != 0:
            sys.exit(process.returncode)

    print()
    print(
        f"Results saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()