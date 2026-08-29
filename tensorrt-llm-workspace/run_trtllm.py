#!/usr/bin/env python3

import argparse
import json
import pathlib
import shutil
import subprocess
import sys


def run(cmd):
    print("\n" + "=" * 100)
    print("Running:")
    print(" ".join(map(str, cmd)))
    print("=" * 100 + "\n")

    subprocess.run(cmd, check=True)


def has_files(path: pathlib.Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def read_json(path: pathlib.Path):
    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def checkpoint_is_fp8(checkpoint_dir: pathlib.Path) -> bool:
    config_path = checkpoint_dir / "config.json"

    config = read_json(config_path)

    if not config:
        return False

    quantization = config.get("quantization", {})
    quant_algo = quantization.get("quant_algo")

    if quant_algo is None:
        return False

    return "fp8" in str(quant_algo).lower()


def sanitize_name(model: str) -> str:
    return model.split("/")[-1]


def infer_base_model(model: str) -> str:
    """
    Qwen/Qwen3-32B-FP8
        ->
    Qwen/Qwen3-32B

    Qwen/Qwen3-32B-AWQ
        ->
    Qwen/Qwen3-32B
    """

    suffixes = [
        "-FP8",
        "-fp8",
        "-AWQ",
        "-awq",
        "-GPTQ",
        "-gptq",
    ]

    for suffix in suffixes:
        if model.endswith(suffix):
            return model[:-len(suffix)]

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Build and serve a quantized model with TensorRT-LLM"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model name, e.g. Qwen/Qwen3-32B-FP8",
    )

    parser.add_argument(
        "--source-model",
        default=None,
        help=(
            "Optional base model used for TRT-LLM quantization. "
            "If omitted, inferred from --model."
        ),
    )

    parser.add_argument(
        "--quant",
        default=None,
        choices=[
            "fp8",
            "int4_awq",
            "w4a8_awq",
            "int4_gptq",
            "full_prec",
        ],
        help="Quantization format. If omitted, inferred from model name.",
    )

    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    # Matches your manual trtllm-build command
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--opt-num-tokens",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--calib-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--calib-max-seq-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--force-quantize",
        action="store_true",
        help="Delete existing TRT-LLM checkpoint and quantize again.",
    )

    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Delete existing engine and build again.",
    )

    args = parser.parse_args()

    #
    # Paths
    #

    model_name = sanitize_name(args.model)

    source_model = args.source_model or infer_base_model(args.model)

    source_model_name = sanitize_name(source_model)

    #
    # Infer quantization
    #

    quant = args.quant

    if quant is None:
        model_lower = args.model.lower()

        if "fp8" in model_lower:
            quant = "fp8"

        elif "awq" in model_lower:
            quant = "int4_awq"

        elif "gptq" in model_lower:
            quant = "int4_gptq"

        else:
            quant = "full_prec"

    print("\nConfiguration")
    print("=" * 100)

    print(f"Requested model : {args.model}")
    print(f"Source model    : {source_model}")
    print(f"Quantization    : {quant}")
    print(f"TP size         : {args.tp_size}")

    print("=" * 100)

    #
    # Directory layout
    #

    models_dir = pathlib.Path("models")

    source_dir = models_dir / source_model_name

    # Matches your manual commands:
    #
    # Qwen3-32B-FP8-checkpoint
    # Qwen3-32B-FP8-engine
    #
    checkpoint_dir = pathlib.Path(
        f"{model_name}-checkpoint"
    )

    engine_dir = pathlib.Path(
        f"{model_name}-engine"
    )

    #
    # TensorRT-LLM repository
    #

    tensor_root = pathlib.Path.home() / "TensorRT-LLM"

    quantize_script = (
        tensor_root
        / "examples"
        / "quantization"
        / "quantize.py"
    )

    #
    # Check dependencies
    #

    if not shutil.which("huggingface-cli"):
        print("ERROR: huggingface-cli not found")
        sys.exit(1)

    if not shutil.which("trtllm-build"):
        print("ERROR: trtllm-build not found")
        sys.exit(1)

    if not shutil.which("trtllm-serve"):
        print("ERROR: trtllm-serve not found")
        sys.exit(1)

    #
    # Clone TensorRT-LLM
    #

    if not tensor_root.exists():

        run([
            "git",
            "clone",
            "--branch",
            "v1.2.1",
            "--depth",
            "1",
            "https://github.com/NVIDIA/TensorRT-LLM.git",
            str(tensor_root),
        ])

    if not quantize_script.exists():

        print(
            f"ERROR: quantize.py not found:\n"
            f"{quantize_script}"
        )

        sys.exit(1)

    #
    # Download source model
    #

    if has_files(source_dir):

        print(
            f"\n✓ Source model already exists:\n"
            f"  {source_dir}"
        )

    else:

        source_dir.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        run([
            "huggingface-cli",
            "download",
            source_model,
            "--local-dir",
            str(source_dir),
        ])

    #
    # Optional checkpoint cleanup
    #

    if args.force_quantize and checkpoint_dir.exists():

        print(
            f"\nRemoving existing checkpoint:\n"
            f"  {checkpoint_dir}"
        )

        shutil.rmtree(checkpoint_dir)

    #
    # Quantization
    #

    if quant == "full_prec":

        print("\nFull precision selected.")

        print(
            "WARNING: This script currently expects "
            "a TRT-LLM checkpoint for trtllm-build."
        )

        print(
            "For full precision, use the dedicated "
            "model conversion workflow."
        )

        sys.exit(1)

    if has_files(checkpoint_dir):

        print(
            f"\n✓ Quantized checkpoint already exists:\n"
            f"  {checkpoint_dir}"
        )

        if quant == "fp8":

            if checkpoint_is_fp8(checkpoint_dir):

                print(
                    "✓ Checkpoint is confirmed as FP8."
                )

            else:

                print(
                    "\nERROR: Existing checkpoint is NOT "
                    "recognized as FP8."
                )

                print(
                    "This likely means it was created using "
                    "convert_checkpoint.py instead of quantize.py."
                )

                print(
                    "\nDelete it or run:"
                )

                print(
                    "  --force-quantize"
                )

                sys.exit(1)

    else:

        quant_cmd = [
            "python",
            str(quantize_script),

            "--model_dir",
            str(source_dir),

            "--output_dir",
            str(checkpoint_dir),

            "--qformat",
            quant,

            "--dtype",
            "bfloat16",

            "--tp_size",
            str(args.tp_size),
        ]

        #
        # FP8 specific options
        #

        if quant == "fp8":

            quant_cmd.extend([
                "--kv_cache_dtype",
                "fp8",

                "--calib_size",
                str(args.calib_size),

                "--calib_max_seq_length",
                str(args.calib_max_seq_length),

                "--batch_size",
                str(args.batch_size),
            ])

        #
        # AWQ specific options
        #

        if quant in [
            "int4_awq",
            "w4a8_awq",
        ]:

            quant_cmd.extend([
                "--calib_size",
                str(args.calib_size),

                "--batch_size",
                str(args.batch_size),
            ])

        run(quant_cmd)

    #
    # Validate FP8 checkpoint
    #

    if quant == "fp8":

        if not checkpoint_is_fp8(checkpoint_dir):

            config_path = checkpoint_dir / "config.json"

            print(
                "\nERROR: Generated checkpoint is not "
                "recognized as FP8."
            )

            print(
                f"\nCheck:\n{config_path}"
            )

            config = read_json(config_path)

            if config:
                print("\nQuantization section:")

                print(
                    json.dumps(
                        config.get(
                            "quantization",
                            {}
                        ),
                        indent=4,
                    )
                )

            sys.exit(1)

        print(
            "\n✓ FP8 checkpoint validation passed."
        )

    #
    # Optional engine cleanup
    #

    if args.force_build and engine_dir.exists():

        print(
            f"\nRemoving existing engine:\n"
            f"  {engine_dir}"
        )

        shutil.rmtree(engine_dir)

    #
    # Build engine
    #

    if has_files(engine_dir):

        print(
            f"\n✓ Engine already exists:\n"
            f"  {engine_dir}"
        )

    else:

        build_cmd = [
            "trtllm-build",

            "--checkpoint_dir",
            str(checkpoint_dir),

            "--output_dir",
            str(engine_dir),

            "--max_batch_size",
            str(args.max_batch_size),

            "--max_seq_len",
            str(args.max_seq_len),

            "--max_num_tokens",
            str(args.max_num_tokens),

            "--opt_num_tokens",
            str(args.opt_num_tokens),

            "--gemm_plugin",
            "auto",

            "--gpt_attention_plugin",
            "auto",
        ]

        run(build_cmd)

    #
    # Serve
    #

    print(
        "\nStarting TensorRT-LLM server..."
    )

    run([
        "trtllm-serve",
        "serve",

        str(engine_dir),

        "--backend",
        "tensorrt",

        # Matches your manual command
        "--tokenizer",
        args.model,

        "--max_batch_size",
        str(args.max_batch_size),

        "--host",
        args.host,

        "--port",
        str(args.port),
    ])


if __name__ == "__main__":
    main()