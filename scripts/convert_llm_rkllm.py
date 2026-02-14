#!/usr/bin/env python3
"""
Convert CosyVoice3 LLM (repackaged as Qwen2) to RKLLM format.

Requires: rkllm-toolkit >= 1.2.3
Target: RK3588 NPU (6 TOPS)

Usage:
    # Activate venv with RKLLM first:
    source /home/user/NAS/functiongemma_convert/venv/bin/activate

    # Then run:
    python convert_llm_rkllm.py [--quantization w8a8]
"""

import os
import sys
import argparse

from rkllm.api import RKLLM

MODEL_DIR = os.environ.get(
    "MODEL_DIR", "./cosyvoice3_qwen2_for_rkllm"
)
OUTPUT_PATH = os.environ.get(
    "OUTPUT_PATH", "./cosyvoice3_llm_rk3588.rkllm"
)


def main():
    parser = argparse.ArgumentParser(description="Convert CosyVoice3 LLM to RKLLM")
    parser.add_argument("--model_dir", default=MODEL_DIR,
                        help="Path to extracted Qwen2 model")
    parser.add_argument("--output", default=OUTPUT_PATH,
                        help="Output .rkllm path")
    parser.add_argument("--quantization", default="w8a8",
                        choices=["w4a16", "w8a8", "w8a16"],
                        help="Quantization type (default: w8a8)")
    parser.add_argument("--optimization_level", type=int, default=1,
                        choices=[0, 1, 2],
                        help="Optimization level (default: 1)")
    parser.add_argument("--target", default="rk3588",
                        choices=["rk3588", "rk3576"],
                        help="Target platform (default: rk3588)")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"ERROR: Model directory not found: {args.model_dir}")
        print(f"  Run: python extract_llm_as_qwen2.py")
        sys.exit(1)

    print(f"=== Converting CosyVoice3 LLM (Qwen2) to RKLLM ===")
    print(f"Model:        {args.model_dir}")
    print(f"Output:       {args.output}")
    print(f"Quantization: {args.quantization}")
    print(f"Target:       {args.target}")
    print(f"Optimization: {args.optimization_level}")
    print()

    llm = RKLLM()

    # Step 1: Load
    print("1. Loading model...")
    ret = llm.load_huggingface(model=args.model_dir)
    if ret != 0:
        print(f"   FAILED to load model: {ret}")
        sys.exit(1)
    print("   Model loaded.")

    # Step 2: Build
    print(f"\n2. Building ({args.quantization}, opt={args.optimization_level})...")
    ret = llm.build(
        do_quantization=True,
        quantized_dtype=args.quantization,
        optimization_level=args.optimization_level,
        target_platform=args.target,
    )
    if ret != 0:
        print(f"   FAILED to build: {ret}")
        sys.exit(1)
    print("   Build complete.")

    # Step 3: Export
    print(f"\n3. Exporting to {args.output}...")
    ret = llm.export_rkllm(args.output)
    if ret != 0:
        print(f"   FAILED to export: {ret}")
        sys.exit(1)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\n=== Done ===")
    print(f"Output: {args.output}")
    print(f"Size:   {size_mb:.1f} MB")
    print(f"\nTo deploy on CM3588:")
    print(f"  scp {args.output} user@cm3588:~/cosyvoice3-rknn/")
    print(f"\nExpected performance on RK3588:")
    print(f"  Qwen2-0.5B w8a8: ~14 tok/s")
    print(f"  Qwen2-0.5B w4a16: ~21 tok/s")


if __name__ == "__main__":
    main()
