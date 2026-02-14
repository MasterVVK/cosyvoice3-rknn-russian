#!/usr/bin/env python3
"""
Convert CosyVoice3 Flow DiT estimator from ONNX to RKNN.

Input: flow_estimator_seqN_sim.onnx (from onnxsim with fixed shapes)
Output: flow_estimator_seqN.rknn (for RK3588 NPU)

Usage:
    python3 convert_flow_rknn.py [--onnx PATH] [--output PATH]
"""

import os
import argparse
from rknn.api import RKNN

DEFAULT_ONNX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cosyvoice3-onnx/flow_estimator_seq500_sim.onnx"
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cosyvoice3-rknn-models/flow_estimator_seq500.rknn"
)


def main():
    parser = argparse.ArgumentParser(description="Convert Flow ONNX to RKNN")
    parser.add_argument("--onnx", default=DEFAULT_ONNX,
                        help="Input ONNX model path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output RKNN model path")
    parser.add_argument("--target", default="rk3588",
                        choices=["rk3588", "rk3576"])
    parser.add_argument("--opt_level", type=int, default=3,
                        choices=[1, 2, 3],
                        help="RKNN optimization level (default: 3)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if not os.path.exists(args.onnx):
        print(f"ERROR: ONNX not found: {args.onnx}")
        return 1

    print(f"=== Converting Flow DiT estimator to RKNN ===")
    print(f"Input:  {args.onnx}")
    print(f"Output: {args.output}")
    print(f"Target: {args.target}")
    print(f"Opt level: {args.opt_level}")

    rknn = RKNN(verbose=True)

    print("\n1. Configuring RKNN...")
    rknn.config(
        target_platform=args.target,
        optimization_level=args.opt_level,
    )

    print("\n2. Loading ONNX...")
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print(f"ERROR: Failed to load ONNX: {ret}")
        return 1
    print("   Loaded.")

    print("\n3. Building RKNN (FP16, no quantization)...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"ERROR: Failed to build: {ret}")
        return 1
    print("   Built.")

    print(f"\n4. Exporting to {args.output}...")
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print(f"ERROR: Failed to export: {ret}")
        return 1

    rknn.release()

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\n=== SUCCESS ===")
    print(f"Output: {args.output} ({size_mb:.1f} MB)")
    print(f"\nDeploy to CM3588:")
    print(f"  scp {args.output} root@192.168.1.173:/root/cosyvoice3-rknn/")
    return 0


if __name__ == "__main__":
    exit(main())
