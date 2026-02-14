#!/usr/bin/env python3
"""
Extract CosyVoice3 Flow front-end components for CPU pipeline.

Extracts:
  1. input_embedding weights (Embedding(6561, 80)) → .npy
  2. spk_embed_affine_layer weights (Linear(192, 80)) → .npy
  3. pre_lookahead_layer conv weights → .npz
  4. Flow config → JSON

Usage:
    cd ~/CosyVoice3/repo
    python3 ~/NAS/tts-rknn-convert/cosyvoice3-rknn/export_flow_frontend.py
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

COSYVOICE_ROOT = os.path.expanduser("~/CosyVoice3/repo")
sys.path.insert(0, COSYVOICE_ROOT)

FLOW_PT = os.path.join(
    COSYVOICE_ROOT,
    "pretrained_models/Fun-CosyVoice3-0.5B/flow.pt"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cosyvoice3-flow-components"
)


def main():
    parser = argparse.ArgumentParser(description="Export Flow front-end components")
    parser.add_argument("--flow_pt", default=FLOW_PT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load state dict directly (no model construction needed)
    print(f"Loading state dict: {args.flow_pt}")
    state = torch.load(args.flow_pt, map_location="cpu", weights_only=True)
    print(f"  {len(state)} keys")

    # 1. Extract input_embedding
    emb_weight = state["input_embedding.weight"].numpy()
    np.save(os.path.join(args.output_dir, "flow_input_embedding.npy"), emb_weight)
    print(f"\n1. flow_input_embedding: {emb_weight.shape}")

    # 2. Extract spk_embed_affine_layer
    spk_w = state["spk_embed_affine_layer.weight"].numpy()
    spk_b = state["spk_embed_affine_layer.bias"].numpy()
    np.save(os.path.join(args.output_dir, "flow_spk_affine_weight.npy"), spk_w)
    np.save(os.path.join(args.output_dir, "flow_spk_affine_bias.npy"), spk_b)
    print(f"2. flow_spk_affine: weight={spk_w.shape}, bias={spk_b.shape}")

    # 3. Extract pre_lookahead_layer conv weights
    conv_weights = {
        "conv1_weight": state["pre_lookahead_layer.conv1.weight"].numpy(),
        "conv1_bias": state["pre_lookahead_layer.conv1.bias"].numpy(),
        "conv2_weight": state["pre_lookahead_layer.conv2.weight"].numpy(),
        "conv2_bias": state["pre_lookahead_layer.conv2.bias"].numpy(),
    }
    npz_path = os.path.join(args.output_dir, "flow_pre_lookahead_weights.npz")
    np.savez(npz_path, **conv_weights)
    print(f"3. pre_lookahead weights:")
    for k, v in conv_weights.items():
        print(f"   {k}: {v.shape}")

    # 4. Save config
    config = {
        "input_size": 80,
        "output_size": 80,
        "vocab_size": 6561,
        "token_mel_ratio": 2,
        "pre_lookahead_len": 3,
        "spk_embed_dim": 192,
        "n_timesteps": 10,
        "inference_cfg_rate": 0.7,
        "sigma_min": 1e-6,
        "solver": "euler",
        "t_scheduler": "cosine",
        "pre_lookahead_channels": 1024,
    }
    config_path = os.path.join(args.output_dir, "flow_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n4. Config: {config_path}")

    # 5. List all flow.pt keys for reference
    print(f"\n5. All state dict key prefixes:")
    prefixes = set()
    for k in state.keys():
        prefix = k.split(".")[0]
        prefixes.add(prefix)
    for p in sorted(prefixes):
        count = sum(1 for k in state.keys() if k.startswith(p + "."))
        print(f"   {p}: {count} keys")

    # Summary
    print(f"\n{'='*60}")
    print(f"EXPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Output dir: {args.output_dir}")
    for f_name in sorted(os.listdir(args.output_dir)):
        f_path = os.path.join(args.output_dir, f_name)
        size = os.path.getsize(f_path)
        print(f"  {f_name} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
