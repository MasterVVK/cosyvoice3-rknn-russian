#!/usr/bin/env python3
"""
Extract CosyVoice3 LLM (Qwen2) and repackage as standard Qwen2ForCausalLM.

CosyVoice3's LLM is a Qwen2ForCausalLM wrapped in Qwen2Encoder/Qwen2LM.
State dict keys have prefix 'llm.model.model.' for transformer layers.
We strip the prefix and save as standard Qwen2 format for RKLLM conversion.

Also extracts CPU-side components (speech_embedding, llm_decoder) as numpy.

Usage:
    python extract_llm_as_qwen2.py [--llm_pt PATH] [--output_dir PATH]
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from safetensors.torch import save_file

# Default paths
DEFAULT_LLM_PT = os.path.expanduser(
    "~/CosyVoice3/repo/pretrained_models/Fun-CosyVoice3-0.5B/llm.pt"
)
DEFAULT_OUTPUT_DIR = "./cosyvoice3_qwen2_for_rkllm"
DEFAULT_EMBEDDINGS_DIR = "./cosyvoice3_embeddings"


def main():
    parser = argparse.ArgumentParser(description="Extract CosyVoice3 LLM as Qwen2")
    parser.add_argument("--llm_pt", default=DEFAULT_LLM_PT,
                        help="Path to CosyVoice3 llm.pt")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for Qwen2 model")
    parser.add_argument("--embeddings_dir", default=DEFAULT_EMBEDDINGS_DIR,
                        help="Output directory for CPU-side embeddings")
    args = parser.parse_args()

    if not os.path.exists(args.llm_pt):
        print(f"ERROR: {args.llm_pt} not found")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.embeddings_dir, exist_ok=True)

    print(f"Loading {args.llm_pt}...")
    state = torch.load(args.llm_pt, map_location="cpu", weights_only=True)
    print(f"  Total keys: {len(state)}")

    # ================================================================
    # Part 1: Extract Qwen2 transformer as standard Qwen2ForCausalLM
    # ================================================================
    print("\n=== Part 1: Qwen2 transformer layers ===")

    qwen2_weights = {}
    cpu_weights = {}
    skipped = []

    for key, tensor in state.items():
        if key.startswith("llm.model.model."):
            # llm.model.model.layers.X.* -> model.layers.X.*
            # llm.model.model.embed_tokens.weight -> model.embed_tokens.weight
            # llm.model.model.norm.weight -> model.norm.weight
            new_key = key.replace("llm.model.model.", "model.")
            qwen2_weights[new_key] = tensor

        elif key == "llm.model.lm_head.weight":
            # Keep lm_head for RKLLM validation
            # Clone to avoid shared memory with embed_tokens (tied weights)
            qwen2_weights["lm_head.weight"] = tensor.clone()

        elif key == "speech_embedding.weight":
            cpu_weights["speech_embedding"] = tensor

        elif key == "llm_decoder.weight":
            cpu_weights["llm_decoder_weight"] = tensor

        elif key == "llm_embedding.weight":
            cpu_weights["llm_embedding"] = tensor

        else:
            skipped.append(key)

    # Count layers
    layer_nums = sorted(set(
        int(k.split(".")[2]) for k in qwen2_weights
        if k.startswith("model.layers.")
    ))
    num_layers = len(layer_nums)

    print(f"  Qwen2 keys: {len(qwen2_weights)}")
    print(f"  CPU keys: {len(cpu_weights)}")
    print(f"  Skipped: {skipped}")
    print(f"  Layers: {num_layers} ({min(layer_nums)}..{max(layer_nums)})")

    # Determine architecture params from weights
    hidden_size = qwen2_weights["model.layers.0.input_layernorm.weight"].shape[0]
    intermediate_size = qwen2_weights["model.layers.0.mlp.gate_proj.weight"].shape[0]
    num_heads = qwen2_weights["model.layers.0.self_attn.q_proj.weight"].shape[0] // \
                (qwen2_weights["model.layers.0.self_attn.q_proj.weight"].shape[0] //
                 (qwen2_weights["model.layers.0.self_attn.q_proj.weight"].shape[0] //
                  qwen2_weights["model.layers.0.self_attn.k_proj.weight"].shape[0]) *
                 qwen2_weights["model.layers.0.self_attn.k_proj.weight"].shape[0] //
                 qwen2_weights["model.layers.0.self_attn.k_proj.weight"].shape[0])

    # Simpler: q_proj [896, 896], k_proj [128, 896]
    q_size = qwen2_weights["model.layers.0.self_attn.q_proj.weight"].shape[0]  # 896
    kv_size = qwen2_weights["model.layers.0.self_attn.k_proj.weight"].shape[0]  # 128
    head_dim = kv_size  # For GQA: kv_size = num_kv_heads * head_dim, with num_kv_heads typically small
    # Actually: q_proj [num_heads * head_dim, hidden_size]
    #           k_proj [num_kv_heads * head_dim, hidden_size]
    # With Qwen2-0.5B: q_proj [896, 896] → num_heads=14 if head_dim=64, or num_heads=7 if head_dim=128
    # k_proj [128, 896] → num_kv_heads=2 if head_dim=64, or num_kv_heads=1 if head_dim=128
    # Actually looking at q_proj.bias [896] and k_proj.bias [128]:
    # If head_dim=64: num_heads=14, num_kv_heads=2
    # If head_dim=128: num_heads=7, num_kv_heads=1
    # Qwen2-0.5B uses head_dim=64, num_heads=14, num_kv_heads=2
    # But let's check: 896 / 14 = 64. 128 / 2 = 64. Yes.
    head_dim = 64
    num_attention_heads = q_size // head_dim  # 896 / 64 = 14
    num_kv_heads = kv_size // head_dim  # 128 / 64 = 2

    vocab_size = qwen2_weights["model.embed_tokens.weight"].shape[0]  # 151936

    print(f"\n  Architecture:")
    print(f"    hidden_size: {hidden_size}")
    print(f"    intermediate_size: {intermediate_size}")
    print(f"    num_layers: {num_layers}")
    print(f"    num_attention_heads: {num_attention_heads}")
    print(f"    num_kv_heads: {num_kv_heads}")
    print(f"    head_dim: {head_dim}")
    print(f"    vocab_size: {vocab_size}")

    # Check if attention has bias (Qwen2 does)
    has_attn_bias = "model.layers.0.self_attn.q_proj.bias" in qwen2_weights
    print(f"    attention_bias: {has_attn_bias}")

    # Save as safetensors
    safetensors_path = os.path.join(args.output_dir, "model.safetensors")
    print(f"\n  Saving {safetensors_path}...")
    save_file(qwen2_weights, safetensors_path)
    size_mb = os.path.getsize(safetensors_path) / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")

    # Save config.json
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": 32768,
        "num_attention_heads": num_attention_heads,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": num_kv_heads,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "use_cache": True,
        "attention_bias": has_attn_bias,
        "attention_dropout": 0.0,
        "torch_dtype": "float32",
        "transformers_version": "4.46.0",
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved {config_path}")

    # Copy tokenizer from CosyVoice3 (Qwen2 uses the same tokenizer)
    tokenizer_src = os.path.dirname(args.llm_pt)
    tokenizer_files = [
        "tokenizer_config.json", "vocab.json", "merges.txt",
        "tokenizer.json", "special_tokens_map.json",
    ]
    # Also try from the CosyVoice3 scripts directory
    alt_tokenizer_dir = os.path.join(
        os.path.dirname(os.path.dirname(args.llm_pt)),
        "CosyVoice-BlankEN"
    )

    for fname in tokenizer_files:
        for src_dir in [tokenizer_src, alt_tokenizer_dir]:
            src = os.path.join(src_dir, fname)
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, os.path.join(args.output_dir, fname))
                print(f"  Copied {fname} from {src_dir}")
                break

    # ================================================================
    # Part 2: Extract CPU-side embeddings as numpy
    # ================================================================
    print(f"\n=== Part 2: CPU embeddings → {args.embeddings_dir} ===")

    # embed_tokens — text token embedding (from Qwen2)
    embed_tokens = qwen2_weights["model.embed_tokens.weight"]
    np.save(
        os.path.join(args.embeddings_dir, "embed_tokens.npy"),
        embed_tokens.numpy()
    )
    print(f"  embed_tokens: {list(embed_tokens.shape)}")

    # speech_embedding — speech token embedding
    if "speech_embedding" in cpu_weights:
        speech_emb = cpu_weights["speech_embedding"]
        np.save(
            os.path.join(args.embeddings_dir, "speech_embedding.npy"),
            speech_emb.numpy()
        )
        print(f"  speech_embedding: {list(speech_emb.shape)}")
    else:
        print("  WARNING: speech_embedding not found in checkpoint!")

    # llm_decoder — linear projection for speech token sampling
    if "llm_decoder_weight" in cpu_weights:
        dec_w = cpu_weights["llm_decoder_weight"]
        np.save(
            os.path.join(args.embeddings_dir, "llm_decoder_weight.npy"),
            dec_w.numpy()
        )
        print(f"  llm_decoder_weight: {list(dec_w.shape)}")
    else:
        print("  WARNING: llm_decoder not found in checkpoint!")

    # llm_embedding — SOS and task_id vectors
    if "llm_embedding" in cpu_weights:
        llm_emb = cpu_weights["llm_embedding"]
        np.save(
            os.path.join(args.embeddings_dir, "llm_embedding.npy"),
            llm_emb.numpy()
        )
        print(f"  llm_embedding: {list(llm_emb.shape)}")
    else:
        print("  WARNING: llm_embedding not found in checkpoint!")
        print("  Will need to extract from full model or use defaults")

    # lm_head — Qwen2's lm_head (not used in CosyVoice3 inference, but saved for reference)
    lm_head = qwen2_weights["lm_head.weight"]
    np.save(
        os.path.join(args.embeddings_dir, "lm_head.npy"),
        lm_head.numpy()
    )
    print(f"  lm_head: {list(lm_head.shape)} (Qwen2 original, not used in TTS)")

    # ================================================================
    # Summary
    # ================================================================
    speech_token_size = cpu_weights.get("speech_embedding", torch.empty(0)).shape[0]

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Qwen2 model: {args.output_dir}/")
    print(f"  - model.safetensors ({size_mb:.1f} MB)")
    print(f"  - config.json (Qwen2ForCausalLM, {num_layers} layers, {hidden_size}D)")
    print(f"CPU embeddings: {args.embeddings_dir}/")
    print(f"  - embed_tokens.npy [{vocab_size}, {hidden_size}]")
    print(f"  - speech_embedding.npy [{speech_token_size}, {hidden_size}]")
    print(f"  - llm_decoder_weight.npy [speech_token_size+3, {hidden_size}]")
    print(f"\nNext step: convert to RKLLM")
    print(f"  python convert_llm_rkllm.py --model_dir {args.output_dir}")


if __name__ == "__main__":
    main()
