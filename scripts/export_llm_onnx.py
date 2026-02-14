#!/usr/bin/env python3
"""
Export CosyVoice3 LLM (Qwen2 transformer) to ONNX with KV-cache.

Uses manual forward pass to avoid transformers' mask machinery
that doesn't trace cleanly with torch.onnx.export.

Architecture: Qwen2 (24 layers, hidden=896, 14 heads, 2 KV heads, head_dim=64)

Usage:
    source /home/user/NAS/functiongemma_convert/venv/bin/activate
    python export_llm_onnx.py
"""

import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = "./cosyvoice3_qwen2_for_rkllm"
OUTPUT_DIR = "./cosyvoice3-llm-onnx"


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to q and k."""
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2ManualForward(nn.Module):
    """Manual forward pass through Qwen2 transformer layers.

    Bypasses transformers' causal mask creation (which doesn't trace to ONNX)
    and computes causal attention using ONNX-friendly operations.
    """

    def __init__(self, qwen2_model, config):
        super().__init__()
        self.layers = qwen2_model.layers
        self.norm = qwen2_model.norm
        self.rotary_emb = qwen2_model.rotary_emb

        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, inputs_embeds, position_ids, *past_kvs):
        """
        Args:
            inputs_embeds: [1, seq_len, hidden_size]
            position_ids: [1, seq_len]
            past_kvs: NUM_LAYERS * 2 tensors, each [1, num_kv_heads, past_len, head_dim]
        Returns:
            hidden_states: [1, seq_len, hidden_size]
            present_kvs: NUM_LAYERS * 2 tensors
        """
        hidden = inputs_embeds

        # Compute RoPE (cos, sin) from position_ids
        cos, sin = self.rotary_emb(hidden, position_ids)

        present_kvs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            past_k = past_kvs[i * 2]      # [1, kv_heads, past_len, head_dim]
            past_v = past_kvs[i * 2 + 1]

            # === Self-attention ===
            residual = hidden
            hidden = layer.input_layernorm(hidden)

            # QKV projections
            q = layer.self_attn.q_proj(hidden)   # [1, seq, num_heads * head_dim]
            k = layer.self_attn.k_proj(hidden)   # [1, seq, kv_heads * head_dim]
            v = layer.self_attn.v_proj(hidden)

            bsz = q.shape[0]
            seq_len = q.shape[1]

            # Reshape to multi-head: [1, heads, seq, head_dim]
            q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # Apply RoPE
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Concat with past KV cache
            k = torch.cat([past_k, k], dim=2)   # [1, kv_heads, total_len, head_dim]
            v = torch.cat([past_v, v], dim=2)
            present_kvs.append(k)
            present_kvs.append(v)

            # Expand KV for GQA: [1, kv_heads, total_len, head_dim] -> [1, num_heads, total_len, head_dim]
            k_exp = k.unsqueeze(2).expand(bsz, self.num_kv_heads, self.num_kv_groups, k.shape[2], self.head_dim)
            k_exp = k_exp.reshape(bsz, self.num_heads, k.shape[2], self.head_dim)
            v_exp = v.unsqueeze(2).expand(bsz, self.num_kv_heads, self.num_kv_groups, v.shape[2], self.head_dim)
            v_exp = v_exp.reshape(bsz, self.num_heads, v.shape[2], self.head_dim)

            # Scaled dot-product attention
            attn_w = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale

            # Causal mask using position_ids and cumsum trick
            # key positions: 0, 1, ..., total_len-1 (computed via cumsum on dynamic shape)
            k_ones = torch.ones_like(k_exp[:, :1, :, :1])  # [1, 1, total_len, 1]
            k_pos = torch.cumsum(k_ones, dim=2) - 1.0       # [1, 1, total_len, 1]
            q_pos = position_ids.float().unsqueeze(1).unsqueeze(3)  # [1, 1, seq_len, 1]
            # mask: -inf where key_pos > query_pos (future positions)
            causal_mask = torch.where(
                k_pos.transpose(2, 3) > q_pos,  # [1, 1, seq_len, total_len]
                torch.tensor(-65504.0, dtype=attn_w.dtype),
                torch.tensor(0.0, dtype=attn_w.dtype),
            )
            attn_w = attn_w + causal_mask

            attn_w = F.softmax(attn_w, dim=-1)
            attn_out = torch.matmul(attn_w, v_exp)  # [1, heads, seq, head_dim]

            # Reshape back: [1, seq, hidden_size]
            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
            attn_out = layer.self_attn.o_proj(attn_out)

            hidden = residual + attn_out

            # === MLP ===
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

        hidden = self.norm(hidden)
        return (hidden, *present_kvs)


def main():
    parser = argparse.ArgumentParser(description="Export CosyVoice3 LLM to ONNX")
    parser.add_argument("--model_dir", default=MODEL_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--fp16", action="store_true", help="Export in FP16")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Exporting CosyVoice3 LLM (Qwen2) to ONNX ===")
    print(f"Model dir: {args.model_dir}")
    print(f"Output dir: {args.output_dir}")

    # Load model and read config
    print("\n1. Loading Qwen2 model...")
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(args.model_dir)
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    hidden_size = config.hidden_size
    print(f"   Config: {num_layers} layers, hidden={hidden_size}, "
          f"{config.num_attention_heads} heads, {num_kv_heads} KV heads, head_dim={head_dim}")

    dtype = torch.float16 if args.fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map="cpu",
    )
    model.eval()

    # Extract the transformer model (without lm_head)
    qwen2_model = model.model  # Qwen2Model
    wrapper = Qwen2ManualForward(qwen2_model, config)
    wrapper.eval()

    print(f"   Loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    # Prepare dummy inputs
    print("\n2. Preparing inputs...")
    batch = 1
    seq_len = 4  # dummy sequence length
    past_len = 8  # non-zero to trace KV-cache path

    inputs_embeds = torch.randn(batch, seq_len, hidden_size, dtype=dtype)
    position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long).unsqueeze(0)

    # Past KV cache
    past_kvs = []
    for _ in range(num_layers):
        past_kvs.append(torch.randn(batch, num_kv_heads, past_len, head_dim, dtype=dtype))  # key
        past_kvs.append(torch.randn(batch, num_kv_heads, past_len, head_dim, dtype=dtype))  # value

    # Input/output names
    input_names = ["inputs_embeds", "position_ids"]
    output_names = ["hidden_states"]
    dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "position_ids": {1: "seq_len"},
        "hidden_states": {1: "seq_len"},
    }

    for i in range(num_layers):
        input_names.append(f"past_key_{i}")
        input_names.append(f"past_value_{i}")
        output_names.append(f"present_key_{i}")
        output_names.append(f"present_value_{i}")
        dynamic_axes[f"past_key_{i}"] = {2: "past_len"}
        dynamic_axes[f"past_value_{i}"] = {2: "past_len"}
        dynamic_axes[f"present_key_{i}"] = {2: "total_len"}
        dynamic_axes[f"present_value_{i}"] = {2: "total_len"}

    all_inputs = (inputs_embeds, position_ids, *past_kvs)

    # First verify PyTorch forward works
    print("\n   Verifying PyTorch forward...")
    with torch.no_grad():
        pt_out = wrapper(*all_inputs)
    print(f"   PyTorch OK: hidden_states shape={pt_out[0].shape}")

    # Export
    output_path = os.path.join(args.output_dir, "qwen2_transformer.onnx")
    print(f"\n3. Exporting to ONNX...")
    print(f"   Inputs: inputs_embeds + position_ids + {num_layers * 2} KV cache tensors")
    print(f"   Outputs: hidden_states + {num_layers * 2} present KV tensors")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            all_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"   Exported: {output_path} ({size_mb:.1f} MB)")

    # Verify with onnxruntime
    print("\n4. Verifying with onnxruntime...")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

        # Test prefill (seq_len=4, past_len=8)
        feeds = {
            "inputs_embeds": inputs_embeds.numpy(),
            "position_ids": position_ids.numpy(),
        }
        for i in range(num_layers):
            feeds[f"past_key_{i}"] = past_kvs[i * 2].numpy()
            feeds[f"past_value_{i}"] = past_kvs[i * 2 + 1].numpy()

        outputs = sess.run(None, feeds)
        hidden = outputs[0]
        pt_hidden = pt_out[0].numpy()
        diff = np.abs(hidden - pt_hidden).mean()
        print(f"   Prefill OK: shape={hidden.shape}, ONNX vs PyTorch diff={diff:.6f}")

        # Test decode (1 token with KV cache from prefill)
        total_past = past_len + seq_len
        new_embeds = np.random.randn(1, 1, hidden_size).astype(
            np.float16 if args.fp16 else np.float32
        )
        decode_pos = np.array([[total_past]], dtype=np.int64)
        feeds2 = {
            "inputs_embeds": new_embeds,
            "position_ids": decode_pos,
        }
        for i in range(num_layers):
            feeds2[f"past_key_{i}"] = outputs[1 + i * 2]
            feeds2[f"past_value_{i}"] = outputs[2 + i * 2]

        outputs2 = sess.run(None, feeds2)
        hidden2 = outputs2[0]
        print(f"   Decode OK: shape={hidden2.shape}, KV total_len={outputs2[1].shape[2]}")

        # Test second decode step
        decode_pos2 = np.array([[total_past + 1]], dtype=np.int64)
        feeds3 = {
            "inputs_embeds": new_embeds,
            "position_ids": decode_pos2,
        }
        for i in range(num_layers):
            feeds3[f"past_key_{i}"] = outputs2[1 + i * 2]
            feeds3[f"past_value_{i}"] = outputs2[2 + i * 2]

        outputs3 = sess.run(None, feeds3)
        print(f"   Decode2 OK: shape={outputs3[0].shape}, KV total_len={outputs3[1].shape[2]}")

        # Test prefill from scratch (past_len=0)
        zero_past = {}
        zero_past["inputs_embeds"] = np.random.randn(1, 6, hidden_size).astype(
            np.float16 if args.fp16 else np.float32
        )
        zero_past["position_ids"] = np.arange(6, dtype=np.int64).reshape(1, -1)
        for i in range(num_layers):
            zero_past[f"past_key_{i}"] = np.zeros((1, num_kv_heads, 0, head_dim),
                                                    dtype=np.float16 if args.fp16 else np.float32)
            zero_past[f"past_value_{i}"] = np.zeros((1, num_kv_heads, 0, head_dim),
                                                     dtype=np.float16 if args.fp16 else np.float32)
        outputs4 = sess.run(None, zero_past)
        print(f"   Fresh prefill OK: shape={outputs4[0].shape}, KV len={outputs4[1].shape[2]}")

    except ImportError:
        print("   onnxruntime not available, skipping verification")

    print(f"\n=== Done ===")
    print(f"ONNX model: {output_path} ({size_mb:.1f} MB)")
    print(f"Architecture: {num_layers} layers, {hidden_size} hidden, {num_kv_heads} KV heads, head_dim={head_dim}")
    print(f"\nDeploy to CM3588:")
    print(f"  scp -r {args.output_dir} root@192.168.1.173:/root/cosyvoice3-rknn/")


if __name__ == "__main__":
    main()
