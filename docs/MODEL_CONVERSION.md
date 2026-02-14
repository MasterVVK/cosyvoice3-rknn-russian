# Model Conversion Guide

How to convert CosyVoice3 models from PyTorch to ONNX/RKNN for the RK3588 pipeline.

All conversion scripts run on an **x86 host** with GPU (optional) and Python 3.10+.

## Prerequisites (x86 host)

```bash
pip install torch transformers onnx onnxruntime onnxsim
pip install rknn-toolkit2  # for RKNN conversion
```

## Source Model

Download [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512):

```bash
pip install huggingface_hub
huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir ./CosyVoice3
```

You also need the CosyVoice repository for model loading:
```bash
git clone https://github.com/FunAudioLLM/CosyVoice.git
```

## Step 1: Extract Qwen2 LLM

Extracts the Qwen2 transformer from CosyVoice3 as a standalone HuggingFace model + numpy embeddings.

```bash
python3 scripts/extract_llm_as_qwen2.py \
    --cosyvoice_dir ./CosyVoice3 \
    --output_model_dir ./cosyvoice3_qwen2_for_rkllm \
    --output_embeddings_dir ./cosyvoice3_embeddings
```

**Output:**
- `cosyvoice3_qwen2_for_rkllm/` — Qwen2 model (HuggingFace format) + tokenizer
- `cosyvoice3_embeddings/` — embed_tokens.npy, speech_embedding.npy, llm_decoder_weight.npy, lm_head.npy

## Step 2: Export LLM to ONNX

Exports Qwen2 transformer to ONNX with explicit KV-cache inputs/outputs.

Uses a manual forward pass (`Qwen2ManualForward`) to avoid transformers' causal mask machinery that doesn't trace cleanly to ONNX. Causal masking is implemented via a cumsum trick.

```bash
python3 scripts/export_llm_onnx.py \
    --model_dir ./cosyvoice3_qwen2_for_rkllm \
    --output_dir ./cosyvoice3-llm-onnx
```

**Output:** `cosyvoice3-llm-onnx/qwen2_transformer.onnx` (~1.4 GB, FP32)

**Architecture:** 24 layers, hidden=896, 14 attention heads, 2 KV heads, head_dim=64

**Verification:** The script automatically verifies:
1. Prefill (seq_len=4, past_len=8)
2. Decode (seq_len=1, using KV cache from prefill)
3. Second decode step
4. Fresh prefill (past_len=0)

## Step 3: Export Flow Frontend

Extracts Flow decoder frontend components (input_embedding, speaker affine, pre-lookahead) as numpy arrays.

```bash
python3 scripts/export_flow_frontend.py \
    --cosyvoice_dir ./CosyVoice3 \
    --output_dir ./cosyvoice3-flow-components
```

**Output:** `cosyvoice3-flow-components/` — flow_config.json, flow_input_embedding.npy, flow_spk_affine_weight.npy, flow_spk_affine_bias.npy, flow_pre_lookahead_weights.npz

## Step 4: Export HiFT Vocoder

Exports HiFT vocoder components: f0_predictor and decode CNN as ONNX, source weights as numpy.

```bash
python3 scripts/export_hift_components.py \
    --cosyvoice_dir ./CosyVoice3 \
    --output_dir ./cosyvoice3-hift-components
```

**Output:** `cosyvoice3-hift-components/` — f0_predictor.onnx, hift_decode_dynamic.onnx, hift_config.json, source_weights.npz

## Step 5: Export Flow Estimator ONNX

The Flow DiT estimator needs to be exported as a fixed-sequence-length ONNX model first, then converted to RKNN.

The ONNX model for the Flow estimator should already exist as `flow.decoder.estimator.fp32.onnx` if you exported from the CosyVoice3 repo. If not, you can export it using the CosyVoice3 tools.

Then simplify it:
```bash
pip install onnxsim
python3 -m onnxsim flow.decoder.estimator.fp32.onnx flow_estimator_sim.onnx
```

## Step 6: Convert Flow to RKNN

Converts Flow estimator ONNX to RKNN format with fixed sequence length.

```bash
# Short sequences (most common)
python3 scripts/convert_flow_rknn.py \
    --onnx_path ./cosyvoice3-onnx/flow_estimator_sim.onnx \
    --seq_len 200 \
    --opt_level 3 \
    --output_dir ./cosyvoice3-rknn-models

# Long sequences
python3 scripts/convert_flow_rknn.py \
    --onnx_path ./cosyvoice3-onnx/flow_estimator_sim.onnx \
    --seq_len 1000 \
    --opt_level 3 \
    --output_dir ./cosyvoice3-rknn-models
```

**Output:** `cosyvoice3-rknn-models/flow_estimator_seq200_opt3.rknn`, etc.

**Recommended seq_len values:**
- `200` — for short texts (up to ~8 words), ~666 MB
- `500` — for medium texts, ~707 MB
- `1000` — for long texts, ~730 MB

The pipeline automatically selects the smallest model that fits the sequence length.

## Step 7 (Optional): Convert LLM to RKLLM

For NPU-accelerated LLM inference (faster but lower quality):

```bash
python3 scripts/convert_llm_rkllm.py \
    --model_dir ./cosyvoice3_qwen2_for_rkllm \
    --quantized_dtype w8a8 \
    --output cosyvoice3_llm_rk3588.rkllm
```

**Note:** ONNX FP32 is recommended for best quality. RKLLM W8A8 is faster (~16 tok/s vs 7.5 tok/s) but may produce truncated or repetitive speech tokens.

## Summary

| Step | Script | Output | Size |
|---|---|---|---|
| 1. Extract LLM | extract_llm_as_qwen2.py | cosyvoice3_qwen2_for_rkllm/ + cosyvoice3_embeddings/ | ~1.1 GB |
| 2. LLM ONNX | export_llm_onnx.py | cosyvoice3-llm-onnx/qwen2_transformer.onnx | 1.4 GB |
| 3. Flow frontend | export_flow_frontend.py | cosyvoice3-flow-components/ | 5 MB |
| 4. HiFT | export_hift_components.py | cosyvoice3-hift-components/ | 83 MB |
| 5. Flow ONNX | (manual/CosyVoice3 tools) | flow_estimator_sim.onnx | ~500 MB |
| 6. Flow RKNN | convert_flow_rknn.py | cosyvoice3-rknn-models/*.rknn | 666-730 MB each |
| 7. LLM RKLLM (opt) | convert_llm_rkllm.py | cosyvoice3_llm_rk3588.rkllm | ~800 MB |

**Total on device:** ~5 GB (LLM ONNX + embeddings + Flow RKNN + HiFT + tokenizer)
