# Setup Guide

Step-by-step installation of CosyVoice3 RKNN pipeline on RK3588.

## Requirements

### Hardware

- **Board**: [FriendlyElec CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588) NAS Kit (RK3588, 8 GB+ RAM)
- **RAM**: 8 GB recommended (4 GB minimum)
- **Storage**: ~5 GB for models

### Software

- **OS**: Linux with RKNN NPU driver 0.9.8+
- **Python**: 3.8+
- **RKNN**: rknn-toolkit-lite2 2.3+ (for NPU inference on device)
- **RKNN-Toolkit2**: 2.3+ (for model conversion on x86 host)

## Installation on Device (CM3588)

### 1. Check NPU driver

```bash
dmesg | grep -i rknpu
# Should show: RKNPU driver loaded
```

### 2. Install Python dependencies

```bash
pip3 install numpy scipy onnxruntime transformers rknn-toolkit-lite2
```

If `--break-system-packages` is needed (Debian/Ubuntu 12+):
```bash
pip3 install numpy scipy onnxruntime transformers rknn-toolkit-lite2 --break-system-packages
```

### 3. Copy models and pipeline

After converting models (see [MODEL_CONVERSION.md](MODEL_CONVERSION.md)):

```bash
# From x86 host
scp cosyvoice3_rknn_pipeline.py root@<DEVICE_IP>:/root/cosyvoice3-rknn/

scp -r cosyvoice3-llm-onnx \
       cosyvoice3_embeddings \
       cosyvoice3-flow-components \
       cosyvoice3-hift-components \
       cosyvoice3-rknn-models \
       cosyvoice3_qwen2_for_rkllm \
       root@<DEVICE_IP>:/root/cosyvoice3-rknn/

# Copy voice prompt
scp -r examples/prompt_russian_v2 root@<DEVICE_IP>:/root/cosyvoice3-rknn/
```

### 4. Test run

```bash
ssh root@<DEVICE_IP>
cd /root/cosyvoice3-rknn

# Basic test (no voice cloning)
python3 cosyvoice3_rknn_pipeline.py \
    --base_dir . \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --flow_rknn auto \
    --text "Hello, this is a test." \
    --output test.wav

# Russian with voice cloning
python3 cosyvoice3_rknn_pipeline.py \
    --base_dir . \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --flow_rknn auto \
    --prompt_dir prompt_russian_v2 \
    --text "Привет, как дела?" \
    --output test_ru.wav
```

### 5. Expected output

```
============================================================
CosyVoice3 TTS on RK3588 NPU (no PyTorch)
============================================================

--- Loading components ---
  Tokenizer loaded: vocab_size=151643
  ONNX LLM: 24 layers, 2 KV heads, head_dim=64
  LLM (ONNX CPU): 6.6s
  Flow estimator RKNN: flow_estimator_seq200_opt3.rknn (666 MB, seq=200)
  Flow estimator RKNN: flow_estimator_seq1000_opt3.rknn (696 MB, seq=1000)
  HiFT vocoder loaded

--- Phase 1: LLM (ONNX CPU) ---
  Generated 53 tokens in 6.9s (7.7 tok/s)

--- Phase 2: Flow (RKNN NPU) ---
  Mel: (80, 106) in 5.9s

--- Phase 3: HiFT vocoder (CPU) ---
  Audio: 50880 samples (2.12s) in 1.1s
  Saved: test.wav
```

## Model Directory Structure

After setup, your device should have:

```
/root/cosyvoice3-rknn/
├── cosyvoice3_rknn_pipeline.py
├── cosyvoice3-llm-onnx/
│   └── qwen2_transformer.onnx          # 1.4 GB
├── cosyvoice3_embeddings/
│   ├── embed_tokens.npy                 # 545 MB
│   ├── speech_embedding.npy             # 24 MB
│   ├── llm_decoder_weight.npy           # 24 MB
│   └── lm_head.npy                      # 545 MB
├── cosyvoice3-flow-components/
│   ├── flow_config.json
│   ├── flow_input_embedding.npy
│   ├── flow_spk_affine_weight.npy
│   ├── flow_spk_affine_bias.npy
│   └── flow_pre_lookahead_weights.npz
├── cosyvoice3-hift-components/
│   ├── hift_config.json
│   ├── f0_predictor.onnx
│   ├── hift_decode_dynamic.onnx
│   └── source_weights.npz
├── cosyvoice3-rknn-models/
│   ├── flow_estimator_seq200_opt3.rknn  # 666 MB
│   └── flow_estimator_seq1000_opt3.rknn # 696 MB
├── cosyvoice3_qwen2_for_rkllm/         # tokenizer files
│   ├── config.json
│   ├── tokenizer.json
│   └── ...
└── prompt_russian_v2/                   # voice prompt
    ├── prompt_text_tokens.npy
    ├── prompt_speech_tokens.npy
    ├── prompt_speech_feat.npy
    └── speaker_embedding.npy
```
