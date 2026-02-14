# CosyVoice3 Russian TTS on RK3588 RKNN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RK3588%20RKNN-blue)]()
[![Language](https://img.shields.io/badge/Language-Russian%20%7C%20English%20%7C%20Chinese-green)]()

**[Русский](README_RU.md)** | **[中文](README_ZH.md)**

Run CosyVoice3 text-to-speech with **Russian voice cloning** on RK3588 built-in NPU (no external accelerator required).



## Features

- **Pure Python pipeline** — single script, no C/C++ binaries, no PyTorch at inference
- **Hybrid CPU/NPU** — LLM on CPU (ONNX FP32, best quality), Flow DiT on RKNN NPU
- **Russian voice cloning** — zero-shot voice cloning with natural prosody
- **No post-processing needed** — HiFT runs on CPU ONNX, no quantization artifacts
- **RKLLM support** — optional W8A8 NPU acceleration for LLM (faster, lower quality)
- **Auto model selection** — automatically picks the best RKNN Flow model for sequence length
- **Full conversion pipeline** — scripts to convert CosyVoice3 models to ONNX/RKNN

## Architecture

```
Text -> [Tokenizer] -> [LLM ONNX CPU] -> [Flow DiT RKNN NPU] -> [HiFT ONNX CPU] -> WAV
         Qwen2 tok      FP32, 7.5 tok/s    FP16, ODE solver      vocoder, clean
```

**Why hybrid CPU/NPU?**
- LLM on CPU FP32 gives significantly better speech token quality than quantized NPU
- Flow DiT benefits from NPU acceleration (10 ODE steps with large tensor ops)
- HiFT on CPU avoids high-frequency artifacts that NPU quantization introduces

## Hardware

Tested on:
- **[FriendlyElec CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588)** NAS Kit — compute module (RK3588 SoC, 8 GB RAM) + NAS carrier board with 4x M.2 NVMe slots, 2.5 GbE

Uses the RK3588's integrated 6 TOPS NPU — no external accelerator card needed.

## Quick Start

### Prerequisites

- **Hardware**: [FriendlyElec CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588) NAS Kit (RK3588, 8 GB+ RAM)
- **OS**: Linux with RKNN driver 0.9.8+
- **Python**: 3.8+ with `numpy`, `scipy`, `onnxruntime`, `transformers`
- **RKNN**: `rknn-toolkit-lite2` 2.3+ (for NPU inference)
- **Models**: Converted from [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)

### 1. Install dependencies on device

```bash
pip3 install numpy scipy onnxruntime transformers rknn-toolkit-lite2
```

### 2. Convert models (on x86 host)

See [docs/MODEL_CONVERSION.md](docs/MODEL_CONVERSION.md) for full instructions.

```bash
# Extract Qwen2 LLM from CosyVoice3
python3 scripts/extract_llm_as_qwen2.py

# Export LLM to ONNX with KV-cache
python3 scripts/export_llm_onnx.py

# Export Flow and HiFT components
python3 scripts/export_flow_frontend.py
python3 scripts/export_hift_components.py

# Convert Flow estimator to RKNN
python3 scripts/convert_flow_rknn.py --seq_len 200
python3 scripts/convert_flow_rknn.py --seq_len 1000
```

### 3. Copy to device

```bash
scp cosyvoice3_rknn_pipeline.py root@<DEVICE_IP>:/root/cosyvoice3-rknn/
scp -r cosyvoice3-llm-onnx cosyvoice3_embeddings cosyvoice3-flow-components \
       cosyvoice3-hift-components cosyvoice3-rknn-models cosyvoice3_qwen2_for_rkllm \
       root@<DEVICE_IP>:/root/cosyvoice3-rknn/
```

### 4. Run

```bash
python3 cosyvoice3_rknn_pipeline.py \
    --base_dir /root/cosyvoice3-rknn \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --flow_rknn auto \
    --text "Привет, как дела? Сегодня хорошая погода." \
    --output output.wav
```

## Usage

```bash
# Basic Russian TTS
python3 cosyvoice3_rknn_pipeline.py --base_dir . \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --flow_rknn auto \
    --text "Ваш текст здесь" --output output.wav

# With Russian voice cloning
python3 cosyvoice3_rknn_pipeline.py --base_dir . \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --flow_rknn auto \
    --prompt_dir prompt_russian_v2 \
    --text "Ваш текст здесь" --output output.wav

# CPU-only mode (no RKNN, slower but works without NPU)
python3 cosyvoice3_rknn_pipeline.py --base_dir . \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --text "Ваш текст здесь" --output output.wav
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--text` | required | Text to synthesize |
| `--output` | `output.wav` | Output WAV file |
| `--llm_model` | auto-detect | Path to LLM model (.onnx or .rkllm) |
| `--flow_rknn` | none | RKNN Flow models: `auto`, `auto:vc`, or comma-separated paths |
| `--prompt_dir` | none | Voice cloning prompt directory |
| `--max_tokens` | 500 | Maximum speech tokens to generate |

## Benchmarks

Tested on CM3588 (RK3588, 4x A76 + 4x A55, 6 TOPS NPU), RKNN driver 0.9.8:

| Metric | Short text | Long text |
|---|---|---|
| LLM tokens | 53 | 156 |
| LLM speed | 7.7 tok/s | 7.2 tok/s |
| Flow (RKNN NPU) | 5.9s (seq200) | 60.1s (seq1000) |
| HiFT vocoder | 1.1s | 3.0s |
| **Total** | **13.9s** | **84.8s** |
| Audio duration | 2.1s | 6.2s |
| Real-time factor | 6.6x | 13.6x |

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for details.

## Model Components

| Component | Format | Size | Runtime |
|---|---|---|---|
| LLM (Qwen2-0.5B) | ONNX FP32 | 1.4 GB | CPU (onnxruntime) |
| Embeddings (embed_tokens, speech_embedding, lm_head) | numpy | 1.1 GB | CPU (numpy) |
| Flow DiT estimator | RKNN FP16 | 700 MB each | NPU (rknn-toolkit-lite2) |
| Flow frontend (input_embedding, spk_affine, etc.) | numpy | 5 MB | CPU (numpy) |
| HiFT vocoder (f0_predictor + decode CNN) | ONNX FP32 | 83 MB | CPU (onnxruntime) |
| Tokenizer (Qwen2) | transformers | 5 MB | CPU |

## Known Issues

| Issue | Cause | Status |
|---|---|---|
| Flow slow on long texts (seq1000) | 69% zero-padding overhead | Add intermediate seq sizes (seq300, seq500) |
| RKLLM LLM can truncate/repeat | W8A8 quantization | Use ONNX FP32 LLM (default) |
| High memory usage (~4 GB) | FP32 LLM + RKNN Flow models | Expected for this architecture |

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more.

## Project Structure

```
cosyvoice3-rknn-russian/
├── cosyvoice3_rknn_pipeline.py    # Main TTS pipeline (runs on device)
├── scripts/                        # Model conversion (runs on x86 host)
│   ├── extract_llm_as_qwen2.py   # Extract Qwen2 from CosyVoice3
│   ├── export_llm_onnx.py        # LLM -> ONNX with KV-cache
│   ├── export_flow_frontend.py    # Flow frontend components
│   ├── export_hift_components.py  # HiFT vocoder components
│   ├── convert_flow_rknn.py       # Flow ONNX -> RKNN
│   └── convert_llm_rkllm.py      # LLM -> RKLLM (optional)
├── docs/                           # Documentation
│   ├── SETUP.md
│   ├── MODEL_CONVERSION.md
│   ├── BENCHMARKS.md
│   ├── RUSSIAN_VOICE.md
│   └── TROUBLESHOOTING.md
├── examples/
│   └── prompt_russian_v2/         # Russian voice prompt
├── README.md
├── README_RU.md
└── LICENSE
```

Models are **not included** (too large for git). See [docs/MODEL_CONVERSION.md](docs/MODEL_CONVERSION.md).

Original model: [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)

## Related Projects

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — original CosyVoice3 by FunAudioLLM/Alibaba
- [cosyvoice3-axera-russian](https://github.com/MasterVVK/cosyvoice3-axera-russian) — CosyVoice3 Russian TTS on AXERA AX650N NPU
- [AXERA-TECH/CosyVoice3.Axera](https://github.com/AXERA-TECH/CosyVoice3.Axera) — AXERA-TECH NPU runtime

## Credits

- [FunAudioLLM / Alibaba](https://github.com/FunAudioLLM/CosyVoice) — CosyVoice3 model
- [Rockchip](https://github.com/rockchip-linux/rknn-toolkit2) — RKNN toolkit and NPU drivers
- [FriendlyElec](https://wiki.friendlyelec.com/wiki/index.php/CM3588) — CM3588 NAS Kit
- [AXERA-TECH](https://github.com/AXERA-TECH) — inspiration from their AX650N CosyVoice3 port

## License

MIT License. See [LICENSE](LICENSE).

The CosyVoice3 model is licensed under Apache-2.0 by Alibaba/FunAudioLLM.
