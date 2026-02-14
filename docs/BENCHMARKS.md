# Benchmarks

Performance measurements on CM3588 (RK3588 SoC, 4x Cortex-A76 + 4x Cortex-A55, 6 TOPS NPU).

RKNN driver: 0.9.8, rknn-toolkit-lite2: 2.3.2, onnxruntime: 1.20.1

## Configuration

- **LLM**: ONNX FP32 on CPU (4 threads)
- **Flow DiT**: RKNN FP16 on NPU (opt_level=3)
- **HiFT**: ONNX FP32 on CPU
- **ODE steps**: 10 (default)

## End-to-End TTS

| Metric | Short text | Long text |
|---|---|---|
| Input text | "Привет, как дела?" | "Здравствуйте, меня зовут Алиса. Сегодня прекрасная погода для прогулки в парке. Давайте обсудим наши планы на выходные." |
| Text tokens | 7 | 42 |
| Speech tokens | 53 | 156 |
| Mel frames | 106 | 312 |
| Audio duration | 2.1s | 6.2s |

### Timing Breakdown

| Phase | Short text | Long text |
|---|---|---|
| Model loading | ~25s (cold start) | ~17s (warm) |
| **LLM (ONNX CPU)** | **6.9s (7.7 tok/s)** | **21.7s (7.2 tok/s)** |
| **Flow (RKNN NPU)** | **5.9s (seq200)** | **60.1s (seq1000)** |
| **HiFT (CPU)** | **1.1s** | **3.0s** |
| **Total (excl. load)** | **13.9s** | **84.8s** |
| Real-time factor | 6.6x | 13.6x |

### Flow NPU Details

| | Short | Long |
|---|---|---|
| mel_len | 106 | 312 |
| RKNN model | seq200 | seq1000 |
| Padding | 47% (94 of 200) | 69% (688 of 1000) |
| Time per ODE step | 0.59s | 6.01s |

The main bottleneck for long texts is **padding overhead** — the seq1000 model processes 69% zero-padded data. Adding intermediate seq sizes (seq300, seq500) would improve this significantly.

## LLM Backend Comparison

| Backend | Speed | Quality | Notes |
|---|---|---|---|
| **ONNX FP32 CPU** | 7.2-7.7 tok/s | Best | Recommended, stable |
| RKLLM W8A8 NPU | ~16 tok/s | Lower | May truncate or repeat |
| RKLLM W8A8_g128 NPU | ~16 tok/s | Lower | Grouped quantization, slightly better |

ONNX FP32 is ~2x slower but produces significantly more reliable speech tokens.

## Memory Usage

| Component | RAM |
|---|---|
| ONNX LLM session | ~1.5 GB |
| Embeddings (numpy) | ~1.1 GB |
| RKNN Flow models (2x) | ~1.4 GB (NPU memory) |
| HiFT ONNX | ~0.1 GB |
| Total | ~4 GB RAM + ~1.4 GB NPU |

## Comparison with AXERA AX650N

| | RK3588 RKNN (ours) | AXERA AX650N |
|---|---|---|
| LLM speed | 7.7 tok/s (CPU FP32) | 5.7 tok/s (NPU w8a16) |
| Flow speed | 5.9s (seq200, 10 steps) | ~5s (10 steps) |
| Total (short text) | 13.9s | ~14s |
| Audio quality | Clean (no artifacts) | LP filter needed (4-6 kHz) |
| Post-processing | None needed | Butterworth LP + fade-out |
| External hardware | None | M.2 NPU card required |
| Power consumption | ~8W (SoC) | ~5W (NPU) + SoC |
