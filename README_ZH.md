# CosyVoice3 俄语语音合成 — RK3588 RKNN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RK3588%20RKNN-blue)]()
[![Language](https://img.shields.io/badge/Language-Russian%20%7C%20English%20%7C%20Chinese-green)]()

**[English](README.md)** | **[Русский](README_RU.md)**

在 RK3588 内置 NPU 上运行 CosyVoice3 语音合成，支持**俄语语音克隆**（无需外部加速器）。

## 特点

- **纯 Python 流水线** — 单一脚本，无需 C/C++ 二进制文件，推理时无需 PyTorch
- **CPU/NPU 混合架构** — LLM 在 CPU 上运行（ONNX FP32，最佳质量），Flow DiT 在 RKNN NPU 上运行
- **俄语语音克隆** — 零样本语音克隆，自然韵律
- **无需后处理** — HiFT 在 CPU ONNX 上运行，无量化伪影
- **支持 RKLLM** — 可选 LLM 的 W8A8 NPU 加速（更快，质量略低）
- **自动模型选择** — 根据序列长度自动选择最佳 RKNN Flow 模型
- **完整转换流程** — 包含将 CosyVoice3 模型转换为 ONNX/RKNN 的脚本

## 架构

```
文本 → [分词器] → [LLM ONNX CPU] → [Flow DiT RKNN NPU] → [HiFT ONNX CPU] → WAV
       Qwen2 tok   FP32, 7.5 tok/s   FP16, ODE solver     声码器, 干净
```

**为什么采用 CPU/NPU 混合架构？**
- CPU FP32 上的 LLM 产生的语音 token 质量明显优于量化 NPU
- Flow DiT 受益于 NPU 加速（大张量运算的 10 步 ODE）
- CPU 上的 HiFT 避免了 NPU 量化引入的高频伪影

## 硬件

测试平台：
- **[FriendlyElec CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588)** NAS 套件 — 计算模块（RK3588 SoC，32 GB RAM）+ NAS 载板，4x M.2 NVMe，2.5 GbE

使用 RK3588 集成 6 TOPS NPU — 无需外部加速器。

## 快速开始

### 前置条件

- **硬件**：[FriendlyElec CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588) NAS 套件（RK3588，8 GB+ RAM）
- **系统**：Linux，RKNN 驱动 0.9.8+
- **Python**：3.8+，安装 `numpy`、`scipy`、`onnxruntime`、`transformers`
- **RKNN**：`rknn-toolkit-lite2` 2.3+
- **模型**：从 [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) 转换

### 1. 在设备上安装依赖

```bash
pip3 install numpy scipy onnxruntime transformers rknn-toolkit-lite2
```

### 2. 转换模型（在 x86 主机上）

详细说明请参阅 [docs/MODEL_CONVERSION.md](docs/MODEL_CONVERSION.md)。

```bash
# 从 CosyVoice3 提取 Qwen2 LLM
python3 scripts/extract_llm_as_qwen2.py

# 导出 LLM 为带 KV-cache 的 ONNX
python3 scripts/export_llm_onnx.py

# 导出 Flow 和 HiFT 组件
python3 scripts/export_flow_frontend.py
python3 scripts/export_hift_components.py

# 将 Flow estimator 转换为 RKNN
python3 scripts/convert_flow_rknn.py --seq_len 200
python3 scripts/convert_flow_rknn.py --seq_len 1000
```

### 3. 复制到设备

```bash
scp cosyvoice3_rknn_pipeline.py root@<设备IP>:/root/cosyvoice3-rknn/
scp -r cosyvoice3-llm-onnx cosyvoice3_embeddings cosyvoice3-flow-components \
       cosyvoice3-hift-components cosyvoice3-rknn-models cosyvoice3_qwen2_for_rkllm \
       root@<设备IP>:/root/cosyvoice3-rknn/
```

### 4. 运行

```bash
python3 cosyvoice3_rknn_pipeline.py \
    --base_dir /root/cosyvoice3-rknn \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --flow_rknn auto \
    --text "你好，今天天气真好。" \
    --output output.wav
```

## 使用方法

```bash
# 基本语音合成
python3 cosyvoice3_rknn_pipeline.py --base_dir . \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --flow_rknn auto \
    --text "您的文本" --output output.wav

# 俄语语音克隆
python3 cosyvoice3_rknn_pipeline.py --base_dir . \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --flow_rknn auto \
    --prompt_dir prompt_russian_v2 \
    --text "Привет, как дела?" --output output.wav

# 纯 CPU 模式（无 RKNN，较慢但无需 NPU）
python3 cosyvoice3_rknn_pipeline.py --base_dir . \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --text "您的文本" --output output.wav
```

### 主要参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--text` | 必填 | 待合成文本 |
| `--output` | `output.wav` | 输出 WAV 文件 |
| `--llm_model` | 自动检测 | LLM 模型路径（.onnx 或 .rkllm） |
| `--flow_rknn` | 无 | RKNN Flow 模型：`auto`、`auto:vc` 或逗号分隔路径 |
| `--prompt_dir` | 无 | 语音克隆提示目录 |
| `--max_tokens` | 500 | 最大语音 token 数 |

## 性能基准

CM3588（RK3588，4x A76 + 4x A55，6 TOPS NPU），RKNN 驱动 0.9.8：

| 指标 | 短文本 | 长文本 |
|---|---|---|
| LLM token 数 | 53 | 156 |
| LLM 速度 | 7.7 tok/s | 7.2 tok/s |
| Flow（RKNN NPU） | 5.9s（seq200） | 60.1s（seq1000） |
| HiFT 声码器 | 1.1s | 3.0s |
| **总计** | **13.9s** | **84.8s** |
| 音频时长 | 2.1s | 6.2s |
| 实时因子 | 6.6x | 13.6x |

详情请参阅 [docs/BENCHMARKS.md](docs/BENCHMARKS.md)。

## 已知问题

| 问题 | 原因 | 状态 |
|---|---|---|
| 长文本 Flow 较慢 | 填充开销（seq1000 中 69%） | 添加中间 seq 尺寸 |
| RKLLM LLM 截断/重复 | W8A8 量化 | 使用 ONNX FP32（默认） |
| 内存占用较高（~4 GB） | FP32 LLM + RKNN Flow 模型 | 该架构的预期行为 |

详情请参阅 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)。

## 相关项目

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — FunAudioLLM/阿里巴巴的 CosyVoice3 原始模型
- [cosyvoice3-axera-russian](https://github.com/MasterVVK/cosyvoice3-axera-russian) — AXERA AX650N NPU 上的 CosyVoice3 俄语 TTS
- [AXERA-TECH/CosyVoice3.Axera](https://github.com/AXERA-TECH/CosyVoice3.Axera) — AXERA-TECH NPU 运行时

## 致谢

- [FunAudioLLM / 阿里巴巴](https://github.com/FunAudioLLM/CosyVoice) — CosyVoice3 模型
- [Rockchip](https://github.com/rockchip-linux/rknn-toolkit2) — RKNN 工具包和 NPU 驱动
- [FriendlyElec](https://wiki.friendlyelec.com/wiki/index.php/CM3588) — CM3588 NAS 套件
- [AXERA-TECH](https://github.com/AXERA-TECH) — AX650N CosyVoice3 移植的启发

## 许可证

MIT 许可证。CosyVoice3 模型由阿里巴巴/FunAudioLLM 以 Apache-2.0 许可证发布。
