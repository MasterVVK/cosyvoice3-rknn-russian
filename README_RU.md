# CosyVoice3 — Русский TTS на RK3588 RKNN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RK3588%20RKNN-blue)]()
[![Language](https://img.shields.io/badge/Language-Russian%20%7C%20English%20%7C%20Chinese-green)]()

**[English](README.md)** | **[中文](README_ZH.md)**

Синтез речи CosyVoice3 с **клонированием русского голоса** на встроенном NPU RK3588 (без внешнего ускорителя).



## Возможности

- **Чистый Python pipeline** — один скрипт, без C/C++ бинарников, без PyTorch
- **Гибрид CPU/NPU** — LLM на CPU (ONNX FP32, лучшее качество), Flow DiT на RKNN NPU
- **Клонирование русского голоса** — zero-shot с естественной просодией
- **Без пост-обработки** — HiFT на CPU ONNX, нет артефактов квантизации
- **Поддержка RKLLM** — опциональное ускорение LLM на NPU (W8A8, быстрее, ниже качество)
- **Авто-выбор моделей** — автоматически подбирает RKNN Flow модель по длине
- **Полный набор конвертации** — скрипты для конвертации CosyVoice3 → ONNX/RKNN

## Архитектура

```
Текст → [Tokenizer] → [LLM ONNX CPU] → [Flow DiT RKNN NPU] → [HiFT ONNX CPU] → WAV
         Qwen2 tok      FP32, 7.5 ток/с    FP16, ODE solver      вокодер, чисто
```

**Почему гибрид CPU/NPU?**
- LLM на CPU FP32 даёт значительно лучшее качество speech tokens чем квантизированный NPU
- Flow DiT выигрывает от ускорения NPU (10 шагов ODE с большими тензорами)
- HiFT на CPU избегает высокочастотных артефактов квантизации NPU

## Железо

Протестировано на:
- **[FriendlyElec CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588)** NAS Kit (RK3588 SoC, встроенный NPU 6 TOPS)

Внешняя NPU карта не нужна — используется встроенный NPU RK3588.

> Подойдёт любая плата на RK3588: Orange Pi 5, Rock 5B, Radxa ROCK 5A и т.д.

## Быстрый старт

### Требования

- **Железо**: Любая плата на RK3588 (CM3588, Orange Pi 5, Rock 5B и т.д.)
- **ОС**: Linux с RKNN driver 0.9.8+
- **Python**: 3.8+ с `numpy`, `scipy`, `onnxruntime`, `transformers`
- **RKNN**: `rknn-toolkit-lite2` 2.3+ (для инференса на NPU)
- **Модели**: Конвертированные из [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)

### 1. Установить зависимости на устройстве

```bash
pip3 install numpy scipy onnxruntime transformers rknn-toolkit-lite2
```

### 2. Конвертировать модели (на x86 хосте)

Подробные инструкции: [docs/MODEL_CONVERSION.md](docs/MODEL_CONVERSION.md)

```bash
# Извлечь Qwen2 LLM из CosyVoice3
python3 scripts/extract_llm_as_qwen2.py

# Экспортировать LLM в ONNX с KV-cache
python3 scripts/export_llm_onnx.py

# Экспортировать Flow и HiFT компоненты
python3 scripts/export_flow_frontend.py
python3 scripts/export_hift_components.py

# Конвертировать Flow estimator в RKNN
python3 scripts/convert_flow_rknn.py --seq_len 200
python3 scripts/convert_flow_rknn.py --seq_len 1000
```

### 3. Скопировать на устройство

```bash
scp cosyvoice3_rknn_pipeline.py root@<IP>:/root/cosyvoice3-rknn/
scp -r cosyvoice3-llm-onnx cosyvoice3_embeddings cosyvoice3-flow-components \
       cosyvoice3-hift-components cosyvoice3-rknn-models cosyvoice3_qwen2_for_rkllm \
       root@<IP>:/root/cosyvoice3-rknn/
```

### 4. Запуск

```bash
python3 cosyvoice3_rknn_pipeline.py \
    --base_dir /root/cosyvoice3-rknn \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --flow_rknn auto \
    --text "Привет, как дела? Сегодня хорошая погода." \
    --output output.wav
```

## Использование

```bash
# Базовый TTS
python3 cosyvoice3_rknn_pipeline.py --base_dir . \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --flow_rknn auto \
    --text "Ваш текст здесь" --output output.wav

# С клонированием русского голоса
python3 cosyvoice3_rknn_pipeline.py --base_dir . \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --flow_rknn auto \
    --prompt_dir prompt_russian_v2 \
    --text "Ваш текст здесь" --output output.wav

# Без NPU (медленнее, но работает без RKNN)
python3 cosyvoice3_rknn_pipeline.py --base_dir . \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --text "Ваш текст здесь" --output output.wav
```

### Параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--text` | обязательный | Текст для синтеза |
| `--output` | `output.wav` | Выходной файл |
| `--llm_model` | авто | Путь к LLM модели (.onnx или .rkllm) |
| `--flow_rknn` | нет | RKNN Flow модели: `auto`, `auto:vc` или пути через запятую |
| `--prompt_dir` | нет | Директория voice cloning промпта |
| `--max_tokens` | 500 | Максимум speech tokens |

## Производительность

CM3588 (RK3588, 4x A76 + 4x A55, NPU 6 TOPS), RKNN driver 0.9.8:

| Метрика | Короткий текст | Длинный текст |
|---|---|---|
| LLM токены | 53 | 156 |
| LLM скорость | 7.7 ток/с | 7.2 ток/с |
| Flow (RKNN NPU) | 5.9с (seq200) | 60.1с (seq1000) |
| HiFT вокодер | 1.1с | 3.0с |
| **Итого** | **13.9с** | **84.8с** |
| Длительность аудио | 2.1с | 6.2с |
| Real-time factor | 6.6x | 13.6x |

Подробнее: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)

## Клонирование русского голоса

Для использования собственного голоса нужно создать prompt — набор embeddings из reference аудио. Готовый русский промпт включён в `examples/prompt_russian_v2/`.

Подробнее: [docs/RUSSIAN_VOICE.md](docs/RUSSIAN_VOICE.md)

## Известные проблемы

| Проблема | Причина | Статус |
|---|---|---|
| Flow медленный на длинных текстах | Padding overhead (69% на seq1000) | Добавить промежуточные seq |
| RKLLM LLM обрезает/повторяет | Квантизация W8A8 | Используйте ONNX FP32 (по умолчанию) |
| Большое потребление RAM (~4 ГБ) | FP32 LLM + RKNN Flow | Ожидаемо для данной архитектуры |

Подробнее: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## Связанные проекты

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — оригинальная модель CosyVoice3 (FunAudioLLM/Alibaba)
- [cosyvoice3-axera-russian](https://github.com/MasterVVK/cosyvoice3-axera-russian) — CosyVoice3 русский TTS на AXERA AX650N NPU
- [AXERA-TECH/CosyVoice3.Axera](https://github.com/AXERA-TECH/CosyVoice3.Axera) — AXERA-TECH NPU runtime

## Благодарности

- [FunAudioLLM / Alibaba](https://github.com/FunAudioLLM/CosyVoice) — модель CosyVoice3
- [Rockchip](https://github.com/rockchip-linux/rknn-toolkit2) — RKNN toolkit и драйверы NPU
- [FriendlyElec](https://wiki.friendlyelec.com/wiki/index.php/CM3588) — CM3588 NAS Kit
- [AXERA-TECH](https://github.com/AXERA-TECH) — вдохновение от их порта CosyVoice3 на AX650N

## Лицензия

MIT. Модель CosyVoice3 под Apache-2.0 (Alibaba/FunAudioLLM).
