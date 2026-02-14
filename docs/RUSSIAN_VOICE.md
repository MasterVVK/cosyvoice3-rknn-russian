# Russian Voice Cloning

CosyVoice3 supports zero-shot voice cloning — synthesizing speech in any voice from a short reference audio sample.

## Quick Start with Included Prompt

A ready-to-use Russian voice prompt is included in `examples/prompt_russian_v2/`:

```bash
python3 cosyvoice3_rknn_pipeline.py --base_dir . \
    --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \
    --flow_rknn auto \
    --prompt_dir prompt_russian_v2 \
    --text "Привет! Сегодня отличная погода для прогулки." \
    --output output.wav
```

## Creating Your Own Voice Prompt

To clone a specific voice, you need to create a prompt directory with embeddings extracted from reference audio.

### Requirements

- Reference audio: 3-10 seconds, clear speech, single speaker, minimal background noise
- Python with `torch`, `torchaudio`, `onnxruntime` (on x86 host)
- CosyVoice3 frontend-onnx models (for embedding extraction)

### Step 1: Prepare Reference Audio

- WAV format, 16 kHz+ sample rate
- Clean recording of a single speaker
- 3-10 seconds of speech
- Transcription of what is said in the audio

### Step 2: Extract Embeddings

Using the AXERA-TECH frontend tools (or CosyVoice3 repo):

```bash
# If using AXERA-TECH tools
python3 process_prompt.py \
    --audio reference.wav \
    --text "You are a helpful assistant.<|endofprompt|>Текст из аудио." \
    --output_dir prompt_files_my_voice
```

### Step 3: Convert to Pipeline Format

The pipeline expects these files in the prompt directory:

| File | Description |
|---|---|
| `prompt_text_tokens.npy` | Tokenized prompt text (int32 array) |
| `prompt_speech_tokens.npy` | Speech tokens from reference audio (int32 array) |
| `prompt_speech_feat.npy` | Mel spectrogram of reference audio (float32, shape [T, 80]) |
| `speaker_embedding.npy` | Speaker embedding vector (float32, shape [192]) |

### Important: System Prefix

The prompt text **MUST** include the system prefix for proper prosody:

```
You are a helpful assistant.<|endofprompt|>Ваш текст транскрипции здесь.
```

Without this prefix, the voice quality degrades significantly — unnatural prosody, foreign accent artifacts.

## Included Prompts

| Prompt | Description |
|---|---|
| `examples/prompt_russian_v2/` | Russian synthetic voice (TTS-generated reference) |

## Tips

- Longer reference audio (5-10s) generally produces better cloning
- The reference should have similar speaking style to what you want to generate
- Russian text works best with Russian voice prompts
- For multilingual synthesis, a native speaker prompt for each language is recommended
