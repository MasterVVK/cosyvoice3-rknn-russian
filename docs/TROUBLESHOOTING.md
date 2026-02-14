# Troubleshooting

Common issues and solutions for CosyVoice3 RKNN pipeline.

## LLM Issues

### RKLLM truncates or repeats speech tokens

**Symptom:** With RKLLM (W8A8), generated speech is too short, loops, or mixes languages.

**Cause:** W8A8 quantization reduces LLM quality, especially for the specialized speech token vocabulary.

**Fix:** Use ONNX FP32 LLM (recommended default):
```bash
--llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx
```

### RKLLM fails to load (truncated file)

**Symptom:** `rkllm_init failed: -1`, error about read size mismatch.

**Cause:** Incomplete file transfer (scp/rsync interrupted).

**Fix:** Re-transfer the file and verify sizes match:
```bash
ls -la cosyvoice3_llm_rk3588.rkllm  # on device
# Compare with source file size
```

### LLM speed is lower than expected

**Symptom:** ONNX LLM runs at 3-5 tok/s instead of 7+ tok/s.

**Cause:** Multiple CPU-intensive processes running simultaneously.

**Fix:** Run one TTS instance at a time. The ONNX LLM uses 4 CPU threads by default.

## Flow (RKNN NPU) Issues

### Flow falls back to ONNX CPU

**Symptom:** Pipeline says "backend=ONNX CPU" instead of "RKNN NPU".

**Cause:** No RKNN model available for the required sequence length, or RKNN models not specified.

**Fix:** Use `--flow_rknn auto` to auto-detect, or specify paths:
```bash
--flow_rknn auto
# or explicitly:
--flow_rknn flow_estimator_seq200_opt3.rknn,flow_estimator_seq1000_opt3.rknn
```

### Flow is slow on long texts

**Symptom:** Flow takes 60+ seconds for long texts.

**Cause:** Padding overhead — seq1000 model processes 69% zero-padded data for mel_len=312.

**Fix:** Add intermediate RKNN models (seq300, seq500) to reduce padding:
```bash
python3 scripts/convert_flow_rknn.py --seq_len 300 --opt_level 3
python3 scripts/convert_flow_rknn.py --seq_len 500 --opt_level 3
```

### RKNN query dynamic range warning

**Symptom:** `Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID`

**Cause:** Normal warning for static-shape RKNN models.

**Fix:** Safe to ignore. The pipeline uses static-shape models by design.

## HiFT Issues

### wave.Error: unknown format: 3

**Symptom:** Error when trying to play output WAV with Python's `wave` module.

**Cause:** Output is IEEE float32 WAV (format tag 3), which Python's `wave` module doesn't support.

**Fix:** Use a media player that supports float32 WAV (VLC, ffplay, audacity), or convert:
```bash
ffmpeg -i output.wav -acodec pcm_s16le output_pcm.wav
```

## Voice Cloning Issues

### Poor prosody / foreign accent

**Symptom:** Russian speech sounds unnatural or has a foreign accent.

**Cause:** Missing system prefix in prompt text.

**Fix:** Ensure prompt text starts with:
```
You are a helpful assistant.<|endofprompt|>
```

### Voice cloning not working

**Symptom:** Output doesn't match the reference voice.

**Cause:** Prompt directory missing required files.

**Fix:** Verify all 4 files exist:
```bash
ls prompt_russian_v2/
# Should have: prompt_text_tokens.npy, prompt_speech_tokens.npy,
#              prompt_speech_feat.npy, speaker_embedding.npy
```

## General Issues

### High memory usage

**Symptom:** System runs out of RAM (~4 GB needed).

**Cause:** ONNX FP32 LLM (1.5 GB) + embeddings (1.1 GB) + RKNN models.

**Fix:** Options:
1. Use RKLLM instead of ONNX LLM (saves ~1.5 GB RAM, uses NPU memory instead)
2. Load fewer RKNN Flow models (each ~700 MB)
3. Use swap if needed: `swapon --show`

### GPU device discovery warning

**Symptom:** `GPU device discovery failed: Failed to open file: "/sys/class/drm/card1/device/vendor"`

**Cause:** ONNX Runtime trying to find GPU on a headless ARM system.

**Fix:** Safe to ignore. The pipeline uses CPU provider only.

### Model loading is slow

**Symptom:** First run takes 20-30 seconds to load.

**Cause:** Loading 1.4 GB ONNX model + 1.1 GB embeddings + RKNN models from storage.

**Fix:** Normal behavior. Use faster storage (NVMe) if available. Subsequent runs with warm filesystem cache are faster.
