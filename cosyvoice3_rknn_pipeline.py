#!/usr/bin/env python3
"""
CosyVoice3 full TTS pipeline on RK3588 (CM3588 NAS Kit).

No PyTorch required — runs entirely on numpy + ONNX Runtime + RKNN.
First known open-source CosyVoice3 implementation on RK3588 RKNN.

Components:
  1. LLM (Qwen2-0.5B) → ONNX Runtime CPU FP32 (best quality)
                         or RKLLM NPU W8A8 (faster, lower quality)
  2. Flow (DiT CFM)   → RKNN NPU FP16 for DiT estimator + numpy for front-end
                         or ONNX Runtime CPU fallback
  3. HiFT (vocoder)   → ONNX Runtime CPU for f0_predictor + decode CNN,
                         numpy for source/STFT/ISTFT

Architecture:
  Text → [Tokenizer] → [LLM ONNX CPU] → [llm_decoder] → speech tokens
  Speech tokens → [Flow front-end] → [ODE solver × N + DiT RKNN NPU] → mel
  Mel → [f0_predictor ONNX] → source → [STFT] → [decode CNN ONNX] → audio

Dependencies on CM3588:
  pip3 install numpy scipy onnxruntime transformers rknn-toolkit-lite2

Usage:
  python3 cosyvoice3_rknn_pipeline.py --base_dir /path/to/models \\
      --llm_model cosyvoice3-llm-onnx/qwen2_transformer.onnx \\
      --flow_rknn auto --text "Привет, как дела?"

GitHub: https://github.com/nickovchinnikov/cosyvoice3-rknn-russian
"""

import os
import sys
import ctypes
import argparse
import time
import json
import numpy as np
import wave as wavmod

# ============================================================
# RKLLM 1.2.3 ctypes bindings (same as Qwen3-TTS pipeline)
# ============================================================

RKLLM_Handle_t = ctypes.c_void_p
RKLLM_RUN_NORMAL = 0
RKLLM_RUN_FINISH = 2
RKLLM_RUN_ERROR = 3
RKLLM_INPUT_TOKEN = 1
RKLLM_INPUT_EMBED = 2
RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_float),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
        ("use_gpu", ctypes.c_bool),
    ]

class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]

class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]

class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput),
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("input_data", RKLLMInputUnion),
    ]

class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]

class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int),
    ]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]

class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]

class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]

callback_type = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int,
)

# ============================================================
# CosyVoice3 Constants
# ============================================================

HIDDEN_SIZE = 896           # Qwen2-0.5B hidden size
SPEECH_TOKEN_SIZE = 6561    # Speech vocabulary size
SOS_TOKEN = 6561            # speech_token_size + 0
EOS_TOKEN = 6562            # speech_token_size + 1
TASK_ID_TOKEN = 6563        # speech_token_size + 2
FILL_TOKEN = 6564           # speech_token_size + 3
SPEECH_EMB_SIZE = 6761      # speech_token_size + 200
LLM_DECODER_SIZE = 6761     # speech_token_size + 200
SAMPLE_RATE = 24000


# ============================================================
# RKLLM wrapper for CosyVoice3 LLM
# ============================================================

class CosyVoiceLLM:
    """CosyVoice3 LLM on RKLLM NPU."""

    def __init__(self, model_path, embeddings_dir, lib_path=None, max_context=512):
        # Load embeddings
        self.embed_tokens = np.load(os.path.join(embeddings_dir, "embed_tokens.npy")).astype(np.float32)
        self.speech_embedding = np.load(os.path.join(embeddings_dir, "speech_embedding.npy")).astype(np.float32)
        self.llm_decoder_weight = np.load(os.path.join(embeddings_dir, "llm_decoder_weight.npy")).astype(np.float32)
        print(f"  embed_tokens: {self.embed_tokens.shape}")
        print(f"  speech_embedding: {self.speech_embedding.shape}")
        print(f"  llm_decoder_weight: {self.llm_decoder_weight.shape}")

        # Init RKLLM
        lib_path = lib_path or self._find_lib()
        self.lib = ctypes.CDLL(lib_path)
        self._setup()
        self._last_hidden = None
        self._c_callback = callback_type(self._callback)

        param = self.lib.rkllm_createDefaultParam()
        param.model_path = model_path.encode()
        param.max_context_len = max_context
        param.max_new_tokens = 1
        param.top_k = ctypes.c_float(25.0)
        param.top_p = ctypes.c_float(0.9)
        param.temperature = ctypes.c_float(1.0)
        param.repeat_penalty = ctypes.c_float(1.0)
        param.skip_special_token = False
        param.is_async = False
        param.use_gpu = True
        param.extend_param.base_domain_id = 1
        param.extend_param.embed_flash = 1

        self.handle = RKLLM_Handle_t()
        ret = self.lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(param), self._c_callback)
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed: {ret}")

    @staticmethod
    def _find_lib():
        for p in ["/usr/local/lib/librkllmrt.so", "/usr/lib/librkllmrt.so"]:
            if os.path.exists(p):
                return p
        raise FileNotFoundError("librkllmrt.so not found")

    def _setup(self):
        self.lib.rkllm_createDefaultParam.restype = RKLLMParam
        self.lib.rkllm_createDefaultParam.argtypes = []
        self.lib.rkllm_init.restype = ctypes.c_int
        self.lib.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type
        ]
        self.lib.rkllm_run.restype = ctypes.c_int
        self.lib.rkllm_run.argtypes = [
            RKLLM_Handle_t, ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p
        ]
        self.lib.rkllm_destroy.restype = ctypes.c_int
        self.lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]

    def _callback(self, result_ptr, userdata, state):
        if state == RKLLM_RUN_NORMAL:
            r = result_ptr.contents
            if r.last_hidden_layer.hidden_states and r.last_hidden_layer.embd_size > 0:
                n = r.last_hidden_layer.num_tokens
                d = r.last_hidden_layer.embd_size
                arr = np.ctypeslib.as_array(
                    r.last_hidden_layer.hidden_states, shape=(n * d,)
                ).copy()
                self._last_hidden = arr.reshape(n, d)

    def get_hidden(self, embeddings, keep_history=1):
        """Feed embeddings [n_tokens, 896] and get hidden states [n_tokens, 896]."""
        self._last_hidden = None
        rk_input = RKLLMInput()
        rk_input.role = None
        rk_input.enable_thinking = False
        rk_input.input_type = RKLLM_INPUT_EMBED

        flat = embeddings.astype(np.float32).flatten()
        c_embed = (ctypes.c_float * len(flat))(*flat)
        rk_input.input_data.embed_input.embed = c_embed
        rk_input.input_data.embed_input.n_tokens = embeddings.shape[0]

        infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        infer_params.mode = RKLLM_INFER_GET_LAST_HIDDEN_LAYER
        infer_params.keep_history = keep_history

        ret = self.lib.rkllm_run(
            self.handle, ctypes.byref(rk_input), ctypes.byref(infer_params), None
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_run failed: {ret}")
        return self._last_hidden

    def build_prefix(self, text_token_ids, prompt_speech_tokens=None):
        """Build LLM prefix: [sos_emb, text_embeddings, task_id_emb, prompt_speech_emb].

        Args:
            text_token_ids: list of int - Qwen2 text token IDs (prompt_text + text concatenated)
            prompt_speech_tokens: list of int - prompt speech token IDs for voice cloning

        Returns:
            prefix: np.array [n_tokens, 896]
        """
        # Text embeddings via Qwen2 embed_tokens
        text_emb = self.embed_tokens[text_token_ids]  # [n_text, 896]

        # Special tokens from speech_embedding
        sos_emb = self.speech_embedding[SOS_TOKEN].reshape(1, -1)  # [1, 896]
        task_id_emb = self.speech_embedding[TASK_ID_TOKEN].reshape(1, -1)  # [1, 896]

        # Prompt speech tokens
        if prompt_speech_tokens is not None and len(prompt_speech_tokens) > 0:
            prompt_emb = self.speech_embedding[prompt_speech_tokens]  # [n_prompt, 896]
        else:
            prompt_emb = np.zeros((0, HIDDEN_SIZE), dtype=np.float32)

        # Concatenate: [sos, text, task_id, prompt_speech]
        prefix = np.concatenate([sos_emb, text_emb, task_id_emb, prompt_emb], axis=0)
        return prefix.astype(np.float32)

    def _nucleus_sampling(self, logits, top_p=0.8, top_k=25):
        """Nucleus (top-p + top-k) sampling. Returns token ID."""
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        sorted_idx = np.argsort(probs)[::-1]
        cum_prob = 0.0
        candidates = []
        candidate_probs = []
        for idx in sorted_idx:
            if cum_prob >= top_p and len(candidates) > 0:
                break
            if len(candidates) >= top_k:
                break
            candidates.append(idx)
            candidate_probs.append(probs[idx])
            cum_prob += probs[idx]
        candidate_probs = np.array(candidate_probs)
        candidate_probs /= candidate_probs.sum()
        return int(candidates[np.random.choice(len(candidates), p=candidate_probs)])

    def _random_sampling(self, logits):
        """Full vocabulary random sampling (fallback for RAS)."""
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    def sample_token(self, hidden_state, decoded_tokens=None,
                     temperature=1.0, top_k=25, top_p=0.8,
                     win_size=10, tau_r=0.1):
        """Sample speech token with RAS + no-repeat n-gram blocking.

        1. No-repeat n-gram: block tokens that would create a repeated 5-gram or 10-gram
        2. Nucleus sampling (top-p + top-k)
        3. RAS: if sampled token appeared in recent window, fall back to random

        Returns token ID from full vocabulary [0..6760].
        Tokens >= SPEECH_TOKEN_SIZE (6561) indicate EOS/stop.
        """
        logits = hidden_state @ self.llm_decoder_weight.T  # [LLM_DECODER_SIZE=6761]
        logits = logits / max(temperature, 1e-6)

        # No-repeat n-gram blocking: prevent phrase-level loops
        if decoded_tokens is not None:
            for ngram_size in [10]:
                n1 = ngram_size - 1
                if len(decoded_tokens) < ngram_size:
                    continue
                prefix = decoded_tokens[-n1:]
                for i in range(len(decoded_tokens) - n1):
                    if decoded_tokens[i:i + n1] == prefix:
                        logits[decoded_tokens[i + n1]] = -1e9

        # Nucleus sampling (top-p + top-k)
        token = self._nucleus_sampling(logits, top_p=top_p, top_k=top_k)

        # RAS: check repetition in recent window
        if decoded_tokens is not None and len(decoded_tokens) >= 1:
            window = decoded_tokens[-win_size:]
            rep_count = sum(1 for t in window if t == token)
            if rep_count >= max(1, int(win_size * tau_r)):
                token = self._random_sampling(logits)

        return token

    def generate_tokens(self, text_token_ids, prompt_speech_tokens=None,
                        max_tokens=500, min_tokens=10, temperature=1.0, top_k=25):
        """Generate speech tokens autoregressively with RAS.

        Returns:
            list of int - speech token IDs
        """
        prefix = self.build_prefix(text_token_ids, prompt_speech_tokens)
        print(f"  LLM prefix: {prefix.shape[0]} tokens")

        # Prefill
        hidden = self.get_hidden(prefix, keep_history=0)
        if hidden is None:
            raise RuntimeError("No hidden states from prefill")

        # Autoregressive generation with RAS (Repetition Aware Sampling)
        out_tokens = []
        for i in range(max_tokens):
            token = self.sample_token(hidden[-1], decoded_tokens=out_tokens,
                                      temperature=temperature, top_k=top_k)

            # Check for stop (token >= SPEECH_TOKEN_SIZE)
            if token >= SPEECH_TOKEN_SIZE:
                if i >= min_tokens:
                    break
                # Before min_tokens: resample until we get a valid speech token
                for _ in range(50):
                    token = self.sample_token(hidden[-1], decoded_tokens=out_tokens,
                                              temperature=temperature, top_k=top_k)
                    if token < SPEECH_TOKEN_SIZE:
                        break
                if token >= SPEECH_TOKEN_SIZE:
                    break  # Give up after 50 retries

            out_tokens.append(token)

            # Feed back speech_embedding[token]
            next_emb = self.speech_embedding[token].reshape(1, HIDDEN_SIZE).astype(np.float32)
            hidden = self.get_hidden(next_emb, keep_history=1)
            if hidden is None:
                break

        return out_tokens

    def destroy(self):
        if self.handle:
            self.lib.rkllm_destroy(self.handle)
            self.handle = None


class CosyVoiceLLM_ONNX:
    """CosyVoice3 LLM on CPU via ONNX Runtime (FP32, no quantization)."""

    def __init__(self, onnx_path, embeddings_dir, num_threads=4):
        import onnxruntime as ort
        # Load embeddings
        self.embed_tokens = np.load(os.path.join(embeddings_dir, "embed_tokens.npy")).astype(np.float32)
        self.speech_embedding = np.load(os.path.join(embeddings_dir, "speech_embedding.npy")).astype(np.float32)
        self.llm_decoder_weight = np.load(os.path.join(embeddings_dir, "llm_decoder_weight.npy")).astype(np.float32)
        print(f"  embed_tokens: {self.embed_tokens.shape}")
        print(f"  speech_embedding: {self.speech_embedding.shape}")
        print(f"  llm_decoder_weight: {self.llm_decoder_weight.shape}")

        # Detect architecture from ONNX model
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = num_threads
        sess_opts.intra_op_num_threads = num_threads
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_opts, providers=["CPUExecutionProvider"])

        # Count layers from input names
        self.num_layers = sum(1 for inp in self.sess.get_inputs() if inp.name.startswith("past_key_"))
        kv_shape = [inp for inp in self.sess.get_inputs() if inp.name == "past_key_0"][0].shape
        self.num_kv_heads = kv_shape[1] if isinstance(kv_shape[1], int) else 2
        self.head_dim = kv_shape[3] if isinstance(kv_shape[3], int) else 64
        print(f"  ONNX LLM: {self.num_layers} layers, {self.num_kv_heads} KV heads, head_dim={self.head_dim}")

        # KV cache
        self._kv_cache = None
        self._pos = 0

    def _reset_kv(self):
        self._kv_cache = {}
        for i in range(self.num_layers):
            self._kv_cache[f"past_key_{i}"] = np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float32)
            self._kv_cache[f"past_value_{i}"] = np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float32)
        self._pos = 0

    def get_hidden(self, embeddings, keep_history=1):
        """Feed embeddings [n_tokens, hidden_size] and get hidden states."""
        if keep_history == 0 or self._kv_cache is None:
            self._reset_kv()

        n_tokens = embeddings.shape[0]
        inputs_embeds = embeddings.reshape(1, n_tokens, -1).astype(np.float32)
        position_ids = np.arange(self._pos, self._pos + n_tokens, dtype=np.int64).reshape(1, -1)

        feeds = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
        }
        feeds.update(self._kv_cache)

        outputs = self.sess.run(None, feeds)
        hidden = outputs[0]  # [1, n_tokens, hidden_size]

        # Update KV cache
        for i in range(self.num_layers):
            self._kv_cache[f"past_key_{i}"] = outputs[1 + i * 2]
            self._kv_cache[f"past_value_{i}"] = outputs[2 + i * 2]
        self._pos += n_tokens

        return hidden[0]  # [n_tokens, hidden_size]

    def build_prefix(self, text_token_ids, prompt_speech_tokens=None):
        """Same as CosyVoiceLLM.build_prefix."""
        text_emb = self.embed_tokens[text_token_ids]
        sos_emb = self.speech_embedding[SOS_TOKEN].reshape(1, -1)
        task_id_emb = self.speech_embedding[TASK_ID_TOKEN].reshape(1, -1)
        if prompt_speech_tokens is not None and len(prompt_speech_tokens) > 0:
            prompt_emb = self.speech_embedding[prompt_speech_tokens]
        else:
            prompt_emb = np.zeros((0, HIDDEN_SIZE), dtype=np.float32)
        prefix = np.concatenate([sos_emb, text_emb, task_id_emb, prompt_emb], axis=0)
        return prefix.astype(np.float32)

    def _nucleus_sampling(self, logits, top_p=0.8, top_k=25):
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        sorted_idx = np.argsort(probs)[::-1]
        cum_prob = 0.0
        candidates, candidate_probs = [], []
        for idx in sorted_idx:
            if cum_prob >= top_p and len(candidates) > 0:
                break
            if len(candidates) >= top_k:
                break
            candidates.append(idx)
            candidate_probs.append(probs[idx])
            cum_prob += probs[idx]
        candidate_probs = np.array(candidate_probs)
        candidate_probs /= candidate_probs.sum()
        return int(candidates[np.random.choice(len(candidates), p=candidate_probs)])

    def _random_sampling(self, logits):
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    def sample_token(self, hidden_state, decoded_tokens=None,
                     temperature=1.0, top_k=25, top_p=0.8,
                     win_size=10, tau_r=0.1):
        """Same sampling logic as CosyVoiceLLM."""
        logits = hidden_state @ self.llm_decoder_weight.T
        logits = logits / max(temperature, 1e-6)
        if decoded_tokens is not None:
            for ngram_size in [10]:
                n1 = ngram_size - 1
                if len(decoded_tokens) < ngram_size:
                    continue
                prefix = decoded_tokens[-n1:]
                for i in range(len(decoded_tokens) - n1):
                    if decoded_tokens[i:i + n1] == prefix:
                        logits[decoded_tokens[i + n1]] = -1e9
        token = self._nucleus_sampling(logits, top_p=top_p, top_k=top_k)
        if decoded_tokens is not None and len(decoded_tokens) >= 1:
            window = decoded_tokens[-win_size:]
            rep_count = sum(1 for t in window if t == token)
            if rep_count >= max(1, int(win_size * tau_r)):
                token = self._random_sampling(logits)
        return token

    def generate_tokens(self, text_token_ids, prompt_speech_tokens=None,
                        max_tokens=500, min_tokens=10, temperature=1.0, top_k=25):
        """Generate speech tokens autoregressively (same interface as RKLLM version)."""
        prefix = self.build_prefix(text_token_ids, prompt_speech_tokens)
        print(f"  LLM prefix: {prefix.shape[0]} tokens")

        hidden = self.get_hidden(prefix, keep_history=0)
        if hidden is None:
            raise RuntimeError("No hidden states from prefill")

        out_tokens = []
        for i in range(max_tokens):
            token = self.sample_token(hidden[-1], decoded_tokens=out_tokens,
                                      temperature=temperature, top_k=top_k)
            if token >= SPEECH_TOKEN_SIZE:
                if i >= min_tokens:
                    break
                for _ in range(50):
                    token = self.sample_token(hidden[-1], decoded_tokens=out_tokens,
                                              temperature=temperature, top_k=top_k)
                    if token < SPEECH_TOKEN_SIZE:
                        break
                if token >= SPEECH_TOKEN_SIZE:
                    break

            out_tokens.append(token)
            next_emb = self.speech_embedding[token].reshape(1, HIDDEN_SIZE).astype(np.float32)
            hidden = self.get_hidden(next_emb, keep_history=1)
            if hidden is None:
                break

        return out_tokens

    def destroy(self):
        self._kv_cache = None
        self.sess = None


# ============================================================
# Flow DiT decoder (ODE solver + ONNX Runtime estimator)
# ============================================================

class FlowDecoder:
    """CosyVoice3 Flow decoder with CPU front-end and RKNN NPU or ONNX CPU estimator.

    Supports multiple RKNN models with different fixed seq_len.
    At inference time, selects the smallest model that fits the mel_len.
    Falls back to ONNX CPU if no RKNN model fits.
    """

    def __init__(self, flow_components_dir, estimator_onnx_path, rknn_paths=None):
        # Load config
        with open(os.path.join(flow_components_dir, "flow_config.json")) as f:
            self.config = json.load(f)

        # Load front-end weights
        self.input_embedding = np.load(
            os.path.join(flow_components_dir, "flow_input_embedding.npy")
        ).astype(np.float32)

        self.spk_affine_weight = np.load(
            os.path.join(flow_components_dir, "flow_spk_affine_weight.npy")
        ).astype(np.float32)
        self.spk_affine_bias = np.load(
            os.path.join(flow_components_dir, "flow_spk_affine_bias.npy")
        ).astype(np.float32)

        pll = np.load(os.path.join(flow_components_dir, "flow_pre_lookahead_weights.npz"))
        self.pll_conv1_w = pll["conv1_weight"].astype(np.float32)
        self.pll_conv1_b = pll["conv1_bias"].astype(np.float32)
        self.pll_conv2_w = pll["conv2_weight"].astype(np.float32)
        self.pll_conv2_b = pll["conv2_bias"].astype(np.float32)

        print(f"  Flow input_embedding: {self.input_embedding.shape}")
        print(f"  Flow spk_affine: {self.spk_affine_weight.shape}")

        # Load estimator(s): RKNN NPU or ONNX CPU
        self.rknn_models = []  # list of (seq_len, rknn_instance), sorted by seq_len
        self.estimator = None
        self._onnx_path = estimator_onnx_path  # for lazy fallback

        if rknn_paths:
            import re
            from rknnlite.api import RKNNLite
            for path in rknn_paths:
                if not os.path.exists(path):
                    continue
                rknn = RKNNLite(verbose=False)
                ret = rknn.load_rknn(path)
                if ret != 0:
                    print(f"  WARNING: Failed to load RKNN: {path}")
                    continue
                ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
                if ret != 0:
                    print(f"  WARNING: Failed to init RKNN runtime: {path}")
                    continue
                basename = os.path.basename(path)
                m = re.search(r'seq(\d+)', basename)
                seq_len = int(m.group(1)) if m else 200
                size_mb = os.path.getsize(path) / 1024 / 1024
                self.rknn_models.append((seq_len, rknn))
                print(f"  Flow estimator RKNN: {basename} ({size_mb:.0f} MB, seq={seq_len})")
            self.rknn_models.sort(key=lambda x: x[0])

        if not self.rknn_models:
            self._load_onnx(estimator_onnx_path)

    def _load_onnx(self, onnx_path):
        """Load ONNX model for CPU inference (fallback)."""
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        self.estimator = ort.InferenceSession(
            onnx_path, sess_options=opts,
            providers=['CPUExecutionProvider']
        )
        print(f"  Flow estimator ONNX loaded (CPU fallback)")

    def _select_rknn(self, mel_len):
        """Select smallest RKNN model that fits mel_len. Returns (seq_len, rknn) or (None, None)."""
        for seq_len, rknn in self.rknn_models:
            if mel_len <= seq_len:
                return seq_len, rknn
        return None, None

    def _pre_lookahead(self, x):
        """PreLookaheadLayer in numpy. x: [1, seq_len, 80] → [1, seq_len, 80]"""
        # Transpose to [1, 80, seq_len] for Conv1d
        out = x.transpose(0, 2, 1)
        # Pad right by pre_lookahead_len=3
        out = np.pad(out, ((0, 0), (0, 0), (0, 3)), mode='constant')
        # Conv1d (conv1_w: [1024, 80, 4])
        out = self._conv1d(out, self.pll_conv1_w, self.pll_conv1_b)
        out = np.maximum(out, 0.01 * out)  # LeakyReLU (default alpha=0.01)
        # Pad left by kernel_size-1=2
        out = np.pad(out, ((0, 0), (0, 0), (2, 0)), mode='constant')
        # Conv1d (conv2_w: [80, 1024, 3])
        out = self._conv1d(out, self.pll_conv2_w, self.pll_conv2_b)
        # Transpose back and residual
        out = out.transpose(0, 2, 1)
        return out + x

    def _conv1d(self, x, weight, bias):
        """Simple Conv1d: x [B, C_in, L], weight [C_out, C_in, K] → [B, C_out, L-K+1]"""
        B, C_in, L = x.shape
        C_out, _, K = weight.shape
        out_len = L - K + 1
        # Unfold approach
        cols = np.lib.stride_tricks.as_strided(
            x,
            shape=(B, C_in, out_len, K),
            strides=(x.strides[0], x.strides[1], x.strides[2], x.strides[2])
        )
        # weight: [C_out, C_in, K] → einsum with cols [B, C_in, out_len, K]
        out = np.einsum('bclk,ock->bol', cols, weight) + bias.reshape(1, -1, 1)
        return out

    def _process_speaker_embedding(self, embedding):
        """Process speaker embedding: normalize → affine → [1, 80]"""
        # L2 normalize
        norm = np.sqrt(np.sum(embedding ** 2, axis=-1, keepdims=True) + 1e-12)
        embedding = embedding / norm
        # Affine: [1, 192] → [1, 80]
        return embedding @ self.spk_affine_weight.T + self.spk_affine_bias

    def _cosine_schedule(self, n_steps):
        """Cosine time schedule for ODE solver."""
        t = np.linspace(0, 1, n_steps + 1).astype(np.float32)
        return (1 - np.cos(t * np.pi / 2)).astype(np.float32)

    def inference(self, speech_tokens, prompt_tokens, prompt_feat, speaker_embedding,
                  force_cpu=False, ode_steps=None, padding_mode='reflect', seed=None):
        """Run Flow inference: speech tokens → mel spectrogram.

        Args:
            speech_tokens: [n_tokens] - generated speech token IDs
            prompt_tokens: [n_prompt] - prompt speech token IDs
            prompt_feat: [n_prompt_mel, 80] - prompt mel features
            speaker_embedding: [192] - speaker embedding vector
            force_cpu: force ONNX CPU backend even when RKNN loaded
            ode_steps: override number of ODE steps (None = use config)
            padding_mode: 'reflect', 'zero', or 'edge' for RKNN padding
            seed: random seed for reproducible initial noise

        Returns:
            mel: [80, mel_len2] - generated mel spectrogram (prompt removed)
        """
        cfg = self.config
        token_mel_ratio = cfg["token_mel_ratio"]

        # 1. Concatenate prompt + generated tokens
        all_tokens = np.concatenate([prompt_tokens, speech_tokens])
        seq_len = len(all_tokens)

        # 2. Token embedding (Flow's own, not LLM's)
        token_emb = self.input_embedding[all_tokens]  # [seq_len, 80]
        token_emb = token_emb.reshape(1, seq_len, 80)

        # 3. PreLookahead layer
        h = self._pre_lookahead(token_emb)

        # 4. Repeat interleave (each token → 2 mel frames)
        h = np.repeat(h, token_mel_ratio, axis=1)  # [1, seq_len*2, 80]

        mel_len1 = prompt_feat.shape[0]  # prompt mel frames
        mel_len2 = h.shape[1] - mel_len1  # generated mel frames
        mel_len = mel_len1 + mel_len2

        # 5. Build conditions
        conds = np.zeros((1, mel_len, 80), dtype=np.float32)
        conds[0, :mel_len1] = prompt_feat
        conds = conds.transpose(0, 2, 1)  # [1, 80, mel_len]

        # 6. Prepare mu and mask
        mu = h.transpose(0, 2, 1).astype(np.float32)  # [1, 80, mel_len]
        mask = np.ones((1, 1, mel_len), dtype=np.float32)

        # 7. Process speaker embedding
        spks = self._process_speaker_embedding(
            speaker_embedding.reshape(1, -1)
        ).astype(np.float32)  # [1, 80]

        # 8. ODE solver (Euler method with CFG)
        n_steps = ode_steps if ode_steps is not None else cfg["n_timesteps"]
        cfg_rate = cfg["inference_cfg_rate"]
        sigma_min = cfg["sigma_min"]

        t_span = self._cosine_schedule(n_steps)

        # Initial noise (with optional fixed seed for reproducibility)
        if seed is not None:
            rng = np.random.RandomState(seed)
            z = rng.randn(1, 80, mel_len).astype(np.float32) * 1.0
        else:
            z = np.random.randn(1, 80, mel_len).astype(np.float32) * 1.0

        # Build CFG batch (batch=2: [conditional, unconditional])
        mu_cfg = np.concatenate([mu, np.zeros_like(mu)], axis=0)          # [2, 80, mel_len]
        mask_cfg = np.concatenate([mask, mask], axis=0)                    # [2, 1, mel_len]
        conds_cfg = np.concatenate([conds, np.zeros_like(conds)], axis=0)  # [2, 80, mel_len]
        spks_cfg = np.concatenate([spks, np.zeros_like(spks)], axis=0)    # [2, 80]

        # Select RKNN model or fall back to ONNX CPU
        rknn_seq, rknn_model = self._select_rknn(mel_len)
        use_rknn_this_call = rknn_model is not None and not force_cpu

        if force_cpu and self.rknn_models:
            print(f"  FORCE CPU: Using ONNX CPU backend (--force_cpu_flow)")

        if not use_rknn_this_call:
            if self.rknn_models and not force_cpu:
                max_seq = self.rknn_models[-1][0]
                print(f"  WARNING: mel_len={mel_len} > max_rknn_seq={max_seq}, falling back to ONNX CPU")
            if self.estimator is None:
                print(f"  Loading ONNX fallback...")
                self._load_onnx(self._onnx_path)

        if use_rknn_this_call:
            backend = f"RKNN NPU (seq={rknn_seq})"
        else:
            backend = "ONNX CPU"
        print(f"  ODE solver: {n_steps} steps, mel_len={mel_len}, backend={backend}")

        # Pre-pad CFG tensors for RKNN (fixed seq_len)
        if use_rknn_this_call:
            pad_w = rknn_seq - mel_len
            if padding_mode == 'zero':
                # Zero padding + mask=0 in padded region
                mu_pad = np.pad(mu_cfg, ((0,0),(0,0),(0,pad_w)), mode='constant', constant_values=0).astype(np.float32)
                mask_real = np.ones((2, 1, mel_len), dtype=np.float32)
                mask_pad = np.pad(mask_real, ((0,0),(0,0),(0,pad_w)), mode='constant', constant_values=0).astype(np.float32)
                conds_pad = np.pad(conds_cfg, ((0,0),(0,0),(0,pad_w)), mode='constant', constant_values=0).astype(np.float32)
            elif padding_mode == 'edge':
                # Edge padding + mask=1
                mu_pad = np.pad(mu_cfg, ((0,0),(0,0),(0,pad_w)), mode='edge').astype(np.float32)
                mask_pad = np.ones((2, 1, rknn_seq), dtype=np.float32)
                conds_pad = np.pad(conds_cfg, ((0,0),(0,0),(0,pad_w)), mode='edge').astype(np.float32)
            else:
                # reflect padding + mask=1 (original behavior)
                mu_pad = np.pad(mu_cfg, ((0,0),(0,0),(0,pad_w)), mode='reflect').astype(np.float32)
                mask_pad = np.ones((2, 1, rknn_seq), dtype=np.float32)
                conds_pad = np.pad(conds_cfg, ((0,0),(0,0),(0,pad_w)), mode='reflect').astype(np.float32)
            print(f"  Padding: mode={padding_mode}, pad_w={pad_w} ({100*pad_w/rknn_seq:.0f}% padding)")

        for step in range(1, len(t_span)):
            t = t_span[step - 1]  # evaluate at START of interval (not end!)
            dt = t_span[step] - t_span[step - 1]

            # Prepare batch input
            x_in = np.concatenate([z, z], axis=0)  # [2, 80, mel_len]
            t_in = np.array([t, t], dtype=np.float32)  # [2]

            if use_rknn_this_call:
                # Pad x to fixed seq_len, run on NPU, crop back
                if padding_mode == 'zero':
                    x_pad = np.pad(x_in, ((0,0),(0,0),(0,pad_w)), mode='constant', constant_values=0).astype(np.float32)
                elif padding_mode == 'edge':
                    x_pad = np.pad(x_in, ((0,0),(0,0),(0,pad_w)), mode='edge').astype(np.float32)
                else:
                    x_pad = np.pad(x_in, ((0,0),(0,0),(0,pad_w)), mode='reflect').astype(np.float32)
                dphi_dt = rknn_model.inference(
                    inputs=[x_pad, mask_pad, mu_pad, t_in, spks_cfg, conds_pad]
                )[0][:, :, :mel_len]  # crop
            else:
                # ONNX Runtime CPU
                dphi_dt = self.estimator.run(
                    None,
                    {
                        "x": x_in,
                        "mask": mask_cfg,
                        "mu": mu_cfg,
                        "t": t_in,
                        "spks": spks_cfg,
                        "cond": conds_cfg,
                    }
                )[0]  # [2, 80, mel_len]

            # CFG blending
            cond_out = dphi_dt[:1]
            uncond_out = dphi_dt[1:]
            dphi_dt = (1.0 + cfg_rate) * cond_out - cfg_rate * uncond_out

            # Euler step
            z = z + dt * dphi_dt

        # 9. Extract generated part (remove prompt)
        mel = z[0, :, mel_len1:]  # [80, mel_len2]
        return mel.astype(np.float32)


# ============================================================
# HiFT vocoder (RKNN NPU for decode CNN + CPU for f0/source)
# ============================================================

class HiFTVocoder:
    """CosyVoice3 HiFT vocoder — PyTorch-free (numpy + ONNX Runtime).

    Components:
      - f0_predictor: ONNX Runtime (mel → f0)
      - SineGen2: numpy (f0 → sine harmonics → source)
      - STFT: numpy (source → source_stft, 16-point DFT)
      - decode CNN: ONNX Runtime (mel + source_stft → raw_output)
      - ISTFT: numpy (raw_output → audio, 16-point IDFT + overlap-add)
    """

    def __init__(self, hift_components_dir):
        import onnxruntime as ort

        # Load config
        with open(os.path.join(hift_components_dir, "hift_config.json")) as f:
            self.cfg = json.load(f)

        # Load source weights
        sw = np.load(os.path.join(hift_components_dir, "source_weights.npz"))
        self.source_linear_w = sw["source_linear_weight"].astype(np.float32)  # [1, 9]
        self.source_linear_b = sw["source_linear_bias"].astype(np.float32)    # [1]
        self.stft_window = sw["stft_window"].astype(np.float32)               # [16]

        # Pre-store random values for causal SineGen2 (same as PyTorch init)
        max_audio_len = 300 * self.cfg["sampling_rate"]  # 7.2M
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        self.sinegen_rand_ini = rng.rand(1, self.cfg["nb_harmonics"] + 1).astype(np.float32)
        self.sinegen_rand_ini[0, 0] = 0.0  # No noise for fundamental
        self.sinegen_sine_waves = rng.rand(1, max_audio_len, self.cfg["nb_harmonics"] + 1).astype(np.float32)
        self.source_uv = rng.rand(1, max_audio_len, 1).astype(np.float32)

        # ONNX sessions
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1

        f0_path = os.path.join(hift_components_dir, "f0_predictor.onnx")
        self.f0_session = ort.InferenceSession(f0_path, sess_options=opts,
                                                providers=['CPUExecutionProvider'])

        decode_path = os.path.join(hift_components_dir, "hift_decode_dynamic.onnx")
        self.decode_session = ort.InferenceSession(decode_path, sess_options=opts,
                                                    providers=['CPUExecutionProvider'])

        print(f"  HiFT vocoder loaded (no PyTorch)")
        print(f"    f0_predictor ONNX: {f0_path}")
        print(f"    decode CNN ONNX: {decode_path}")

    def _sinegen2(self, f0_upsampled):
        """SineGen2 in numpy. f0: [1, audio_len, 1] → sine_waves [1, audio_len, 9]"""
        cfg = self.cfg
        sr = cfg["sampling_rate"]
        upsample_scale = cfg["upsample_scale"]
        n_harmonics = cfg["nb_harmonics"]
        sine_amp = cfg["sine_amp"]
        noise_std = cfg["noise_std"]
        voiced_threshold = cfg["voiced_threshold"]

        # f0_upsampled: [1, audio_len, 1]
        f0 = f0_upsampled

        # Generate harmonics: f0 * [1, 2, 3, ..., 9]
        harmonics = np.arange(1, n_harmonics + 2, dtype=np.float32).reshape(1, 1, -1)
        fn = f0 * harmonics  # [1, audio_len, 9]

        # _f02sine (causal version)
        rad_values = (fn / sr) % 1.0  # [1, audio_len, 9]

        # Add initial phase (causal: use pre-stored rand_ini)
        rad_values[:, 0, :] = rad_values[:, 0, :] + self.sinegen_rand_ini

        # Downsample → cumsum → upsample → sin (efficient phase accumulation)
        # interpolate: scale_factor = 1/upsample_scale, mode='linear'
        audio_len = rad_values.shape[1]
        down_len = int(np.ceil(audio_len / upsample_scale))

        # Simple linear interpolation downsampling
        indices = np.linspace(0, audio_len - 1, down_len).astype(np.float32)
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.minimum(idx_floor + 1, audio_len - 1)
        frac = (indices - idx_floor).reshape(1, -1, 1)
        rad_down = rad_values[:, idx_floor] * (1 - frac) + rad_values[:, idx_ceil] * frac

        # Cumulative sum for phase
        phase_down = np.cumsum(rad_down, axis=1) * 2 * np.pi

        # Upsample back (nearest mode for causal)
        phase_up = np.repeat(phase_down, upsample_scale, axis=1)[:, :audio_len]
        phase_up = phase_up * upsample_scale

        sines = np.sin(phase_up) * sine_amp  # [1, audio_len, 9]

        # UV detection
        uv = (f0 > voiced_threshold).astype(np.float32)  # [1, audio_len, 1]

        # Noise (causal: use pre-stored sine_waves)
        noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
        noise = noise_amp * self.sinegen_sine_waves[:, :audio_len]

        # Combine: voiced sine + noise
        sine_waves = sines * uv + noise  # [1, audio_len, 9]

        return sine_waves, uv

    def _source_module(self, f0_upsampled):
        """SourceModuleHnNSF in numpy. f0: [1, audio_len, 1] → source [1, 1, audio_len]"""
        sine_waves, uv = self._sinegen2(f0_upsampled)

        # Linear merge: [1, audio_len, 9] @ [9, 1] + bias → [1, audio_len, 1]
        sine_merge = sine_waves @ self.source_linear_w.T + self.source_linear_b
        sine_merge = np.tanh(sine_merge)

        # Noise for noise branch (causal: use pre-stored uv)
        audio_len = f0_upsampled.shape[1]
        noise = self.source_uv[:, :audio_len] * self.cfg["sine_amp"] / 3

        # source: [1, 1, audio_len]
        source = sine_merge.transpose(0, 2, 1)  # [1, 1, audio_len]
        return source

    def _stft(self, source):
        """STFT in numpy. source: [1, 1, audio_len] → source_stft [1, 18, stft_len]

        n_fft=16, hop_len=4, window=hann(16), center=True
        """
        n_fft = self.cfg["n_fft"]
        hop_len = self.cfg["hop_len"]
        window = self.stft_window

        x = source[0, 0]  # [audio_len]
        # Center padding
        pad_len = n_fft // 2
        x_padded = np.pad(x, (pad_len, pad_len), mode='reflect')

        # Frame the signal
        num_frames = (len(x_padded) - n_fft) // hop_len + 1
        frames = np.lib.stride_tricks.as_strided(
            x_padded,
            shape=(num_frames, n_fft),
            strides=(x_padded.strides[0] * hop_len, x_padded.strides[0])
        ).copy()

        # Apply window and DFT
        frames = frames * window
        spectrum = np.fft.rfft(frames, n=n_fft, axis=1)  # [num_frames, n_fft//2+1]

        # Split into real and imag
        real = spectrum.real.T  # [n_fft//2+1, num_frames]
        imag = spectrum.imag.T  # [n_fft//2+1, num_frames]

        # Concatenate: [1, 18, stft_len] (9 real + 9 imag)
        source_stft = np.concatenate([
            real.reshape(1, n_fft // 2 + 1, num_frames),
            imag.reshape(1, n_fft // 2 + 1, num_frames)
        ], axis=1).astype(np.float32)

        return source_stft

    def _istft(self, magnitude, phase):
        """ISTFT in numpy. magnitude/phase: [1, 9, T] → audio [audio_len]

        n_fft=16, hop_len=4, window=hann(16)
        """
        n_fft = self.cfg["n_fft"]
        hop_len = self.cfg["hop_len"]
        window = self.stft_window

        # Clip magnitude
        magnitude = np.clip(magnitude[0], a_min=None, a_max=1e2)  # [9, T]
        phase = phase[0]  # [9, T]

        # Build complex spectrum
        real = magnitude * np.cos(phase)
        imag = magnitude * np.sin(phase)
        spectrum = real + 1j * imag  # [9, T]

        num_frames = spectrum.shape[1]
        # IRFFT
        frames = np.fft.irfft(spectrum.T, n=n_fft, axis=1)  # [T, n_fft]

        # Apply window
        frames = frames * window

        # Overlap-add
        audio_len = (num_frames - 1) * hop_len + n_fft
        audio = np.zeros(audio_len, dtype=np.float32)
        window_sum = np.zeros(audio_len, dtype=np.float32)
        for i in range(num_frames):
            start = i * hop_len
            audio[start:start + n_fft] += frames[i]
            window_sum[start:start + n_fft] += window ** 2

        # Normalize by window sum
        mask = window_sum > 1e-8
        audio[mask] /= window_sum[mask]

        # Remove padding (center=True)
        pad_len = n_fft // 2
        audio = audio[pad_len:-pad_len] if pad_len > 0 else audio

        return audio

    def inference(self, mel):
        """Convert mel spectrogram to audio. No PyTorch required.

        Args:
            mel: np.array [80, mel_len] - mel spectrogram

        Returns:
            audio: np.array [samples] - audio waveform
        """
        mel_input = mel.reshape(1, 80, -1).astype(np.float32)
        mel_len = mel_input.shape[2]

        # 1. F0 prediction (ONNX)
        f0 = self.f0_session.run(None, {"mel": mel_input})[0]  # [1, mel_len]

        # 2. Upsample f0 by total_upsample * hop_len = 480
        upsample_factor = self.cfg["total_upsample"] * self.cfg["hop_len"]
        f0_up = np.repeat(f0[:, :, np.newaxis], upsample_factor, axis=1)  # simple nearest
        # Actually need to match torch.nn.Upsample: scale f0 [1, 1, mel_len] → [1, 1, mel_len*480]
        f0_1d = f0.reshape(1, 1, mel_len)
        # Linear interpolation to match torch Upsample
        audio_len = mel_len * upsample_factor
        x_old = np.linspace(0, 1, mel_len, dtype=np.float32)
        x_new = np.linspace(0, 1, audio_len, dtype=np.float32)
        f0_upsampled = np.interp(x_new, x_old, f0_1d[0, 0]).reshape(1, audio_len, 1).astype(np.float32)

        # 3. Source generation (numpy)
        source = self._source_module(f0_upsampled)  # [1, 1, audio_len]

        # 4. STFT (numpy)
        source_stft = self._stft(source)  # [1, 18, stft_len]

        # 5. Decode CNN (ONNX)
        raw_output = self.decode_session.run(
            None,
            {"mel": mel_input, "source_stft": source_stft}
        )[0]  # [1, 18, out_len]

        n_fft = self.cfg["n_fft"]
        # 6. Post-process: exp for magnitude, sin for phase
        magnitude = np.exp(raw_output[:, :n_fft // 2 + 1, :])  # [1, 9, T]
        phase = np.sin(raw_output[:, n_fft // 2 + 1:, :])       # [1, 9, T]

        # 7. ISTFT (numpy)
        audio = self._istft(magnitude, phase)

        # 8. Clamp
        audio = np.clip(audio, -self.cfg["audio_limit"], self.cfg["audio_limit"])

        return audio


# ============================================================
# Tokenizer
# ============================================================

class CosyVoiceTokenizer:
    """Simple Qwen2 tokenizer wrapper."""

    def __init__(self, tokenizer_dir):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        print(f"  Tokenizer loaded: vocab_size={self.tokenizer.vocab_size}")

    def encode(self, text):
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False)


# ============================================================
# Main pipeline
# ============================================================

def _load_txt_values(filepath, dtype=float):
    """Load values from text file (newline or comma separated)."""
    data = open(filepath).read().strip()
    sep = '\n' if '\n' in data else ','
    return [dtype(x) for x in data.split(sep)]


def load_prompt_dir(prompt_dir):
    """Load voice cloning prompt data from AX650N-format directory.

    Returns dict with: prompt_text_tokens, prompt_speech_tokens,
                       prompt_speech_feat, speaker_embedding
    """
    result = {}

    # prompt_text.txt — text token IDs (prepended to user text)
    pt_path = os.path.join(prompt_dir, "prompt_text.txt")
    if os.path.exists(pt_path):
        result["prompt_text_tokens"] = _load_txt_values(pt_path, int)
        print(f"  prompt_text: {len(result['prompt_text_tokens'])} tokens")

    # llm_prompt_speech_token.txt — speech tokens for LLM prefix
    st_path = os.path.join(prompt_dir, "llm_prompt_speech_token.txt")
    if os.path.exists(st_path):
        result["prompt_speech_tokens"] = _load_txt_values(st_path, int)
        print(f"  prompt_speech: {len(result['prompt_speech_tokens'])} tokens")

    # flow_prompt_speech_token.txt — speech tokens for Flow (usually same as LLM)
    fst_path = os.path.join(prompt_dir, "flow_prompt_speech_token.txt")
    if os.path.exists(fst_path):
        result["flow_prompt_speech_tokens"] = _load_txt_values(fst_path, int)

    # prompt_speech_feat.txt — mel spectrogram features [N*80] → reshape to [N, 80]
    feat_path = os.path.join(prompt_dir, "prompt_speech_feat.txt")
    if os.path.exists(feat_path):
        vals = _load_txt_values(feat_path, float)
        n_frames = len(vals) // 80
        result["prompt_speech_feat"] = np.array(vals[:n_frames * 80], dtype=np.float32).reshape(n_frames, 80)
        print(f"  prompt_speech_feat: {result['prompt_speech_feat'].shape}")

    # flow_embedding.txt or llm_embedding.txt — 192-dim speaker embedding
    emb_path = os.path.join(prompt_dir, "flow_embedding.txt")
    if not os.path.exists(emb_path):
        emb_path = os.path.join(prompt_dir, "llm_embedding.txt")
    if os.path.exists(emb_path):
        result["speaker_embedding"] = np.array(_load_txt_values(emb_path, float), dtype=np.float32)
        print(f"  speaker_embedding: {result['speaker_embedding'].shape}")

    # Also support .npy files (pre-converted for faster loading)
    for npy_name, key in [
        ("prompt_speech_tokens.npy", "prompt_speech_tokens"),
        ("prompt_speech_feat.npy", "prompt_speech_feat"),
        ("speaker_embedding.npy", "speaker_embedding"),
        ("prompt_text_tokens.npy", "prompt_text_tokens"),
    ]:
        npy_path = os.path.join(prompt_dir, npy_name)
        if os.path.exists(npy_path) and key not in result:
            arr = np.load(npy_path)
            if key in ("prompt_speech_tokens", "prompt_text_tokens"):
                result[key] = arr.tolist()
            else:
                result[key] = arr.astype(np.float32)
            print(f"  {key} (npy): {arr.shape}")

    return result


def main():
    parser = argparse.ArgumentParser(description="CosyVoice3 TTS on RK3588 NPU")
    parser.add_argument("--base_dir", default=".", help="Base dir with all model files")
    parser.add_argument("--llm_model", default=None, help="Path to .rkllm or .onnx LLM model")
    parser.add_argument("--embeddings_dir", default=None, help="Dir with embed_tokens.npy etc.")
    parser.add_argument("--flow_components_dir", default=None, help="Dir with flow front-end weights")
    parser.add_argument("--flow_estimator_onnx", default=None, help="Path to flow estimator ONNX")
    parser.add_argument("--hift_components_dir", default=None, help="Dir with HiFT ONNX + weights")
    parser.add_argument("--tokenizer_dir", default=None, help="Dir with tokenizer files")
    parser.add_argument("--text", default="Привет, как дела?", help="Text to synthesize")
    parser.add_argument("--flow_rknn", default=None, help="Flow RKNN model(s): comma-separated paths, 'auto' (seq200+seq1000), 'auto:vc' (seq500+seq1000), or 'auto:small' (seq200+seq500)")
    parser.add_argument("--prompt_dir", default=None, help="Dir with voice cloning prompt files (AX650N format)")
    parser.add_argument("--prompt_speech_tokens", default=None, help="Path to .npy with prompt speech tokens")
    parser.add_argument("--prompt_mel", default=None, help="Path to .npy with prompt mel features")
    parser.add_argument("--speaker_embedding", default=None, help="Path to .npy with speaker embedding")
    parser.add_argument("--no_buffer", action="store_true", help="Disable buffer phrase")
    parser.add_argument("--output", default="output.wav", help="Output WAV file")
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--save_intermediates", action="store_true",
                        help="Save intermediate .npy files (tokens, mel) to /tmp/ for diagnostics")
    parser.add_argument("--force_cpu_flow", action="store_true",
                        help="Force ONNX CPU for Flow even when RKNN models are loaded")
    parser.add_argument("--ode_steps", type=int, default=None,
                        help="Override number of ODE solver steps (default: from flow_config.json)")
    parser.add_argument("--padding_mode", default="zero",
                        choices=["reflect", "zero", "edge"],
                        help="Padding mode for RKNN fixed-size inputs (default: zero)")
    parser.add_argument("--load_speech_tokens", default=None,
                        help="Load speech tokens from .npy file (skip LLM generation)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible noise in Flow ODE")
    parser.add_argument("--mel_offset", type=float, default=0.0,
                        help="Offset to add to mel spectrogram (compensate RKNN NPU bias)")
    args = parser.parse_args()

    # Resolve paths relative to base_dir
    bd = args.base_dir
    # Auto-detect LLM model: check for ONNX first, then RKLLM
    if args.llm_model:
        llm_model = args.llm_model
    else:
        onnx_llm = os.path.join(bd, "cosyvoice3-llm-onnx", "qwen2_transformer.onnx")
        rkllm_default = os.path.join(bd, "cosyvoice3_llm_rk3588.rkllm")
        llm_model = rkllm_default  # default to RKLLM
    embeddings_dir = args.embeddings_dir or os.path.join(bd, "cosyvoice3_embeddings")
    flow_components_dir = args.flow_components_dir or os.path.join(bd, "cosyvoice3-flow-components")
    flow_estimator_onnx = args.flow_estimator_onnx or os.path.join(bd, "flow.decoder.estimator.fp32.onnx")
    hift_components_dir = args.hift_components_dir or os.path.join(bd, "cosyvoice3-hift-components")
    tokenizer_dir = args.tokenizer_dir or os.path.join(bd, "cosyvoice3_qwen2_for_rkllm")

    print("=" * 60)
    print("CosyVoice3 TTS on RK3588 NPU (no PyTorch)")
    print("=" * 60)

    # Load components
    print("\n--- Loading components ---")

    t0 = time.time()
    tokenizer = CosyVoiceTokenizer(tokenizer_dir)
    print(f"  Tokenizer: {time.time()-t0:.1f}s")

    t0 = time.time()
    if llm_model.endswith(".onnx"):
        llm = CosyVoiceLLM_ONNX(llm_model, embeddings_dir)
        print(f"  LLM (ONNX CPU): {time.time()-t0:.1f}s")
    else:
        llm = CosyVoiceLLM(llm_model, embeddings_dir)
        print(f"  LLM (RKLLM): {time.time()-t0:.1f}s")

    # Resolve RKNN model paths
    flow_rknn_paths = []
    if args.flow_rknn:
        if args.flow_rknn.lower().startswith('auto'):
            import glob as globmod
            import re as remod
            # Prefer opt3 models (better quality), fall back to any seq*.rknn
            opt3_paths = sorted(globmod.glob(os.path.join(bd, "flow_estimator_seq*_opt3.rknn")))
            if opt3_paths:
                all_paths = opt3_paths
            else:
                all_paths = sorted(globmod.glob(os.path.join(bd, "flow_estimator_seq*.rknn")))
            def _get_seq(p):
                m = remod.search(r'seq(\d+)', os.path.basename(p))
                return int(m.group(1)) if m else 0
            seq_map = {_get_seq(p): p for p in all_paths}
            seqs = sorted(seq_map.keys())
            # NPU memory fits ~2 models. Profiles select best pair:
            #   auto      = smallest + largest (general: fast short + all lengths)
            #   auto:vc   = mid + largest (voice cloning: better mid-range)
            #   auto:small = smallest + mid (no long text, fast short+mid)
            profile = args.flow_rknn.split(':')[1] if ':' in args.flow_rknn else ''
            if len(seqs) <= 2:
                flow_rknn_paths = [seq_map[s] for s in seqs]
            elif profile == 'vc' and len(seqs) >= 3:
                # Pick second smallest + largest
                pick = [seqs[1], seqs[-1]]
                flow_rknn_paths = [seq_map[s] for s in pick]
            elif profile == 'small':
                # Pick two smallest
                pick = seqs[:2]
                flow_rknn_paths = [seq_map[s] for s in pick]
            else:
                # Default: smallest + largest
                pick = [seqs[0], seqs[-1]]
                flow_rknn_paths = [seq_map[s] for s in pick]
            names = [os.path.basename(p) for p in flow_rknn_paths]
            print(f"  Auto-detected {len(all_paths)} RKNN models, selected {len(names)}: {', '.join(names)}")
        else:
            for p in args.flow_rknn.split(','):
                p = p.strip()
                if not os.path.isabs(p):
                    p = os.path.join(bd, p)
                flow_rknn_paths.append(p)
    t0 = time.time()
    flow = FlowDecoder(flow_components_dir, flow_estimator_onnx, rknn_paths=flow_rknn_paths or None)
    print(f"  Flow decoder: {time.time()-t0:.1f}s")

    t0 = time.time()
    hift = HiFTVocoder(hift_components_dir)
    print(f"  HiFT vocoder: {time.time()-t0:.1f}s")

    # Load prompt data (voice cloning)
    prompt_text_tokens = []
    prompt_speech_tokens = None
    flow_prompt_speech_tokens = None
    prompt_mel = np.zeros((0, 80), dtype=np.float32)
    speaker_embedding = np.zeros(192, dtype=np.float32)

    if args.prompt_dir:
        print(f"\n--- Loading voice cloning prompts ---")
        print(f"  Dir: {args.prompt_dir}")
        pdata = load_prompt_dir(args.prompt_dir)
        prompt_text_tokens = pdata.get("prompt_text_tokens", [])
        prompt_speech_tokens = pdata.get("prompt_speech_tokens")
        flow_prompt_speech_tokens = pdata.get("flow_prompt_speech_tokens", prompt_speech_tokens)
        if "prompt_speech_feat" in pdata:
            prompt_mel = pdata["prompt_speech_feat"]
        if "speaker_embedding" in pdata:
            speaker_embedding = pdata["speaker_embedding"]

    # Legacy individual file args (override prompt_dir if specified)
    if args.prompt_speech_tokens:
        prompt_speech_tokens = np.load(args.prompt_speech_tokens).tolist()
    if args.prompt_mel:
        prompt_mel = np.load(args.prompt_mel).astype(np.float32)
    if args.speaker_embedding:
        speaker_embedding = np.load(args.speaker_embedding).astype(np.float32).flatten()

    has_voice_cloning = prompt_speech_tokens is not None and len(prompt_speech_tokens) > 0

    # Tokenize text
    print(f"\n--- Tokenizing ---")
    text_tokens_user = tokenizer.encode(args.text)

    # Buffer phrase (optional, for RKLLM which may truncate without guidance)
    use_buffer = not args.no_buffer and not llm_model.endswith(".onnx")
    BUFFER_PHRASE = " Вот так."
    if use_buffer:
        text_tokens_user_with_buffer = tokenizer.encode(args.text + BUFFER_PHRASE)
        n_buffer_tokens = len(text_tokens_user_with_buffer) - len(text_tokens_user)
        text_tokens_for_llm = text_tokens_user_with_buffer
        print(f"  Text: '{args.text}' (+buffer '{BUFFER_PHRASE}')")
    else:
        n_buffer_tokens = 0
        text_tokens_for_llm = text_tokens_user
        print(f"  Text: '{args.text}'")

    # Prepend prompt text tokens (voice cloning prefix)
    if prompt_text_tokens:
        all_text_tokens = prompt_text_tokens + text_tokens_for_llm
        print(f"  Prompt text: {len(prompt_text_tokens)} tokens + User text: {len(text_tokens_for_llm)} tokens = {len(all_text_tokens)}")
    else:
        all_text_tokens = text_tokens_for_llm
        print(f"  Tokens ({len(all_text_tokens)}): {all_text_tokens[:20]}...")

    if has_voice_cloning:
        print(f"  Voice cloning: ON ({len(prompt_speech_tokens)} prompt speech tokens, {prompt_mel.shape[0]} mel frames)")
    else:
        print(f"  Voice cloning: OFF (zero speaker embedding)")

    # Token limits
    n_user_text = len(text_tokens_user)
    if use_buffer and n_buffer_tokens > 0:
        # RKLLM: use min/max to compensate for quantization instability
        effective_min = max(10, n_user_text * 4)
        text_based_max = n_user_text * 8
        effective_max = min(args.max_tokens, max(50, text_based_max))
    else:
        # ONNX FP32: no artificial limits, trust the model
        effective_min = 10
        effective_max = args.max_tokens
    print(f"  Token limits: min={effective_min}, max={effective_max} (text_len={n_user_text})")

    # Phase 1: LLM — generate speech tokens
    if args.load_speech_tokens:
        print(f"\n--- Phase 1: LLM (SKIPPED — loading from {args.load_speech_tokens}) ---")
        t_llm = 0.0
        speech_tokens = np.load(args.load_speech_tokens).tolist()
        n_tokens = len(speech_tokens)
        print(f"  Loaded {n_tokens} speech tokens")
        print(f"  First 10: {speech_tokens[:10]}")
    else:
        llm_backend = "ONNX CPU" if llm_model.endswith(".onnx") else "RKLLM NPU"
        print(f"\n--- Phase 1: LLM ({llm_backend}) ---")
        t_llm = time.time()
        speech_tokens = llm.generate_tokens(
            all_text_tokens,
            prompt_speech_tokens=prompt_speech_tokens,
            max_tokens=effective_max,
            min_tokens=effective_min,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        t_llm = time.time() - t_llm
        n_tokens = len(speech_tokens)
        print(f"  Generated {n_tokens} tokens in {t_llm:.1f}s ({n_tokens/t_llm:.1f} tok/s)")
        print(f"  First 10: {speech_tokens[:10]}")

    if n_tokens == 0:
        print("ERROR: No speech tokens generated!")
        llm.destroy()
        return

    # Save intermediate tokens (AFTER trimming)
    if args.save_intermediates:
        np.save('/tmp/rknn_speech_tokens.npy', np.array(speech_tokens))
        print(f"  Saved /tmp/rknn_speech_tokens.npy ({n_tokens} tokens)")

    # Phase 2: Flow — speech tokens → mel spectrogram
    flow_backend = "ONNX CPU (forced)" if args.force_cpu_flow else ("RKNN NPU" if flow.rknn_models else "ONNX CPU")
    ode_str = f", ode_steps={args.ode_steps}" if args.ode_steps else ""
    print(f"\n--- Phase 2: Flow ({flow_backend}, padding={args.padding_mode}{ode_str}) ---")
    # Use flow-specific prompt tokens if available, otherwise same as LLM
    flow_pt = flow_prompt_speech_tokens or (prompt_speech_tokens if prompt_speech_tokens else [])
    flow_prompt_array = np.array(flow_pt, dtype=np.int64)
    t_flow = time.time()
    mel = flow.inference(
        np.array(speech_tokens),
        flow_prompt_array,
        prompt_mel,
        speaker_embedding,
        force_cpu=args.force_cpu_flow,
        ode_steps=args.ode_steps,
        padding_mode=args.padding_mode,
        seed=args.seed,
    )
    t_flow = time.time() - t_flow
    print(f"  Mel: {mel.shape}, range [{mel.min():.3f}, {mel.max():.3f}] in {t_flow:.1f}s")

    # Apply mel offset correction (compensate RKNN NPU systematic bias)
    if args.mel_offset != 0.0:
        mel = mel + args.mel_offset
        print(f"  Applied mel_offset={args.mel_offset:+.2f}, new range [{mel.min():.3f}, {mel.max():.3f}]")

    # Save intermediate mel
    if args.save_intermediates:
        suffix = "cpu" if args.force_cpu_flow else "npu"
        mel_path = f'/tmp/rknn_mel_{suffix}.npy'
        np.save(mel_path, mel)
        print(f"  Saved {mel_path}")

    # Phase 3: HiFT — mel → audio
    print(f"\n--- Phase 3: HiFT vocoder (CPU) ---")
    t_hift = time.time()
    audio = hift.inference(mel)
    t_hift = time.time() - t_hift
    audio_dur = len(audio) / SAMPLE_RATE
    print(f"  Audio: {len(audio)} samples ({audio_dur:.2f}s) in {t_hift:.1f}s")

    # Post-processing: Butterworth LP filter + fade-out
    from scipy.signal import butter, sosfilt
    cutoff_hz = 5000
    sos = butter(8, cutoff_hz, btype='low', fs=SAMPLE_RATE, output='sos')
    audio = sosfilt(sos, audio).astype(np.float32)
    # Fade-out last 50ms
    fade_len = int(SAMPLE_RATE * 0.05)
    if len(audio) > fade_len:
        audio[-fade_len:] *= np.linspace(1, 0, fade_len, dtype=np.float32)
    audio = np.clip(audio, -0.99, 0.99)

    # Save WAV
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wavmod.open(args.output, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    print(f"  Saved: {args.output}")

    # Summary
    total = t_llm + t_flow + t_hift
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Text: '{args.text}'")
    llm_speed = f"{n_tokens/t_llm:.1f} tok/s" if t_llm > 0 else "skipped"
    print(f"LLM:   {n_tokens} tokens in {t_llm:.1f}s ({llm_speed})")
    print(f"Flow:  mel {mel.shape} in {t_flow:.1f}s")
    print(f"HiFT:  {audio_dur:.2f}s audio in {t_hift:.1f}s")
    print(f"Total: {total:.1f}s for {audio_dur:.1f}s audio (RTF={total/audio_dur:.1f}x)")
    print(f"Output: {args.output}")

    llm.destroy()


if __name__ == "__main__":
    main()
