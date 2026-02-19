#!/usr/bin/env python3
"""
Compare CosyVoice3 speech token quality: RKLLM (quantized) vs ONNX FP32 (reference).

Runs on CM3588 with both backends. Compares:
1. Hidden state cosine similarity per step (quantization error)
2. Logit distribution KL-divergence per step
3. Token sequences: edit distance, length, repetitions
4. Greedy-decoded tokens (deterministic comparison)
5. Optionally: mel spectrogram comparison via Flow+HiFT

Usage on CM3588:
  python3 compare_token_quality.py \
      --rkllm_model /root/cosyvoice3-rknn/cosyvoice3_llm_rk3588_w8a16.rkllm \
      --onnx_model /root/cosyvoice3-rknn/cosyvoice3-llm-onnx/qwen2_transformer.onnx \
      --embeddings /root/cosyvoice3-rknn/cosyvoice3_embeddings \
      --tokenizer /root/cosyvoice3-rknn/cosyvoice3_qwen2_for_rkllm \
      --prompt_dir /root/cosyvoice3-rknn/prompt_russian_v2 \
      --text "Привет, как дела? Сегодня хорошая погода."
"""

import os
import sys
import argparse
import time
import numpy as np

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


SPEECH_TOKEN_SIZE = 6561
HIDDEN_SIZE = 896


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def kl_divergence(logits_p, logits_q):
    """KL(P||Q) where P=reference (ONNX), Q=test (RKLLM)."""
    # Softmax
    p = np.exp(logits_p - logits_p.max())
    p /= p.sum()
    q = np.exp(logits_q - logits_q.max())
    q /= q.sum()
    # Avoid log(0)
    q = np.clip(q, 1e-10, None)
    p = np.clip(p, 1e-10, None)
    return float(np.sum(p * np.log(p / q)))


def edit_distance(seq1, seq2):
    """Levenshtein edit distance."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def count_repetitions(tokens, window=10):
    """Count how many tokens repeat within a sliding window."""
    reps = 0
    for i in range(1, len(tokens)):
        start = max(0, i - window)
        if tokens[i] in tokens[start:i]:
            reps += 1
    return reps


def greedy_decode(hidden_state, llm_decoder_weight):
    """Greedy (argmax) token selection — deterministic."""
    logits = hidden_state @ llm_decoder_weight.T
    return int(np.argmax(logits))


def generate_step_by_step(llm, text_token_ids, prompt_speech_tokens,
                          max_tokens=200, min_tokens=10, temperature=1.0, top_k=25):
    """Generate tokens step by step, returning hidden states and logits at each step.

    Returns:
        tokens: list of int (greedy-decoded for deterministic comparison)
        hiddens: list of np.array [hidden_size] per step
        logits_list: list of np.array [vocab_size] per step
    """
    prefix = llm.build_prefix(text_token_ids, prompt_speech_tokens)
    hidden = llm.get_hidden(prefix, keep_history=0)
    if hidden is None:
        raise RuntimeError("No hidden states from prefill")

    tokens = []
    hiddens = []
    logits_list = []

    for i in range(max_tokens):
        h = hidden[-1]  # [hidden_size]
        hiddens.append(h.copy())

        # Compute logits (before temperature/sampling)
        raw_logits = h @ llm.llm_decoder_weight.T  # [vocab_size]
        logits_list.append(raw_logits.copy())

        # Greedy decode for deterministic comparison
        token = int(np.argmax(raw_logits))

        # Check EOS
        if token >= SPEECH_TOKEN_SIZE and i >= min_tokens:
            break

        # If EOS before min_tokens, pick best non-EOS
        if token >= SPEECH_TOKEN_SIZE:
            valid_logits = raw_logits[:SPEECH_TOKEN_SIZE]
            token = int(np.argmax(valid_logits))

        tokens.append(token)

        # Feed back
        next_emb = llm.speech_embedding[token].reshape(1, HIDDEN_SIZE).astype(np.float32)
        hidden = llm.get_hidden(next_emb, keep_history=1)
        if hidden is None:
            break

    return tokens, hiddens, logits_list


def generate_sampled(llm, text_token_ids, prompt_speech_tokens,
                     max_tokens=200, min_tokens=10, temperature=0.6, top_k=15,
                     seed=42):
    """Generate with nucleus sampling (stochastic), for audio quality test."""
    np.random.seed(seed)
    return llm.generate_tokens(
        text_token_ids, prompt_speech_tokens,
        max_tokens=max_tokens, min_tokens=min_tokens,
        temperature=temperature, top_k=top_k
    )


def compare_results(ref_tokens, ref_hiddens, ref_logits,
                    test_tokens, test_hiddens, test_logits,
                    label="RKLLM"):
    """Compare reference (ONNX) vs test (RKLLM) results."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: ONNX FP32 (reference) vs {label}")
    print(f"{'='*60}")

    # Token counts
    print(f"\n--- Token Counts ---")
    print(f"  ONNX:  {len(ref_tokens)} tokens")
    print(f"  {label}: {len(test_tokens)} tokens")
    print(f"  Difference: {len(test_tokens) - len(ref_tokens)}")

    # Token match
    min_len = min(len(ref_tokens), len(test_tokens))
    matches = sum(1 for i in range(min_len) if ref_tokens[i] == test_tokens[i])
    print(f"\n--- Token Match (greedy, first {min_len} steps) ---")
    print(f"  Exact matches: {matches}/{min_len} ({100*matches/max(min_len,1):.1f}%)")
    print(f"  Edit distance: {edit_distance(ref_tokens, test_tokens)}")

    # First divergence
    for i in range(min_len):
        if ref_tokens[i] != test_tokens[i]:
            print(f"  First divergence at step {i}: ONNX={ref_tokens[i]}, {label}={test_tokens[i]}")
            break

    # Repetitions
    ref_reps = count_repetitions(ref_tokens)
    test_reps = count_repetitions(test_tokens)
    print(f"\n--- Repetitions (window=10) ---")
    print(f"  ONNX:  {ref_reps} ({100*ref_reps/max(len(ref_tokens),1):.1f}%)")
    print(f"  {label}: {test_reps} ({100*test_reps/max(len(test_tokens),1):.1f}%)")

    # Hidden state comparison
    n_compare = min(len(ref_hiddens), len(test_hiddens))
    print(f"\n--- Hidden State Cosine Similarity ({n_compare} steps) ---")
    cosines = []
    for i in range(n_compare):
        cs = cosine_similarity(ref_hiddens[i], test_hiddens[i])
        cosines.append(cs)

    if cosines:
        print(f"  Step 0 (prefill):  {cosines[0]:.6f}")
        if len(cosines) > 5:
            print(f"  Step 5:            {cosines[5]:.6f}")
        if len(cosines) > 10:
            print(f"  Step 10:           {cosines[10]:.6f}")
        if len(cosines) > 20:
            print(f"  Step 20:           {cosines[20]:.6f}")
        print(f"  Last ({n_compare-1}):       {cosines[-1]:.6f}")
        print(f"  Mean:              {np.mean(cosines):.6f}")
        print(f"  Min:               {np.min(cosines):.6f}")

        # Check for drift
        if len(cosines) > 10:
            early = np.mean(cosines[:5])
            late = np.mean(cosines[-5:])
            drift = early - late
            print(f"  Drift (early-late): {drift:.6f} {'(DRIFTING!)' if drift > 0.01 else '(stable)'}")

    # KL divergence of logit distributions
    n_kl = min(len(ref_logits), len(test_logits))
    print(f"\n--- Logit KL-Divergence ({n_kl} steps) ---")
    kls = []
    for i in range(n_kl):
        kl = kl_divergence(ref_logits[i], test_logits[i])
        kls.append(kl)

    if kls:
        print(f"  Step 0:  {kls[0]:.4f}")
        if len(kls) > 5:
            print(f"  Step 5:  {kls[5]:.4f}")
        if len(kls) > 10:
            print(f"  Step 10: {kls[10]:.4f}")
        print(f"  Mean:    {np.mean(kls):.4f}")
        print(f"  Max:     {np.max(kls):.4f}")

        # Interpret
        mean_kl = np.mean(kls)
        if mean_kl < 0.01:
            verdict = "EXCELLENT — nearly identical distributions"
        elif mean_kl < 0.1:
            verdict = "GOOD — minor quantization noise"
        elif mean_kl < 1.0:
            verdict = "FAIR — noticeable distribution shift"
        else:
            verdict = "POOR — significant distribution divergence"
        print(f"  Verdict: {verdict}")

    # Top-1 agreement (would greedy produce same token?)
    top1_agree = sum(1 for i in range(n_kl)
                     if np.argmax(ref_logits[i]) == np.argmax(test_logits[i]))
    print(f"\n--- Top-1 Agreement ---")
    print(f"  {top1_agree}/{n_kl} ({100*top1_agree/max(n_kl,1):.1f}%)")

    # Overall verdict
    print(f"\n{'='*60}")
    print(f"OVERALL VERDICT")
    print(f"{'='*60}")
    quality_ok = True
    issues = []
    if cosines and np.mean(cosines) < 0.90:
        issues.append(f"Hidden similarity low ({np.mean(cosines):.4f})")
        quality_ok = False
    if kls and np.mean(kls) > 0.5:
        issues.append(f"KL divergence high ({np.mean(kls):.4f})")
        quality_ok = False
    if test_reps > ref_reps * 2 and test_reps > 5:
        issues.append(f"Excessive repetitions ({test_reps} vs {ref_reps})")
        quality_ok = False
    if abs(len(test_tokens) - len(ref_tokens)) > len(ref_tokens) * 0.3:
        issues.append(f"Token count differs significantly ({len(test_tokens)} vs {len(ref_tokens)})")
        quality_ok = False

    if quality_ok:
        print("  PASS — RKLLM quality appears acceptable for dual-NPU")
    else:
        print("  FAIL — Issues detected:")
        for issue in issues:
            print(f"    - {issue}")

    return {
        "cosine_mean": float(np.mean(cosines)) if cosines else 0,
        "kl_mean": float(np.mean(kls)) if kls else 999,
        "top1_agree_pct": 100 * top1_agree / max(n_kl, 1),
        "edit_distance": edit_distance(ref_tokens, test_tokens),
        "ref_tokens": len(ref_tokens),
        "test_tokens": len(test_tokens),
        "ref_reps": ref_reps,
        "test_reps": test_reps,
        "quality_ok": quality_ok,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare RKLLM vs ONNX token quality")
    parser.add_argument("--rkllm_model", required=True, help="Path to .rkllm model")
    parser.add_argument("--onnx_model", required=True, help="Path to .onnx transformer model")
    parser.add_argument("--embeddings", required=True, help="Dir with embed_tokens.npy etc.")
    parser.add_argument("--tokenizer", required=True, help="Dir with Qwen2 tokenizer files")
    parser.add_argument("--prompt_dir", default=None, help="Dir with voice cloning prompt data")
    parser.add_argument("--text", default="Привет, как дела? Сегодня хорошая погода для прогулки.",
                        help="Text to synthesize")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--save_tokens", action="store_true", help="Save token files for audio comparison")
    args = parser.parse_args()

    # Lazy imports (heavy libraries)
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Prepare text tokens
    # CosyVoice3 format: system prompt + text
    system_prompt = "You are a helpful assistant."
    prompt_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
    full_text = prompt_text + args.text
    text_token_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Load prompt data (supports both .npy and .txt formats)
    prompt_speech_tokens = None
    prompt_text_tokens = None
    if args.prompt_dir:
        # Speech tokens
        for name in ["prompt_speech_tokens.npy", "llm_prompt_speech_token.txt"]:
            path = os.path.join(args.prompt_dir, name)
            if os.path.exists(path):
                if path.endswith('.npy'):
                    prompt_speech_tokens = np.load(path).astype(int).tolist()
                else:
                    data = open(path).read().strip()
                    sep = '\n' if '\n' in data else ','
                    prompt_speech_tokens = [int(x) for x in data.split(sep)]
                print(f"Prompt speech tokens: {len(prompt_speech_tokens)}")
                break
        # Text tokens (prepend to user text)
        for name in ["prompt_text_tokens.npy", "prompt_text.txt"]:
            path = os.path.join(args.prompt_dir, name)
            if os.path.exists(path):
                if path.endswith('.npy'):
                    prompt_text_tokens = np.load(path).astype(int).tolist()
                else:
                    data = open(path).read().strip()
                    sep = '\n' if '\n' in data else ','
                    prompt_text_tokens = [int(x) for x in data.split(sep)]
                print(f"Prompt text tokens: {len(prompt_text_tokens)}")
                break

    # Prepend prompt text tokens if available
    if prompt_text_tokens:
        text_token_ids = prompt_text_tokens + text_token_ids
        print(f"Prepended {len(prompt_text_tokens)} prompt text tokens")

    print(f"Text: '{args.text}'")
    print(f"Total text tokens: {len(text_token_ids)}")
    if prompt_speech_tokens:
        print(f"Prompt speech tokens: {len(prompt_speech_tokens)}")

    # =========================================================
    # Phase 1: ONNX FP32 reference
    # =========================================================
    print(f"\n{'='*60}")
    print("Phase 1: ONNX FP32 (reference)")
    print(f"{'='*60}")

    # Import from the pipeline module (check multiple locations)
    for candidate in [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tts-rknn-convert', 'cosyvoice3-rknn'),
        '/root/cosyvoice3-rknn',
    ]:
        if os.path.exists(os.path.join(candidate, 'cosyvoice3_rknn_pipeline.py')):
            sys.path.insert(0, candidate)
            break
    from cosyvoice3_rknn_pipeline import CosyVoiceLLM_ONNX, CosyVoiceLLM

    print("Loading ONNX model...")
    t0 = time.time()
    onnx_llm = CosyVoiceLLM_ONNX(args.onnx_model, args.embeddings)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("\nGenerating (greedy, step-by-step)...")
    t0 = time.time()
    ref_tokens, ref_hiddens, ref_logits = generate_step_by_step(
        onnx_llm, text_token_ids, prompt_speech_tokens,
        max_tokens=args.max_tokens
    )
    t_onnx = time.time() - t0
    print(f"  {len(ref_tokens)} tokens in {t_onnx:.1f}s ({len(ref_tokens)/t_onnx:.1f} tok/s)")

    # Also generate sampled tokens for audio test
    if args.save_tokens:
        print("\nGenerating (sampled, for audio)...")
        ref_sampled = generate_sampled(
            onnx_llm, text_token_ids, prompt_speech_tokens,
            max_tokens=args.max_tokens, temperature=args.temperature,
            top_k=args.top_k, seed=42
        )
        np.save('/tmp/onnx_tokens_sampled.npy', np.array(ref_sampled))
        print(f"  Saved /tmp/onnx_tokens_sampled.npy ({len(ref_sampled)} tokens)")

    onnx_llm.destroy()

    # =========================================================
    # Phase 2: RKLLM (quantized)
    # =========================================================
    print(f"\n{'='*60}")
    print(f"Phase 2: RKLLM ({os.path.basename(args.rkllm_model)})")
    print(f"{'='*60}")

    print("Loading RKLLM model...")
    t0 = time.time()
    rkllm = CosyVoiceLLM(args.rkllm_model, args.embeddings)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("\nGenerating (greedy, step-by-step)...")
    t0 = time.time()
    test_tokens, test_hiddens, test_logits = generate_step_by_step(
        rkllm, text_token_ids, prompt_speech_tokens,
        max_tokens=args.max_tokens
    )
    t_rkllm = time.time() - t0
    print(f"  {len(test_tokens)} tokens in {t_rkllm:.1f}s ({len(test_tokens)/t_rkllm:.1f} tok/s)")

    # Also generate sampled tokens
    if args.save_tokens:
        print("\nGenerating (sampled, for audio)...")
        test_sampled = generate_sampled(
            rkllm, text_token_ids, prompt_speech_tokens,
            max_tokens=args.max_tokens, temperature=args.temperature,
            top_k=args.top_k, seed=42
        )
        np.save('/tmp/rkllm_tokens_sampled.npy', np.array(test_sampled))
        print(f"  Saved /tmp/rkllm_tokens_sampled.npy ({len(test_sampled)} tokens)")

    rkllm.destroy()

    # =========================================================
    # Phase 3: Compare
    # =========================================================
    label = os.path.basename(args.rkllm_model).replace('.rkllm', '')
    results = compare_results(
        ref_tokens, ref_hiddens, ref_logits,
        test_tokens, test_hiddens, test_logits,
        label=label
    )

    # Speed comparison
    print(f"\n--- Speed ---")
    print(f"  ONNX FP32: {len(ref_tokens)/t_onnx:.1f} tok/s")
    print(f"  RKLLM:     {len(test_tokens)/t_rkllm:.1f} tok/s")
    print(f"  Speedup:   {t_onnx/t_rkllm:.1f}x")

    if args.save_tokens:
        print(f"\n--- Saved files for audio comparison ---")
        print(f"  /tmp/onnx_tokens_sampled.npy")
        print(f"  /tmp/rkllm_tokens_sampled.npy")
        print(f"  Run Flow+HiFT on each to compare audio quality")

    return 0 if results["quality_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
