#!/usr/bin/env python3
"""
Export CosyVoice3 HiFT components for PyTorch-free inference on CM3588.

Exports:
  1. f0_predictor → ONNX (dynamic mel_len)
  2. decode CNN → ONNX (dynamic mel_len)
  3. source_module weights → .npz (l_linear, SineGen2 pre-stored values)
  4. Config JSON

Usage:
    cd ~/CosyVoice3/repo
    python3 ~/NAS/tts-rknn-convert/cosyvoice3-rknn/export_hift_components.py
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

COSYVOICE_ROOT = os.path.expanduser("~/CosyVoice3/repo")
sys.path.insert(0, COSYVOICE_ROOT)

HIFT_PT = os.path.join(
    COSYVOICE_ROOT,
    "pretrained_models/Fun-CosyVoice3-0.5B/hift.pt"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cosyvoice3-hift-components"
)


class F0PredictorONNX(nn.Module):
    """Wrapper for CausalConvRNNF0Predictor that flattens causal padding for ONNX."""

    def __init__(self, f0_predictor):
        super().__init__()
        self.condnet = f0_predictor.condnet
        self.classifier = f0_predictor.classifier

    def forward(self, x):
        # finalize=True path: pad with zeros and forward
        # CausalConv1d.forward with empty cache → pad with causal_padding zeros
        for layer in self.condnet:
            if hasattr(layer, 'causal_padding'):
                # CausalConv1d: pad appropriately
                pad_size = layer.causal_padding
                if layer.causal_type == 'left':
                    x_padded = F.pad(x, (pad_size, 0), value=0.0)
                else:
                    x_padded = F.pad(x, (0, pad_size), value=0.0)
                # Call Conv1d.forward (parent class)
                x = nn.Conv1d.forward(layer, x_padded)
            else:
                x = layer(x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class DecodeCNNDynamic(nn.Module):
    """Decode CNN wrapper with dynamic shapes."""

    def __init__(self, hift):
        super().__init__()
        self.conv_pre = hift.conv_pre
        self.ups = hift.ups
        self.source_downs = hift.source_downs
        self.source_resblocks = hift.source_resblocks
        self.resblocks = hift.resblocks
        self.conv_post = hift.conv_post
        self.reflection_pad = hift.reflection_pad
        self.num_upsamples = hift.num_upsamples
        self.num_kernels = hift.num_kernels
        self.lrelu_slope = hift.lrelu_slope

    def forward(self, mel, source_stft):
        # Flatten CausalConv1d calls for ONNX compatibility
        # conv_pre is CausalConv1d with causal_type='right'
        pad = self.conv_pre.causal_padding
        x = F.pad(mel, (0, pad), value=0.0)
        x = nn.Conv1d.forward(self.conv_pre, x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)

            # CausalConv1dUpsample: upsample + pad + conv
            up = self.ups[i]
            x = up.upsample(x)
            up_pad = up.causal_padding
            x = F.pad(x, (up_pad, 0), value=0.0)
            x = nn.Conv1d.forward(up, x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # Source fusion - source_downs can be CausalConv1d or CausalConv1dDownSample
            sd = self.source_downs[i]
            if hasattr(sd, 'causal_type'):
                # CausalConv1d
                sd_pad = sd.causal_padding
                if sd.causal_type == 'left':
                    si = F.pad(source_stft, (sd_pad, 0), value=0.0)
                else:
                    si = F.pad(source_stft, (0, sd_pad), value=0.0)
                si = nn.Conv1d.forward(sd, si)
            else:
                # CausalConv1dDownSample
                sd_pad = sd.causal_padding
                si = F.pad(source_stft, (sd_pad, 0), value=0.0)
                si = nn.Conv1d.forward(sd, si)

            si = self._resblock_forward(self.source_resblocks[i], si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j]
                if xs is None:
                    xs = self._resblock_forward(rb, x)
                else:
                    xs = xs + self._resblock_forward(rb, x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        # conv_post is CausalConv1d with causal_type='left'
        post_pad = self.conv_post.causal_padding
        x = F.pad(x, (post_pad, 0), value=0.0)
        x = nn.Conv1d.forward(self.conv_post, x)
        return x

    def _resblock_forward(self, rb, x):
        """ResBlock forward with causal conv flattened."""
        for idx, (c1, c2) in enumerate(zip(rb.convs1, rb.convs2)):
            xt = rb.activations1[idx](x)  # Snake activation (NOT leaky_relu!)
            # c1: CausalConv1d with causal_type='left'
            p1 = c1.causal_padding
            xt = F.pad(xt, (p1, 0), value=0.0)
            xt = nn.Conv1d.forward(c1, xt)
            xt = rb.activations2[idx](xt)  # Snake activation (NOT leaky_relu!)
            # c2: CausalConv1d with causal_type='left'
            p2 = c2.causal_padding
            xt = F.pad(xt, (p2, 0), value=0.0)
            xt = nn.Conv1d.forward(c2, xt)
            x = xt + x
        return x


def load_hift():
    """Load CausalHiFTGenerator."""
    from cosyvoice.hifigan.generator import CausalHiFTGenerator
    from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor

    f0_predictor = CausalConvRNNF0Predictor(
        num_class=1, in_channels=80, cond_channels=512,
    )
    model = CausalHiFTGenerator(
        in_channels=80, base_channels=512, nb_harmonics=8,
        sampling_rate=24000, nsf_alpha=0.1, nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1, audio_limit=0.99, conv_pre_look_right=4,
        f0_predictor=f0_predictor,
    )
    state = torch.load(HIFT_PT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded HiFT: {len(state)} keys")
    return model


def remove_weight_norms(model):
    """Remove weight norm from all modules."""
    from torch.nn.utils.parametrize import remove_parametrizations
    count = 0
    for module in model.modules():
        if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations:
            remove_parametrizations(module, 'weight')
            count += 1
    print(f"  Removed weight norm from {count} modules")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mel_len", type=int, default=200)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hift = load_hift()
    remove_weight_norms(hift)

    # ===== 1. Export f0_predictor to ONNX =====
    print("\n=== 1. Exporting f0_predictor to ONNX ===")
    f0_onnx = F0PredictorONNX(hift.f0_predictor)
    f0_onnx.eval()

    mel_dummy = torch.randn(1, 80, args.mel_len)
    with torch.no_grad():
        f0_test = f0_onnx(mel_dummy)
    print(f"  Input: mel {list(mel_dummy.shape)}")
    print(f"  Output: f0 {list(f0_test.shape)}")

    f0_path = os.path.join(args.output_dir, "f0_predictor.onnx")
    torch.onnx.export(
        f0_onnx, mel_dummy, f0_path,
        input_names=["mel"],
        output_names=["f0"],
        dynamic_axes={"mel": {2: "mel_len"}, "f0": {1: "mel_len"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"  Saved: {f0_path} ({os.path.getsize(f0_path)/1024/1024:.1f} MB)")

    # ===== 2. Export decode CNN to ONNX (dynamic shapes) =====
    print("\n=== 2. Exporting decode CNN to ONNX (dynamic) ===")
    decode_cnn = DecodeCNNDynamic(hift)
    decode_cnn.eval()

    total_upsample = int(np.prod([8, 5, 3]))  # 120
    hop_len = 4
    stft_len = args.mel_len * total_upsample + 1  # = mel_len * 120 + 1
    source_stft_dummy = torch.randn(1, 18, stft_len)

    with torch.no_grad():
        raw_out = decode_cnn(mel_dummy, source_stft_dummy)
    print(f"  Input: mel {list(mel_dummy.shape)}, source_stft {list(source_stft_dummy.shape)}")
    print(f"  Output: raw {list(raw_out.shape)}")

    decode_path = os.path.join(args.output_dir, "hift_decode_dynamic.onnx")
    torch.onnx.export(
        decode_cnn, (mel_dummy, source_stft_dummy), decode_path,
        input_names=["mel", "source_stft"],
        output_names=["raw_output"],
        dynamic_axes={
            "mel": {2: "mel_len"},
            "source_stft": {2: "stft_len"},
            "raw_output": {2: "out_len"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"  Saved: {decode_path} ({os.path.getsize(decode_path)/1024/1024:.1f} MB)")

    # ===== 3. Export source module weights =====
    print("\n=== 3. Exporting source module weights ===")
    source = hift.m_source

    # SineGen2 parameters
    sinegen = source.l_sin_gen
    # NOTE: sinegen.rand_ini, sinegen.sine_waves, source.uv are NOT in state_dict.
    # They're random tensors created at init time. We can recreate them on CM3588.
    weights = {
        # Source linear merge: Linear(9, 1) — this IS trained
        "source_linear_weight": source.l_linear.weight.detach().numpy(),
        "source_linear_bias": source.l_linear.bias.detach().numpy(),
        # STFT window (Hann)
        "stft_window": hift.stft_window.detach().numpy(),
    }
    for k, v in weights.items():
        print(f"  {k}: {v.shape}")

    npz_path = os.path.join(args.output_dir, "source_weights.npz")
    np.savez(npz_path, **weights)
    print(f"  Saved: {npz_path} ({os.path.getsize(npz_path)/1024/1024:.1f} MB)")

    # ===== 4. Save config =====
    print("\n=== 4. Saving config ===")
    config = {
        "sampling_rate": int(hift.sampling_rate),
        "nb_harmonics": int(sinegen.harmonic_num),
        "sine_amp": float(sinegen.sine_amp),
        "noise_std": float(sinegen.noise_std),
        "voiced_threshold": float(sinegen.voiced_threshold),
        "upsample_scale": int(sinegen.upsample_scale),
        "n_fft": int(hift.istft_params["n_fft"]),
        "hop_len": int(hift.istft_params["hop_len"]),
        "audio_limit": float(hift.audio_limit),
        "total_upsample": int(total_upsample),
        "upsample_rates": [int(x) for x in hift.upsample_rates],
    }
    config_path = os.path.join(args.output_dir, "hift_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {config_path}")

    # ===== 5. Verify full pipeline =====
    print("\n=== 5. Verification ===")

    # Run full PyTorch inference for reference
    with torch.no_grad():
        audio_ref, source_ref = hift.inference(mel_dummy)
    print(f"  Reference audio: {list(audio_ref.shape)}")

    # Run our split pipeline
    with torch.no_grad():
        # f0
        f0 = f0_onnx(mel_dummy)
        print(f"  f0: {list(f0.shape)}, range [{f0.min():.1f}, {f0.max():.1f}]")

        # source (PyTorch reference)
        s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = hift.m_source(s)
        s = s.transpose(1, 2)

        # stft
        spec = torch.stft(s.squeeze(1), 16, 4, 16,
                          window=hift.stft_window, return_complex=True)
        spec = torch.view_as_real(spec)
        source_stft = torch.cat([spec[..., 0], spec[..., 1]], dim=1)
        print(f"  source_stft: {list(source_stft.shape)}")

        # decode CNN
        raw = decode_cnn(mel_dummy, source_stft)
        magnitude = torch.exp(raw[:, :9, :])
        phase = torch.sin(raw[:, 9:, :])

        # istft
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        audio_split = torch.istft(
            torch.complex(real, imag), 16, 4, 16,
            window=hift.stft_window
        )
        audio_split = torch.clamp(audio_split, -0.99, 0.99)

    diff = (audio_ref.squeeze() - audio_split.squeeze()).abs().max().item()
    print(f"  Max diff ref vs split: {diff:.6f}")

    # Save reference data
    np.save(os.path.join(args.output_dir, "test_mel.npy"), mel_dummy.numpy())
    np.save(os.path.join(args.output_dir, "test_audio_ref.npy"), audio_ref.numpy())
    np.save(os.path.join(args.output_dir, "test_f0.npy"), f0.numpy())
    np.save(os.path.join(args.output_dir, "test_source_stft.npy"), source_stft.numpy())

    # Summary
    print(f"\n{'='*60}")
    print(f"EXPORT SUMMARY")
    print(f"{'='*60}")
    for f_name in sorted(os.listdir(args.output_dir)):
        f_path = os.path.join(args.output_dir, f_name)
        size = os.path.getsize(f_path)
        print(f"  {f_name} ({size/1024/1024:.1f} MB)")
    print(f"\nPipeline on CM3588 (no PyTorch):")
    print(f"  1. f0_predictor.onnx    → ONNX Runtime")
    print(f"  2. Source gen + STFT     → numpy (SineGen2 + 16-point DFT)")
    print(f"  3. hift_decode_dynamic   → ONNX Runtime")
    print(f"  4. ISTFT                 → numpy (16-point IDFT + overlap-add)")


if __name__ == "__main__":
    main()
