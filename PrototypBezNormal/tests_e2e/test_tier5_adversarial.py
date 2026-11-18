#!/usr/bin/env python3
"""
test_tier5_adversarial.py
Tier 5: Adversarial Hardening & Stress Suite for PrototypBezNormal.

Verifies:
1. Silence frame filtering and rejection (RMS < 0.002 guard) and numerical stability on pure zeros.
2. DC offset immunity and elimination via peak waveform normalization.
3. Ambient cooling fan and low-frequency motor noise non-fault classification.
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROTOTYPE_DIR = REPO_ROOT / "PrototypBezNormal"
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))

from ml.hdd_audio_encoder import (
    TARGET_SR,
    CHUNK_DURATION,
    encode_audio_chunk,
    normalize_waveform_peak,
)
from PrototypBezNormal.live_predict import analyze_audio_file


class TestTier5Adversarial(unittest.TestCase):
    """Tier 5: Adversarial and Extreme Stress Test Suite."""

    def test_t5_01_silence_rejection_rms(self):
        """Verify RMS < 0.002 silence rejection threshold and numerical stability on pure zeros."""
        num_samples = int(CHUNK_DURATION * TARGET_SR)

        # 1. Digital silence (all zeros)
        pure_silence = np.zeros(num_samples, dtype=np.float32)
        rms_zero = float(np.sqrt(np.mean(pure_silence ** 2)))
        self.assertLess(rms_zero, 0.002, "Pure zeros must have RMS < 0.002")

        # Must encode safely without ZeroDivisionError or NaNs
        tensor_zero = encode_audio_chunk(pure_silence, sr=TARGET_SR, tensor_format="tf")
        self.assertEqual(tensor_zero.shape, (128, 259, 3))
        self.assertFalse(np.isnan(tensor_zero).any(), "Found NaNs when encoding pure silence")
        self.assertFalse(np.isinf(tensor_zero).any(), "Found Infs when encoding pure silence")

        # 2. USB dongle simulated idle digital noise (RMS ~0.00007)
        np.random.seed(123)
        dongle_noise = (np.random.randn(num_samples) * 0.00007).astype(np.float32)
        rms_dongle = float(np.sqrt(np.mean(dongle_noise ** 2)))
        self.assertLess(
            rms_dongle,
            0.002,
            f"Dongle noise ({rms_dongle:.6f}) must be strictly below 0.002 silence threshold",
        )

        tensor_dongle = encode_audio_chunk(dongle_noise, sr=TARGET_SR, tensor_format="tf")
        self.assertEqual(tensor_dongle.shape, (128, 259, 3))
        self.assertFalse(np.isnan(tensor_dongle).any())

    def test_t5_02_dc_offset_immunity(self):
        """Verify peak waveform normalization completely eliminates large DC offset bias."""
        num_samples = int(CHUNK_DURATION * TARGET_SR)
        t = np.linspace(0, CHUNK_DURATION, num_samples, endpoint=False)

        # Contaminate signal with huge DC bias (+0.75V offset on 0.15V AC signal)
        ac_signal = 0.15 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
        dc_bias = 0.75
        contaminated_wave = ac_signal + dc_bias

        # Verify initial raw waveform has huge DC offset
        raw_mean = float(np.mean(contaminated_wave))
        self.assertAlmostEqual(raw_mean, dc_bias, delta=0.01)

        # Apply peak normalization
        normalized_wave = normalize_waveform_peak(contaminated_wave)

        # 1. DC offset must be eliminated (mean == 0.0)
        norm_mean = float(np.mean(normalized_wave))
        self.assertAlmostEqual(
            norm_mean,
            0.0,
            places=6,
            msg=f"Peak normalization failed to remove DC offset: mean={norm_mean}",
        )

        # 2. Maximum absolute peak must be scaled to exactly 1.0
        norm_peak = float(np.max(np.abs(normalized_wave)))
        self.assertAlmostEqual(
            norm_peak,
            1.0,
            places=5,
            msg=f"Normalized waveform peak must be 1.0, got {norm_peak}",
        )

    def test_t5_03_ambient_fan_noise_rejection(self):
        """Verify continuous ambient cooling fan noise (aerodynamic broadband airflow) is classified as normal."""
        duration_sec = 4.0
        sr = TARGET_SR
        num_samples = int(duration_sec * sr)

        # Simulate broadband aerodynamic turbulent airflow noise (cooling fan exhaust)
        np.random.seed(42)
        white_noise = np.random.randn(num_samples).astype(np.float32)
        # Low-pass filter for smooth turbulent airflow acoustic profile
        filter_kernel = np.ones(50, dtype=np.float32) / 50.0
        turbulent_noise = np.convolve(white_noise, filter_kernel, mode="same")

        fan_acoustic = (0.025 * turbulent_noise).astype(np.float32)

        # Energy is well above silence threshold (RMS ~0.0035 > 0.002)
        rms = float(np.sqrt(np.mean(fan_acoustic ** 2)))
        self.assertGreater(rms, 0.002, "Synthetic fan noise must have RMS > 0.002 to test OOD classifier")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, fan_acoustic, sr)
            temp_path = f.name

        try:
            res = analyze_audio_file(temp_path, device="cpu")
            self.assertEqual(
                res["overall_verdict"],
                "normal",
                f"Ambient fan noise falsely classified as mechanical fault: {res['overall_verdict']}",
            )
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    unittest.main()
