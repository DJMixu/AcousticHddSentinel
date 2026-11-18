#!/usr/bin/env python3
"""
test_tier3_workflows.py
Tier 3: Workflows & Integration Suite for PrototypBezNormal.

Verifies:
1. Subprocess invocation of live_predict.py on audio input.
2. Return schema of audio prediction (file, duration_s, total_windows_analyzed, overall_verdict, windows/window_details).
3. CLI argument parsing in record_and_predict.py (--duration, --output, etc.).
4. Hardware device guard enforcing Realtek ALC1220 and rejecting USB dongle.
5. Zero-padding and encoding for short audio (< 3.0 seconds).
"""

import json
import subprocess
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

from ml.hdd_audio_encoder import TARGET_SR, CHUNK_DURATION, encode_audio_chunk
from PrototypBezNormal.live_predict import analyze_audio_file
import PrototypBezNormal.record_and_predict as rap_module


class TestTier3Workflows(unittest.TestCase):
    """Tier 3: Workflow Integration and CLI Boundary Suite."""

    @classmethod
    def setUpClass(cls):
        # Create a temporary synthetic 3.5-second WAV file for CLI testing
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_wav = Path(cls.temp_dir.name) / "test_synth_3.5s.wav"

        duration_sec = 3.5
        t = np.linspace(0, duration_sec, int(duration_sec * TARGET_SR), endpoint=False)
        # Moderate amplitude acoustic tone (well above silence RMS 0.002)
        y = (0.2 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
        sf.write(str(cls.temp_wav), y, TARGET_SR)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_t3_01_live_predict_cli_call(self):
        """Execute live_predict.py as an external CLI subprocess and verify clean exit."""
        cmd = [
            sys.executable,
            str(PROTOTYPE_DIR / "live_predict.py"),
            str(self.temp_wav),
            "--device", "cpu",
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )

        self.assertEqual(
            result.returncode,
            0,
            f"live_predict.py failed with returncode {result.returncode}.\nStderr: {result.stderr}",
        )
        self.assertIn("RAPORT DIAGNOSTYCZNY AUDIO", result.stdout)
        self.assertIn("OGÓLNY WERDYKT", result.stdout)

    def test_t3_02_live_predict_schema(self):
        """Verify returned dictionary schema from audio file analysis."""
        res = analyze_audio_file(self.temp_wav, device="cpu")

        self.assertIsInstance(res, dict)
        # Required top-level keys
        self.assertIn("file", res)
        self.assertIn("duration_s", res)
        self.assertIn("total_windows_analyzed", res)
        self.assertIn("overall_verdict", res)

        self.assertEqual(Path(res["file"]).resolve(), self.temp_wav.resolve())
        self.assertAlmostEqual(res["duration_s"], 3.5, delta=0.1)
        self.assertGreaterEqual(res["total_windows_analyzed"], 1)
        self.assertIn(res["overall_verdict"], ["normal", "beeping", "clicking", "scratching"])

        # Window list representation (window_details or windows)
        windows = res.get("window_details", res.get("windows"))
        self.assertIsNotNone(windows, "Expected 'window_details' or 'windows' in prediction response")
        self.assertIsInstance(windows, list)
        self.assertEqual(len(windows), res["total_windows_analyzed"])

        # Validate structure of individual window result
        first_window = windows[0]
        self.assertIn("predicted_class", first_window)
        self.assertIn("confidence", first_window)

    def test_t3_03_record_cli_arguments(self):
        """Verify record_and_predict.py argument parser exposes --duration and --output options."""
        cmd = [
            sys.executable,
            str(PROTOTYPE_DIR / "record_and_predict.py"),
            "--help",
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )

        self.assertEqual(result.returncode, 0, f"record_and_predict.py --help failed: {result.stderr}")
        self.assertIn("--duration", result.stdout)
        self.assertIn("--output", result.stdout)

    def test_t3_04_record_mic_device_guard(self):
        """Verify ALC1220 physical microphone guard and strict rejection of USB dongle."""
        expected_active_source = "alsa_input.pci-0000_7b_00.6.analog-stereo"
        forbidden_usb_dongle = "alsa_input.usb-C-Media_Electronics_Inc._USB_Advanced_Audio_Device-00.analog-stereo"

        # Module constant check
        active_source = getattr(rap_module, "PHYSICAL_MIC_SOURCE", None)
        self.assertEqual(
            active_source,
            expected_active_source,
            f"record_and_predict.py must target Realtek ALC1220 Analog, got {active_source}",
        )
        self.assertNotEqual(active_source, forbidden_usb_dongle)

        # Cross-check with scripts/audio_hardware_config.json Single Source of Truth
        config_path = REPO_ROOT / "scripts" / "audio_hardware_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.assertEqual(cfg["active_microphone"]["source_name"], expected_active_source)
        inactive_sources = [m["source_name"] for m in cfg["inactive_microphones"]]
        self.assertIn(forbidden_usb_dongle, inactive_sources)

    def test_t3_05_short_audio_padding(self):
        """Verify that audio shorter than 3.0s is zero-padded without error to full shape (128, 259, 3)."""
        short_durations = [0.5, 1.0, 2.0, 2.9]

        for dur in short_durations:
            samples = int(dur * TARGET_SR)
            short_wave = np.sin(np.linspace(0, 2 * np.pi * 440 * dur, samples, endpoint=False)).astype(np.float32)

            tensor_tf = encode_audio_chunk(short_wave, sr=TARGET_SR, tensor_format="tf")
            self.assertEqual(
                tensor_tf.shape,
                (128, 259, 3),
                f"Padding failed for duration {dur}s: got shape {tensor_tf.shape}",
            )
            self.assertFalse(np.isnan(tensor_tf).any())
            self.assertFalse(np.isinf(tensor_tf).any())

            tensor_pt = encode_audio_chunk(short_wave, sr=TARGET_SR, tensor_format="pytorch")
            self.assertEqual(
                tensor_pt.shape,
                (3, 128, 259),
                f"PyTorch padding failed for duration {dur}s: got shape {tensor_pt.shape}",
            )


if __name__ == "__main__":
    unittest.main()
