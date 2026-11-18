#!/usr/bin/env python3
"""
test_tier1_invariants.py
Tier 1: Feature & Invariant Coverage for PrototypBezNormal.

Verifies:
1. AST syntax parsing for all Python source files in PrototypBezNormal/.
2. ADR-001 digital signal processing constants in ml/hdd_audio_encoder.py.
3. Zero Data Leakage invariant in dataset_encoded/train and dataset_encoded/validate (0 normal samples, exact counts).
4. Audio encoding tensor format, dimensional shapes, and numerical validity (no NaNs/Infs).
5. Audio hardware configuration enforcing ALC1220 analog mic and forbidding USB dongle.
"""

import ast
import json
import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROTOTYPE_DIR = REPO_ROOT / "PrototypBezNormal"
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))

from ml.hdd_audio_encoder import (
    TARGET_SR,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    CHUNK_DURATION,
    encode_audio_chunk,
)
from PrototypBezNormal.dataset_no_normal import FaultOnlyNpyDataset


class TestTier1Invariants(unittest.TestCase):
    """Tier 1: Invariant and Specification Verification Suite."""

    def test_t1_01_ast_syntax(self):
        """AST parse all Python scripts in PrototypBezNormal/ to ensure syntax validity."""
        py_files = sorted(PROTOTYPE_DIR.rglob("*.py"))
        # Exclude __pycache__ and hidden files
        py_files = [p for p in py_files if "__pycache__" not in p.parts and not p.name.startswith(".")]

        self.assertGreaterEqual(
            len(py_files),
            5,
            f"Expected at least 5 Python files in {PROTOTYPE_DIR}, found {len(py_files)}",
        )

        syntax_errors = []
        for file_path in py_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source_code = f.read()
                ast.parse(source_code, filename=str(file_path))
            except SyntaxError as e:
                syntax_errors.append(f"{file_path.name}:{e.lineno} - {e.msg}")

        self.assertEqual(
            len(syntax_errors),
            0,
            f"Syntax errors detected in PrototypBezNormal: {syntax_errors}",
        )

    def test_t1_02_adr001_constants(self):
        """Verify ADR-001 DSP constants in ml/hdd_audio_encoder.py."""
        self.assertEqual(TARGET_SR, 22050, f"Expected TARGET_SR=22050, got {TARGET_SR}")
        self.assertEqual(N_FFT, 1024, f"Expected N_FFT=1024, got {N_FFT}")
        self.assertEqual(HOP_LENGTH, 256, f"Expected HOP_LENGTH=256, got {HOP_LENGTH}")
        self.assertEqual(N_MELS, 128, f"Expected N_MELS=128, got {N_MELS}")
        self.assertEqual(FMIN, 50.0, f"Expected FMIN=50.0, got {FMIN}")
        self.assertEqual(FMAX, 10000.0, f"Expected FMAX=10000.0, got {FMAX}")
        self.assertEqual(CHUNK_DURATION, 3.0, f"Expected CHUNK_DURATION=3.0, got {CHUNK_DURATION}")

    def test_t1_03_zero_normal_leakage(self):
        """Verify dataset_encoded/train and validate contain 0 normal samples and exact counts."""
        dataset_root = REPO_ROOT / "dataset_encoded"
        train_dir = dataset_root / "train"
        val_dir = dataset_root / "validate"

        self.assertTrue(train_dir.exists(), f"Train directory missing: {train_dir}")
        self.assertTrue(val_dir.exists(), f"Validate directory missing: {val_dir}")

        ds_train = FaultOnlyNpyDataset(train_dir, include_normal=False, preload_to_ram=False)
        ds_val = FaultOnlyNpyDataset(val_dir, include_normal=False, preload_to_ram=False)

        # Invariant 1: Exact sample counts
        self.assertEqual(
            len(ds_train),
            3421,
            f"Expected exactly 3,421 fault tensors in train, got {len(ds_train)}",
        )
        self.assertEqual(
            len(ds_val),
            131,
            f"Expected exactly 131 fault tensors in validate, got {len(ds_val)}",
        )

        # Invariant 2: 0 normal samples in train
        train_normal_count = sum(1 for _, _, class_name in ds_train.samples if class_name == "normal")
        self.assertEqual(
            train_normal_count,
            0,
            f"Data leakage detected! {train_normal_count} normal samples found in train set.",
        )

        # Invariant 3: 0 normal samples in validate
        val_normal_count = sum(1 for _, _, class_name in ds_val.samples if class_name == "normal")
        self.assertEqual(
            val_normal_count,
            0,
            f"Data leakage detected! {val_normal_count} normal samples found in validate set.",
        )

        # Verify class distribution covers only the 3 fault classes
        expected_classes = {"beeping", "clicking", "scratching"}
        self.assertEqual(set(ds_train.classes), expected_classes)
        self.assertEqual(set(ds_val.classes), expected_classes)

    def test_t1_04_tensor_format_and_shape(self):
        """Verify encode_audio_chunk produces valid tensors in both Channel-Last and Channel-First formats."""
        # Synthesize a standard 3.0-second chunk (66,150 samples)
        num_samples = int(CHUNK_DURATION * TARGET_SR)
        t = np.linspace(0, CHUNK_DURATION, num_samples, endpoint=False)
        test_waveform = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)

        # 1. Channel-Last format ("tf" / disk .npy standard)
        tensor_tf = encode_audio_chunk(test_waveform, sr=TARGET_SR, tensor_format="tf")
        self.assertIsInstance(tensor_tf, np.ndarray)
        self.assertEqual(
            tensor_tf.shape,
            (128, 259, 3),
            f"Expected shape (128, 259, 3), got {tensor_tf.shape}",
        )
        self.assertEqual(tensor_tf.dtype, np.float32)
        self.assertFalse(np.isnan(tensor_tf).any(), "Found NaNs in Channel-Last tensor")
        self.assertFalse(np.isinf(tensor_tf).any(), "Found Infs in Channel-Last tensor")

        # 2. Channel-First format ("pytorch" / memory standard)
        tensor_pt = encode_audio_chunk(test_waveform, sr=TARGET_SR, tensor_format="pytorch")
        self.assertIsInstance(tensor_pt, np.ndarray)
        self.assertEqual(
            tensor_pt.shape,
            (3, 128, 259),
            f"Expected shape (3, 128, 259), got {tensor_pt.shape}",
        )
        self.assertEqual(tensor_pt.dtype, np.float32)
        self.assertFalse(np.isnan(tensor_pt).any(), "Found NaNs in Channel-First tensor")
        self.assertFalse(np.isinf(tensor_pt).any(), "Found Infs in Channel-First tensor")

    def test_t1_05_audio_hardware_config(self):
        """Verify scripts/audio_hardware_config.json mandates ALC1220 and forbids USB dongle."""
        config_path = REPO_ROOT / "scripts" / "audio_hardware_config.json"
        self.assertTrue(config_path.exists(), f"Configuration file missing: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Active microphone check
        active_mic = cfg.get("active_microphone", {})
        active_source = active_mic.get("source_name")
        self.assertEqual(
            active_source,
            "alsa_input.pci-0000_7b_00.6.analog-stereo",
            f"Active microphone must be ALC1220 PCI analog stereo, got {active_source}",
        )
        self.assertEqual(active_mic.get("status"), "PRIMARY_ACTIVE_MIC")

        # Inactive / forbidden dongle check
        inactive_mics = cfg.get("inactive_microphones", [])
        forbidden_sources = [m.get("source_name") for m in inactive_mics]
        self.assertIn(
            "alsa_input.usb-C-Media_Electronics_Inc._USB_Advanced_Audio_Device-00.analog-stereo",
            forbidden_sources,
            "Forbidden USB dongle not listed in inactive microphones list",
        )


if __name__ == "__main__":
    unittest.main()
