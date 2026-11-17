#!/usr/bin/env python3
"""
test_prototype_logic.py
Automatyczny test integralności i poprawności działania modelu PrototypBezNormal.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

PROTOTYPE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROTOTYPE_DIR.parent
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_no_normal import AcousticFaultOnlyCNN
from dataset_no_normal import FaultOnlyNpyDataset, create_dataloaders
from live_predict import analyze_audio_file


class TestPrototypBezNormal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights_path = PROTOTYPE_DIR / "output" / "best_model_no_normal.pt"
        cls.device = torch.device("cpu")
        cls.model = AcousticFaultOnlyCNN(num_fault_classes=3, embedding_dim=128)
        if cls.weights_path.exists():
            ckpt = torch.load(cls.weights_path, map_location=cls.device, weights_only=False)
            cls.model.load_state_dict(ckpt["state_dict"])
            cls.model.eval()

    def test_01_model_weights_exist(self):
        """Weryfikacja obecności pliku wag."""
        self.assertTrue(self.weights_path.exists(), "Plik wag best_model_no_normal.pt nie istnieje!")
        self.assertGreater(self.weights_path.stat().st_size, 1_000_000, "Rozmiar wag jest zbyt mały (< 1 MB).")

    def test_02_calibration_buffers(self):
        """Weryfikacja kalibracji prototypów metrycznych."""
        self.assertTrue(self.model.is_calibrated.item(), "Model nie został poprawnie wykalibrowany!")
        self.assertEqual(self.model.prototypes.shape, (3, 128))
        self.assertGreater(self.model.global_threshold.item(), 0.5)

    def test_03_forward_shapes(self):
        """Sprawdzenie obsługi Channel-First (B, 3, 128, 259) i Channel-Last (B, 128, 259, 3)."""
        x_first = torch.randn(2, 3, 128, 259)
        logits_first, embed_first = self.model(x_first)
        self.assertEqual(logits_first.shape, (2, 3))
        self.assertEqual(embed_first.shape, (2, 128))

        x_last = torch.randn(2, 128, 259, 3)
        logits_last, embed_last = self.model(x_last)
        self.assertEqual(logits_last.shape, (2, 3))
        self.assertEqual(embed_last.shape, (2, 128))

    def test_04_predict_with_normal(self):
        """Weryfikacja predykcji 4-klasowej."""
        x = torch.randn(1, 3, 128, 259)
        res = self.model.predict_with_normal(x)
        self.assertIn("pred_class", res)
        self.assertIn("probabilities", res)
        self.assertEqual(res["probabilities"].shape, (1, 4))
        # Suma prawdopodobieństw powinna wynosić 1.0
        self.assertAlmostEqual(float(res["probabilities"].sum()), 1.0, places=4)

    def test_05_dataset_no_normal_in_train(self):
        """Upewnienie się, że train_loader nie zawiera próbek 'normal'."""
        train_ds = FaultOnlyNpyDataset(PROJECT_ROOT / "dataset_encoded" / "train", include_normal=False)
        for _, _, cname in train_ds.samples:
            self.assertNotEqual(cname, "normal", "Zbiór treningowy zawiera klasę 'normal'!")

    def test_06_live_prediction_on_real_audio(self):
        """Test działania live_predict na fizycznym nagraniu dysku."""
        test_wav = PROJECT_ROOT / "tests" / "recordings_4min_study" / "WD" / "WD_5400rpm_healthy_idle_60s.wav"
        if test_wav.exists():
            res = analyze_audio_file(test_wav, weights_path=str(self.weights_path), device="cpu")
            self.assertEqual(res["overall_verdict"], "normal")


if __name__ == "__main__":
    unittest.main()
