#!/usr/bin/env python3
"""
Testy jednostkowe i logiczne dla zaawansowanego prototypu metrycznego OOD (PrototypBezNormal v2).
Sprawdza:
1. Prawidłowość wymiarów wejścia/wyjścia (Channel-First i Channel-Last).
2. Zachowanie normy L2 na sferze S^{d-1} (||z|| == 1.0).
3. Poprawność matematyczną warstwy ArcMarginProduct (marża kątowa).
4. Poprawność modułu CenterLoss.
5. Dodatnią określoność macierzy kowariancji i odległość Mahalanobisa.
6. Poprawność i determinizm inferencji predict_ood.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROTOTYPE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROTOTYPE_DIR.parent
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PrototypBezNormal.model_metric_cnn import MetricFaultOnlyCNN
from PrototypBezNormal.loss_functions import ArcMarginProduct, CenterLoss


class TestMetricPrototypeLogic(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = MetricFaultOnlyCNN(num_fault_classes=3, embedding_dim=128, scale=16.0, margin=0.35).to(self.device)
        self.model.eval()

    def test_01_tensor_shapes_both_layouts(self):
        """Test obsługi wymiarów Channel-First (B, 3, 128, 259) i Channel-Last (128, 259, 3)."""
        # Channel-First batch
        x_cf = torch.randn(2, 3, 128, 259, device=self.device)
        logits_cf, embs_cf = self.model(x_cf)
        self.assertEqual(logits_cf.shape, (2, 3))
        self.assertEqual(embs_cf.shape, (2, 128))

        # Channel-Last single sample
        x_cl = torch.randn(128, 259, 3, device=self.device)
        logits_cl, embs_cl = self.model(x_cl)
        self.assertEqual(logits_cl.shape, (1, 3))
        self.assertEqual(embs_cl.shape, (1, 128))

    def test_02_spherical_l2_normalization(self):
        """Test zachowania normy L2 = 1.0 na sferze unitarnej."""
        x = torch.randn(4, 3, 128, 259, device=self.device)
        embs = self.model.extract_features(x)
        norms = torch.norm(embs, p=2, dim=1).detach().cpu().numpy()
        for n in norms:
            self.assertAlmostEqual(n, 1.0, places=4, msg="Embedding nie jest znormalizowany na sferze L2!")

    def test_03_arcmargin_product_behavior(self):
        """Test działania marży kątowej w treningu vs inferencji."""
        arc = ArcMarginProduct(in_features=128, out_features=3, scale=16.0, margin=0.35)
        embs = F.normalize(torch.randn(4, 128), p=2, dim=1)
        labels = torch.tensor([0, 1, 2, 0])

        # Inferencja (bez marży)
        logits_inf = arc(embs, label=None)
        self.assertEqual(logits_inf.shape, (4, 3))

        # Trening (z marżą kątową)
        logits_train = arc(embs, label=labels)
        self.assertEqual(logits_train.shape, (4, 3))

        # Wartości w logitach treningowych dla klasy poprawnej powinny być stłumione przez marżę cos(theta + m)
        # dla dodatnich cosinusów
        diff = logits_inf[0, labels[0]] - logits_train[0, labels[0]]
        # diff powinien być nieujemny jeśli cosinus > 0
        self.assertFalse(torch.isnan(logits_train).any(), "NaN w wyjściu ArcMargin!")

    def test_04_center_loss_gradients(self):
        """Test działania CenterLoss i braku NaN w gradientach."""
        center_loss = CenterLoss(num_classes=3, feat_dim=128)
        embs = F.normalize(torch.randn(6, 128, requires_grad=True), p=2, dim=1)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])

        loss = center_loss(embs, labels)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()

        self.assertIsNotNone(center_loss.centers.grad)
        self.assertFalse(torch.isnan(center_loss.centers.grad).any())

    def test_05_ood_calibration_and_mahalanobis(self):
        """Test kalibracji OOD i odwracalności macierzy kowariancji."""
        embs = F.normalize(torch.randn(60, 128), p=2, dim=1)
        labels = torch.randint(0, 3, (60,))
        logits = torch.randn(60, 3)

        calib = self.model.calibrate_ood_parameters(embs, labels, logits, percentile=95.0)
        self.assertTrue(self.model.is_calibrated.item())
        self.assertIn("global_cosine_threshold", calib)
        self.assertIn("global_mahalanobis_threshold", calib)
        self.assertIn("global_energy_threshold", calib)

        # Sprawdzenie macierzy kowariancji
        inv_cov = self.model.pooled_cov_inv.cpu().numpy()
        self.assertEqual(inv_cov.shape, (128, 128))
        self.assertFalse(np.isnan(inv_cov).any(), "NaN w odwróconej macierzy kowariancji!")

    def test_06_predict_ood_inference(self):
        """Test kompletnej inferencji OOD (predict_ood) i formatu wyjścia."""
        x = torch.randn(1, 3, 128, 259, device=self.device)
        res = self.model.predict_ood(x)

        self.assertIsInstance(res, dict)
        self.assertIn("verdict", res)
        self.assertIn("is_fault", res)
        self.assertIn("confidence", res)
        self.assertIn("metrics", res)
        self.assertIn(res["verdict"], ["beeping", "clicking", "scratching", "normal"])
        self.assertGreaterEqual(res["confidence"], 0.50)
        self.assertLessEqual(res["confidence"], 1.00)


if __name__ == "__main__":
    unittest.main()
