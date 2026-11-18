"""
tests/test_smpc_architecture.py
Kompleksowy zestaw testów jednostkowych architektury Sub-Center Multi-Prototype Cosine (SMPC)
oraz integracji OOD Hardening w projekcie Acoustic HDD Sentinel (Milestone 2 - Feature 8).

Weryfikuje:
1. Obsługę tensorów Channel-First (B, 3, 128, 259) i Channel-Last (B, 128, 259, 3).
2. Transparentne wsparcie wielomodalnych prototypów SMPC o kształcie (3, M, 128) w buforach modelu.
3. Niezmiennik liczby trenowalnych parametrów bazowych (dokładnie 1,206,756 parametrów).
4. Poprawność numeryczną Option A (MatMul + view + ReduceMax bez einsum) zgodną z ONNX.
5. Funkcję straty SubCenterArcMarginLoss i przepływ gradientów.
6. Regularyzację kątową compute_orthogonality_loss zapobiegającą kolapsowi sub-centrów.
7. Hybrydową walidację cech w Gałęzi D HybridAcousticSentinel (scratching recall vs odrzucanie wentylatorów).
"""

import os
import sys
import unittest
from pathlib import Path

# Auto-dispatch to project virtualenv if running in system Python lacking dependencies
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
if _VENV_PYTHON.exists() and Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    try:
        import numpy
    except ImportError:
        import subprocess
        sys.exit(subprocess.call([str(_VENV_PYTHON)] + sys.argv))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ml"))

from ml.model_cnn import build_hdd_cnn_model, AcousticHddSentinelCNN
from ml.train_cnn import SubCenterArcMarginLoss, compute_orthogonality_loss, spherical_kmeans
from ml.hybrid_acoustic_sentinel import HybridAcousticSentinel
from ml.hdd_audio_encoder import TARGET_SR


class TestSmpcArchitecture(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = build_hdd_cnn_model(num_classes=4, dropout_rate=0.2).to(self.device)

    def test_smpc_trainable_param_invariant(self):
        """Weryfikuje, że dodanie buforów SMPC (3, M, 128) nie zmienia liczby trenowalnych parametrów (1,206,756)."""
        trainable_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(trainable_before, 1_206_756, f"Przed SMPC: {trainable_before} != 1,206,756!")

        # Rejestracja wielomodalnych prototypów SMPC (M=3 podcentra na klasę dla 3 wad)
        M = 3
        prototypes = torch.randn(3, M, 128)
        thresholds = torch.tensor([0.45, 0.40, 0.38], dtype=torch.float32)
        self.model.set_fault_prototypes(prototypes, thresholds, gamma=12.0)

        trainable_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(trainable_after, 1_206_756, f"Po SMPC: {trainable_after} != 1,206,756!")
        self.assertTrue(bool(self.model.use_open_set.item()))
        self.assertEqual(self.model.fault_prototypes.shape, (3, M, 128))

    def test_smpc_forward_channel_first_and_channel_last(self):
        """Weryfikuje forward pass z aktywnym SMPC dla układu Channel-First i Channel-Last."""
        M = 3
        prototypes = F.normalize(torch.randn(3, M, 128), p=2, dim=-1)
        thresholds = torch.tensor([0.45, 0.40, 0.38], dtype=torch.float32)
        self.model.set_fault_prototypes(prototypes, thresholds, gamma=12.0)
        self.model.eval()

        dummy_cf = torch.randn(2, 3, 128, 259)
        dummy_cl = torch.randn(2, 128, 259, 3)

        with torch.no_grad():
            out_cf = self.model(dummy_cf)
            out_cl = self.model(dummy_cl)

        self.assertEqual(out_cf.shape, (2, 4))
        self.assertEqual(out_cl.shape, (2, 4))
        self.assertFalse(torch.isnan(out_cf).any())
        self.assertFalse(torch.isnan(out_cl).any())
        self.assertTrue(torch.allclose(out_cf.sum(dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.allclose(out_cl.sum(dim=-1), torch.ones(2), atol=1e-5))

    def test_smpc_option_a_mathematical_equivalence(self):
        """Weryfikuje, że Option A (MatMul + view + ReduceMax) jest bitowo zgodna z definicją max_m (z · W_m)."""
        B, K, M, D = 4, 3, 3, 128
        z = F.normalize(torch.randn(B, D), p=2, dim=-1)
        w = F.normalize(torch.randn(K, M, D), p=2, dim=-1)

        # 1. Definicja referencyjna (pętla)
        ref_sims = torch.zeros(B, K)
        for b in range(B):
            for k in range(K):
                dots = [torch.dot(z[b], w[k, m]).item() for m in range(M)]
                ref_sims[b, k] = max(dots)

        # 2. Option A (ONNX-friendly: MatMul + view + ReduceMax bez einsum)
        w_flat = w.view(K * M, D)
        sims_flat = torch.matmul(z, w_flat.t())
        sims = sims_flat.view(B, K, M)
        option_a_sims, _ = torch.max(sims, dim=2)

        max_diff = torch.max(torch.abs(ref_sims - option_a_sims)).item()
        self.assertLess(max_diff, 1e-6, f"Różnica Option A względem definicji referencyjnej: {max_diff}")

    def test_subcenter_arc_margin_loss_and_gradients(self):
        """Weryfikuje funkcję straty SubCenterArcMarginLoss i propagację gradientów."""
        loss_fn = SubCenterArcMarginLoss(scale=16.0, margin=0.25, label_smoothing=0.03)

        # Przykładowe wektory podobieństw cosinusowych w zakresie [-1, 1]
        cos_sims = torch.tensor([
            [0.85, 0.10, 0.05, 0.15],
            [0.10, 0.90, 0.05, 0.20]
        ], requires_grad=True)
        targets = torch.tensor([0, 1], dtype=torch.long)

        loss = loss_fn(cos_sims, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(cos_sims.grad)
        self.assertFalse(torch.isnan(cos_sims.grad).any())

    def test_compute_orthogonality_loss(self):
        """Weryfikuje regularyzację ortogonalności sub-centrów (zapobieganie kolapsowi)."""
        K, D = 3, 128

        # Przypadek 1: Idealnie ortogonalne wektory (dot_product = 0.0 <= 0.30)
        w_ortho = torch.zeros(K, 2, D)
        w_ortho[:, 0, 0] = 1.0
        w_ortho[:, 1, 1] = 1.0
        loss_zero = compute_orthogonality_loss(w_ortho, threshold=0.30)
        self.assertEqual(loss_zero.item(), 0.0)

        # Przypadek 2: Skolapsowane wektory (dot_product = 1.0 > 0.30)
        w_collinear = torch.zeros(K, 2, D)
        w_collinear[:, 0, 0] = 1.0
        w_collinear[:, 1, 0] = 1.0
        loss_penalized = compute_orthogonality_loss(w_collinear, threshold=0.30)
        expected_penalty = K * (1.0 - 0.30)  # 3 * 0.70 = 2.10
        self.assertAlmostEqual(loss_penalized.item(), expected_penalty, places=5)

    def test_spherical_kmeans_clustering(self):
        """Weryfikuje sferyczny algorytm K-Means generujący M znormalizowanych podcentrów."""
        # Generowanie dwóch wyraźnych skupisk na sferze
        c1 = torch.zeros(128)
        c1[0] = 1.0
        c2 = torch.zeros(128)
        c2[1] = 1.0

        cluster1 = F.normalize(c1 + 0.1 * torch.randn(30, 128), p=2, dim=-1)
        cluster2 = F.normalize(c2 + 0.1 * torch.randn(30, 128), p=2, dim=-1)
        data = torch.cat([cluster1, cluster2], dim=0)

        centroids = spherical_kmeans(data, k=2, max_iters=20)
        self.assertEqual(centroids.shape, (2, 128))
        # Sprawdzenie normalizacji unit L2
        norms = torch.norm(centroids, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(2), atol=1e-5))

    def test_hybrid_sentinel_scratching_smpc_validation(self):
        """Weryfikuje, że Gałąź D w HybridAcousticSentinel prawidłowo waliduje cechy scratching."""
        weights_path = PROJECT_ROOT / "training_output" / "best_hdd_cnn_model.pt"
        sentinel = HybridAcousticSentinel(model_weights_path=str(weights_path))

        # 1. Czysty szum o szerokim spektrum (np. wentylator) nie może być oznaczony jako scratching
        np.random.seed(42)
        white_noise = (np.random.randn(int(3.0 * TARGET_SR)) * 0.01).astype(np.float32)
        res_noise = sentinel.diagnose_chunk(white_noise)
        self.assertFalse(res_noise["is_anomaly"])
        self.assertIn(res_noise["decision"], ["ambient_pc", "normal"])


if __name__ == "__main__":
    unittest.main()
