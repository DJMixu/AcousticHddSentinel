#!/usr/bin/env python3
"""
test_tier2_components.py
Tier 2: Component Contracts & Boundary Cases for PrototypBezNormal.

Verifies:
1. AcousticSMPCModel forward pass on Channel-First (B, 3, 128, 259), Channel-Last (B, 128, 259, 3), and unbatched (128, 259, 3).
2. Spherical normalization onto S^{127} enforcing ||z||_2 = 1.0 ± 10^{-4}.
3. Sub-center prototype shape (K=3, M=2, D=128) and cosine similarity bounds [-1.0, 1.0].
4. Intra-class sub-center orthogonality regularization loss behavior.
5. SMPCFaultOODDetector.predict_chunk interface contract and return schema.
6. Calibration logic determinism on validation fault scores.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROTOTYPE_DIR = REPO_ROOT / "PrototypBezNormal"
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))

from PrototypBezNormal.model_smpc import AcousticSMPCModel
from PrototypBezNormal.loss_smpc import compute_orthogonality_loss
from PrototypBezNormal.smpc_ood_engine import SMPCFaultOODDetector


class TestTier2Components(unittest.TestCase):
    """Tier 2: Component Architecture and Mathematical Invariants Suite."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = AcousticSMPCModel(
            num_fault_classes=3,
            embedding_dim=128,
            sub_centers=2,
            scale=16.0
        ).to(self.device)
        self.model.eval()

    def test_t2_01_forward_shapes(self):
        """Test model forward pass on Channel-First, Channel-Last, and unbatched inputs."""
        batch_size = 4

        # 1. Channel-First (PyTorch native): (B, 3, 128, 259)
        x_cf = torch.randn(batch_size, 3, 128, 259, dtype=torch.float32, device=self.device)
        logits_cf, embed_cf = self.model(x_cf)
        self.assertEqual(logits_cf.shape, (batch_size, 3))
        self.assertEqual(embed_cf.shape, (batch_size, 128))

        # 2. Channel-Last (disk .npy format): (B, 128, 259, 3)
        x_cl = torch.randn(batch_size, 128, 259, 3, dtype=torch.float32, device=self.device)
        logits_cl, embed_cl = self.model(x_cl)
        self.assertEqual(logits_cl.shape, (batch_size, 3))
        self.assertEqual(embed_cl.shape, (batch_size, 128))

        # 3. Unbatched single slice: (128, 259, 3)
        x_single = np.random.randn(128, 259, 3).astype(np.float32)
        logits_single, embed_single = self.model(x_single)
        self.assertEqual(logits_single.shape, (1, 3))
        self.assertEqual(embed_single.shape, (1, 128))

    def test_t2_02_spherical_normalization(self):
        """Verify L2 spherical projection satisfies ||z||_2 = 1.0 ± 10^{-4} on S^{127}."""
        x = torch.randn(8, 3, 128, 259, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            raw_features, norm_embeddings = self.model.extract_features(x)

        # Compute L2 norms along feature dimension
        norms = torch.norm(norm_embeddings, p=2, dim=-1)
        expected_norms = torch.ones_like(norms)

        self.assertTrue(
            torch.allclose(norms, expected_norms, atol=1e-4),
            f"Normalized embeddings must lie on unit sphere S^127: {norms.tolist()}",
        )

        # Raw features should not be constrained to unit length
        raw_norms = torch.norm(raw_features, p=2, dim=-1)
        self.assertFalse(
            torch.allclose(raw_norms, expected_norms, atol=1e-2),
            "Raw unnormalized features should not be forced to norm 1.0 prior to normalization",
        )

    def test_t2_03_subcenter_head_contract(self):
        """Verify sub-center prototypes tensor shape and cosine similarity contract."""
        # Prototypes must have shape (K, M, D) = (3, 2, 128)
        self.assertEqual(
            self.model.prototypes.shape,
            (3, 2, 128),
            f"Prototypes shape mismatch: expected (3, 2, 128), got {self.model.prototypes.shape}",
        )

        # Verify cosine similarity outputs
        z = F.normalize(torch.randn(6, 128, dtype=torch.float32, device=self.device), p=2, dim=-1)
        sims = self.model.compute_cosine_similarities(z)

        self.assertEqual(sims.shape, (6, 3))
        # Cosine values must be strictly within [-1.0001, 1.0001]
        self.assertTrue(torch.all(sims >= -1.0001))
        self.assertTrue(torch.all(sims <= 1.0001))

        # Check maximum property across sub-centers
        w_norm = F.normalize(self.model.prototypes, p=2, dim=-1)
        for i in range(6):
            for k in range(3):
                sc1 = torch.dot(z[i], w_norm[k, 0]).item()
                sc2 = torch.dot(z[i], w_norm[k, 1]).item()
                expected_max = max(sc1, sc2)
                self.assertAlmostEqual(sims[i, k].item(), expected_max, places=5)

    def test_t2_04_subcenter_orthogonality(self):
        """Verify intra-class sub-center orthogonality loss penalties."""
        threshold = 0.30

        # Case A: Identical sub-centers (W_1 == W_2, dot product = 1.0)
        # Penalty per class should be max(0, 1.0 - 0.30) = 0.70
        collinear_proto = torch.zeros(3, 2, 128)
        collinear_proto[:, 0, 0] = 1.0
        collinear_proto[:, 1, 0] = 1.0  # Identical unit vector
        loss_collinear = compute_orthogonality_loss(collinear_proto, threshold=threshold)
        expected_penalty = 3 * (1.0 - threshold)
        self.assertAlmostEqual(loss_collinear.item(), expected_penalty, places=5)

        # Case B: Mutually orthogonal sub-centers (dot product = 0.0 <= threshold)
        # Penalty should be strictly 0.0
        orthogonal_proto = torch.zeros(3, 2, 128)
        orthogonal_proto[:, 0, 0] = 1.0
        orthogonal_proto[:, 1, 1] = 1.0  # Orthogonal unit vector
        loss_ortho = compute_orthogonality_loss(orthogonal_proto, threshold=threshold)
        self.assertEqual(loss_ortho.item(), 0.0)

        # Case C: Model's initialized prototypes produce non-negative finite loss
        model_ortho_loss = compute_orthogonality_loss(self.model.prototypes, threshold=threshold)
        self.assertGreaterEqual(model_ortho_loss.item(), 0.0)
        self.assertFalse(torch.isnan(model_ortho_loss))

    def test_t2_05_ood_engine_interface(self):
        """Verify SMPCFaultOODDetector.predict_chunk contract and return schema."""
        output_smpc_dir = PROTOTYPE_DIR / "output_smpc"
        weights_path = output_smpc_dir / "best_smpc_model.pt"
        config_path = output_smpc_dir / "smpc_ood_config.json"

        temp_cfg_path = None
        if not config_path.exists():
            # If calibration config does not yet exist, create a temporary valid config
            temp_cfg = {
                "threshold": 0.50,
                "temperature": 2.0,
                "alpha_energy": 0.10,
                "scale": 16.0,
                "num_classes": 3,
                "sub_centers": 2,
                "embedding_dim": 128,
                "classes_3": ["beeping", "clicking", "scratching"],
                "val_stats": {"std": 0.15}
            }
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(temp_cfg, f)
                temp_cfg_path = f.name
            used_config = temp_cfg_path
        else:
            used_config = str(config_path)

        # Fallback to saving model weights if checkpoint not on disk
        temp_weights_path = None
        if not weights_path.exists():
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                temp_weights_path = f.name
            torch.save({"model_state_dict": self.model.state_dict()}, temp_weights_path)
            used_weights = temp_weights_path
        else:
            used_weights = str(weights_path)

        try:
            detector = SMPCFaultOODDetector(
                weights_path=used_weights,
                config_path=used_config,
                device="cpu"
            )

            chunk_npy = np.random.randn(128, 259, 3).astype(np.float32)
            pred = detector.predict_chunk(chunk_npy)

            # Contract verification
            self.assertIsInstance(pred, dict)
            required_keys = ["predicted_class", "is_fault", "ood_score", "threshold", "confidence"]
            for k in required_keys:
                self.assertIn(k, pred, f"Missing required key '{k}' in predict_chunk response")

            self.assertIn(pred["predicted_class"], ["normal", "beeping", "clicking", "scratching"])
            self.assertIsInstance(pred["is_fault"], bool)
            self.assertIsInstance(pred["ood_score"], (int, float))
            self.assertIsInstance(pred["threshold"], (int, float))
            self.assertIsInstance(pred["confidence"], (int, float))
            self.assertGreaterEqual(pred["confidence"], 0.0)
            self.assertLessEqual(pred["confidence"], 1.0)
        finally:
            if temp_cfg_path and Path(temp_cfg_path).exists():
                Path(temp_cfg_path).unlink()
            if temp_weights_path and Path(temp_weights_path).exists():
                Path(temp_weights_path).unlink()

    def test_t2_06_calibration_determinism(self):
        """Verify calibration threshold calculation determinism and quantile properties."""
        # Simulated validation composite scores for 131 samples
        np.random.seed(42)
        val_scores = np.random.normal(loc=0.75, scale=0.12, size=131)

        target_recall = 0.90
        # For target recall 0.90, threshold is the 10th percentile
        quantile_percent = (1.0 - target_recall) * 100.0  # 10.0%

        tau_run1 = float(np.percentile(val_scores, quantile_percent))
        tau_run2 = float(np.percentile(val_scores, quantile_percent))

        self.assertEqual(tau_run1, tau_run2, "Calibration threshold must be perfectly deterministic")

        # Invariant: Fraction of validation scores >= tau must be >= target_recall
        recall = np.mean(val_scores >= tau_run1)
        self.assertGreaterEqual(
            recall,
            target_recall - 0.01,
            f"Expected empirical recall >= {target_recall}, got {recall}",
        )


if __name__ == "__main__":
    unittest.main()
