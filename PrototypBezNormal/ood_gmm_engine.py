"""
ood_gmm_engine.py
Silnik detekcji Out-Of-Distribution (OOD) dla architektury Fault-Only.
Wykorzystuje modelowanie gęstości prawdopodobieństwa w przestrzeni ukrytej 128D
za pomocą mieszaniny rozkładów Gaussa (Gaussian Mixture Model - GMM).

Single Source of Truth: ADR-001, Bishop (Pattern Recognition and Machine Learning),
Bishop OOD Density Framework.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PrototypBezNormal.model_no_normal import AcousticFaultOnlyCNN


class GMMFaultOODDetector:
    """
    Zintegrowany klasyfikator i detektor OOD dla diagnostyki dysków HDD.
    Wykorzystuje sieć 2D CNN (nauczoną na 3 awariach) oraz estymator gęstości GMM.
    """
    def __init__(
        self,
        model_weights_path: str or Path = "PrototypBezNormal/output_v2/best_fault_cnn_v2.pt",
        gmm_model_path: str or Path = "PrototypBezNormal/output_v2/gmm_density_model.joblib",
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.fault_classes = ["beeping", "clicking", "scratching"]
        self.eval_classes = ["beeping", "clicking", "normal", "scratching"]

        # 1. Ładowanie modelu CNN
        ckpt = torch.load(model_weights_path, map_location=self.device, weights_only=False)
        self.model = AcousticFaultOnlyCNN(num_fault_classes=3, embedding_dim=128).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        # 2. Ładowanie estymatora GMM i parametrów progowych
        gmm_data = joblib.load(gmm_model_path)
        self.gmm = gmm_data["gmm"]
        self.threshold = float(gmm_data["threshold"])
        self.val_mean = float(gmm_data.get("val_mean", 190.0))
        self.val_std = float(gmm_data.get("val_std", 35.0))

    def _ensure_channel_first(self, x: torch.Tensor) -> torch.Tensor:
        """Automatyczna transpozycja z (128, 259, 3) na (1, 3, 128, 259)."""
        if x.dim() == 3 and x.shape[-1] == 3:
            return x.permute(2, 0, 1).unsqueeze(0).contiguous()
        elif x.dim() == 4 and x.shape[-1] == 3:
            return x.permute(0, 3, 1, 2).contiguous()
        elif x.dim() == 3 and x.shape[0] == 3:
            return x.unsqueeze(0).contiguous()
        return x

    def predict(self, x: torch.Tensor or np.ndarray) -> dict:
        """
        Główna funkcja inferencyjna dla pojedynczego okna 3.0s lub batcha.
        Zwraca werdykt ('normal' vs rodzaj awarii) oraz szczegółowe metryki prawdopodobieństwa.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        x = x.to(self.device)
        x = self._ensure_channel_first(x)

        with torch.no_grad():
            raw_e, norm_e = self.model.extract_features(x)
            h = F.relu(raw_e)
            logits = self.model.classifier(h)
            probs = F.softmax(logits, dim=1).cpu().numpy()

            norm_e_np = norm_e.cpu().numpy()
            logits_np = logits.cpu().numpy()

        # Obliczenie gęstości prawdopodobieństwa GMM
        log_densities = self.gmm.score_samples(norm_e_np)

        results = []
        for i in range(len(norm_e_np)):
            log_d = float(log_densities[i])
            best_fault_idx = int(np.argmax(logits_np[i]))
            best_fault_name = self.fault_classes[best_fault_idx]
            fault_prob = float(probs[i, best_fault_idx])

            # Decyzja OOD:
            # Jeśli gęstość prawdopodobieństwa leży poniżej progu klastrów awarii -> DYSK SPRAWNY (NORMAL)
            is_normal = (log_d < self.threshold)

            if is_normal:
                verdict = "normal"
                # Pewność normalności mierzona odchyleniem od progu
                diff = self.threshold - log_d
                conf = float(np.clip(0.60 + 0.40 * (diff / (self.val_std + 1e-6)), 0.60, 0.99))
            else:
                verdict = best_fault_name
                conf = float(np.clip(fault_prob, 0.60, 0.99))

            results.append({
                "verdict": verdict,
                "is_fault": not is_normal,
                "confidence": conf,
                "log_likelihood": log_d,
                "threshold": self.threshold,
                "fault_candidate": best_fault_name,
                "fault_probabilities": {self.fault_classes[k]: float(probs[i, k]) for k in range(3)}
            })

        if len(results) == 1:
            return results[0]
        return results
