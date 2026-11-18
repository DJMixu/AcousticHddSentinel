#!/usr/bin/env python3
"""
smpc_ood_engine.py
Silnik detekcji Out-Of-Distribution (OOD) oparty na architekturze SMPC
dla projektu Acoustic HDD Sentinel (PrototypBezNormal).

Single Source of Truth: ADR-001, AGENTS.md, PROJECT.md, SCOPE.md
Klasa SMPCFaultOODDetector integruje wyuczoną reprezentację sferyczną modelu
z kalibrowanym progiem tau na zbiorze walidacyjnym awarii.
"""

from typing import Dict, Any, Tuple, Union, Optional
import json
import sys
from pathlib import Path

# Zapewnienie dostępu do głównego katalogu repozytorium w sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from PrototypBezNormal.model_smpc import AcousticSMPCModel
from PrototypBezNormal.dataset_no_normal import FAULT_CLASSES_3


class SMPCFaultOODDetector:
    """
    Silnik detektora OOD opartego na modelu SMPC.
    Klasyfikuje 3-sekundowe wycinki do jednej z klas:
    - 'normal': dysk sprawny (gdy wynik kompozytowy OOD < próg tau)
    - 'beeping' / 'clicking' / 'scratching': wykryte uszkodzenie mechaniczne.
    """

    def __init__(
        self,
        weights_path: str = "PrototypBezNormal/output_smpc/best_smpc_model.pt",
        config_path: str = "PrototypBezNormal/output_smpc/smpc_ood_config.json",
        device: str = "cpu"
    ):
        self.weights_path = Path(weights_path)
        self.config_path = Path(config_path)

        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku konfiguracji OOD: {self.config_path}")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku wag modelu SMPC: {self.weights_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.threshold = float(self.config.get("threshold", 0.50))
        self.temperature = float(self.config.get("temperature", 2.0))
        self.alpha_energy = float(self.config.get("alpha_energy", 0.10))
        self.scale = float(self.config.get("scale", 16.0))
        self.num_classes = int(self.config.get("num_classes", 3))
        self.sub_centers = int(self.config.get("sub_centers", 2))
        self.embedding_dim = int(self.config.get("embedding_dim", 128))
        self.fault_classes = self.config.get("classes_3", FAULT_CLASSES_3)

        val_stats = self.config.get("val_stats", {})
        self.val_std = float(val_stats.get("std", 0.15))

        # Ładowanie wag modelu
        ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))

        self.model = AcousticSMPCModel(
            num_fault_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
            sub_centers=self.sub_centers,
            scale=self.scale
        ).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def compute_ood_score(
        self,
        x: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[float, int, float, torch.Tensor]:
        """
        Oblicza wynik OOD dla pojedynczego tensora (lub pierwszego z batcha).
        Zwraca:
            (ood_score: float, candidate_fault_class_idx: int, confidence: float, probs: torch.Tensor)
        """
        x_pt = self.model._ensure_channel_first(x).to(self.device)

        with torch.no_grad():
            logits, norm_embed = self.model(x_pt)
            cos_sims = self.model.compute_cosine_similarities(norm_embed)  # (B, 3)

            max_cos, cand_indices = torch.max(cos_sims, dim=-1)
            energy = torch.logsumexp(logits / self.temperature, dim=-1)
            composite = max_cos + self.alpha_energy * energy
            probs = F.softmax(logits, dim=-1)

        ood_score = float(composite[0].item())
        cand_idx = int(cand_indices[0].item())
        fault_prob = float(probs[0, cand_idx].item())

        # Obliczenie estymacji pewności
        if ood_score >= self.threshold:
            # Awaria
            conf = float(np.clip(fault_prob, 0.60, 0.99))
        else:
            # Normal / Dysk sprawny (im dalej poniżej progu, tym wyższa pewność normalności)
            diff = self.threshold - ood_score
            conf = float(np.clip(0.60 + 0.40 * (diff / (self.val_std + 1e-6)), 0.60, 0.99))

        return ood_score, cand_idx, conf, probs[0]

    def predict_chunk(
        self,
        chunk_tensor: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Dokonuje pełnej predykcji dla pojedynczego 3-sekundowego okna audio.
        Zgodne z interfejsem ADR-001 i SCOPE.md.
        """
        ood_score, cand_idx, conf, probs = self.compute_ood_score(chunk_tensor)
        fault_candidate = self.fault_classes[cand_idx]
        is_fault = (ood_score >= self.threshold)

        predicted_class = fault_candidate if is_fault else "normal"

        probs_dict = {
            self.fault_classes[k]: float(probs[k].item())
            for k in range(len(self.fault_classes))
        }

        return {
            "predicted_class": predicted_class,
            "verdict": predicted_class,
            "is_fault": is_fault,
            "ood_score": float(ood_score),
            "threshold": float(self.threshold),
            "confidence": float(conf),
            "fault_candidate": fault_candidate,
            "fault_probabilities": probs_dict
        }

    def predict_batch(
        self,
        batch_tensors: Union[np.ndarray, torch.Tensor]
    ) -> list[Dict[str, Any]]:
        """
        Szybka predykcja dla batcha wielu okien czasowych jednocześnie.
        """
        x_pt = self.model._ensure_channel_first(batch_tensors).to(self.device)

        with torch.no_grad():
            logits, norm_embed = self.model(x_pt)
            cos_sims = self.model.compute_cosine_similarities(norm_embed)

            max_cos, cand_indices = torch.max(cos_sims, dim=-1)
            energy = torch.logsumexp(logits / self.temperature, dim=-1)
            composite = max_cos + self.alpha_energy * energy
            probs = F.softmax(logits, dim=-1)

        comp_np = composite.cpu().numpy()
        cands_np = cand_indices.cpu().numpy()
        probs_np = probs.cpu().numpy()

        results = []
        for i in range(len(comp_np)):
            ood_score = float(comp_np[i])
            cand_idx = int(cands_np[i])
            fault_candidate = self.fault_classes[cand_idx]
            fault_prob = float(probs_np[i, cand_idx])
            is_fault = (ood_score >= self.threshold)

            if is_fault:
                predicted_class = fault_candidate
                conf = float(np.clip(fault_prob, 0.60, 0.99))
            else:
                predicted_class = "normal"
                diff = self.threshold - ood_score
                conf = float(np.clip(0.60 + 0.40 * (diff / (self.val_std + 1e-6)), 0.60, 0.99))

            probs_dict = {
                self.fault_classes[k]: float(probs_np[i, k])
                for k in range(len(self.fault_classes))
            }

            results.append({
                "predicted_class": predicted_class,
                "verdict": predicted_class,
                "is_fault": is_fault,
                "ood_score": ood_score,
                "threshold": float(self.threshold),
                "confidence": conf,
                "fault_candidate": fault_candidate,
                "fault_probabilities": probs_dict
            })

        return results
