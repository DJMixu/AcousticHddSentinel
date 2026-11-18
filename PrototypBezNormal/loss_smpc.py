#!/usr/bin/env python3
"""
loss_smpc.py
Funkcje strat dla architektury Sub-Center Spherical Prototypes (SMPC-OOD).

Single Source of Truth: ADR-001, AGENTS.md, PROJECT.md
Zawiera:
1. SubCenterArcMarginLoss: Additive Angular Margin Loss (ArcFace) na sferze S^{127}
   z parametrami scale=16.0, margin=0.20 i wagami klas.
2. compute_orthogonality_loss: Regularyzacja ortogonalności sub-centrów
   sum_{k=0}^{K-1} max(0, W_{k,1} · W_{k,2} - 0.30) wymuszająca rozbieżność
   akustyczną modów w obrębie tej samej awarii (np. ciągłe tarcie vs stick-slip).
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubCenterArcMarginLoss(nn.Module):
    """
    Sub-Center Additive Angular Margin Loss (ArcFace).
    Nakłada karę kątową margin (m) na maksymalne podobieństwo cosinusowe klasy docelowej:
        cos(theta_{y_i} + m) = cos(theta) * cos(m) - sin(theta) * sin(m)
    Następnie skaluje logity czynnikiem scale (s) i wylicza CrossEntropyLoss.
    """

    def __init__(
        self,
        scale: float = 16.0,
        margin: float = 0.20,
        easy_margin: bool = False,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.label_smoothing = label_smoothing

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, cosine_similarities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        cosine_similarities: Tensor (B, num_classes) z cos(theta) lub skalowanymi logitami
        targets: Tensor (B,) z indeksami klas awarii
        """
        # Obsługa zarówno czystych cosinusów [-1, 1], jak i przeskalowanych logitów s * cos
        if cosine_similarities.max() > 1.05 or cosine_similarities.min() < -1.05:
            cosine = cosine_similarities / self.scale
        else:
            cosine = cosine_similarities

        # Stabilizacja numeryczna przed liczeniem sinusa
        eps = 1e-7
        cosine_clamped = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
        sine = torch.sqrt(1.0 - torch.pow(cosine_clamped, 2))

        # Obliczenie cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine_clamped * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine_clamped > 0, phi, cosine_clamped)
        else:
            phi = torch.where(cosine_clamped > self.th, phi, cosine_clamped - self.mm)

        # Maska one-hot dla klas docelowych
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1.0)

        # Nałożenie marży wyłącznie na klasę docelową
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        weights = self.class_weights if self.class_weights is not None else None
        loss = F.cross_entropy(output, targets, weight=weights, label_smoothing=self.label_smoothing)
        return loss


def compute_orthogonality_loss(prototypes_weight: torch.Tensor, threshold: float = 0.30) -> torch.Tensor:
    """
    Wylicza karę ortogonalności sub-centrów:
        L_ortho = sum_{k=0}^{K-1} max(0, W_{k,1} · W_{k,2} - threshold)
    na znormalizowanych sferycznie prototypach (K, M, D).

    Dla M=2:
        Wymusza, aby dwa sub-centry tej samej klasy awarii miały iloczyn skalarny <= threshold,
        zapobiegając kolapsowi do pojedynczego punktu.
    """
    w_norm = F.normalize(prototypes_weight, p=2, dim=-1)  # (K, M, D)
    K, M, D = w_norm.shape

    if M < 2:
        return torch.tensor(0.0, device=prototypes_weight.device, dtype=prototypes_weight.dtype)

    if M == 2:
        w1 = w_norm[:, 0, :]  # (K, D)
        w2 = w_norm[:, 1, :]  # (K, D)
        dot_prods = (w1 * w2).sum(dim=-1)  # (K,)
        penalties = torch.clamp(dot_prods - threshold, min=0.0)
        return penalties.sum()

    # Przypadek ogólny dla M > 2
    total_penalty = torch.tensor(0.0, device=prototypes_weight.device, dtype=prototypes_weight.dtype)
    for k in range(K):
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                dot_prod = torch.dot(w_norm[k, m1], w_norm[k, m2])
                total_penalty = total_penalty + torch.clamp(dot_prod - threshold, min=0.0)

    return total_penalty
