#!/usr/bin/env python3
"""
model_smpc.py
Architektura sieci 2D CNN z głowicą Sub-Center Prototypes (SMPC-OOD)
dla projektu Acoustic HDD Sentinel (Eksperyment PrototypBezNormal).

Single Source of Truth: ADR-001, AGENTS.md, PROJECT.md
Cechy modelu:
1. 4 bloki konwolucyjne 2D CNN (32 -> 64 -> 128 -> 256) z BatchNorm, MaxPool i Dropout2d.
2. Global Average Pooling (256D) -> Linear Projection (128D) + BatchNorm1d + L2 sferyczna normalizacja na S^{127}.
3. Wielomodalna głowica sub-centrów (M=2 sub-centry per klasa awarii) pozwalająca na uchwycenie
   różnych pod-reżimów uszkodzenia (np. tarcie ciągłe vs stick-slip chatter dla scratching).
4. Przeliczanie podobieństwa cosinusowego: cos theta_k = max_{m=1..M} (z · W_{k,m}).
5. Logity kątowe: logits = scale * cos_theta (s=16.0).
6. Transparentna obsługa formatów Channel-First (B, 3, 128, 259) i Channel-Last (B, 128, 259, 3) / unbatched.
"""

from typing import Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Stałe wejściowe zgodne z ADR-001
INPUT_SHAPE_NPY = (128, 259, 3)     # Channel-Last na dysku
INPUT_SHAPE_PT = (3, 128, 259)      # Channel-First w PyTorch
DEFAULT_NUM_FAULT_CLASSES = 3       # beeping (0), clicking (1), scratching (2)
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_SUB_CENTERS = 2
DEFAULT_SCALE = 16.0


class AcousticSMPCModel(nn.Module):
    """
    Model 2D CNN z reprezentacją Sub-Center Spherical Prototypes.
    Uczony wyłącznie na 3 klasach uszkodzeń mechanicznych (Zero Normal Leakage).
    """

    def __init__(
        self,
        num_fault_classes: int = DEFAULT_NUM_FAULT_CLASSES,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        sub_centers: int = DEFAULT_SUB_CENTERS,
        scale: float = DEFAULT_SCALE
    ):
        super().__init__()
        self.num_fault_classes = num_fault_classes
        self.embedding_dim = embedding_dim
        self.sub_centers = sub_centers
        self.scale = scale
        self.fault_classes = ["beeping", "clicking", "scratching"]

        # =====================================================================
        # BLOK 1: (B, 3, 128, 259) -> (B, 32, 64, 129)
        # =====================================================================
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p=0.2)

        # =====================================================================
        # BLOK 2: (B, 32, 64, 129) -> (B, 64, 32, 64)
        # =====================================================================
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(p=0.2)

        # =====================================================================
        # BLOK 3: (B, 64, 32, 64) -> (B, 128, 16, 32)
        # =====================================================================
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(p=0.2)

        # =====================================================================
        # BLOK 4: (B, 128, 16, 32) -> (B, 256, 8, 16)
        # =====================================================================
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout2d(p=0.2)

        # =====================================================================
        # Global Average Pooling & Projekcja L2
        # =====================================================================
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_proj = nn.Linear(256, embedding_dim, bias=False)
        self.bn_proj = nn.BatchNorm1d(embedding_dim)

        # =====================================================================
        # Wagi prototypów sub-centrów: (K, M, D)
        # =====================================================================
        self.prototypes = nn.Parameter(
            torch.empty(num_fault_classes, sub_centers, embedding_dim)
        )
        self._init_prototypes()

    def _init_prototypes(self):
        """Inicjalizacja wag prototypów Xavier / sferyczna unitarna."""
        flattened = self.prototypes.view(self.num_fault_classes * self.sub_centers, self.embedding_dim)
        nn.init.xavier_uniform_(flattened)

    @property
    def prototypes_weight(self) -> torch.Tensor:
        """Alias dla parametru wag prototypów."""
        return self.prototypes

    def _ensure_channel_first(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Zapewnia format Channel-First (B, 3, 128, 259) w PyTorch.
        Obsługuje Channel-Last (B, 128, 259, 3) oraz wycinki pojedyncze (128, 259, 3) i numpy.ndarray.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if x.dtype != torch.float32:
            x = x.float()

        if x.ndim == 3:
            # Pojedynczy wycinek
            if x.shape[0] == 3 and x.shape[1] == 128 and x.shape[2] == 259:
                x = x.unsqueeze(0)
            elif x.shape[-1] == 3 and x.shape[0] == 128 and x.shape[1] == 259:
                x = x.permute(2, 0, 1).unsqueeze(0)
            elif x.shape[0] == 3:
                x = x.unsqueeze(0)
            elif x.shape[-1] == 3:
                x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.ndim == 4:
            # Batch
            if x.shape[1] != 3 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)

        return x.contiguous()

    def extract_features(self, x: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ekstrakcja cech widmowych.
        Zwraca:
            raw_features: (B, 128) po projekcji liniowej i BatchNorm1d
            norm_embeddings: (B, 128) znormalizowane wektory jednostkowe na S^{127}
        """
        x = self._ensure_channel_first(x)

        # Blok 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.drop1(self.pool1(x))

        # Blok 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.drop2(self.pool2(x))

        # Blok 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.drop3(self.pool3(x))

        # Blok 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.drop4(self.pool4(x))

        # GAP -> projekcja 128D
        gap_vec = torch.flatten(self.gap(x), 1)
        raw_features = self.bn_proj(self.fc_proj(gap_vec))
        norm_embeddings = F.normalize(raw_features, p=2, dim=-1)

        return raw_features, norm_embeddings

    def compute_cosine_similarities(self, z: torch.Tensor) -> torch.Tensor:
        """
        Oblicza maksymalne podobieństwo cosinusowe dla każdej klasy uszkodzenia
        względem jej M sub-centrów:
            cos theta_k = max_{m=1..M} (z · W_{k,m})
        Zwraca:
            (B, num_fault_classes) z wartościami w zakresie [-1, 1].
        """
        z_norm = F.normalize(z, p=2, dim=-1)
        w_norm = F.normalize(self.prototypes, p=2, dim=-1)  # (K, M, D)

        # Podobieństwo macierzowe (B, K, M)
        sims = torch.einsum("bd,kmd->bkm", z_norm, w_norm)
        # Maksymalne podobieństwo w obrębie sub-centrów danej klasy
        max_sims, _ = torch.max(sims, dim=-1)  # (B, K)
        return max_sims

    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Przejście w przód modelu SMPC.
        Zwraca:
            logits: (B, num_fault_classes) = scale * cos_similarities
            norm_embeddings: (B, 128) znormalizowany wektor cech na sferze
        """
        raw_features, norm_embeddings = self.extract_features(x)
        cos_sims = self.compute_cosine_similarities(norm_embeddings)
        logits = self.scale * cos_sims
        return logits, norm_embeddings
