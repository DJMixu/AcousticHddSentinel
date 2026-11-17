#!/usr/bin/env python3
"""
model_no_normal.py
Architektura sieci 2D CNN trenowanej WYŁĄCZNIE na błędach mechanicznych (Fault-Only Learning)
dla projektu Acoustic HDD Sentinel (Eksperyment PrototypBezNormal).

Klasy treningowe (3 klasy awarii):
  0: beeping
  1: clicking
  2: scratching

Detekcja braku awarii ('normal' / dysk sprawny):
  Realizowana w przestrzeni metrycznej 128D (Deep Metric Prototypes)
  w połączeniu z modelem energii logitów (Energy-Based Out-of-Fault Detection).
  Jeśli embedding dźwięku leży w odległości d_min > threshold od znanych awarii,
  układ automatycznie kwalifikuje sygnał jako 'normal'.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AcousticFaultOnlyCNN(nn.Module):
    def __init__(self, num_fault_classes: int = 3, embedding_dim: int = 128):
        super().__init__()
        self.num_fault_classes = num_fault_classes
        self.embedding_dim = embedding_dim
        self.fault_classes = ["beeping", "clicking", "scratching"]

        # Blok 1: Wejście (3, 128, 259) -> (32, 64, 129)
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.2)

        # Blok 2: (32, 64, 129) -> (64, 32, 64)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.2)

        # Blok 3: (64, 32, 64) -> (128, 16, 32)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.2)

        # Blok 4: (128, 16, 32) -> (256, 8, 16)
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(0.2)

        # Global Average Pooling -> wektor 256
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Ekstraktor embeddingów 128D
        self.fc_embed = nn.Linear(256, embedding_dim, bias=False)
        self.bn_embed = nn.BatchNorm1d(embedding_dim)
        self.drop_embed = nn.Dropout(0.35)

        # Głowica klasyfikacji 3 awarii
        self.classifier = nn.Linear(embedding_dim, num_fault_classes)

        # Bufory metryczne dla detekcji 'normal' (kalibrowane po treningu na zbiorze train)
        self.register_buffer("prototypes", torch.zeros(num_fault_classes, embedding_dim))
        self.register_buffer("thresholds", torch.zeros(num_fault_classes))
        self.register_buffer("global_threshold", torch.tensor(1.15))
        self.register_buffer("energy_threshold", torch.tensor(-2.5))
        self.register_buffer("is_calibrated", torch.tensor(False))

    def _ensure_channel_first(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != 3 and x.shape[-1] == 3:
                return x.permute(0, 3, 1, 2).contiguous()
        elif x.ndim == 3:
            if x.shape[0] != 3 and x.shape[-1] == 3:
                return x.permute(2, 0, 1).unsqueeze(0).contiguous()
            elif x.shape[0] == 3:
                return x.unsqueeze(0).contiguous()
        return x

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Zwraca (raw_embed, norm_embed) w przestrzeni cech 128D.
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

        # GAP + Dense 128D
        x = self.gap(x)
        x = torch.flatten(x, 1)
        raw_embed = self.bn_embed(self.fc_embed(x))
        norm_embed = F.normalize(raw_embed, p=2, dim=-1)

        return raw_embed, norm_embed

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Standardowe przejście w przód używane podczas treningu na 3 awariach.
        Zwraca (logits, norm_embed).
        """
        raw_embed, norm_embed = self.extract_features(x)
        h = F.relu(raw_embed)
        h = self.drop_embed(h)
        logits = self.classifier(h)
        return logits, norm_embed

    def calibrate_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor, percentile: float = 95.0):
        """
        Kalibruje prototypy i progi odległości na danych treningowych 3 awarii.
        """
        self.eval()
        device = self.prototypes.device
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        protos = torch.zeros(self.num_fault_classes, self.embedding_dim, device=device)
        threshs = torch.zeros(self.num_fault_classes, device=device)

        all_distances = []
        for c in range(self.num_fault_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                c_embeds = embeddings[mask]
                proto = c_embeds.mean(dim=0)
                proto = F.normalize(proto, p=2, dim=0)
                protos[c] = proto

                dists = torch.norm(c_embeds - proto, p=2, dim=-1)
                threshs[c] = torch.quantile(dists, percentile / 100.0)
                all_distances.extend(dists.detach().cpu().tolist())

        self.prototypes.copy_(protos)
        self.thresholds.copy_(threshs)

        # Globalny próg: maksymalny promień plus margines bezpieczeństwa
        glob_thresh = float(np.percentile(all_distances, 98.0)) + 0.15
        self.global_threshold.copy_(torch.tensor(glob_thresh, device=device))
        self.is_calibrated.copy_(torch.tensor(True, device=device))

    def predict_with_normal(self, x: torch.Tensor, temp: float = 1.0) -> dict:
        """
        Inferencja testowa: klasyfikacja 4-klasowa (normal, beeping, clicking, scratching)
        pomimo wytrenowania wyłącznie na awariach!
        """
        self.eval()
        with torch.no_grad():
            logits, norm_embed = self.forward(x)
            device = logits.device

            # 1. Obliczenie energii logitów (Energy score)
            # Im niższa energia (ujemna, duża wartość), tym silniejsza pewność awarii.
            # Dla szumu/normalnego dysku energia jest wysoka (bliska zeru).
            energy = -temp * torch.logsumexp(logits / temp, dim=-1)

            # 2. Obliczenie odległości metrycznych od prototypów 3 awarii
            # norm_embed: (B, 128), prototypes: (3, 128)
            diff = norm_embed.unsqueeze(1) - self.prototypes.unsqueeze(0)  # (B, 3, 128)
            dists = torch.norm(diff, p=2, dim=-1)  # (B, 3)
            min_dist, closest_fault_idx = torch.min(dists, dim=-1)  # (B,)

            # 3. Wyznaczenie prawdopodobieństwa braku awarii ('normal')
            # Używamy sigmoidy na różnicy odległości względem wykalibrowanego progu
            thresh = self.global_threshold.item()
            # normal_score rośnie gdy odległość przekracza próg
            margin = (min_dist - thresh) * 4.0
            p_normal = torch.sigmoid(margin)  # (B,)

            # Jeśli energia jest wysoka (brak silnej aktywacji żadnej awarii), podbijamy p_normal
            energy_score = torch.sigmoid((energy - (-2.0)) * 2.0)
            p_normal = torch.max(p_normal, energy_score)

            # Prawdopodobieństwa 3 awarii z Softmaxu
            fault_probs = F.softmax(logits, dim=-1)  # (B, 3)

            # Prawdopodobieństwa końcowe dla 4 klas: [beeping, clicking, normal, scratching]
            # Skalujemy awarie czynnikiem (1 - p_normal)
            b = logits.shape[0]
            final_probs = torch.zeros(b, 4, device=device)
            # Indeksy: 0: beeping, 1: clicking, 2: normal, 3: scratching
            final_probs[:, 0] = fault_probs[:, 0] * (1.0 - p_normal)
            final_probs[:, 1] = fault_probs[:, 1] * (1.0 - p_normal)
            final_probs[:, 2] = p_normal
            final_probs[:, 3] = fault_probs[:, 2] * (1.0 - p_normal)

            pred_class_idx = torch.argmax(final_probs, dim=-1)
            classes_4 = ["beeping", "clicking", "normal", "scratching"]
            pred_class_names = [classes_4[idx] for idx in pred_class_idx.cpu().tolist()]

            return {
                "pred_class": pred_class_names,
                "pred_idx": pred_class_idx.cpu().numpy(),
                "probabilities": final_probs.cpu().numpy(),
                "min_distance": min_dist.cpu().numpy(),
                "threshold": thresh,
                "energy": energy.cpu().numpy(),
                "is_fault": (p_normal < 0.50).cpu().numpy()
            }
