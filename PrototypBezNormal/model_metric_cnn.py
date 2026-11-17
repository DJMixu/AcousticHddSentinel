"""
Zaawansowany model 2D CNN dla klasyfikacji uszkodzeń HDD z uczeniem metrycznym (ArcFace + Center Loss)
i wielowymiarową detekcją Out-Of-Distribution (OOD) dla klasy 'normal'.

Single Source of Truth: ADR-001, ArcFace, Mahalanobis OOD (Lee et al., 2018), Free Energy OOD (Liu et al., 2020).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PrototypBezNormal.loss_functions import ArcMarginProduct


class MetricFaultOnlyCNN(nn.Module):
    """
    Sieć konwolucyjna 2D dla detekcji i klasyfikacji uszkodzeń mechanicznych HDD.
    Trenowana wyłącznie na 3 klasach awarii (beeping, clicking, scratching).
    Stan 'normal' jest wykrywany analitycznie i geometrycznie jako Out-Of-Distribution.
    """
    def __init__(
        self,
        num_fault_classes: int = 3,
        embedding_dim: int = 128,
        scale: float = 16.0,
        margin: float = 0.35,
        dropout_rate: float = 0.25
    ):
        super().__init__()
        self.num_fault_classes = num_fault_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin
        self.fault_classes = ["beeping", "clicking", "scratching"]

        # === Blok 1: Niskopoziomowe cechy widmowo-czasowe ===
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(dropout_rate * 0.5)

        # === Blok 2: Transjenty i rezonanse pośrednie ===
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(dropout_rate * 0.75)

        # === Blok 3: Struktury harmoniczne i wzorce impulsowe ===
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(dropout_rate)

        # === Blok 4: Wysokopoziomowe sygnatury uszkodzeń ===
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout2d(dropout_rate)

        # === Global Average Pooling i Rzutowanie do Przestrzeni Metrycznej ===
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_embed = nn.Linear(256, embedding_dim, bias=False)
        self.bn_embed = nn.BatchNorm1d(embedding_dim)

        # === Warstwa Klasyfikatora Kątowego (ArcMargin) ===
        self.arc_margin = ArcMarginProduct(
            in_features=embedding_dim,
            out_features=num_fault_classes,
            scale=scale,
            margin=margin
        )

        # === Bufory Kalibracyjne OOD (zapisywane w wagach modelu) ===
        self.register_buffer("is_calibrated", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("class_centers", torch.zeros(num_fault_classes, embedding_dim))
        self.register_buffer("pooled_cov_inv", torch.eye(embedding_dim))
        self.register_buffer("global_cosine_threshold", torch.tensor(0.50))
        self.register_buffer("class_cosine_thresholds", torch.zeros(num_fault_classes))
        self.register_buffer("global_mahalanobis_threshold", torch.tensor(10.0))
        self.register_buffer("global_energy_threshold", torch.tensor(-1.5))

    def _ensure_channel_first(self, x: torch.Tensor) -> torch.Tensor:
        """Automatyczna transpozycja (B, 128, 259, 3) -> (B, 3, 128, 259)."""
        if x.dim() == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
            return x.permute(0, 3, 1, 2).contiguous()
        elif x.dim() == 3 and x.shape[-1] == 3:
            return x.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        elif x.dim() == 3 and x.shape[0] == 3:
            return x.unsqueeze(0).contiguous()
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zwraca znormalizowany wektor cech na hiperkuli unitarnej S^{d-1} (B, embedding_dim).
        """
        x = self._ensure_channel_first(x)

        # Blok 1
        x = F.leaky_relu(self.bn1_1(self.conv1_1(x)), 0.1)
        x = F.leaky_relu(self.bn1_2(self.conv1_2(x)), 0.1)
        x = self.drop1(self.pool1(x))

        # Blok 2
        x = F.leaky_relu(self.bn2_1(self.conv2_1(x)), 0.1)
        x = F.leaky_relu(self.bn2_2(self.conv2_2(x)), 0.1)
        x = self.drop2(self.pool2(x))

        # Blok 3
        x = F.leaky_relu(self.bn3_1(self.conv3_1(x)), 0.1)
        x = F.leaky_relu(self.bn3_2(self.conv3_2(x)), 0.1)
        x = self.drop3(self.pool3(x))

        # Blok 4
        x = F.leaky_relu(self.bn4_1(self.conv4_1(x)), 0.1)
        x = F.leaky_relu(self.bn4_2(self.conv4_2(x)), 0.1)
        x = self.drop4(self.pool4(x))

        # GAP + Normalizacja
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.bn_embed(self.fc_embed(x))
        norm_embed = F.normalize(x, p=2, dim=-1)
        return norm_embed

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Standardowe przejście w przód używane w treningu i inferencji.
        Zwraca: (logits, norm_embed).
        """
        norm_embed = self.extract_features(x)
        logits = self.arc_margin(norm_embed, labels)
        return logits, norm_embed

    def calibrate_ood_parameters(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        percentile: float = 95.0,
        regularization: float = 1e-4
    ):
        """
        Rygorystyczna kalibracja parametrów OOD WYŁĄCZNIE na danych awarii (zbiór walidacyjny / treningowy).
        Zero dostępu do próbek 'normal' w trakcie wyznaczania progów!
        """
        device = embeddings.device
        embs_np = embeddings.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()

        num_classes = self.num_fault_classes
        feat_dim = self.embedding_dim

        # 1. Obliczenie centroidów klas na sferze
        centers = np.zeros((num_classes, feat_dim), dtype=np.float32)
        for c in range(num_classes):
            mask = (labels_np == c)
            if np.sum(mask) > 0:
                mean_c = np.mean(embs_np[mask], axis=0)
                centers[c] = mean_c / (np.linalg.norm(mean_c) + 1e-9)
            else:
                centers[c] = np.zeros(feat_dim, dtype=np.float32)

        self.class_centers.copy_(torch.from_numpy(centers).to(device))

        # 2. Obliczenie wspólnej macierzy kowariancji (Pooled Covariance Matrix)
        pooled_cov = np.zeros((feat_dim, feat_dim), dtype=np.float32)
        total_samples = len(embs_np)

        for c in range(num_classes):
            mask = (labels_np == c)
            if np.sum(mask) > 0:
                diff = embs_np[mask] - centers[c]
                pooled_cov += np.dot(diff.T, diff)

        pooled_cov /= max(1, total_samples)
        pooled_cov += regularization * np.eye(feat_dim, dtype=np.float32)
        inv_cov = np.linalg.inv(pooled_cov).astype(np.float32)
        self.pooled_cov_inv.copy_(torch.from_numpy(inv_cov).to(device))

        # 3. Wyznaczenie progów cosinusowych dla awarii (dolny percentyl)
        # Próbki awarii mają wysoki cosinus. Próg odcina (100 - percentile)% najsłabszych próbek.
        lower_percentile = 100.0 - percentile  # np. 5.0%
        class_thresh = np.zeros(num_classes, dtype=np.float32)
        all_correct_cosines = []

        for c in range(num_classes):
            mask = (labels_np == c)
            if np.sum(mask) > 0:
                cos_sims = np.dot(embs_np[mask], centers[c])
                class_thresh[c] = float(np.percentile(cos_sims, lower_percentile))
                all_correct_cosines.extend(cos_sims.tolist())
            else:
                class_thresh[c] = 0.50

        self.class_cosine_thresholds.copy_(torch.from_numpy(class_thresh).to(device))
        global_cos = float(np.percentile(all_correct_cosines, lower_percentile)) if all_correct_cosines else 0.50
        self.global_cosine_threshold.fill_(global_cos)

        # 4. Wyznaczenie progu Mahalanobisa (górny percentyl awarii)
        # Próbki awarii mają MAŁY dystans Mahalanobisa do swojego centroidu.
        all_mahals = []
        for i in range(len(embs_np)):
            c = labels_np[i]
            diff = embs_np[i] - centers[c]
            m_dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
            all_mahals.append(m_dist)

        global_mahal = float(np.percentile(all_mahals, percentile))
        self.global_mahalanobis_threshold.fill_(global_mahal)

        # 5. Wyznaczenie progu Free Energy (górny percentyl energii)
        # E(x) = -T * log(sum(exp(l / T)))
        T = 1.0
        energies = -T * np.log(np.sum(np.exp(logits_np / T), axis=1) + 1e-9)
        global_energy = float(np.percentile(energies, percentile))
        self.global_energy_threshold.fill_(global_energy)

        self.is_calibrated.fill_(True)

        return {
            "global_cosine_threshold": global_cos,
            "class_cosine_thresholds": {self.fault_classes[i]: float(class_thresh[i]) for i in range(num_classes)},
            "global_mahalanobis_threshold": global_mahal,
            "global_energy_threshold": global_energy,
            "calibrated_samples": total_samples
        }

    def predict_ood(self, x: torch.Tensor) -> dict:
        """
        Kompletna inferencja OOD dla pojedynczego okna lub batcha okien.
        Zwraca werdykt ('normal' vs rodzaj awarii) oraz szczegółowe metryki decyzyjne.
        """
        self.eval()
        with torch.no_grad():
            x = self._ensure_channel_first(x)
            logits, embs = self.forward(x, labels=None)

            # Cosinusy do wag ArcMargin (wagi są znormalizowane)
            w_norm = F.normalize(self.arc_margin.weight, p=2, dim=1)  # (3, 128)
            cos_matrix = torch.matmul(embs, w_norm.t())  # (B, 3)

            # Najbardziej prawdopodobna awaria i maksymalny cosinus
            max_cos, fault_indices = torch.max(cos_matrix, dim=1)
            fault_probs = F.softmax(logits, dim=1)

            # Odległość Mahalanobisa
            embs_np = embs.cpu().numpy()
            inv_cov_np = self.pooled_cov_inv.cpu().numpy()
            centers_np = self.class_centers.cpu().numpy()

            results = []
            for b in range(len(embs_np)):
                f_idx = fault_indices[b].item()
                f_name = self.fault_classes[f_idx]
                cos_val = float(max_cos[b].item())

                # Mahalanobis do centroidu przewidywanej awarii
                diff = embs_np[b] - centers_np[f_idx]
                mahal_dist = float(np.sqrt(np.dot(np.dot(diff, inv_cov_np), diff.T)))

                # Free Energy
                l = logits[b].cpu().numpy()
                energy = float(-np.log(np.sum(np.exp(l)) + 1e-9))

                # Progi decyzyjne
                cos_th = float(self.class_cosine_thresholds[f_idx].item()) if self.is_calibrated else 0.50
                mahal_th = float(self.global_mahalanobis_threshold.item()) if self.is_calibrated else 12.0
                energy_th = float(self.global_energy_threshold.item()) if self.is_calibrated else -1.0

                # Decyzja OOD:
                # Jeśli kąt do stożka awarii jest zbyt duży (cosinus za mały)
                # LUB dystans Mahalanobisa jest za duży
                # LUB Free Energy wskazuje na brak dopasowania do uszkodzeń:
                is_normal = (cos_val < cos_th) or (mahal_dist > mahal_th) or (energy > energy_th)

                if is_normal:
                    verdict = "normal"
                    # Pewność normalności mierzona odchyleniem od progu cosinusa
                    confidence = float(np.clip(1.0 - (cos_val / max(cos_th, 1e-4)), 0.50, 0.99))
                else:
                    verdict = f_name
                    # Pewność awarii to prawdopodobieństwo softmax skorygowane o dopasowanie kątowe
                    raw_prob = float(fault_probs[b, f_idx].item())
                    confidence = float(np.clip(raw_prob * (cos_val / max(cos_th, 1e-4)), 0.50, 0.99))

                results.append({
                    "verdict": verdict,
                    "is_fault": not is_normal,
                    "confidence": confidence,
                    "fault_candidate": f_name,
                    "fault_probability": float(fault_probs[b, f_idx].item()),
                    "metrics": {
                        "cosine_similarity": cos_val,
                        "cosine_threshold": cos_th,
                        "mahalanobis_distance": mahal_dist,
                        "mahalanobis_threshold": mahal_th,
                        "free_energy": energy,
                        "energy_threshold": energy_th
                    }
                })

            if len(results) == 1:
                return results[0]
            return results
