"""
Moduł zaawansowanych funkcji kosztu metrycznego dla uczenia bez klasy 'normal'.
Single Source of Truth: ADR-001, ArcFace (Deng et al., 2019), Center Loss (Wen et al., 2016).

Zawiera:
1. CenterLoss: Wymusza zwartość wewnątrzklasową embeddingów w przestrzeni R^d.
2. ArcMarginProduct: Wymusza marżę kątową na hiperkuli sferycznej S^{d-1}.
3. CombinedMetricLoss: Hybrydowa funkcja straty (ArcFace + lambda * CenterLoss + Class Weights).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """
    Center Loss: L_center = 0.5 * sum(||z_i - c_{y_i}||_2^2)
    Uczy środki klastrów (centroidy) dla każdej z klas awarii,
    ściągając próbki tej samej klasy do gęstego, zwartego jądra.
    """
    def __init__(self, num_classes: int = 3, feat_dim: int = 128, device: torch.device = None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device or torch.device("cpu")
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=self.device))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: Tensor (B, feat_dim)
        labels: Tensor (B,) z indeksami klas
        """
        batch_size = features.size(0)
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(features, self.centers.t(), beta=1.0, alpha=-2.0)

        classes = torch.arange(self.num_classes, device=features.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss


class ArcMarginProduct(nn.Module):
    """
    ArcMargin (ArcFace): Additive Angular Margin Classifier.
    Wagi W i wektory cech z są rzutowane na sferę unitarną S^{d-1}.
    Logit klasy poprawnej to s * cos(theta_y + m).
    Dzięki marży kątowej m każda klasa tworzy stożek o ograniczonym kącie rozwarcia.
    Próbki tła / zdrowego dysku (normal) mają cos(theta) < cos(theta_thresh),
    co pozwala na ich bezbłędną identyfikację jako Out-Of-Distribution (OOD).
    """
    def __init__(
        self,
        in_features: int = 128,
        out_features: int = 3,
        scale: float = 16.0,
        margin: float = 0.35,
        easy_margin: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input_features: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        """
        input_features: znormalizowane cechy (B, in_features)
        label: opcjonalne etykiety do nałożenia marży kątowej w treningu.
               Jeśli None (inferencja), zwraca s * cos(theta).
        """
        # Normalizacja cech i wag na sferze S^{d-1}
        x_norm = F.normalize(input_features, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Czysty cosinus kąta między cechami a wagami klas
        cosine = F.linear(x_norm, w_norm)
        
        if label is None:
            # W trybie inferencji zwracamy skalowane logity cosinusowe
            return cosine * self.scale

        # Obliczenie cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        sine = torch.sqrt(1.0 - torch.clamp(torch.pow(cosine, 2), 0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask dla klas poprawnych
        one_hot = torch.zeros(cosine.size(), device=input_features.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output
