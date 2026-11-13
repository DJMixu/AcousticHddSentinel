"""
HDD Acoustic Fault Detection - 2D CNN Model Architecture (Pure PyTorch)
Single Source of Truth: ADR-001 & AGENTS.md

Moduł definiujący architekturę sieci 2D CNN w PyTorch przystosowaną
do przetwarzania 3-kanałowych reprezentacji czasowo-częstotliwościowych [Log-Mel, Delta, Delta-Delta]
wycinków audio dysków HDD.
Obsługuje tensory (128, 259, 3) oraz (3, 128, 259).
"""

from typing import Tuple, Optional, Union, List, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Stałe DSP / wejściowe zgodne z ADR-001
INPUT_SHAPE: Tuple[int, int, int] = (128, 259, 3)       # Channel-Last (H, W, C) - format dyskowy .npy
INPUT_SHAPE_PT: Tuple[int, int, int] = (3, 128, 259)    # Channel-First (C, H, W) - format wewnętrzny PyTorch
DEFAULT_NUM_CLASSES: int = 4                            # np. clicking, beeping, scratching, normal


class AcousticHddSentinelCNN(nn.Module):
    """
    Model 2D CNN w PyTorch dedykowany do akustycznej detekcji i klasyfikacji uszkodzeń HDD.
    Architektura oparta na 4 blokach konwolucyjnych z Batch Normalization (eps=1e-3),
    warstwami poolingowymi oraz Global Average Pooling, co zapobiega przeuczeniu
    (overfitting) na małych i średnich zbiorach akustycznych, zachowując wysoką
    czułość na impulsowe transjenty (kanał Delta) oraz wąskopasmowe piski (Log-Mel).
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout_rate: float = 0.3,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE_PT
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.input_shape = input_shape

        # =========================================================================
        # BLOK 1: Niskopoziomowa ekstrakcja cech widmowych i transjentów
        # Wejście: (B, 3, 128, 259) -> Wyjście po poolingu: (B, 32, 64, 129)
        # =========================================================================
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)
        self.relu1_1 = nn.ReLU(inplace=False)

        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)
        self.relu1_2 = nn.ReLU(inplace=False)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.s_dropout1 = nn.Dropout2d(p=dropout_rate * 0.5)

        # =========================================================================
        # BLOK 2: Średniopoziomowe wzorce czasowo-częstotliwościowe (klikanie głowicy)
        # Wejście: (B, 32, 64, 129) -> Wyjście po poolingu: (B, 64, 32, 64)
        # =========================================================================
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        self.relu2_1 = nn.ReLU(inplace=False)

        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        self.relu2_2 = nn.ReLU(inplace=False)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.s_dropout2 = nn.Dropout2d(p=dropout_rate * 0.75)

        # =========================================================================
        # BLOK 3: Wysokopoziomowa dynamika zmian (zatarte łożyska, rezonanse)
        # Wejście: (B, 64, 32, 64) -> Wyjście po poolingu: (B, 128, 16, 32)
        # =========================================================================
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128, eps=1e-3, momentum=0.01)
        self.relu3_1 = nn.ReLU(inplace=False)

        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128, eps=1e-3, momentum=0.01)
        self.relu3_2 = nn.ReLU(inplace=False)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.s_dropout3 = nn.Dropout2d(p=dropout_rate)

        # =========================================================================
        # BLOK 4: Semantyczna reprezentacja akustyczna
        # Wejście: (B, 128, 16, 32) -> Wyjście po poolingu: (B, 256, 8, 16)
        # =========================================================================
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256, eps=1e-3, momentum=0.01)
        self.relu4_1 = nn.ReLU(inplace=False)

        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(256, eps=1e-3, momentum=0.01)
        self.relu4_2 = nn.ReLU(inplace=False)

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(p=dropout_rate)

        # =========================================================================
        # AGREGACJA PRZESTRZENNO-CZASOWA (Global Average Pooling) -> (B, 256)
        # =========================================================================
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Gęsta warstwa projekcyjna (Dense Projection Layer)
        self.dense_features = nn.Linear(256, 128, bias=False)
        self.bn_dense = nn.BatchNorm1d(128, eps=1e-3, momentum=0.01)
        self.relu_dense = nn.ReLU(inplace=False)
        self.dense_dropout = nn.Dropout(p=dropout_rate)

        # =========================================================================
        # WARSTWA WYJŚCIOWA (Klasyfikator)
        # =========================================================================
        self.predictions = nn.Linear(128, num_classes)

        # =========================================================================
        # OOD PROTOTYPE REJECTION (Open-Set Recognition wewnątrz sieci)
        # Bufory dla znanych klas uszkodzeń: [beeping (0), clicking (1), scratching (3)]
        # =========================================================================
        self.register_buffer("use_open_set", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("fault_prototypes", torch.zeros((3, 128), dtype=torch.float32))
        self.register_buffer("fault_thresholds", torch.full((3,), 0.40, dtype=torch.float32))
        self.register_buffer("gamma", torch.tensor(12.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass sieci.
        Obsługuje przezroczyście tensory wejściowe:
        - Channel-First (B, 3, 128, 259) - natywny PyTorch
        - Channel-Last (B, 128, 259, 3) - zgodny z plikami .npy (automatyczna transpozycja)
        """
        if x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        elif x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.ndim == 3 and x.shape[0] == 3:
            x = x.unsqueeze(0)

        # BLOK 1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        x = self.pool1(x)
        x = self.s_dropout1(x)

        # BLOK 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)

        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        x = self.pool2(x)
        x = self.s_dropout2(x)

        # BLOK 3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)

        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu3_2(x)

        x = self.pool3(x)
        x = self.s_dropout3(x)

        # BLOK 4
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu4_1(x)

        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu4_2(x)

        x = self.pool4(x)
        x = self.dropout4(x)

        # GAP & Flatten
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        # Dense projection
        feat = self.dense_features(x)
        feat = self.bn_dense(feat)
        feat = self.relu_dense(feat)
        dense_feat = feat

        # Regularyzacja dropout (aktywna tylko w treningu)
        feat = self.dense_dropout(feat)

        # Klasyfikator
        logits = self.predictions(feat)

        # OOD Open-Set Rejection w fazie inferencji (działa również natywnie w ONNX!)
        if bool(self.use_open_set.item()) and not self.training:
            # 1. Normalizacja cech L2
            z_norm = F.normalize(dense_feat, p=2, dim=1)
            p_norm = F.normalize(self.fault_prototypes, p=2, dim=1)
            # 2. Podobieństwo cosinusowe do 3 prototypów wad [beeping(0), clicking(1), scratching(3)]
            cos_sim = torch.matmul(z_norm, p_norm.t())  # (B, 3)
            # 3. Bramka OOD Sigmoid: jeśli cos_sim < threshold, gating -> 0
            gating = torch.sigmoid(self.gamma * (cos_sim - self.fault_thresholds))  # (B, 3)
            # 4. Tłumienie logitów uszkodzeń (log(0.0001) = -9.21)
            suppression = torch.log(gating + 1e-4)  # (B, 3)

            # Indeksy klas w logitach: 0: beeping, 1: clicking, 2: normal, 3: scratching
            b_supp = suppression[:, 0:1]
            c_supp = suppression[:, 1:2]
            s_supp = suppression[:, 2:3]
            zero_pad = torch.zeros_like(b_supp)

            full_suppression = torch.cat([b_supp, c_supp, zero_pad, s_supp], dim=1)  # (B, 4)
            logits = logits + full_suppression

        if return_logits:
            return logits

        if self.num_classes == 1:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zwraca wektor cech 128-d (penultimate layer) przed klasyfikatorem.
        """
        if x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        elif x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.ndim == 3 and x.shape[0] == 3:
            x = x.unsqueeze(0)

        x = self.relu1_2(self.bn1_2(self.conv1_2(self.relu1_1(self.bn1_1(self.conv1_1(x))))))
        x = self.pool1(x)
        x = self.relu2_2(self.bn2_2(self.conv2_2(self.relu2_1(self.bn2_1(self.conv2_1(x))))))
        x = self.pool2(x)
        x = self.relu3_2(self.bn3_2(self.conv3_2(self.relu3_1(self.bn3_1(self.conv3_1(x))))))
        x = self.pool3(x)
        x = self.relu4_2(self.bn4_2(self.conv4_2(self.relu4_1(self.bn4_1(self.conv4_1(x))))))
        x = self.pool4(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dense_features(x)
        x = self.bn_dense(x)
        x = self.relu_dense(x)
        return x

    def set_fault_prototypes(self, prototypes: torch.Tensor, thresholds: torch.Tensor, gamma: float = 12.0):
        """
        Inicjalizuje prototypy cech dla znanych klas uszkodzeń (beeping, clicking, scratching)
        oraz progi podobieństwa cosinusowego do odrzucania próbek Out-Of-Distribution (OOD).
        """
        self.fault_prototypes.copy_(prototypes)
        self.fault_thresholds.copy_(thresholds)
        self.gamma.copy_(torch.tensor(gamma, dtype=torch.float32))
        self.use_open_set.fill_(True)

    @torch.no_grad()
    def predict_proba(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Pomocnicza metoda inferencyjna zwracająca prawdopodobieństwa (Softmax) jako np.ndarray.
        """
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x).float().to(device)
        else:
            x_t = x.to(device)

        probs = self.forward(x_t, return_logits=False)
        result = probs.detach().cpu().numpy()

        if was_training:
            self.train()
        return result

    @torch.no_grad()
    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Zwraca indeksy klas z najwyższym prawdopodobieństwem (argmax).
        """
        probs = self.predict_proba(x)
        if self.num_classes == 1:
            return (probs > 0.5).astype(np.int64)
        return np.argmax(probs, axis=-1)


class FusedEnsemblePyTorch(nn.Module):
    """
    Komitet N modeli PyTorch z uśrednianiem Softmax (Soft Voting).
    Redukuje wariancję błędu i podnosi stabilność klasyfikacji na próbkach brzegowych.
    """
    def __init__(self, models: List[AcousticHddSentinelCNN]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = models[0].num_classes if models else DEFAULT_NUM_CLASSES

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        probs = [m(x, return_logits=False) for m in self.models]
        avg_prob = torch.stack(probs, dim=0).mean(dim=0)
        if return_logits:
            return torch.log(avg_prob + 1e-9)
        return avg_prob

    @torch.no_grad()
    def predict_proba(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x).float().to(device)
        else:
            x_t = x.to(device)
        probs = self.forward(x_t, return_logits=False)
        result = probs.detach().cpu().numpy()
        if was_training:
            self.train()
        return result

    @torch.no_grad()
    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=-1)


def build_hdd_cnn_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE_PT,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout_rate: float = 0.3,
    **kwargs
) -> AcousticHddSentinelCNN:
    """
    Główna funkcja fabryczna tworząca model 2D CNN w PyTorch.
    """
    return AcousticHddSentinelCNN(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        input_shape=input_shape
    )


class TransferLearningMobileNetV2(nn.Module):
    """
    Model transfer learningu oparty na architekturze MobileNetV2 w PyTorch.
    """
    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        pretrained: bool = False,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        try:
            import torchvision.models as tv_models
            weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            backbone = tv_models.mobilenet_v2(weights=weights)
        except Exception:
            import torchvision.models as tv_models
            backbone = tv_models.mobilenet_v2(pretrained=pretrained)

        self.features = backbone.features
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_gap = nn.BatchNorm1d(1280, eps=1e-3, momentum=0.01)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(1280, 128)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        if x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.bn_gap(x)
        x = self.dropout1(x)
        x = self.relu(self.dense(x))
        x = self.dropout2(x)
        logits = self.classifier(x)
        if return_logits:
            return logits
        return F.softmax(logits, dim=-1)


def build_transfer_learning_model(
    backbone_name: str = "mobilenetv2",
    num_classes: int = DEFAULT_NUM_CLASSES,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.3,
    **kwargs
) -> nn.Module:
    """
    Tworzy model transfer learningu w PyTorch.
    """
    if backbone_name.lower() == "mobilenetv2":
        return TransferLearningMobileNetV2(
            num_classes=num_classes,
            pretrained=False,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    raise ValueError(f"Nieobsługiwany szkielet: {backbone_name}. Wybierz: mobilenetv2.")


if __name__ == "__main__":
    print("=" * 70)
    print("AKUSTYCZNA DETEKCJA AWARII HDD - ARCHITEKTURA 2D CNN (PyTorch)")
    print("=" * 70)
    model = build_hdd_cnn_model(num_classes=4)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Liczba parametrów: {total_params:,}")
    dummy = torch.randn(2, 3, 128, 259)
    out = model(dummy)
    print(f"Wyjście Forward pass: {out.shape}")
    print("[OK] Model PyTorch działa poprawnie.")
