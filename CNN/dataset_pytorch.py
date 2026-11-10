"""
HDD Acoustic Fault Detection - PyTorch Dataset & DataLoader
Single Source of Truth: ADR-001 & AGENTS.md

Moduł ładowania danych w formacie PyTorch (Dataset & DataLoader) z plików .npy
wygenerowanych przez hdd_audio_encoder.py w dataset_encoded/{train,validate,test}.
Zawiera implementację SpecAugment (Time & Frequency Masking) oraz ważenie klas.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

# Stałe wejściowe zgodne z ADR-001
INPUT_SHAPE_TF = (128, 259, 3)   # Na dysku .npy
INPUT_SHAPE_PT = (3, 128, 259)   # Channel-First w pamięci PyTorch


class SpecAugmentPyTorch:
    """
    Augmentacja SpecAugment w locie (Frequency Masking & Time Masking)
    dla tensora PyTorch (3, 128, 259) lub (128, 259, 3).
    """
    def __init__(self, max_freq_mask: int = 16, max_time_mask: int = 24):
        self.max_freq_mask = max_freq_mask
        self.max_time_mask = max_time_mask

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Oczekuje tensora (C, H, W) = (3, 128, 259)
        c, h, w = tensor.shape
        augmented = tensor.clone()

        # Frequency Masking
        if self.max_freq_mask > 0:
            f_len = int(torch.randint(0, self.max_freq_mask + 1, (1,)).item())
            if f_len > 0 and f_len < h:
                f0 = int(torch.randint(0, h - f_len, (1,)).item())
                augmented[:, f0:f0 + f_len, :] = 0.0

        # Time Masking
        if self.max_time_mask > 0:
            t_len = int(torch.randint(0, self.max_time_mask + 1, (1,)).item())
            if t_len > 0 and t_len < w:
                t0 = int(torch.randint(0, w - t_len, (1,)).item())
                augmented[:, :, t0:t0 + t_len] = 0.0

        return augmented


class HddNpyDataset(Dataset):
    """
    PyTorch Dataset wczytujący tensory .npy dla wycinków akustycznych HDD.
    Obsługuje wstępne załadowanie do pamięci RAM (preload=True), eliminując narzut dyskowy I/O.
    """
    def __init__(
        self,
        file_paths: List[Path],
        labels: List[int],
        is_training: bool = False,
        augment: bool = False,
        max_freq_mask: int = 16,
        max_time_mask: int = 24,
        preload: bool = True
    ):
        self.file_paths = [Path(p) for p in file_paths]
        self.labels = labels
        self.is_training = is_training
        self.augment = augment and is_training
        self.spec_augment = SpecAugmentPyTorch(max_freq_mask, max_time_mask) if self.augment else None
        self.preload = preload

        if self.preload:
            self.tensors = []
            for p in self.file_paths:
                data = np.load(p).astype(np.float32)
                if data.shape == INPUT_SHAPE_TF:
                    data = np.transpose(data, (2, 0, 1))
                self.tensors.append(torch.from_numpy(data))
        else:
            self.tensors = None

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.tensors is not None:
            tensor = self.tensors[idx]
        else:
            file_path = self.file_paths[idx]
            data = np.load(file_path).astype(np.float32)
            if data.shape == INPUT_SHAPE_TF:
                data = np.transpose(data, (2, 0, 1))
            tensor = torch.from_numpy(data)

        if self.augment and self.spec_augment is not None:
            tensor = self.spec_augment(tensor.clone())

        label = int(self.labels[idx])
        return tensor, label


def resolve_dataset_dir(custom_path: Optional[str] = None) -> Path:
    """Lokalizuje katalog z danymi dataset_encoded."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Nie znaleziono katalogu danych: {custom_path}")

    current_dir = Path(__file__).resolve().parent
    candidates = [
        Path("dataset_encoded"),
        Path("CNN/dataset_encoded"),
        current_dir / "dataset_encoded",
        current_dir / "CNN" / "dataset_encoded",
    ]
    for c in candidates:
        if c.exists() and (c / "train").exists():
            return c.resolve()

    return (current_dir / "dataset_encoded").resolve()


def discover_classes_and_files(
    split_dir: Path,
    expected_classes: Optional[List[str]] = None
) -> Tuple[List[str], List[Path], List[int]]:
    """Wykrywa klasy i pliki .npy w danym podkatalogu."""
    if not split_dir.exists():
        return [], [], []

    subdirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if not subdirs:
        npy_files = sorted(list(split_dir.glob("*.npy")))
        return ["default"], npy_files, [0] * len(npy_files)

    class_names = expected_classes if expected_classes is not None else [d.name for d in subdirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    all_files: List[Path] = []
    all_labels: List[int] = []

    for c_name in class_names:
        c_folder = split_dir / c_name
        if not c_folder.exists():
            continue
        files = sorted(list(c_folder.glob("*.npy")))
        all_files.extend(files)
        all_labels.extend([class_to_idx[c_name]] * len(files))

    return class_names, all_files, all_labels


def compute_balanced_class_weights(
    labels: List[int],
    num_classes: int,
    use_sqrt_smoothing: bool = True
) -> torch.Tensor:
    """
    Oblicza zrównoważone wagi klas (z opcjonalnym wygładzaniem pierwiastkowym np.clip(sqrt(w), 0.5, 3.5)).
    Zwraca wagowy tensor PyTorch do przekazania do nn.CrossEntropyLoss(weight=weights).
    """
    unique_present = np.unique(labels)
    computed = compute_class_weight(class_weight="balanced", classes=unique_present, y=labels)
    weight_dict = dict(zip(unique_present, computed))

    weights = []
    for i in range(num_classes):
        w = weight_dict.get(i, 1.0)
        if use_sqrt_smoothing:
            w = float(np.clip(np.sqrt(w), 0.5, 3.5))
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)


def create_pytorch_dataloaders(
    dataset_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 0,
    augment_train: bool = True,
    preload: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], torch.Tensor]:
    """
    Tworzy kompletne DataLoader'y dla train, validate i test.
    """
    dataset_path = Path(dataset_dir)
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "validate"
    test_dir = dataset_path / "test"

    classes, train_files, train_labels = discover_classes_and_files(train_dir)
    _, val_files, val_labels = discover_classes_and_files(val_dir, expected_classes=classes)
    _, test_files, test_labels = discover_classes_and_files(test_dir, expected_classes=classes)

    class_weights = compute_balanced_class_weights(train_labels, num_classes=len(classes))

    train_ds = HddNpyDataset(train_files, train_labels, is_training=True, augment=augment_train, preload=preload)
    val_ds = HddNpyDataset(val_files, val_labels, is_training=False, augment=False, preload=preload)
    test_ds = HddNpyDataset(test_files, test_labels, is_training=False, augment=False, preload=preload)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )

    return train_loader, val_loader, test_loader, classes, class_weights


if __name__ == "__main__":
    ds_dir = resolve_dataset_dir()
    print(f"Lokalizacja dataset_encoded: {ds_dir}")
    train_loader, val_loader, test_loader, classes, weights = create_pytorch_dataloaders(ds_dir, batch_size=16)
    print(f"Klasy: {classes}")
    print(f"Wagi klas: {weights}")
    print(f"Batch treningowy: {len(train_loader)} paczek, walidacyjny: {len(val_loader)}, testowy: {len(test_loader)}")
    x, y = next(iter(train_loader))
    print(f"Przykładowy batch wejściowy: {x.shape}, etykiety: {y.shape}")
