#!/usr/bin/env python3
"""
dataset_no_normal.py
Dataset i DataLoadery dla PrototypBezNormal.

Zasada działania:
- Dla train i validate: wczytuje WYŁĄCZNIE 3 klasy awarii: beeping (0), clicking (1), scratching (2).
  Klasa 'normal' jest w 100% odfiltrowana.
- Dla test: wczytuje WSZYSTKIE 4 klasy: beeping (0), clicking (1), normal (2), scratching (3),
  umożliwiając rzetelną ewaluację zdolności modelu do odrzucania/rozpoznawania zdrowych dysków.
"""

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

FAULT_CLASSES_3 = ["beeping", "clicking", "scratching"]
EVAL_CLASSES_4 = ["beeping", "clicking", "normal", "scratching"]


class SpecAugment(object):
    """Augmentacja w locie w domenie czasowo-częstotliwościowej."""
    def __init__(self, freq_mask_max: int = 16, time_mask_max: int = 24, p: float = 0.5):
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.p:
            return tensor
        x = tensor.clone()
        _, n_mels, time_steps = x.shape

        # Maskowanie pasm częstotliwości (Mel)
        f = np.random.randint(0, self.freq_mask_max)
        f0 = np.random.randint(0, max(1, n_mels - f))
        x[:, f0:f0 + f, :] = 0.0

        # Maskowanie kroków czasowych
        t = np.random.randint(0, self.time_mask_max)
        t0 = np.random.randint(0, max(1, time_steps - t))
        x[:, :, t0:t0 + t] = 0.0

        return x


class FaultOnlyNpyDataset(Dataset):
    def __init__(
        self,
        split_dir: str or Path,
        include_normal: bool = False,
        augment: bool = False,
        preload_to_ram: bool = True
    ):
        self.split_dir = Path(split_dir)
        self.include_normal = include_normal
        self.augment = augment
        self.spec_augment = SpecAugment() if augment else None

        if self.include_normal:
            self.classes = EVAL_CLASSES_4
        else:
            self.classes = FAULT_CLASSES_3

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []

        for class_name in self.classes:
            class_folder = self.split_dir / class_name
            if not class_folder.exists():
                continue
            idx = self.class_to_idx[class_name]
            for npy_file in sorted(class_folder.glob("*.npy")):
                self.samples.append((str(npy_file), idx, class_name))

        self.preload_to_ram = preload_to_ram
        self.preloaded_data = []

        if self.preload_to_ram and len(self.samples) > 0:
            for filepath, label, _ in self.samples:
                arr = np.load(filepath).astype(np.float32)
                # Zamiana (128, 259, 3) -> (3, 128, 259)
                if arr.ndim == 3 and arr.shape[-1] == 3:
                    arr = np.transpose(arr, (2, 0, 1))
                self.preloaded_data.append((torch.from_numpy(arr), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self.preload_to_ram:
            tensor, label = self.preloaded_data[idx]
            tensor = tensor.clone()
        else:
            filepath, label, _ = self.samples[idx]
            arr = np.load(filepath).astype(np.float32)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = np.transpose(arr, (2, 0, 1))
            tensor = torch.from_numpy(arr)

        if self.augment and self.spec_augment is not None:
            tensor = self.spec_augment(tensor)

        return tensor, label


def compute_fault_class_weights(dataset: FaultOnlyNpyDataset) -> torch.Tensor:
    """Ważenie klas metodą Inverse Frequency dla 3 awarii."""
    labels = [s[1] for s in dataset.samples]
    num_classes = len(dataset.classes)
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * np.maximum(counts, 1).astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    dataset_encoded_dir: str or Path = "dataset_encoded",
    batch_size: int = 32
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Tworzy:
    - train_loader: 3 klasy awarii (beeping, clicking, scratching) z augmentacją SpecAugment
    - val_loader: 3 klasy awarii (czyste)
    - test_loader: 4 klasy (łącznie z normal!) dla pełnego testu ewaluacyjnego
    - fault_weights: wagi dla CrossEntropyLoss
    """
    root = Path(dataset_encoded_dir)

    train_ds = FaultOnlyNpyDataset(root / "train", include_normal=False, augment=True, preload_to_ram=True)
    val_ds = FaultOnlyNpyDataset(root / "validate", include_normal=False, augment=False, preload_to_ram=True)
    test_ds = FaultOnlyNpyDataset(root / "test", include_normal=True, augment=False, preload_to_ram=True)

    fault_weights = compute_fault_class_weights(train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader, fault_weights
