#!/usr/bin/env python3
"""
tune_threshold.py
Analiza optymalnego progu odległości (Threshold Tuning) dla modelu Fault-Only.
"""

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report
from model_no_normal import AcousticFaultOnlyCNN
from dataset_no_normal import FaultOnlyNpyDataset, EVAL_CLASSES_4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights_path = Path("PrototypBezNormal/output/best_model_no_normal.pt")
ckpt = torch.load(weights_path, map_location=device, weights_only=False)

model = AcousticFaultOnlyCNN(num_fault_classes=3, embedding_dim=128).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

test_ds = FaultOnlyNpyDataset("dataset_encoded/test", include_normal=True, augment=False, preload_to_ram=True)

embeddings = []
all_logits = []
targets = []

with torch.no_grad():
    for i in range(len(test_ds)):
        x, y = test_ds[i]
        x_in = x.unsqueeze(0).to(device)
        logits, norm_embed = model(x_in)
        embeddings.append(norm_embed)
        all_logits.append(logits)
        targets.append(y)

embeddings = torch.cat(embeddings, dim=0)
all_logits = torch.cat(all_logits, dim=0)
targets = np.array(targets)

diff = embeddings.unsqueeze(1) - model.prototypes.unsqueeze(0)
dists = torch.norm(diff, p=2, dim=-1)
min_dists, closest_c = torch.min(dists, dim=-1)
min_dists = min_dists.cpu().numpy()
closest_c = closest_c.cpu().numpy()

print(f"{'Próg D_min':12s} | {'Acc Ogólna':12s} | {'Macro F1':10s} | {'Normal F1':11s} | {'Normal Rec':12s} | {'Fault Acc':10s}")
print("-" * 80)

best_f1 = -1.0
best_th = 0.85

for th in np.arange(0.60, 1.30, 0.05):
    preds = []
    for i in range(len(targets)):
        md = min_dists[i]
        if md > th:
            preds.append(2)  # normal
        else:
            fc = closest_c[i]
            preds.append(3 if fc == 2 else fc)
    preds = np.array(preds)
    acc = accuracy_score(targets, preds)
    f1_m = f1_score(targets, preds, average="macro", zero_division=0)
    norm_rec = np.sum((targets == 2) & (preds == 2)) / np.sum(targets == 2)
    norm_prec = np.sum((targets == 2) & (preds == 2)) / max(1, np.sum(preds == 2))
    norm_f1 = 2 * (norm_prec * norm_rec) / max(1e-9, norm_prec + norm_rec)
    fault_mask = (targets != 2)
    fault_acc = np.sum((targets[fault_mask] == preds[fault_mask])) / np.sum(fault_mask)
    if f1_m > best_f1:
        best_f1 = f1_m
        best_th = th
    print(f"th = {th:5.2f}    | {acc*100:6.2f}%     | {f1_m:7.4f}   | {norm_f1:7.4f}    | {norm_rec*100:6.2f}%     | {fault_acc*100:5.2f}%")

print("\n" + "="*80)
print(f"Najlepszy próg pod kątem Macro F1: th = {best_th:.2f} (Macro F1: {best_f1:.4f})")
print("="*80)
