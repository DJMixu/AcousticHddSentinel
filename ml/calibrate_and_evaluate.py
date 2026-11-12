"""
Kalibracja prototypów Open-Set OOD oraz eksport modelu.
Single Source of Truth: ADR-001 & AGENTS.md
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))

from model_cnn import AcousticHddSentinelCNN
from dataset_pytorch import create_pytorch_dataloaders
from export_model import export_pipeline

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")
    
    dataset_dir = Path("dataset_encoded")
    output_dir = Path("training_output")
    best_model_path = output_dir / "best_hdd_cnn_model.pt"
    
    train_loader, val_loader, test_loader, classes, weights = create_pytorch_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=32,
        preload=True,
        num_workers=2
    )
    
    print(f"Klasy: {classes}")
    model = AcousticHddSentinelCNN(num_classes=len(classes)).to(device)
    ckpt = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    print("\n[1/3] Obliczanie prototypów i progów OOD dla klas uszkodzeń...")
    all_features = []
    all_targets = []
    with torch.no_grad():
        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            f_b = model.extract_features(x_b)
            all_features.append(f_b.cpu())
            all_targets.append(y_b.cpu())
            
    all_f = torch.cat(all_features, dim=0)
    all_t = torch.cat(all_targets, dim=0)
    
    # Mapowanie: 0: beeping, 1: clicking, 2: normal, 3: scratching
    fault_indices = [0, 1, 3]
    prototypes = torch.zeros((3, 128), dtype=torch.float32)
    thresholds = torch.zeros(3, dtype=torch.float32)
    
    for i, f_idx in enumerate(fault_indices):
        mask = (all_t == f_idx)
        class_f = all_f[mask]
        class_f_norm = F.normalize(class_f, p=2, dim=1)
        centroid = class_f_norm.mean(dim=0)
        centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)
        prototypes[i] = centroid
        
        sims = torch.matmul(class_f_norm, centroid)
        thresh = torch.quantile(sims, 0.05).item()
        thresh = max(0.50, min(0.72, thresh))
        thresholds[i] = thresh
        cls_name = classes[f_idx]
        print(f"  - Prototyp {cls_name:12s}: {len(class_f)} próbek, próg cosinusowy: {thresh:.3f}")
        
    model.set_fault_prototypes(prototypes.to(device), thresholds.to(device), gamma=16.0)
    
    ckpt["state_dict"] = model.state_dict()
    ckpt["open_set_calibrated"] = True
    torch.save(ckpt, best_model_path)
    print(f"-> Zaktualizowano wagi w {best_model_path}")
    
    print("\n[2/3] Ewaluacja na zbiorze testowym...")
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(device)
            probs = model(x_b)
            preds = torch.argmax(probs, dim=-1)
            y_true.extend(y_b.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            
    report_text = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=True, zero_division=0)
    print("\n" + report_text)
    
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
        
    summary = {
        "classes": classes,
        "detailed_metrics": report_dict,
        "total_test_samples": len(y_true)
    }
    with open(output_dir / "test_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n[3/3] Eksport do ONNX z wbudowaną detekcją OOD...")
    export_pipeline(
        weights_path=str(best_model_path),
        output_dir="exported_models",
        classes_path=str(output_dir / "classes.json")
    )
    
    # Kopiowanie wyeksportowanego modelu ONNX do katalogu assets aplikacji Android
    android_assets_dir = Path("android/app/src/main/assets")
    if android_assets_dir.exists():
        import shutil
        shutil.copy2("exported_models/best_hdd_cnn_model.onnx", android_assets_dir / "best_hdd_cnn_model.onnx")
        print(f"-> Skopiowano best_hdd_cnn_model.onnx do {android_assets_dir}")
        
    print("\n[OK] Kalibracja, ewaluacja i eksport zakończone sukcesem!")

if __name__ == "__main__":
    main()
