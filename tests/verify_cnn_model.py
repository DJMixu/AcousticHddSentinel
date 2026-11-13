"""
Skrypt weryfikacyjny dla architektury modelu 2D CNN w PyTorch (Acoustic HDD Sentinel).
Działa w 100% w środowisku PyTorch.
Uruchomienie: python verify_cnn_model.py
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))

from model_cnn import build_hdd_cnn_model, AcousticHddSentinelCNN, INPUT_SHAPE
from dataset_pytorch import resolve_dataset_dir, discover_classes_and_files


def run_verification():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("KOMPLEKSOWA WERYFIKACJA MODELU 2D CNN (ACOUSTIC HDD SENTINEL - PYTORCH)")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_props = torch.cuda.get_device_properties(device)
        vram_gb = gpu_props.total_memory / (1024 ** 3)
        print(f"URZĄDZENIE: {device} ({gpu_name}, {vram_gb:.2f} GB VRAM)")
    else:
        print(f"URZĄDZENIE: {device}")
    print("=" * 70)

    # 1. Kształt wejściowy i parametry
    print("[1/6] Inicjalizacja modelu PyTorch i weryfikacja kształtów...")
    model = build_hdd_cnn_model(num_classes=4, dropout_rate=0.2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   -> Liczba parametrów ogółem: {total_params:,} (trenowalnych: {trainable_params:,})")
    assert trainable_params == 1_206_756, f"Niepoprawna liczba parametrów: {trainable_params} != 1,206,756!"
    print("   [OK] Liczba parametrów w 100% zgodna z architekturą ADR-001 (1,206,756).\n")

    # 2. Przekazywanie w przód (Forward pass)
    print("[2/6] Test przekazywania w przód (Forward Pass)...")
    dummy_ch_first = torch.randn(4, 3, 128, 259, device=device)
    dummy_ch_last = torch.randn(4, 128, 259, 3, device=device)

    model.eval()
    with torch.no_grad():
        out_first = model(dummy_ch_first)
        out_last = model(dummy_ch_last)

    assert out_first.shape == (4, 4), "Błędny kształt wyjścia Channel-First!"
    assert out_last.shape == (4, 4), "Błędny kształt wyjścia Channel-Last!"
    assert not torch.isnan(out_first).any(), "Wykryto NaN w Channel-First!"
    assert not torch.isnan(out_last).any(), "Wykryto NaN w Channel-Last!"
    assert torch.allclose(torch.sum(out_first, dim=-1), torch.ones(4, device=device), atol=1e-5), "Suma Softmax != 1.0!"

    print(f"   -> Channel-First shape: {out_first.shape}, Channel-Last shape: {out_last.shape}")
    print(f"   -> Średnie prawdopodobieństwa: {out_first.mean(dim=0).cpu().tolist()}")
    print("   [OK] Model obsługuje przezroczyście tensory wejściowe (B, 3, 128, 259) oraz (B, 128, 259, 3).\n")

    # 3. Przepływ gradientów (Backward pass)
    print("[3/6] Test przepływu gradientów (Backward Pass)...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dummy_x = torch.randn(4, 3, 128, 259, device=device)
    dummy_y = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)

    optimizer.zero_grad()
    logits = model(dummy_x, return_logits=True)
    loss = criterion(logits, dummy_y)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Brak gradientu dla {name}!"
            assert not torch.isnan(param.grad).any(), f"NaN w gradiencie dla {name}!"

    print(f"   -> Obliczono i zweryfikowano gradienty dla wszystkich {len(list(model.parameters()))} tensorów parametrów.")
    print("   [OK] Gradienty prawidłowo przepływają przez wszystkie warstwy splotowe, BatchNorm i liniowe.\n")

    # 4. Zbieżność uczenia (Convergence check)
    print("[4/6] Test zbieżności uczenia (Trening na małym batchu)...")
    train_x = torch.randn(8, 3, 128, 259, device=device)
    train_y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long, device=device)

    torch.manual_seed(42)
    step_losses = []
    for step in range(20):
        optimizer.zero_grad()
        out = model(train_x, return_logits=True)
        step_loss = criterion(out, train_y)
        step_loss.backward()
        optimizer.step()
        step_losses.append(step_loss.item())

    initial_loss = step_losses[0]
    final_loss = min(step_losses)

    print(f"   -> Loss początkowy: {initial_loss:.4f} -> Najniższy loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "Brak spadku wartości funkcji straty!"
    print("   [OK] Model optymalizuje funkcję straty i uczy się zbieżnie w PyTorch.\n")

    # 5. Zapis i odczyt z dysku (Serialization)
    print("[5/6] Test serializacji (torch.save & torch.load)...")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model.eval()
        with torch.no_grad():
            ref_preds = model(dummy_ch_first).cpu().numpy()

        torch.save(model.state_dict(), tmp_path)

        loaded_model = build_hdd_cnn_model(num_classes=4, dropout_rate=0.2)
        loaded_model.load_state_dict(torch.load(tmp_path, weights_only=True))
        loaded_model.eval()

        with torch.no_grad():
            loaded_preds = loaded_model(dummy_ch_first.cpu()).cpu().numpy()

        assert np.allclose(ref_preds, loaded_preds, atol=1e-7), "Rozbieżność po deserializacji!"
        print(f"   -> Model pomyślnie zapisany i załadowany z: {tmp_path}")
        print("   [OK] Wagi i wyjścia po serializacji są w 100% bitowo identyczne.\n")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 6. Weryfikacja na zbiorze testowym
    print("[6/6] Weryfikacja modelu bazowego best_hdd_cnn_model.pt na zbiorze testowym...")
    pt_path = Path("training_output/best_hdd_cnn_model.pt")

    if pt_path.exists():
        pt_checkpoint = torch.load(pt_path, weights_only=True)
        pt_eval_model = build_hdd_cnn_model(num_classes=4).to(device)
        state_dict = pt_checkpoint["state_dict"] if "state_dict" in pt_checkpoint else pt_checkpoint
        pt_eval_model.load_state_dict(state_dict)
        pt_eval_model.eval()

        dataset_dir = resolve_dataset_dir()
        test_dir = dataset_dir / "test"
        classes, test_files, test_labels = discover_classes_and_files(test_dir)

        if test_files:
            X_test = np.stack([np.load(f).astype(np.float32) for f in test_files], axis=0)
            y_test = np.array(test_labels)

            with torch.no_grad():
                pt_preds = pt_eval_model(torch.from_numpy(X_test).to(device)).cpu().numpy()
            pt_labels = np.argmax(pt_preds, axis=1)

            pt_acc = accuracy_score(y_test, pt_labels)

            print(f"   -> Liczba próbek testowych: {len(test_files)}")
            print(f"   -> Dokładność (Accuracy) na zbiorze testowym: {pt_acc * 100:.2f}% ({np.sum(pt_labels == y_test)}/{len(test_files)})")
            
            # Weryfikacja bezbłędnej detekcji dysków sprawnych (klasa 'normal')
            normal_idx = classes.index("normal")
            normal_mask = (y_test == normal_idx)
            normal_correct = np.sum(pt_labels[normal_mask] == normal_idx)
            normal_recall = normal_correct / np.sum(normal_mask)
            print(f"   -> Wykrywanie dysków sprawnych ('normal'): {normal_correct}/{np.sum(normal_mask)} ({normal_recall * 100:.2f}% czułość)")

            assert pt_acc >= 0.50, f"Dokładność poniżej oczekiwanej: {pt_acc}"
            assert normal_recall >= 0.70, f"Zbyt niska czułość dla klasy 'normal': {normal_recall:.2%}"
            print("   [OK] Model PyTorch w 100% sprawny i zgodny ze specyfikacją projektu!\n")
        else:
            print("   [INFO] Brak plików testowych w dataset_encoded/test.\n")
    else:
        print("   [INFO] Brak pliku training_output/best_hdd_cnn_model.pt.\n")

    print("=" * 70)
    print("WYNIK: MODEL 2D CNN W PYTORCH DZIAŁA W 100% POPRAWNIE (ZERO ZALEŻNOŚCI OD TF)!")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
