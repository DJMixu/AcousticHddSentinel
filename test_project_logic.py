"""
Kompaktowy tester integralności i logiki projektu AcousticHddSentinel.
Działa w czystym Pythonie (standardowa biblioteka), weryfikując:
1. Składnię i AST wszystkich modułów Python.
2. Niezmiennik ADR-001: brak Data Leakage (podział na poziomie plików źródłowych).
3. Niezmiennik ADR-001: stały kształt tensorów (128, 259, 3) oraz format float32 (<f4).
4. Integralność zbioru surowego (pliki WAV, nagłówki RIFF).
5. Spójność mapowania klas i artefaktów treningowych.
"""

import ast
import json
import os
import re
import struct
from pathlib import Path

def parse_npy_header(filepath: Path):
    """
    Parsuje nagłówek pliku .npy w czystym Pythonie bez użycia biblioteki numpy.
    Format nagłówka NPY:
    - 6 bajtów: magic string b'\\x93NUMPY'
    - 1 bajt: major version (zwykle 1, 2 lub 3)
    - 1 bajt: minor version (zwykle 0)
    - 2 bajty (dla v1) lub 4 bajty (dla v2): długość nagłówka (header_len)
    - header_len bajtów: słownik Pythona w formacie tekstowym
    """
    with open(filepath, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"Niepoprawny nagłówek NPY w {filepath}")
        major, minor = struct.unpack("BB", f.read(2))
        if major == 1:
            h_len = struct.unpack("<H", f.read(2))[0]
        else:
            h_len = struct.unpack("<I", f.read(4))[0]
        header_str = f.read(h_len).decode("latin1").strip()
        
        # Ekstrakcja shape i descr za pomocą regex
        shape_match = re.search(r"'shape':\s*\(([^)]+)\)", header_str)
        descr_match = re.search(r"'descr':\s*'([^']+)'", header_str)
        fortran_match = re.search(r"'fortran_order':\s*(True|False)", header_str)
        
        shape = tuple(int(dim.strip()) for dim in shape_match.group(1).split(",") if dim.strip()) if shape_match else None
        descr = descr_match.group(1) if descr_match else None
        fortran_order = (fortran_match.group(1) == "True") if fortran_match else None
        
        return shape, descr, fortran_order


def run_tests():
    print("=" * 70)
    print("KRÓTKI TEST INTEGRALNOŚCI PROJEKTU (ACOUSTIC HDD SENTINEL)")
    print("=" * 70)
    
    passed_tests = 0
    total_tests = 5

    # -------------------------------------------------------------
    # TEST 1: Składnia i AST wszystkich skryptów Python
    # -------------------------------------------------------------
    print("\n[TEST 1/5] Walidacja składniowa AST modułów Python...")
    scripts = [
        "hdd_audio_encoder.py",
        "model_cnn.py",
        "train_cnn.py",
        "prepare_audio_dataset.py",
        "verify_cnn_model.py",
        "profile_normalization_quality.py",
    ]
    all_ast_ok = True
    for s in scripts:
        p = Path(s)
        if not p.exists():
            print(f"  [BŁĄD] Brak pliku: {s}")
            all_ast_ok = False
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=s)
            func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            print(f"  - {s:32s} [OK] ({len(func_names)} funkcji)")
        except SyntaxError as e:
            print(f"  - {s:32s} [BŁĄD SKŁADNI]: {e}")
            all_ast_ok = False
            
    assert all_ast_ok, "Błąd składniowy w modułach Python!"
    passed_tests += 1
    print("  -> Wynik Testu 1: ZDANY (Wszystkie skrypty poprawne syntaktycznie)")

    # -------------------------------------------------------------
    # TEST 2: Niezmiennik ADR-001 — Brak Data Leakage (Izolacja plików źródłowych)
    # -------------------------------------------------------------
    print("\n[TEST 2/5] Weryfikacja braku Data Leakage w podziale zbiorów...")
    splits = ["train", "validate", "test"]
    stems_per_split = {s: set() for s in splits}
    total_tensors = 0

    for split in splits:
        split_dir = Path("dataset_encoded") / split
        if not split_dir.exists():
            continue
        for npy_file in split_dir.rglob("*.npy"):
            total_tensors += 1
            # Format nazwy: <source_stem>_chunk<idx>.npy
            stem_base = npy_file.stem.split("_chunk")[0]
            stems_per_split[split].add(stem_base)

    print(f"  - Unikalne nagrania źródłowe:")
    print(f"    * TRAIN:    {len(stems_per_split['train'])} nagrań")
    print(f"    * VALIDATE: {len(stems_per_split['validate'])} nagrań")
    print(f"    * TEST:     {len(stems_per_split['test'])} nagrań")

    # Przecięcia zbiorów źródłowych (powinny być puste)
    leak_train_val = stems_per_split["train"] & stems_per_split["validate"]
    leak_train_test = stems_per_split["train"] & stems_per_split["test"]
    leak_val_test = stems_per_split["validate"] & stems_per_split["test"]

    print(f"  - Przeciek Train <-> Validate: {len(leak_train_val)} (oczekiwano 0)")
    print(f"  - Przeciek Train <-> Test:     {len(leak_train_test)} (oczekiwano 0)")
    print(f"  - Przeciek Val <-> Test:       {len(leak_val_test)} (oczekiwano 0)")

    assert len(leak_train_val) == 0, f"DATA LEAKAGE: {leak_train_val}"
    assert len(leak_train_test) == 0, f"DATA LEAKAGE: {leak_train_test}"
    assert len(leak_val_test) == 0, f"DATA LEAKAGE: {leak_val_test}"
    passed_tests += 1
    print("  -> Wynik Testu 2: ZDANY (Zero przecieku wycinków między zbiorami)")

    # -------------------------------------------------------------
    # TEST 3: Niezmiennik ADR-001 — Stały kształt tensora (128, 259, 3) i typ <f4
    # -------------------------------------------------------------
    print("\n[TEST 3/5] Weryfikacja kształtu i typu wszystkich wygenerowanych tensorów .npy...")
    target_shape = (128, 259, 3)
    target_dtype = "<f4"  # float32 little-endian
    
    checked_count = 0
    shape_errors = 0
    dtype_errors = 0

    for split in splits:
        split_dir = Path("dataset_encoded") / split
        if not split_dir.exists():
            continue
        for npy_file in split_dir.rglob("*.npy"):
            checked_count += 1
            shape, descr, fortran = parse_npy_header(npy_file)
            if shape != target_shape:
                shape_errors += 1
            if descr != target_dtype:
                dtype_errors += 1

    print(f"  - Zbadano plików .npy: {checked_count}")
    print(f"  - Błędne wymiary tensora: {shape_errors}")
    print(f"  - Błędne typy danych:    {dtype_errors}")
    assert checked_count > 0, "Brak plików .npy do weryfikacji!"
    assert shape_errors == 0, f"Wykryto {shape_errors} tensorów o złym kształcie!"
    assert dtype_errors == 0, f"Wykryto {dtype_errors} tensorów o złym typie!"
    passed_tests += 1
    print(f"  -> Wynik Testu 3: ZDANY (Wszystkie {checked_count} tensorów mają idealny kształt (128, 259, 3) float32)")

    # -------------------------------------------------------------
    # TEST 4: Integralność bazy źródłowej WAV
    # -------------------------------------------------------------
    print("\n[TEST 4/5] Weryfikacja bazy surowych plików audio (CNN/dataset)...")
    wav_dir = Path("CNN/dataset")
    wav_files = list(wav_dir.glob("*.wav"))
    valid_wavs = 0
    for w in wav_files:
        if w.stat().st_size > 44:  # Nagłówek RIFF to min 44 bajty
            with open(w, "rb") as f:
                header = f.read(12)
                if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
                    valid_wavs += 1

    print(f"  - Znaleziono plików WAV: {len(wav_files)}")
    print(f"  - Prawidłowe nagłówki RIFF/WAVE: {valid_wavs}")
    assert valid_wavs == len(wav_files), "Wykryto uszkodzone pliki WAV!"
    passed_tests += 1
    print("  -> Wynik Testu 4: ZDANY (Wszystkie surowe nagrania WAV są sprawne)")

    # -------------------------------------------------------------
    # TEST 5: Spójność artefaktów modelu i mapowania klas
    # -------------------------------------------------------------
    print("\n[TEST 5/5] Weryfikacja artefaktów treningowych...")
    out_dir = Path("training_output")
    classes_json_path = out_dir / "classes.json"
    model_path = out_dir / "best_hdd_cnn_model.keras"
    cm_path = out_dir / "confusion_matrix.png"
    curves_path = out_dir / "training_curves.png"

    assert classes_json_path.exists(), "Brak classes.json!"
    assert model_path.exists(), "Brak pliku wag modelu best_hdd_cnn_model.keras!"
    assert model_path.stat().st_size > 1_000_000, "Plik modelu jest zbyt mały!"
    assert cm_path.exists(), "Brak confusion_matrix.png!"
    assert curves_path.exists(), "Brak training_curves.png!"

    classes_data = json.loads(classes_json_path.read_text(encoding="utf-8"))
    print(f"  - Skonfigurowane klasy w modelu: {classes_data['classes']}")
    print(f"  - Kształt wejściowy zdefiniowany w JSON: {classes_data['input_shape']}")
    print(f"  - Rozmiar pliku wag modelu: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    assert classes_data["input_shape"] == [128, 259, 3]
    passed_tests += 1
    print("  -> Wynik Testu 5: ZDANY (Artefakty modelu i konfiguracja w 100% kompletne)")

    print("\n" + "=" * 70)
    print(f"PODSUMOWANIE: {passed_tests}/{total_tests} TESTÓW ZDANYCH POMYŚLNIE (100% SUKCES)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
