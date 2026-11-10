# Projekt: HDD Acoustic Fault Detection (2D CNN w PyTorch)
**Instrukcja dla agentów AI (Antigravity CLI / AGY) oraz developerów**

## 1. Cel i zakres projektu
System akustycznej detekcji i wieloklasowej klasyfikacji uszkodzeń mechanicznych dysków twardych (HDD) z wykorzystaniem głębokich sieci konwolucyjnych (2D CNN w PyTorch). Analiza opiera się na 3-sekundowych wycinkach audio przetworzonych do 3-kanałowej reprezentacji czasowo-częstotliwościowej:
- **Kanał 0:** Log-Mel Spektrogram (struktura widmowa, rezonanse, piski)
- **Kanał 1:** Delta – 1. pochodna czasowa (transjenty, impulsowe klikanie głowic)
- **Kanał 2:** Delta-Delta – 2. pochodna czasowa (dynamika przyspieszeń i zmian obrotowych)

---

## 2. Niezmienniki techniczne (Invariants) — ZAKAZ ŁAMANIA PRZEZ AGENTA
Każdy agent AGY modyfikujący lub rozwijający projekt **MUSI bezwzględnie** przestrzegać poniższych reguł zdefiniowanych w [ADR-001](file:///home/DJMixu/Pulpit/AcousticHddSentinel/docx/adr/adr.md):

1. **Brak Data Leakage (Podział na poziomie plików):**
   Podział na `train` (70%), `validate` (15%) i `test` (15%) **MUSI** odbywać się na całych nagraniach źródłowych przed ich pocięciem na 3-sekundowe okna czasowe. Nigdy nie stosuj losowego `train_test_split` bezpośrednio na wygenerowanych plikach `.npy`.
2. **Kształt tensora wejściowego:**
   - Na dysku w plikach `.npy`: tensor `float32` o stałych wymiarach **`(128, 259, 3)`** (Channel-Last).
   - W pamięci modelu PyTorch: tensor `(B, 3, 128, 259)` (Channel-First).
   - Klasa `AcousticHddSentinelCNN` w [model_cnn.py](file:///home/DJMixu/Pulpit/AcousticHddSentinel/model_cnn.py) transparentnie akceptuje oba układy wymiarów (automatyczna transpozycja).
3. **Parametry DSP (Digital Signal Processing):**
   - `TARGET_SR = 22050` Hz
   - `N_FFT = 1024` (okno ~46.4 ms)
   - `HOP_LENGTH = 256` (krok ~11.6 ms – kluczowy dla wysokiego Crest Factor)
   - `N_MELS = 128`
   - `FMIN = 50.0` Hz, `FMAX = 10000.0` Hz
   - `CHUNK_DURATION = 3.0` s (66 150 próbek)
   - `OVERLAP = 1.5` s dla zbioru `train` (50% augmentacji), `0.0` s dla `validate`/`test`.
4. **Izolacja funkcji normalizacji:**
   Podczas pisania skryptów ewaluacyjnych, testowych lub inferencyjnych zawsze importuj `encode_audio_chunk`, `normalize_waveform_peak` oraz `normalize_tensor_zscore` bezpośrednio z [hdd_audio_encoder.py](file:///home/DJMixu/Pulpit/AcousticHddSentinel/hdd_audio_encoder.py). Nie duplikuj kodu normalizacji.
5. **Loader PyTorch `DataLoader` z `HddNpyDataset`:**
   Wczytując pliki `.npy` z dysku, używaj `HddNpyDataset` i `create_pytorch_dataloaders` z [dataset_pytorch.py](file:///home/DJMixu/Pulpit/AcousticHddSentinel/dataset_pytorch.py) oraz funkcji ważenia klas `compute_balanced_class_weights`.

---

## 3. Mapa plików i architektura repozytorium

```
AcousticHddSentinel/
├── AGENTS.md                          # Główny przewodnik i Single Source of Truth dla agentów AGY
├── README.md                          # Dokumentacja repozytorium dla ludzi i developerów
├── requirements.txt                   # Zależności Python (PyTorch, Librosa, Scikit-learn, etc.)
├── .gitignore                         # Wykluczenia git (duże zbiory .npy, venv, cache)
├── model_cnn.py                       # Architektura sieci 2D CNN w PyTorch (AcousticHddSentinelCNN)
├── model_cnn_pytorch.py               # Alias / moduł PyTorch
├── train_cnn.py                       # Kompletny pipeline treningowy w PyTorch (DataLoader, SpecAugment, callbacki)
├── train_pytorch.py                   # Dedykowany moduł treningowy PyTorch
├── dataset_pytorch.py                 # PyTorch Dataset i DataLoader dla wycinków .npy zgodny z ADR-001
├── verify_cnn_model.py                # 6-etapowy test weryfikacyjny sprawności modelu PyTorch
├── verify_pytorch_model.py            # Dedykowany test weryfikacyjny PyTorch
├── hdd_audio_encoder.py               # Enkoder audio do tensorów .npy zgodny z ADR-001
├── prepare_audio_dataset.py           # Kategoryzator i generator próbek audio
├── profile_normalization_quality.py   # Profiler i weryfikator jakości normalizacji
├── normalization_quality_report.png   # Diagnostyczny wykres jakości normalizacji
├── normalization_quality_metrics.json # Metryki numeryczne jakości normalizacji
│
├── training_output/                   # Wytrenowany model i wyniki ewaluacji
│   ├── best_hdd_cnn_model.pt          # Wagi najlepszego modelu w PyTorch (1.2M parametrów)
│   ├── confusion_matrix.png           # Wykres macierzy pomyłek (liczby i procenty)
│   ├── training_curves.png            # Krzywe funkcji straty i dokładności (Train & Val)
│   ├── training_history.csv           # Log wszystkich epok
│   ├── classification_report.txt      # Raport Precision, Recall, F1 per klasa
│   ├── test_evaluation_summary.json   # Podsumowanie metryk testowych
│   └── classes.json                   # Mapowanie indeksów klas
│
├── CNN/                               # Podkatalog modułów DSP i bazy audio
│   ├── dataset/                       # 91 surowych plików WAV (Northwind + Aesonlabs)
│   ├── audio_dataset_profile.csv      # Profil akustyczny zebranych próbek
│   └── ...                            # Kopie modułów zgodne ze ścieżkami podkatalogu
│
└── docx/adr/
    └── adr.md                         # ADR-001: Architektura przetwarzania sygnału i enkodowania
```

---

## 4. Polecenia szybkiego startu (Quickstart dla AGY)

Każdy agent lub programista może uruchomić kluczowe etapy za pomocą gotowych poleceń:

### A. Sprawdzenie sprawności modelu:
```bash
python verify_cnn_model.py
```

### B. Weryfikacja jakości normalizacji:
```bash
python profile_normalization_quality.py
```

### C. Regeneracja zbioru tensorów (w razie potrzeby):
```bash
python prepare_audio_dataset.py
python hdd_audio_encoder.py
```

### D. Uruchomienie pełnego treningu (PyTorch):
```bash
python train_cnn.py --epochs 30 --batch-size 32 --output-dir ./training_output
```

---

## 5. Aktualne wyniki modelu (Stan obecny)
- **Model:** Wytrenowany na 695 tensorach treningowych z ważeniem klas `class_weight`.
- **Wykrywanie dysków sprawnych (`normal`):** **100% precyzja, 100% czułość (F1 = 1.000)** na zbiorze testowym.
- **Wykrywanie stukania głowicy (`clicking`):** **90.9% precyzja** na zbiorze testowym.
- **Średnia ważona precyzja:** **74.55%**.

---

## 6. Sugerowany backlog dla kolejnego agenta AGY:
1. **Eksport do ONNX / TorchScript / ExecuTorch:** Użycie `torch.onnx.export` lub `torch.jit.trace` z kwantyzacją `int8` lub `float16` w celu uruchomienia lekkiej inferencji na telefonie Android lub mikrokontrolerach brzegowych (zgodnie ze `docx/specs.txt`).
2. **Augmentacja spektrogramów w locie:** Dodanie warstwy SpecAugment (TimeMasking / FreqMasking) bezpośrednio do `train_cnn.py` w celu poprawy czułości na rzadsze klasy.
3. **Zwiększenie bazy dla klasy `scratching`:** Klasa zarysowania talerza miała tylko 6 próbek testowych – warto wyodrębnić więcej fragmentów z nagrań tarcia głowic.
4. **Seed Ensembling (Komitet modeli):** Wdrożenie zespołu 3–5 modeli trenowanych na różnych ziarnach losowości (`seed = 42, 123, 777...`) z uśrednianiem prawdopodobieństw Softmax (Soft Voting) w celu redukcji wariancji błędu i podniesienia F1-score (szczegóły w [NOTES.md](file:///home/DJMixu/Pulpit/AcousticHddSentinel/NOTES.md)).
