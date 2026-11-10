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
├── kontekst.md                        # Całościowe podsumowanie dotychczasowych etapów i rozmów
├── fineTuning.md                      # Dokumentacja prac optymalizacyjnych i wyników w CNN/TESTY/
├── local_llm_bridge.py                # Most orkiestrujący lokalne modele Ollama (27B + Coder 14B)
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
├── CNN/TESTY/                         # Zoptymalizowane środowisko (F1 >= 0.70 dla wszystkich klas)
│   ├── training_output/best_hdd_cnn_model.pt # Model z F1: beeping 86.4%, clicking 78.5%, scratching 73.7%
│   ├── encoder_testy.py               # Enkoder audio z filtracją ciszy i zbalansowaną augmentacją
│   ├── train_testy.py                 # Pipeline treningowy PyTorch z akceleracją ROCm
│   └── ...
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

### E. Akceleracja GPU (AMD Radeon RX 9060 XT / ROCm HIP):
Projekt wspiera natywnie akcelerację GPU na architekturze RDNA 4 (`gfx1200`) poprzez AMD ROCm (`torch+rocm` z modułem `amd-torch-device-gfx1200`):
- Pamięć VRAM: 16 GB, taktowanie do 2.78 GHz.
- Czas epoki treningowej: **~1.9 s** na GPU (vs ~16 s na CPU – ok. **8.5x–9x przyspieszenie**).
- Asynchroniczny transfer PCIe DMA (`to(device, non_blocking=True)` + `pin_memory=True`).

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

---

## 7. Automatyczna Integracja z Lokalnymi Modelami LLM (Ollama & RX 9060 XT)
W systemie użytkownika działa lokalna instalacja Ollama (`http://localhost:11434` oraz Open WebUI na `http://localhost:8080`) z dwoma modelami open-weights:
1. **Architekt / Planista (128k kontekstu, 12 GB VRAM):** `hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S`
2. **Programista Wykonawczy / Coder (9 GB VRAM):** `qwen2.5-coder:14b`

### Zasady automatycznego wykorzystania przez każdego agenta:
- **Automatyczne delegowanie zadań programistycznych:**
  Gdy użytkownik zleca projektowanie nowych modułów, implementację zaawansowanych algorytmów (np. eksport ONNX, skrypty Android/Kotlin, testy jednostkowe), agent powinien **automatycznie** wykorzystać tandem modeli za pomocą skryptu:
  ```bash
  python local_llm_bridge.py --task "<opis zadania>" [--context <plik_kontekstu>] [--output <plik_wyjścia>]
  ```
- **Niezmiennik VRAM (16 GB):**
  Modele 27B i 14B nie mogą znajdować się w VRAM równocześnie. Skrypt `local_llm_bridge.py` automatycznie wyładowuje modele (`keep_alive=0`). Przed rozpoczęciem jakiegokolwiek treningu PyTorch na GPU agent **MUSI** upewnić się, że VRAM jest zwolniony (`ollama ps` puste).

