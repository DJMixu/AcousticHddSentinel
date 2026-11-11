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
   Podział na `train` (68%), `validate` (14%) i `test` (18%) **MUSI** odbywać się na całych nagraniach źródłowych przed ich pocięciem na 3-sekundowe okna czasowe. Nigdy nie stosuj losowego `train_test_split` bezpośrednio na wygenerowanych plikach `.npy`.
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
   - Odrzucanie ramek ciszy i szumu tła: $RMS < 0.002$.
4. **Izolacja funkcji normalizacji:**
   Podczas pisania skryptów ewaluacyjnych, testowych lub inferencyjnych zawsze importuj `encode_audio_chunk`, `normalize_waveform_peak` oraz `normalize_tensor_zscore` z [ml/hdd_audio_encoder.py](file:///home/DJMixu/Pulpit/AcousticHddSentinel/ml/hdd_audio_encoder.py). Nie duplikuj kodu normalizacji.
5. **Loader PyTorch `DataLoader` z `HddNpyDataset`:**
   Wczytując pliki `.npy` z dysku, używaj `HddNpyDataset` i `create_pytorch_dataloaders` z [ml/dataset_pytorch.py](file:///home/DJMixu/Pulpit/AcousticHddSentinel/ml/dataset_pytorch.py) oraz funkcji ważenia klas `compute_balanced_class_weights`.

---

## 3. Mapa plików i uporządkowana architektura repozytorium

```
AcousticHddSentinel/
├── AGENTS.md                          # Główny przewodnik i Single Source of Truth dla agentów AGY
├── README.md                          # Główna dokumentacja projektu dla ludzi i developerów
├── kontekst.md                        # Całościowe podsumowanie dotychczasowych etapów i audytów
├── fineTuning.md                      # Dokumentacja prac optymalizacyjnych i wyników F1 >= 0.70
├── NOTES.md                           # Notatki badawcze (koncepcja Seed Ensembling)
├── requirements.txt                   # Zależności Python (PyTorch, Librosa, Scikit-learn, etc.)
├── .gitignore                         # Wykluczenia git (duże zbiory .npy, venv, cache)
│
├── ml/                                # 🧠 Pipeline uczenia maszynowego (PyTorch 2D CNN, DSP, eksport)
│   ├── model_cnn.py                   # Architektura sieci 2D CNN w PyTorch (AcousticHddSentinelCNN & FusedEnsemblePyTorch)
│   ├── dataset_pytorch.py             # PyTorch Dataset z preloadem RAM i SpecAugment (HddNpyDataset)
│   ├── hdd_audio_encoder.py           # Enkoder audio do tensorów .npy zgodny z ADR-001
│   ├── prepare_audio_dataset.py       # Kategoryzator bazy audio i generator klasy 'normal'
│   ├── train_cnn.py                   # Kompletny pipeline treningowy w PyTorch (AdamW, Label Smoothing, ROCm)
│   ├── train_ensemble.py              # Pipeline Seed Ensembling (komitet modeli o różnych ziarnach)
│   ├── export_model.py                # Eksport do formatów ONNX (FP32/INT8), TorchScript z auto-sync do Androida
│   └── compare_ensembles.py           # Narzędzie ewaluacji porównawczej komitetu modeli z modelem bazowym
│
├── tests/                             # 🧪 Testy logiczne, weryfikacja wag, profilowanie i stress-testy
│   ├── test_project_logic.py          # Automatyczny test integralności projektu i niezmienników ADR-001 (6/6)
│   ├── verify_cnn_model.py            # 6-etapowy test weryfikacyjny sprawności modelu PyTorch
│   ├── profile_normalization_quality.py # Profiler i weryfikator jakości normalizacji DSP
│   ├── stress_test.py                 # Test ciągły odporności potoku DSP + CNN (latencja, wycieki RAM, edge cases)
│   ├── stress_test_vram.py            # Test obciążeniowy pamięci VRAM karty GPU
│   └── testEpok.py                    # Diagnostyka stabilności uczenia dla różnej liczby epok
│
├── scripts/                           # 🛠️ Skrypty narzędziowe i orkiestracja lokalnych modeli LLM
│   ├── local_llm_bridge.py            # Most orkiestrujący lokalne modele Ollama (27B + Coder 14B)
│   ├── start_services.sh              # Skrypt uruchamiania usług Ollama i Open WebUI
│   └── stop_services.sh               # Skrypt zatrzymywania usług Ollama i Open WebUI
│
├── android/                           # 📱 Kompletny moduł natywnej aplikacji mobilnej (Android Studio / Gradle / Kotlin)
│   ├── app/
│   │   ├── src/main/assets/           # Skwantyzowany model ONNX INT8, classes.json, model_card.json
│   │   ├── src/main/java/             # Pakiety: audio (AudioRecord), data (Room SQLite), dsp (ADR-001 DSP), ml (ONNX), ui (MVVM)
│   │   ├── src/main/res/              # Zasoby UI (layouty, motywy Material 3, kolory, ikony)
│   │   ├── src/test/java/             # Testy jednostkowe DSP (AudioPreprocessorTest)
│   │   ├── build.gradle.kts           # Zależności: ONNX Runtime Mobile, Room, Coroutines
│   │   └── AndroidManifest.xml        # Uprawnienia RECORD_AUDIO, deklaracje aktywności
│   ├── build.gradle.kts / settings.gradle.kts # Konfiguracja projektu Gradle
│   ├── gradlew / gradlew.bat          # Wrapper narzędzia Gradle
│   └── README.md                      # Instrukcja uruchomienia i testowania w Android Studio
│
├── data/                              # Baza surowych nagrań laboratoryjnych (Ground Truth)
│   └── raw/
│       ├── northwind/                 # 82 autentyczne pliki MP3 z nazwami laboratoryjnymi
│       ├── additional/                # 13 nowych nagrań laboratoryjnych uszkodzeń HDD
│       └── aesonlabs/                 # 9 nagrań uszkodzeń Aesonlabs Data Recovery
│
├── dataset_categorized/               # 134 nagrania podzielone na 4 klasy (beeping, clicking, normal, scratching)
├── dataset_encoded/                   # 1598 tensorów .npy (1389 train, 92 validate, 117 test)
│
├── training_output/                   # Produkcyjny model, wagi i raporty ewaluacji (Accuracy 83.8%, Macro F1 84.6%)
│   ├── best_hdd_cnn_model.pt          # Wagi najlepszego pojedynczego modelu PyTorch (seed 777)
│   ├── fused_ensemble_model.pt        # Wagi zintegrowanego komitetu modeli
│   ├── ensemble_model_1_seed_42.pt    # Model składowy komitetu (seed 42)
│   ├── ensemble_model_2_seed_123.pt   # Model składowy komitetu (seed 123)
│   ├── ensemble_model_3_seed_777.pt   # Model składowy komitetu (seed 777)
│   ├── confusion_matrix.png           # Wykres macierzy pomyłek (liczby i procenty)
│   ├── training_curves_seed_777.png   # Krzywe funkcji straty i dokładności (Train & Val)
│   ├── classification_report.txt      # Raport Precision, Recall, F1 per klasa
│   ├── test_evaluation_summary.json   # Podsumowanie metryk testowych w JSON
│   ├── normalization_quality_report.png # Wykres profilowania jakości normalizacji
│   ├── normalization_quality_metrics.json # Metryki numeryczne normalizacji
│   └── classes.json                   # Mapowanie klas i parametrów wejścia
│
├── exported_models/                   # Wyeksportowane modele dla urządzeń brzegowych i aplikacji mobilnych
│   ├── best_hdd_cnn_model.onnx        # Model ONNX FP32 z dynamiczną osią batcha (4.61 MB)
│   ├── best_hdd_cnn_model_int8.onnx   # Skwantyzowany model ONNX INT8 (1.17 MB - redukcja o 74.6%)
│   ├── best_hdd_cnn_model_torchscript.pt # Model TorchScript / PyTorch Mobile (4.68 MB)
│   ├── AcousticClassifierAndroid.kt   # Szablon integracyjny w Kotlin dla Androida
│   ├── model_card.json                # Karta modelu z parametrami DSP i specyfikacją
│   └── classes.json                   # Mapowanie indeksów klas dla aplikacji
│
└── docx/
    ├── adr/adr.md                     # ADR-001: Architektura DSP i niezmienniki techniczne
    ├── diagramOverView.svg            # Schemat architektury systemu
    └── specs.txt                      # Specyfikacja techniczna wymagań
```

---

## 4. Polecenia szybkiego startu (Quickstart dla AGY)

Każdy agent lub programista może uruchomić kluczowe etapy za pomocą gotowych poleceń:

### A. Sprawdzenie integralności i testów logicznych projektu:
```bash
python tests/test_project_logic.py
```

### B. Weryfikacja sprawności modelu na GPU/CPU:
```bash
python tests/verify_cnn_model.py
```

### C. Profilowanie jakości normalizacji DSP:
```bash
python tests/profile_normalization_quality.py
```

### D. Regeneracja zbioru audio i tensorów:
```bash
python ml/prepare_audio_dataset.py
python ml/hdd_audio_encoder.py
```

### E. Uruchomienie pełnego treningu (PyTorch z akceleracją ROCm):
```bash
python ml/train_cnn.py --epochs 35 --batch-size 32 --output-dir ./training_output
```

### F. Trening komitetu modeli (Seed Ensembling):
```bash
python ml/train_ensemble.py --epochs 35 --batch-size 32 --output-dir ./training_output
```

### G. Eksport do formatów produkcyjnych (ONNX, TorchScript, Android Kotlin):
```bash
python ml/export_model.py --weights training_output/best_hdd_cnn_model.pt --output-dir exported_models
```

### H. Testy modułu aplikacji mobilnej Android:
```bash
cd android && ./gradlew test
```

---

## 5. Aktualne wyniki modelu produkcyjnego (Stan obecny)
- **Model:** Wytrenowany na 1389 zbalansowanych tensorach treningowych z ważeniem klas `CrossEntropyLoss(label_smoothing=0.03)` oraz `AdamW`.
- **Ewaluacja:** Zbiór testowy w 100% czysty (117 próbek bez nakładania okien i bez sztucznej augmentacji).
- **Wykrywanie dysków sprawnych (`normal`):** **100% precyzja, 100% czułość (F1 = 1.000)** na zbiorze testowym.
- **Wykrywanie stukania głowicy (`clicking`):** **Precision: 70.5%, Recall: 88.6%, F1 = 0.785**.
- **Wykrywanie zatarcia / zablokowania (`beeping`):** **Precision: 86.4%, Recall: 86.4%, F1 = 0.864**.
- **Wykrywanie tarcia głowic o talerz (`scratching`):** **Precision: 87.5%, Recall: 63.6%, F1 = 0.737**.
- **Dokładność ogólna (Accuracy):** **83.76%** (98/117 próbek testowych).
- **Średnia Makro (Macro F1):** **0.8463 (84.63%)** — warunek $F1 \ge 0.70$ dla wszystkich klas w pełni spełniony!

---

## 6. Automatyczna Integracja z Lokalnymi Modelami LLM (Ollama & RX 9060 XT)
W systemie użytkownika działa lokalna instalacja Ollama (`http://localhost:11434` oraz Open WebUI na `http://localhost:8080`) z dwoma modelami open-weights:
1. **Architekt / Planista (128k kontekstu, 12 GB VRAM):** `hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S`
2. **Programista Wykonawczy / Coder (9 GB VRAM):** `qwen2.5-coder:14b`

### Zasady automatycznego wykorzystania przez każdego agenta:
- **Uruchomienie usług LLM w razie potrzeby:**
  Przed pracą z modelami lokalnymi agent może upewnić się o działaniu usług za pomocą `./scripts/start_services.sh` (lub zatrzymać je po pracy przez `./scripts/stop_services.sh`).
- **Automatyczne delegowanie zadań programistycznych:**
  Gdy użytkownik zleca projektowanie nowych modułów, implementację zaawansowanych algorytmów, agent powinien **automatycznie** wykorzystać tandem modeli za pomocą skryptu:
  ```bash
  python scripts/local_llm_bridge.py --task "<opis zadania>" [--context <plik_kontekstu>] [--output <plik_wyjścia>]
  ```
- **Niezmiennik VRAM (16 GB) i architektura iGPU + dGPU:**
  Monitor i pulpit Linux działają na iGPU (Ryzen 7 7700), dzięki czemu karta **AMD Radeon RX 9060 XT (16 GB)** ma 100% VRAM (15.9 GB) wolne na czyste obliczenia AI. Dzięki FlashAttention i Q4_0 KV Cache w Ollama, model Qwen 3.8 27B obsługuje bezpiecznie kontekst **aż do 128k (`131072`)** bez wycieku do RAM CPU.
  Modele 27B i 14B nie mogą znajdować się w VRAM równocześnie. Skrypt `local_llm_bridge.py` automatycznie wyładowuje modele (`keep_alive=0`). Przed rozpoczęciem jakiegokolwiek treningu PyTorch na GPU agent **MUSI** upewnić się, że VRAM jest zwolniony (`ollama ps` puste).
