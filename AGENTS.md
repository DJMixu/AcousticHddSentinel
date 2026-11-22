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
├── .agents/                           # Customizacje agentów AGY (Antigravity CLI)
│   ├── rules/
│   │   ├── adr-001-audio-pipeline.md  # Reguła: niezmienniki DSP pipeline (always_on)
│   │   └── local-llm-orchestration.md # Reguła: kooperacja z lokalnymi LLM Ollama (always_on)
│   └── skills/
│       ├── local-llm-teamwork/        # Skill: Hybryda /teamwork + lokalne agenty LLM (Qwen 3.8 + Coder 14B)
│       │   └── SKILL.md               # Instrukcja integracji swarm, briefy i zarządzanie VRAM
│       └── dsp-adversarial-sentinel/  # Skill: Przetwarzanie sygnałów DSP, strażnicy fizyczni i testy adwersarialne
│           └── SKILL.md               # Standard analizy widmowej, RPM, filtracji wentylatorów i walidacji
├── .gitignore                         # Wykluczenia git (duże zbiory .npy, venv, cache)
│
├── ml/                                # 🧠 Pipeline uczenia maszynowego (PyTorch 2D CNN, DSP, eksport)
│   ├── model_cnn.py                   # Architektura sieci 2D CNN w PyTorch (AcousticHddSentinelCNN, SMPC Option A & FusedEnsemblePyTorch)
│   ├── dataset_pytorch.py             # PyTorch Dataset z preloadem RAM i SpecAugment (HddNpyDataset)
│   ├── hdd_audio_encoder.py           # Enkoder audio do tensorów .npy zgodny z ADR-001
│   ├── rpm_harmonics_analyzer.py      # Analizator pików częstotliwościowych, stacjonarności pitch w czasie (5 niespójnych przesunięć, interpolacja paraboliczna), ostrości piku (sharpness >= 6.0) oraz grzebienia harmonicznego RPM (5400/7200/10000)
│   ├── hybrid_acoustic_sentinel.py    # Hybrydowa fuzja decyzyjna (DSP RPM Guard + OOD Ambient Rejection + SMPC Gałąź D + 2D CNN)
│   ├── dsp_adversarial_guard.py       # Strażnik adwersarialny (detekcja wentylatorów PC slope<=-1.80, kurtoza, clipping, VAD)
│   ├── prepare_audio_dataset.py       # Kategoryzator bazy audio i generator klasy 'normal'
│   ├── train_cnn.py                   # Kompletny pipeline treningowy w PyTorch (AdamW, Label Smoothing, ROCm, SMPC)
│   ├── train_ensemble.py              # Pipeline Seed Ensembling (komitet modeli o różnych ziarnach)
│   ├── export_model.py                # Eksport do formatów ONNX (FP32/INT8), TorchScript z auto-sync do Androida
│   └── compare_ensembles.py           # Narzędzie ewaluacji porównawczej komitetu modeli z modelem bazowym
│
├── tests/                             # 🧪 Testy logiczne, weryfikacja wag, profilowanie i stress-testy
│   ├── test_project_logic.py          # Automatyczny test integralności projektu i niezmienników ADR-001 (6/6)
│   ├── test_smpc_architecture.py      # Testy jednostkowe architektury Sub-Center Multi-Prototype Cosine (SMPC Option A, 7/7)
│   ├── test_rpm_harmonics_analyzer.py # Testy jednostkowe analizatora harmonicznych obrotowych RPM i odrzucania głosu (9/9)
│   ├── test_computer_sound_rejection.py # Testy odporności fuzji hybrydowej na szumy wentylatorów, GPU i cichy sygnał 45 dB (6/6)
│   ├── test_full_dataset_fault_detection_rate.py # Weryfikacja wykrywalności awarii na 100% całego datasetu (6 840 tensorów, >80% recall)
│   ├── evaluate_app_pipeline_on_dataset.py # Pełna ewaluacja potoku aplikacji (z DSP vs bez DSP, 0% FPR na dyskach)
│   ├── verify_binary_verdict_and_real_drives.py # Test 320 okien rzeczywistych dysków laboratoryjnych
│   ├── verify_dsp_parity_python_java.py # Weryfikacja numerycznej parzystości DSP (Python vs Android Java 17)
│   ├── verify_cnn_model.py            # 6-etapowy test weryfikacyjny sprawności modelu PyTorch
│   ├── profile_normalization_quality.py # Profiler i weryfikator jakości normalizacji DSP
│   ├── stress_test.py                 # Test ciągły odporności potoku DSP + CNN (latencja, wycieki RAM, edge cases)
│   ├── stress_test_vram.py            # Test obciążeniowy pamięci VRAM karty GPU
│   └── testEpok.py                    # Diagnostyka stabilności uczenia dla różnej liczby epok
│
├── scripts/                           # 🛠️ Skrypty narzędziowe i orkiestracja lokalnych modeli LLM
│   ├── install_apk.sh                 # Automatyczna instalacja APK i uruchomienie na telefonie Android
│   ├── record_mic.sh                  # Skrypt nagrywania z fizycznego mikrofonu (ALC1220 Analog)
│   ├── audio_hardware_config.json     # Konfiguracja fizycznych źródeł audio w systemie Linux
│   ├── local_llm_bridge.py            # Most orkiestrujący lokalne modele Ollama (27B + Coder 14B) z 7 presetami
│   ├── start_services.sh              # Skrypt uruchamiania usług Ollama i Open WebUI
│   └── stop_services.sh               # Skrypt zatrzymywania usług Ollama i Open WebUI
│
├── android/                           # 📱 Kompletny moduł natywnej aplikacji mobilnej (Android Studio / Gradle / Java 17, Material 3 UI)
│   ├── app/
│   │   ├── src/main/assets/           # Skwantyzowany model ONNX INT8, classes.json, model_card.json
│   │   ├── src/main/java/             # Pakiety: audio (AudioRecord, RAW mode), data (SQLite), dsp (Zero GC Churn), ml (Wzorzec Strategii Dual-Model), ui (MVVM & LiveData)
│   │   ├── src/main/res/              # Zasoby UI (układ radarowy, motywy Material 3, kolory, ikony)
│   │   ├── src/test/java/             # 13 klas testowych (74 testy jednostkowe w czystej Javie 17: AudioChunk, AudioPreferences, Decymacja, Preprocessor, RPM, Strażnik fizyczny, Audio, Baza SQLite, Klasyfikacja, ModelStrategy, DspEdgeCases, DatasetAudioPipeline, OnnxInference)
│   │   ├── build.gradle.kts           # Zależności: ONNX Runtime Mobile, AndroidX AppCompat, Material 3, Lifecycle (0 Kotlina)
│   │   └── AndroidManifest.xml        # Uprawnienia RECORD_AUDIO, deklaracje aktywności
│   ├── build.gradle.kts / settings.gradle.kts # Konfiguracja projektu Gradle
│   ├── gradlew / gradlew.bat          # Wrapper narzędzia Gradle
│   └── README.md                      # Instrukcja uruchomienia i testowania w Android Studio
│
├── data/                              # Baza surowych nagrań laboratoryjnych (Ground Truth)
│   └── raw/                           # 353 autentyczne nagrania audio (northwind, aesonlabs, archive, freesound, web, verified_real)
│       └── PROCESSED_SOURCES_REGISTRY.json # Rejestr unikalnych próbek i sum kontrolnych MD5
│
├── dataset_categorized/               # Nagrania źródłowe podzielone na 4 klasy
├── dataset_encoded/                   # 6 840 tensorów .npy (Zero Data Leakage: train 6045, validate 279, test 516)
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
│   ├── evaluation_with_vs_without_dsp.json # Raport porównawczy modelu z DSP vs bez DSP
│   ├── full_dataset_fault_detection_summary.json # Podsumowanie ewaluacji 100% całego datasetu (6 840 tensorów)
│   ├── normalization_quality_report.png # Wykres profilowania jakości normalizacji
│   ├── normalization_quality_metrics.json # Metryki numeryczne normalizacji
│   └── classes.json                   # Mapowanie klas i parametrów wejścia
│
├── exported_models/                   # Wyeksportowane modele dla urządzeń brzegowych i aplikacji mobilnych
│   ├── best_hdd_cnn_model.onnx        # Model ONNX FP32 z dynamiczną osią batcha (4.61 MB)
│   ├── best_hdd_cnn_model_int8.onnx   # Skwantyzowany model ONNX INT8 (1.17 MB - redukcja o 74.6%)
│   ├── best_hdd_cnn_model_torchscript.pt # Model TorchScript / PyTorch Mobile (4.68 MB)
│   ├── AcousticClassifierAndroid.java # Szablon integracyjny w czystej Javie 17 dla Androida (ADR-002)
│   ├── model_card.json                # Karta modelu z parametrami DSP i specyfikacją
│   └── classes.json                   # Mapowanie indeksów klas dla aplikacji
│
└── docx/
    ├── adr/
    │   ├── adr.md                     # ADR-001: Architektura DSP i niezmienniki techniczne
    │   └── adr-002-mobile-architecture.md # ADR-002: Architektura aplikacji mobilnej Android
    ├── diagramOverView.svg            # Schemat architektury systemu
    └── specs.txt                      # Zaktualizowana specyfikacja techniczna wymagań
```

---

## 4. Polecenia szybkiego startu (Quickstart dla AGY)

Każdy agent lub programista może uruchomić kluczowe etapy za pomocą gotowych poleceń:

### A. Sprawdzenie integralności i 7 oficjalnych zestawów testowych systemu:
```bash
# 1. Automatyczny test integralności projektu i niezmienników ADR-001 (6/6):
python tests/test_project_logic.py

# 2. Test architektury wielomodalnych podcentrów (SMPC Option A, 7/7):
python tests/test_smpc_architecture.py

# 3. Test analizatora harmonicznych obrotowych RPM i odrzucania głosu (9/9):
python tests/test_rpm_harmonics_analyzer.py

# 4. Test odporności fuzji hybrydowej na szumy wentylatorów PC i cichy sygnał 45 dB (6/6):
python tests/test_computer_sound_rejection.py

# 5. Weryfikacja wykrywalności awarii na 100% całego datasetu (6 840 tensorów, >80% recall):
python tests/test_full_dataset_fault_detection_rate.py

# 6. Pełna ewaluacja potoku aplikacji (z DSP vs bez DSP, 0% FPR na dyskach laboratoryjnych):
python tests/evaluate_app_pipeline_on_dataset.py

# 7. Zestaw testów jednostkowych aplikacji mobilnej Android (Czysta Java 17, 13 klas, 74 testy):
cd android && ./gradlew test && cd ..
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

### G. Eksport do formatów produkcyjnych (ONNX, TorchScript, Android Java 17):
```bash
python ml/export_model.py --weights training_output/best_hdd_cnn_model.pt --output-dir exported_models
```

### H. Testy modułu aplikacji mobilnej Android:
```bash
cd android && ./gradlew test
```

---

## 5. Aktualne wyniki modelu produkcyjnego i ewaluacja całego datasetu
- **Pojedynczy Model Produkcyjny (Test Split 117 próbek):**
  - **Wykrywanie dysków sprawnych (`normal`):** **100% precyzja, 100% czułość (F1 = 1.000)**.
  - **Wykrywanie stukania głowicy (`clicking`):** **Precision: 70.5%, Recall: 88.6%, F1 = 0.785**.
  - **Wykrywanie zatarcia / zablokowania (`beeping`):** **Precision: 86.4%, Recall: 86.4%, F1 = 0.864**.
  - **Wykrywanie tarcia głowic o talerz (`scratching`):** **Precision: 87.5%, Recall: 63.6%, F1 = 0.737**.
  - **Dokładność ogólna (Accuracy):** **83.76%** (98/117 próbek testowych).
  - **Średnia Makro (Macro F1):** **0.8463 (84.63%)** — warunek $F1 \ge 0.70$ dla wszystkich klas w pełni spełniony!

- **Kompleksowa Ewaluacja na 100% Całego Datasetu (6 840 tensorów, 3 765 awarii):**
  - **Czułość wykrywania awarii przy progu operacyjnym $P(\text{fault}) \ge 0.70$:** **80.05%** w całym zbiorze (Test: 70.89%, Validate: 73.28%, Train: 80.88%).
  - **Czułość wykrywania awarii przy standardowym progu $P(\text{fault}) \ge 0.50$:** **89.03%** w całym zbiorze (Test: 85.92%, Validate: 86.26%, Train: 89.33%).
  - **Wykrywalność per klasa uszkodzenia (próg 0.70):**
    * `beeping`: **86.71%** (842 / 971 tensorów)
    * `scratching`: **81.58%** (1 311 / 1 607 tensorów)
    * `clicking`: **72.54%** (861 / 1 187 tensorów)
  - **False Positive Rate (FPR) na dyskach sprawnych:** zaledwie 8–12% na zaszumionych wycinkach laboratoryjnych oraz **0.0% FPR** na 320 oknach rzeczywistych dysków (Toshiba, WD, Seagate, HGST).

---

## 6. Automatyczna Integracja z Lokalnymi Modelami LLM (Ollama & RX 9060 XT)
W systemie użytkownika działa lokalna instalacja Ollama (`http://localhost:11434` oraz Open WebUI na `http://localhost:8080`) z dwoma modelami open-weights zoptymalizowanymi pod architekturę RDNA4:
1. **Architekt / Planista (128k kontekstu, 12 GB VRAM):** `hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S` (alias Ollama: `qwen3.8-architect`)
   - **Rola:** Główny Architekt Oprogramowania & DSP. Analizuje zadania, weryfikuje niezmienniki ADR-001, projektuje sygnatury i pseudokod.
   - **Parametry generowania:** `temperature: 0.2`, `top_p: 0.95`, `repeat_penalty: 1.05`, `num_ctx: do 131072`, `num_predict: -1` (brak limitu długości).
   - **Reasoning Stream:** Model typu RCO z kanałem myślenia (`thinking`), separowany od właściwej specyfikacji (`content`).
2. **Programista Wykonawczy / Coder (9 GB VRAM):** `qwen2.5-coder:14b` (alias Ollama: `qwen2.5-coder-expert`)
   - **Rola:** Senior Staff Software Engineer (PyTorch / DSP). Pisze gotowy do wdrożenia kod w Pythonie bez skrótów i placeholderów.
   - **Parametry generowania:** `temperature: 0.1` (deterministyczna), `top_p: 0.90`, `repeat_penalty: 1.05`, `num_predict: -1`.

### Zasady automatycznego wykorzystania przez każdego agenta:
- **Uruchomienie usług LLM w razie potrzeby:**
  Przed pracą z modelami lokalnymi agent może upewnić się o działaniu usług za pomocą `./scripts/start_services.sh` (lub zatrzymać je po pracy przez `./scripts/stop_services.sh`).
- **Automatyczne delegowanie zadań programistycznych:**
  Gdy użytkownik zleca projektowanie nowych modułów, implementację zaawansowanych algorytmów, agent powinien **automatycznie** wykorzystać tandem modeli za pomocą skryptu:
  ```bash
  python scripts/local_llm_bridge.py --task "<opis zadania>" [--context <plik_kontekstu>] [--output <plik_wyjścia>]
  ```
  *Skrypt automatycznie wstrzykuje kluczowe niezmienniki ADR-001 z AGENTS.md, obsługuje strumieniowanie myślenia Architekta i natychmiast zwalnia pamięć VRAM.*
- **Niezmiennik VRAM (16 GB) i architektura iGPU + dGPU:**
  Monitor i pulpit Linux działają na iGPU (Ryzen 7 7700), dzięki czemu karta **AMD Radeon RX 9060 XT (16 GB)** ma 100% VRAM (15.9 GB) wolne na czyste obliczenia AI. Dzięki FlashAttention i Q4_0 KV Cache w Ollama, model Qwen 3.8 27B obsługuje bezpiecznie kontekst **aż do 128k (`131072`)** bez wycieku do RAM CPU.
  Modele 27B i 14B nie mogą znajdować się w VRAM równocześnie. Skrypt `local_llm_bridge.py` automatycznie wyładowuje modele (`keep_alive=0`). Przed rozpoczęciem jakiegokolwiek treningu PyTorch na GPU agent **MUSI** upewnić się, że VRAM jest zwolniony (`ollama ps` puste).


---

## 7. Autonomiczne Zarządzanie Subagentami AI (Subagents & Parallel Workers)
Agent AGY (Antigravity) posiada pełne uprawnienie i zalecenie do autonomicznego tworzenia i orkiestracji niezależnych subagentów AI:
1. **Subagenci `self` (Klonowanie agenta głównego):**
   - Agent może w dowolnym momencie powoływać niezależnych subagentów `self` dziedziczących pełne uprawnienia, prompt, narzędzia i środowisko.
   - Stosować do długotrwałych zadań równoległych (np. wielogodzinny crawling archiwów, scraping forów dyskowych, generowanie syntetycznych ataków adversarial, niezależny audyt kodu).
2. **Subagenci wyspecjalizowani (`define_subagent`):**
   - Tworzenie dedykowanych workerów zadaniowych z precyzyjnie dobranym system promptem i zakresem uprawnień (np. `hdd-archive-scraper`, `dsp-verifier`, `code-reviewer`).
3. **Zasady koordynacji:**
   - Subagenci raportują wyniki asynchronicznie poprzez system wiadomości (`send_message`).
   - Agent główny nie polluje w pętli – oczekuje reaktywnie na powiadomienia o zakończeniu prac subagentów.

### 7.1 Oficjalny Standard Zespołowy: Natywny `swarm` z Modelami Qwen (Zastępuje generyczny `/teamwork`)
W projekcie obowiązuje **bezwzględny priorytet natywnego narzędzia `swarm`** nad generycznym `/teamwork`.
Skill `local-llm-teamwork` ([.agents/skills/local-llm-teamwork/SKILL.md](file:///home/DJMixu/Pulpit/AcousticHddSentinel/.agents/skills/local-llm-teamwork/SKILL.md)) definiuje oficjalne procedury pracy swarma:

1. **Dlaczego `swarm` jest standardem projektu?**
   - **Oszczędność ~90% tokenów chmurowych:** Koordynator zarządza jedynie zwięzłą tablicą wiadomości. Nie duplikuje gigantycznego kontekstu repozytorium do chmury, eliminując błędy `429 Resource Exhausted`.
   - **Czysty kod produkcyjny:** Wyspecjalizowany model `qwen2.5-coder:14b` na GPU (`temp=0.1`) generuje kompletne implementacje bez ucięć, skrótów i placeholderów `TODO`.
   - **Architekt 128k VRAM:** Model `Qwen 3.8 27B` (IQ3_S, RCO reasoning) analizuje całą architekturę w 12 GB VRAM z pełnym kanałem myślenia.
   - **ZAKAZ PISANIA KODU Z GŁOWY PRZEZ CHMURĘ:** Koordynator i chmurowe workery mają zakaz generowania kodu z głowy – kod MUSI pochodzić z lokalnego GPU poprzez `scripts/local_llm_bridge.py`.

2. **Role workerów w `swarm` i profile zadań (`--preset`):**
   | Worker / Rola | Narzędzie Swarm | Delegacja na GPU | Flaga / Preset |
   |:---|:---|:---|:---|
   | `architect` | `swarm{action:"add", name:"architect"}` | ✅ Qwen 3.8 27B | `--arch-only` (128k ctx) |
   | `coder` | `swarm{action:"add", name:"coder"}` | ✅ Qwen 2.5 Coder 14B | `--coder-only` (temp=0.1) |
   | `implementer` | `swarm{action:"add", name:"implementer"}` | ✅ Tandem obu modeli | Pełny potok |
   | `verifier` / `reviewer` | `swarm{action:"add", name:"verifier"}` | ✅ Qwen 2.5 Coder 14B | `--preset code-review` |
   | `dsp-translator` | `swarm{action:"add", name:"dsp-translator"}` | ✅ Qwen 2.5 Coder 14B | `--preset dsp-translate` |
   | `adversarial-gen` | `swarm{action:"add", name:"adversarial-gen"}` | ✅ Qwen 3.8 27B | `--preset adversarial-gen` |
   | `android-opt` | `swarm{action:"add", name:"android-opt"}` | ✅ Qwen 2.5 Coder 14B | `--preset android-opt` |
   | `smpc-architect` | `swarm{action:"add", name:"smpc-architect"}` | ✅ Qwen 3.8 27B | `--preset smpc-arch` |
   | `num-auditor` | `swarm{action:"add", name:"num-auditor"}` | ✅ Qwen 2.5 Coder 14B | `--preset num-audit` |
   | `telemetry-analyst` | `swarm{action:"add", name:"telemetry-analyst"}` | ✅ Qwen 3.8 27B | `--preset smart-telemetry` |
   | `tester` | `swarm{action:"add", name:"tester"}` | ❌ Brak (AGY / Bash) | — |

3. **Sekwencyjność GPU i porządek zwolnienia pamięci:**
   Tylko jeden proces na raz może korzystać z GPU (16 GB VRAM). Kolejność w potoku:
   `Architect` $\to$ `Coder` $\to$ `Verifier` $\to$ czyszczenie VRAM przed jakimkolwiek treningiem PyTorch.

4. **Wczesne Uruchamianie (Front-Loaded Execution — ZAKAZ OPÓŹNIANIA):**
   Zadania dla modeli Qwen na GPU **MUSZĄ być zlecane w pierwszej kolejności na starcie każdego cyklu pracy**, eliminując przestoje zespołu.

5. **Sprawdzanie i zwalnianie VRAM:**
   ```bash
   python scripts/local_llm_bridge.py --check-ready   # Stan Ollama i VRAM
   python scripts/local_llm_bridge.py --free-vram      # Wyładowanie modeli przed treningiem
   ```

---

## 8. Niezmiennik Sprzętowy: Konfiguracja Mikrofonu Fizycznego Użytkownika (ZAKAZ POMYŁEK)
W systemie Linux użytkownika zainstalowane są różne urządzenia wejściowe audio w PipeWire / PulseAudio:
- **JEDYNY WŁAŚCIWY MIKROFON FIZYCZNY:**
  `alsa_input.pci-0000_7b_00.6.analog-stereo`
  (Opis urządzenia: *Ryzen HD Audio Controller / Realtek ALC1220 Analog* na płycie głównej komputera).
  Do tego złącza podłączony jest fizyczny mikrofon, do którego użytkownik mówi, puszcza dźwięki otoczenia i szum.
- **BEZWZGLĘDNY ZAKAZ UŻYWANIA:**
  `alsa_input.usb-C-Media_Electronics_Inc._USB_Advanced_Audio_Device-00.analog-stereo`
  (Jest to dongle / słuchawki bez podpiętego mikrofonu – rejestruje wyłącznie sztuczny szum cyfrowy o energii RMS ~0.00007).
  Nigdy nie używać źródła `default` bez uprzedniego upewnienia się, że wskazuje na kartę PCI ALC1220!
- **Gotowy skrypt rejestracji audio:**
  Każdy agent nagrywający dźwięk z otoczenia MUSI korzystać z gotowego skryptu lub jawnego źródła:
  ```bash
  # Użycie gotowego skryptu:
  ./scripts/record_mic.sh <czas_w_sekundach> <ścieżka_wyjściowa.wav>
  
  # Lub bezpośrednio przez ffmpeg:
  ffmpeg -y -f pulse -i alsa_input.pci-0000_7b_00.6.analog-stereo -t 10 -ar 22050 -ac 1 output.wav
  ```
  Przed nagraniem agent może upewnić się o poprawnym źródle za pomocą:
  `pactl set-default-source alsa_input.pci-0000_7b_00.6.analog-stereo`
- **Plik konfiguracyjny Single Source of Truth:**
  [scripts/audio_hardware_config.json](file:///home/DJMixu/Pulpit/AcousticHddSentinel/scripts/audio_hardware_config.json)

