# Acoustic HDD Sentinel 🛡️💽
> **System akustycznej detekcji i klasyfikacji awarii mechanicznych dysków twardych (HDD) przy użyciu głębokich sieci konwolucyjnych (2D CNN w PyTorch).**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-F1%20%3E%200.70%20(Verified)-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/Acceleration-AMD%20ROCm%20HIP%20(RX%209060%20XT)-purple.svg)]()

---

## 📖 Spis treści
1. [O projekcie](#-o-projekcie)
2. [Architektura przetwarzania audio (ADR-001)](#-architektura-przetwarzania-audio-adr-001)
3. [Architektura sieci 2D CNN](#-architektura-sieci-2d-cnn)
4. [Instalacja i wymagania](#-instalacja-i-wymagania)
5. [Szybki start i instrukcja obsługi](#-szybki-start-i-instrukcja-obsługi)
6. [Wyniki treningu i ewaluacja testowa](#-wyniki-treningu-i-ewaluacja-testowa)
7. [Eksport modeli produkcyjnych (ONNX / Mobile / Android)](#-eksport-modeli-produkcyjnych-onnx--mobile--android)
8. [Uporządkowana struktura projektu](#-uporządkowana-struktura-projektu)
9. [Współpraca z Antigravity (AGY)](#-współpraca-z-antigravity-agy)

---

## 🎯 O projekcie
Acoustic HDD Sentinel analizuje krótkie, 3-sekundowe nagrania dźwiękowe pracy dysków twardych, wykrywając uszkodzenia mechaniczne i klasyfikując ich typ przed katastrofalną utratą danych:
- **`clicking`** – stukanie głowicy o ogranicznik (*Click of Death*, uszkodzenie pozycjonera / zespołu głowic).
- **`beeping`** – pisk / melodia zatartego silnika wrzeciona (*stuck spindle*, *stiction* głowic na talerzu).
- **`scratching`** – szerokopasmowe tarcie głowicy po magnetycznej powierzchni talerza (*head crash / platter scoring*).
- **`normal`** – prawidłowa, stabilna praca dysku twardego.

---

## 🔬 Architektura przetwarzania audio (ADR-001)
Zgodnie z decyzją architektoniczną **ADR-001** (`docx/adr/adr.md`), każde 3-sekundowe okno audio jest transformowane do **3-kanałowego tensora `(128, 259, 3)`**:
1. **Kanał 0:** Log-Mel Spektrogram (128 pasm Mel, skala decybelowa)
2. **Kanał 1:** Delta – 1. pochodna czasowa (detekcja ostrych transjentów i uderzeń)
3. **Kanał 2:** Delta-Delta – 2. pochodna czasowa (dynamika przyspieszeń obrotowych)

### Normalizacja i filtracja:
- **Peak Normalization:** Surowy wycinek audio jest skalowany szczytowo do `[-1.0, 1.0]`, co zachowuje wysoki współczynnik szczytu (*Crest Factor* sięgający w nagraniach awarii aż `131.7`).
- **Per-Channel Z-score:** Każdy z 3 kanałów tensora jest niezależnie standaryzowany ($\mu = 0.0$, $\sigma = 1.0$), uniezależniając model od 430-krotnego rozrzutu głośności mikrofonów.
- **Filtracja ciszy:** Odrzucanie fragmentów o energii RMS $< 0.002$ w celu eliminacji szumu własnego przetworników.

---

## 🧠 Architektura sieci 2D CNN
Model w `model_cnn.py` został zaprojektowany z myślą o specyfice danych akustycznych:
- **Format tensora:** Natywny `(B, 3, 128, 259)` (Channel-First) z przezroczystą obsługą `(B, 128, 259, 3)` (Channel-Last).
- **4 Bloki splotowe:** `Conv2D` (32 -> 64 -> 128 -> 256 filtrów $3 \times 3$), `BatchNorm2d`, `ReLU`, `MaxPool2d(2, 2)`, `Dropout2d` oraz regularyzacja wag.
- **Agregacja:** `AdaptiveAvgPool2d((1, 1))` (Global Average Pooling redukujący wymiar do 256 bez eksplozji parametrów, całkowita liczba wag: **1.21 mln**).
- **Klasyfikator:** `Linear(256, 128, bias=False)` + `BatchNorm1d` + `Dropout(0.35)` + `Linear(128, num_classes)`.
- **Komitet modeli (Ensemble):** `FusedEnsemblePyTorch` integrujący modele trenowane na różnych ziarnach losowości z uśrednianiem Softmax (Soft Voting).

---

## 💻 Instalacja i wymagania

Wymagany Python 3.10+ oraz wirtualne środowisko:

```bash
git clone git@github.com:DJMixu/AcousticHddSentinel.git
cd AcousticHddSentinel

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Szybki start i instrukcja obsługi

### 1. Weryfikacja integralności projektu
```bash
python tests/test_project_logic.py
```

### 2. Sprawdzenie sprawności modelu na GPU/CPU
```bash
python tests/verify_cnn_model.py
```

### 3. Sprawdzenie jakości normalizacji danych
```bash
python tests/profile_normalization_quality.py
```

### 4. Przygotowanie danych i enkodowanie
```bash
# 1. Kategoryzacja nagrań i wygenerowanie klasy 'normal'
python ml/prepare_audio_dataset.py

# 2. Enkodowanie do tensorów .npy z podziałem bez wycieku danych (Zero Data Leakage)
python ml/hdd_audio_encoder.py
```

### 5. Uruchomienie treningu (PyTorch)
```bash
# Trening pojedynczego modelu:
python ml/train_cnn.py --epochs 35 --batch-size 32 --output-dir ./training_output

# Trening komitetu modeli (Seed Ensembling):
python ml/train_ensemble.py --epochs 35 --batch-size 32 --output-dir ./training_output
```

---

## 📊 Wyniki treningu i ewaluacja testowa

Ewaluacja produkcyjnego modelu na niezależnym, w 100% czystym zbiorze testowym (`TEST` – 117 tensorów z niewidzianych wcześniej dysków, 0% overlapu, brak sztucznej augmentacji):

| Klasa awarii | Precyzja (Precision) | Czułość (Recall) | F1-Score | Próbki testowe | Warunek ($F1 \ge 0.70$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`normal` (sprawny dysk)** | **100.0%** | **100.0%** | **1.0000** | 27 | **SPEŁNIONY** |
| **`beeping` (zatarte wrzeciono)** | **86.4%** | **86.4%** | **0.8636** | 22 | **SPEŁNIONY** |
| **`clicking` (stukanie głowic)** | **70.5%** | **88.6%** | **0.7848** | 35 | **SPEŁNIONY** |
| **`scratching` (tarcie talerza)** | **87.5%** | **63.6%** | **0.7368** | 33 | **SPEŁNIONY** |
| **Średnia Makro (Macro F1)** | **86.08%** | **84.64%** | **0.8463** | **117** | **SPEŁNIONY** |
| **Dokładność ogólna (Accuracy)** | — | — | **83.76%** | **117** | — |

Wizualizacje i artefakty:
- `training_output/confusion_matrix.png` – macierz pomyłek z wartościami liczbowymi i procentami
- `training_output/training_curves_seed_777.png` – krzywe funkcji straty i dokładności
- `training_output/best_hdd_cnn_model.pt` – wagi modelu referencyjnego

---

## 📱 Eksport modeli produkcyjnych (ONNX / Mobile / Android)

Model można natychmiast wyeksportować do formatów zoptymalizowanych pod urządzenia mobilne i brzegowe:
```bash
python ml/export_model.py --weights training_output/best_hdd_cnn_model.pt --output-dir exported_models
```
Wygenerowane artefakty w `exported_models/`:
- `best_hdd_cnn_model.onnx` – ONNX FP32 (4.61 MB)
- `best_hdd_cnn_model_int8.onnx` – Skwantyzowany ONNX INT8 (1.17 MB – **redukcja o 74.6%**)
- `best_hdd_cnn_model_torchscript.pt` – TorchScript / PyTorch Mobile (4.68 MB)
- `AcousticClassifierAndroid.kt` – Gotowa klasa integracyjna w Kotlin dla Androida z ONNX Runtime Mobile
- `model_card.json` – Karta modelu z parametrami DSP i specyfikacją wejść/wyjść

---

## 📁 Uporządkowana struktura projektu

```
AcousticHddSentinel/
├── AGENTS.md                          # Wytyczne i architektura dla agentów AI (Antigravity CLI)
├── kontekst.md                        # Całościowe podsumowanie dotychczasowych etapów i audytów
├── fineTuning.md                      # Dokumentacja prac optymalizacyjnych
├── NOTES.md                           # Notatki badawcze (Seed Ensembling)
├── README.md                          # Niniejsza dokumentacja projektu
├── requirements.txt                   # Zależności biblioteczne Pythona
├── .gitignore                         # Wykluczenia git (duże tensory .npy, venv, cache)
│
├── ml/                                # 🧠 Moduł uczenia maszynowego i potoku DSP
│   ├── model_cnn.py                   # Architektura 2D CNN (AcousticHddSentinelCNN & FusedEnsemblePyTorch)
│   ├── dataset_pytorch.py             # PyTorch Dataset z preloadem RAM i SpecAugment (HddNpyDataset)
│   ├── hdd_audio_encoder.py           # Enkoder audio do tensorów .npy zgodny z ADR-001
│   ├── prepare_audio_dataset.py       # Kategoryzator bazy audio i generator klasy 'normal'
│   ├── train_cnn.py                   # Pipeline treningowy PyTorch (AdamW, Label Smoothing, ROCm)
│   ├── train_ensemble.py              # Trening komitetu modeli (Seed Ensembling)
│   ├── export_model.py                # Eksport do ONNX (FP32/INT8), TorchScript z auto-sync do Androida
│   └── compare_ensembles.py           # Porównanie komitetu modeli z modelem bazowym
│
├── tests/                             # 🧪 Moduł testów integralności i weryfikacji
│   ├── test_project_logic.py          # Test integralności projektu i niezmienników ADR-001 (6/6)
│   ├── verify_cnn_model.py            # 6-etapowa weryfikacja poprawności modelu PyTorch
│   ├── profile_normalization_quality.py # Profiler i weryfikator jakości normalizacji DSP
│   ├── stress_test.py                 # Test ciągły odporności potoku DSP + CNN (latencja, wycieki RAM, edge cases)
│   ├── stress_test_vram.py            # Test obciążeniowy pamięci VRAM karty graficznej
│   └── testEpok.py                    # Diagnostyka stabilności uczenia dla różnej liczby epok
│
├── scripts/                           # 🛠️ Skrypty narzędziowe i orkiestracja lokalnych usług LLM
│   ├── local_llm_bridge.py            # Most orkiestrujący lokalne modele LLM (Qwen 3.8 + Coder 14B)
│   ├── start_services.sh              # Skrypt uruchamiania usług Ollama i Open WebUI
│   └── stop_services.sh               # Skrypt zatrzymywania usług Ollama i Open WebUI
│
├── data/raw/                          # Baza surowych nagrań laboratoryjnych (Ground Truth)
│   ├── northwind/                     # 82 pliki MP3 z nazwami laboratoryjnymi
│   ├── additional/                    # 13 nowych nagrań laboratoryjnych uszkodzeń HDD
│   └── aesonlabs/                     # 9 nagrań uszkodzeń Aesonlabs Data Recovery
│
├── dataset_categorized/               # 134 nagrania podzielone na 4 klasy
├── dataset_encoded/                   # 1598 tensorów .npy (train/val/test)
├── training_output/                   # Produkcyjny model, wagi i raporty ewaluacji
├── exported_models/                   # Wyeksportowane modele produkcyjne ONNX i Kotlin
│
├── android/                           # Kompletny moduł natywnej aplikacji mobilnej (Android Studio / Gradle / Kotlin)
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
└── docx/                              # Dokumentacja specyfikacji i decyzji (ADR-001, ADR-002)
```

---

## 🤖 Współpraca z Antigravity (AGY)
Projekt jest w pełni przystosowany do pracy w trybie pair programming z agentami **Antigravity CLI (AGY)**:
- Szczegółowe niezmienniki i wytyczne dla agenta znajdują się w [AGENTS.md](AGENTS.md).
- Reguły automatyczne zdefiniowano w [.agents/rules/adr-001-audio-pipeline.md](.agents/rules/adr-001-audio-pipeline.md).
- Decyzje architektoniczne opisano w [docx/adr/adr.md](docx/adr/adr.md) (ADR-001) oraz [docx/adr/adr-002-mobile-architecture.md](docx/adr/adr-002-mobile-architecture.md) (ADR-002).

---

## 📄 Licencja
Projekt rozwijany w celach badawczych i diagnostycznych.
