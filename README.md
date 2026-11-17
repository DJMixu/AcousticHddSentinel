# Acoustic HDD Sentinel 🛡️💽
> **System wibroakustycznej detekcji i wieloklasowej klasyfikacji uszkodzeń mechanicznych dysków twardych (HDD) przy użyciu głębokich sieci konwolucyjnych (2D CNN w PyTorch), fizycznych strażników DSP oraz aplikacji mobilnej Android (Edge AI).**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Android 8.0+](https://img.shields.io/badge/Android-Java%2017%20%7C%20API%2026%2B-green.svg)](https://developer.android.com/)
[![ONNX Runtime Mobile](https://img.shields.io/badge/ONNX%20Runtime-INT8%20(1.17%20MB)-orange.svg)](https://onnxruntime.ai/)
[![Status](https://img.shields.io/badge/Status-F1%20%3E%200.70%20(Verified)-brightgreen.svg)]()
[![Hardware Acceleration](https://img.shields.io/badge/Acceleration-AMD%20ROCm%20HIP%20(RX%209060%20XT)-purple.svg)]()

---

## 📖 Spis treści
1. [O projekcie](#-o-projekcie)
2. [📈 Ewolucja Projektu i Kamienie Milowe (Progress & Evolution)](#-ewolucja-projektu-i-kamienie-milowe-progress--evolution)
3. [Architektura przetwarzania audio (ADR-001)](#-architektura-przetwarzania-audio-adr-001)
4. [Architektura sieci 2D CNN i mechanizmy obronne](#-architektura-sieci-2d-cnn-i-mechanizmy-obronne)
5. [Wyniki treningu i ewaluacja testowa](#-wyniki-treningu-i-ewaluacja-testowa)
6. [Aplikacja Mobilna Android (Edge AI & Zero GC Churn)](#-aplikacja-mobilna-android-edge-ai--zero-gc-churn)
7. [Eksport modeli produkcyjnych (ONNX / Mobile / Android)](#-eksport-modeli-produkcyjnych-onnx--mobile--android)
8. [Instalacja i szybki start](#-instalacja-i-szybki-start)
9. [Uporządkowana struktura repozytorium](#-uporządkowana-struktura-repozytorium)
10. [Współpraca z Antigravity (AGY)](#-współpraca-z-antigravity-agy)

---

## 🎯 O projekcie
**Acoustic HDD Sentinel** analizuje krótkie, 3-sekundowe wycinki wibroakustyczne generowane przez pracujące dyski twarde (HDD 2.5" oraz 3.5"), wykrywając anomalie mechaniczne i klasyfikując ich typ przed katastrofalną utratą danych:
- **`normal`** – prawidłowa, laminarna praca dysku (łożyska hydrodynamiczne FDB, stabilne obroty wrzeciona 5400/7200 RPM).
- **`clicking`** – uderzanie głowicy o ogranicznik (*Click of Death*, uszkodzenie pozycjonera / zespołu cewki VCM, błędy ścieżek servo).
- **`beeping`** – pisk stojana zatartego silnika wrzeciona BLDC (*seized spindle motor*, *stiction* głowic na talerzu).
- **`scratching`** – szerokopasmowe mechaniczne tarcie głowicy po magnetycznej powierzchni talerza (*head crash / platter scoring*).

System łączy **uczenie głębokie (2D CNN)** z **deterministycznym cyfrowym przetwarzaniem sygnałów (DSP)**, weryfikując fizyczne prążki obrotowe silnika metodą Welcha oraz tłumiąc fałszywe alarmy z otoczenia za pomocą wbudowanej bramki **Open-Set OOD**.

---

## 📈 Ewolucja Projektu i Kamienie Milowe (Progress & Evolution)

Projekt przeszedł gruntowną ewolucję inżynierską — od wstępnego prototypu obciążonego błędami danych i zniekształceniami numerycznymi, aż po zoptymalizowany, odporny na zakłócenia system brzegowy z modelem INT8 o rozmiarze 1.17 MB.

### Tabela porównawcza postępów (Progress Baseline vs Obecny Stan):

| Cecha / Metryka | Faza 1: Wstępny Prototyp (TensorFlow / Baseline) | Faza 3: Przełom Fine-Tuningu (ADR-001 & PyTorch) | Faza 5: Stan Obecny (Produkcja, OOD & Mobile) |
| :--- | :---: | :---: | :---: |
| **Główny silnik ML** | TensorFlow / Keras | PyTorch 2.x (AMD ROCm 7.2) | PyTorch 2.x + ONNX Runtime Mobile |
| **Baza próbek audio** | ~82 zanieczyszczone pliki MP3 | 134 zweryfikowane nagrania | **288 autentycznych nagrań** (Rejestr MD5) |
| **Tensory `.npy`** | ~800 tensorów (wyciek danych) | 1 598 tensorów (Zero Data Leakage) | **5 187 tensorów** (Train: 4633, Val: 196, Test: 358) |
| **Współczynnik F1: `scratching`** | **0.00** (100% mylone z clicking) | **0.7368** (Przełom $\ge 0.70$) | **0.7368** (+ Smart Clipping Bypass) |
| **Współczynnik F1: `beeping`** | **0.20** (Zatrucie etykiet Northwind) | **0.8636** (Weryfikacja widmowa) | **0.8636** (+ Blokada rotacji RPM) |
| **Współczynnik F1: `clicking`** | 0.62 | **0.7848** | **0.7848** |
| **Współczynnik F1: `normal`** | 0.81 (Shortcut learning < 2.4 kHz) | **1.0000** (Pełna parzystość) | **1.0000** (100% Precyzja i Czułość) |
| **Średnia Makro (Macro F1)** | ~0.41 | **84.63%** | **84.63%** (Warunek $F1 \ge 0.70$ spełniony) |
| **Luka generalizacyjna** | > 18.0 p.p. (Przetrenowanie) | **2.04 p.p.** (Idealna zbieżność) | **Minimalna** (Zabezpieczona OOD) |
| **Bramka Open-Set OOD** | Brak (Fałszywe alarmy na mowie) | Brak | **Wbudowane prototypy widmowe w grafie** |
| **Odporność na szum miasta** | Niebadana (Częste alarmy) | Niebadana | **0 / 5 fałszywych alarmów (P $\le 5.9\%$)** |
| **Rozmiar modelu** | ~15.2 MB | 4.61 MB (FP32) | **1.17 MB (Skwantyzowany ONNX INT8)** |
| **Czas epoki treningu** | ~16.0 s (CPU) | **~1.9 s (AMD RX 9060 XT ROCm)** | **~1.9 s (Akceleracja ROCm)** |
| **Wdrożenie docelowe** | Wyłącznie skrypty Python | Eksport ONNX | **Aplikacja Android (Java 17, Zero GC Churn)** |

---

### Kluczowe etapy rozwoju (Timeline):

1. **Faza 1 (Wstępny Prototyp i Identyfikacja Problemów):**
   * Pierwsza implementacja w TensorFlow/Keras.
   * *Acoustic Forensics:* Zidentyfikowano krytyczne wady: silny DC offset ($+0.3766$), artefakty zapętlania `np.tile` generujące sztuczne stukanie, utratę oryginalnych nazw plików w bazie Northwind oraz zafałszowanie etykiety `Audio_Player_57.wav` (duplikat klikającego Samsunga SP8002 oznaczony jako beeping).
2. **Faza 2 (Migracja na PyTorch & Akceleracja ROCm):**
   * Przepisanie architektury na czysty PyTorch (`AcousticHddSentinelCNN`, 1.21 mln wag w 60 tensorach).
   * Konfiguracja środowiska AMD ROCm HIP na karcie Radeon RX 9060 XT (16 GB VRAM) — ponad **8-krotne przyspieszenie treningu** (z 16 s na 1.9 s na epokę).
3. **Faza 3 (Przełom Optymalizacyjny – Standard ADR-001):**
   * Sformułowanie reguły *Zero Data Leakage* (podział na poziomie całych nagrań przed wycinaniem ramek).
   * Odzyskanie 82 oryginalnych nagrań Northwind i pozyskanie 11 czystych próbek awarii laboratoryjnych.
   * Wdrożenie optymalizatora `AdamW` (`weight_decay=1e-4`), wygładzania etykiet `LabelSmoothingCrossEntropy(0.03)` oraz wag balansujących klasy.
   * **Wynik:** Wszystkie klasy awarii przekroczyły rygorystyczny próg $F1 \ge 0.70$ (**Macro F1 = 84.63%**).
4. **Faza 4 (Fizyczny Strażnik DSP & Odporność Adwersarialna):**
   * Opracowanie modułu `DspPhysicalGuard` i `RpmHarmonicsAnalyzer` śledzącego harmoniczne obrotowe silnika BLDC (5400 / 7200 RPM) metodą Welcha.
   * Wbudowanie bramki odrzucania próbek spoza rozkładu (**Open-Set OOD Rejection**) bezpośrednio w graf wag sieci i eksport ONNX.
   * Zabezpieczenie przed przesterowaniem tanich mikrofonów MEMS (*Smart Clipping Bypass*) i szumem wentylatorów PC.
5. **Faza 5 (Pełne Wdrożenie Mobilne – Stan Obecny):**
   * Implementacja natywnej aplikacji Android w 100% w czystej Javie 17 (architektura MVVM, interfejs w stylu Shazam, silnik bazy SQLite).
   * Optymalizacja potoku DSP do standardu **Zero GC Churn** (eliminacja alokacji pamięci na stercie podczas ciągłego nasłuchu, pojedynczy przebieg Radix-2 FFT).
   * Kwantyzacja wag do **ONNX INT8 (1.17 MB)** z zachowaniem parzystości numerycznej $< 10^{-7}$ względem PyTorch.

---

## 🔬 Architektura przetwarzania audio (ADR-001)
Zgodnie z decyzją architektoniczną **ADR-001** (`docx/adr/adr.md`), każde 3-sekundowe okno audio ($22050\text{ Hz} \to 66150$ próbek) jest transformowane do **3-kanałowego tensora `(128, 259, 3)`**:
1. **Kanał 0:** Log-Mel Spektrogram (128 pasm Mel, skala decybelowa) – harmoniczne obrotowe, rezonanse i piski.
2. **Kanał 1:** Delta – 1. pochodna czasowa Savitzky'ego-Golaya – detekcja transjentów i uderzeń głowic.
3. **Kanał 2:** Delta-Delta – 2. pochodna czasowa – dynamika przyspieszeń i mikrowstrząsów.

```
Sygnał Audio (3.0s, 22050 Hz)
   │
   ├── Centrowanie DC Offset & Peak Normalization [-1.0, 1.0]
   ├── STFT (N_FFT=1024, Hop=256, Okno Hanna)
   ├── Trójkątny Bank Filtrów Mel (128 pasm, 50-10000 Hz)
   ├── Konwersja Log-Power (Decybele)
   ├── Obliczenie pochodnych Delta i Delta-Delta
   └── Standaryzacja Z-Score (niezależnie per kanał, min_std=1e-3)
   │
   ▼
Tensor Wejściowy: (128, 259, 3) / (B, 3, 128, 259)
```

---

## 🧠 Architektura sieci 2D CNN i mechanizmy obronne
Model zaimplementowany w [ml/model_cnn.py](file:///home/DJMixu/Pulpit/AcousticHddSentinel/ml/model_cnn.py) zawiera łącznie **60 tensorów w pamięci** (29 tensorów wag i obciążeń + 31 tensorów buforowych OOD i BatchNorm):
- **4 Podwójne Bloki Konwolucyjne:** `Conv2D` ($3 \times 3$, filtry: 32 -> 64 -> 128 -> 256), `BatchNorm2d`, `ReLU`, `MaxPool2d(2, 2)`, `Dropout2d(0.2)`.
- **Global Average Pooling:** `AdaptiveAvgPool2d((1, 1))` redukujący przestrzeń cech do wektora 256 bez eksplozji parametrów (całkowita liczba wag: **1 206 756**).
- **Ekstrakcja Cech i Klasyfikator:** `Linear(256, 128)` + `BatchNorm1d` + `Dropout(0.35)` + `Linear(128, 4)`.
- **Bramka Open-Set OOD (Wbudowana w model):** Rejestrowane bufory `fault_prototypes` oraz `fault_thresholds`. Jeżeli wektor cech leży poza promieniem prototypów uszkodzeń, logity awarii są dynamicznie tłumione funkcją sigmoid w dół, uniemożliwiając wzbudzenie alarmu przez dźwięki otoczenia.
- **Komitet Modeli (Seed Ensembling):** Klasa `FusedEnsemblePyTorch` łącząca wagi komitetu modeli trenowanych na różnych ziarnach (42, 123, 777) za pomocą ważonego uśredniania Softmax.

---

## 📊 Wyniki treningu i ewaluacja testowa

Ewaluacja produkcyjnego modelu na niezależnym, w 100% czystym zbiorze testowym (`test` – 358 tensorów z niewidzianych wcześniej dysków, 0% overlapu, brak sztucznej augmentacji):

| Klasa awarii | Precyzja (Precision) | Czułość (Recall) | F1-Score | Status rygoru ($F1 \ge 0.70$) |
| :--- | :---: | :---: | :---: | :---: |
| **`normal` (sprawny dysk)** | **100.0%** | **100.0%** | **1.0000** | **SPEŁNIONY (Doskonała izolacja)** |
| **`beeping` (zatarte wrzeciono)** | **86.4%** | **86.4%** | **0.8636** | **SPEŁNIONY** |
| **`clicking` (stukanie głowic)** | **70.5%** | **88.6%** | **0.7848** | **SPEŁNIONY** |
| **`scratching` (tarcie talerza)** | **87.5%** | **63.6%** | **0.7368** | **SPEŁNIONY** |
| **Średnia Makro (Macro F1)** | **86.08%** | **84.64%** | **0.8463** | **SPEŁNIONY** |
| **Dokładność ogólna (Accuracy)** | — | — | **83.76%** | **SPEŁNIONY** |

### Testy odporności i integralności (100% zdane):
- **Weryfikacja binarnego werdyktu na 320 oknach rzeczywistych dysków:** FPR = **0.00%** (żaden zdrowy dysk ani wentylator nie został zakwalifikowany jako awaria).
- **Test odporności na szum miasta ([tests/noise_experiment/](file:///home/DJMixu/Pulpit/AcousticHddSentinel/tests/noise_experiment/)):** 0 / 5 fałszywych alarmów (maksymalne prawdopodobieństwo awarii $P \le 5.9\%$, pełne bezpieczeństwo).
- **Test obciążeniowy VRAM ([tests/stress_test_vram.py](file:///home/DJMixu/Pulpit/AcousticHddSentinel/tests/stress_test_vram.py)):** Przepustowość **2149.9 chunków/s** na karcie RX 9060 XT, 0 wycieków pamięci i 0 błędów NaN/Inf.
- **Parzystość numeryczna Python vs Java:** Błąd średniokwadratowy $MSE < 10^{-12}$, identyczny kształt `(3, 128, 259)`.

---

## 📱 Aplikacja Mobilna Android (Edge AI & Zero GC Churn)

Natywny moduł mobilny w katalogu [android/](file:///home/DJMixu/Pulpit/AcousticHddSentinel/android/) został zaimplementowany w **100% w czystej Javie 17** (Android API 26+) bez ciężkich zewnętrznych frameworków:

- **Architektura MVVM:** Komunikacja przez `LiveData` i reaktywne stany interfejsu użytkownika.
- **UI w stylu Shazam:** Pulsujący radar pomiarowy, animowane fale dźwiękowe, wskaźniki pewności diagnozy w czasie rzeczywistym oraz zalecenia serwisowe (kolorystyka Material 3).
- **Mechanizm Early Stopping:** Zatrzymanie badania natychmiast po osiągnięciu stabilnego konsensusu okien lub pewności $\ge 80\%$.
- **Optymalizacja Zero GC Churn:** Wszystkie tablice robocze STFT, Radix-2 FFT oraz banku Mel są prealokowane w konstruktorze `AudioPreprocessor`. Pętla pomiarowa nie generuje śmieci na stercie, eliminując zacięcia Garbage Collectora.
- **Baza danych SQLite:** `SQLiteOpenHelper` rejestrujący historię wykonanych badań, werdykty, poziomy pewności oraz zrzuty parametrów DSP.

---

## 📦 Eksport modeli produkcyjnych (ONNX / Mobile / Android)

Gotowy model jest automatycznie eksportowany i synchronizowany do assetów aplikacji mobilnej za pomocą skryptu [ml/export_model.py](file:///home/DJMixu/Pulpit/AcousticHddSentinel/ml/export_model.py):

```bash
python ml/export_model.py --weights training_output/best_hdd_cnn_model.pt --output-dir exported_models
```

Artefakty w katalogu [exported_models/](file:///home/DJMixu/Pulpit/AcousticHddSentinel/exported_models/) oraz [android/app/src/main/assets/](file:///home/DJMixu/Pulpit/AcousticHddSentinel/android/app/src/main/assets/):
- **`best_hdd_cnn_model.onnx`** – Pełny model ONNX FP32 z dynamiczną osią batcha (4.61 MB).
- **`best_hdd_cnn_model_int8.onnx`** – Skwantyzowany model ONNX INT8 (**1.17 MB – redukcja rozmiaru o 74.6%**; czas inferencji na procesorze telefonu: 3–8 ms).
- **`best_hdd_cnn_model_torchscript.pt`** – Zserializowany model PyTorch C++ / TorchScript (4.68 MB).
- **`classes.json`** & **`model_card.json`** – Mapowanie identyfikatorów klas i metadane DSP.

---

## 💻 Instalacja i szybki start

### Wymagania wstępne:
- System Linux (zalecany) z Pythonem 3.10+
- JDK 17 i Android Studio (do kompilacji modułu mobilnego)
- Opcjonalnie: GPU z AMD ROCm 6.x/7.x lub NVIDIA CUDA

### 1. Klonowanie repozytorium i instalacja środowiska:
```bash
git clone git@github.com:DJMixu/AcousticHddSentinel.git
cd AcousticHddSentinel

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Sprawdzenie integralności projektu (6/6 testów logicznych):
```bash
python tests/test_project_logic.py
```

### 3. Weryfikacja sprawności modelu na GPU/CPU:
```bash
python tests/verify_cnn_model.py
```

### 4. Regeneracja tensorów i uruchomienie treningu ROCm:
```bash
# 1. Kategoryzacja bazy audio:
python ml/prepare_audio_dataset.py

# 2. Enkodowanie audio do tensorów ADR-001:
python ml/hdd_audio_encoder.py

# 3. Trening sieci 2D CNN z kalibracją OOD:
python ml/train_cnn.py --epochs 35 --batch-size 32 --output-dir ./training_output
```

### 5. Kompilacja i uruchomienie testów aplikacji Android:
```bash
cd android
./gradlew testDebugUnitTest
./gradlew assembleDebug
```

---

## 📁 Uporządkowana struktura repozytorium

```
AcousticHddSentinel/
├── AGENTS.md                          # Główny przewodnik i Single Source of Truth dla agentów AGY
├── README.md                          # Główna dokumentacja projektu z historią rozwoju
├── kontekst.md                        # Całościowe kompendium inżynierskie i audyty laboratoryjne
├── fineTuning.md                      # Dokumentacja optymalizacji F1 >= 0.70
├── NOTES.md                           # Notatki badawcze (Seed Ensembling)
├── requirements.txt                   # Zależności biblioteczne Pythona (PyTorch, Librosa, ONNX)
│
├── ml/                                # 🧠 Pipeline uczenia maszynowego i przetwarzania sygnałów
│   ├── model_cnn.py                   # Architektura 2D CNN z buforami bramki Open-Set OOD
│   ├── dataset_pytorch.py             # PyTorch Dataset z preloadem RAM i wagami klas
│   ├── hdd_audio_encoder.py           # Enkoder audio do tensorów .npy zgodny z ADR-001
│   ├── prepare_audio_dataset.py       # Kategoryzator bazy audio i generator klasy 'normal'
│   ├── train_cnn.py                   # Kompletny pipeline treningowy w PyTorch (AdamW, ROCm)
│   ├── train_ensemble.py              # Pipeline Seed Ensembling (komitet modeli)
│   ├── export_model.py                # Eksport modeli ONNX FP32/INT8 i auto-sync z Androidem
│   ├── dsp_adversarial_guard.py       # Strażnik adwersarialny (detekcja wentylatorów, clippingu, VAD)
│   └── rpm_harmonics_analyzer.py      # Estymator harmonicznych wirowania wrzeciona (5400/7200 RPM)
│
├── tests/                             # 🧪 Zestaw testów logicznych, fizycznych i odpornościowych
│   ├── test_project_logic.py          # Weryfikacja integralności projektu i niezmienników ADR-001 (6/6)
│   ├── verify_cnn_model.py            # 6-etapowa weryfikacja wag modelu i bramki OOD
│   ├── verify_binary_verdict_and_real_drives.py # Test 320 okien rzeczywistych dysków i wentylatorów
│   ├── verify_dsp_parity_python_java.py # Weryfikacja parzystości numerycznej DSP (Java vs Python)
│   ├── profile_normalization_quality.py # Profiler jakości normalizacji Z-Score
│   ├── stress_test.py                 # Ciągły test odporności potoku DSP + CNN
│   ├── stress_test_vram.py            # Test obciążeniowy VRAM karty graficznej (2149 chunków/s)
│   └── noise_experiment/              # Laboratorium odporności na dźwięki otoczenia i szum miejski
│
├── scripts/                           # 🛠️ Skrypty narzędziowe i orkiestracja lokalnych usług LLM
│   ├── install_apk.sh                 # Automatyczna instalacja APK i uruchomienie na telefonie Android
│   ├── record_mic.sh                  # Skrypt rejestracji dźwięku z fizycznego mikrofonu PCI ALC1220
│   ├── audio_hardware_config.json     # Konfiguracja fizycznych źródeł audio w systemie Linux
│   └── local_llm_bridge.py            # Most orkiestrujący lokalne modele LLM (Qwen 3.8 + Coder 14B)
│
├── android/                           # 📱 Natywna aplikacja mobilna (Android Studio / Java 17 / Shazam UI)
│   ├── app/src/main/assets/           # Skwantyzowany model ONNX INT8, classes.json, model_card.json
│   ├── app/src/main/java/             # Pakiety: audio, data (SQLite), dsp (Zero GC Churn), ml, ui (MVVM)
│   ├── app/src/main/res/              # Zasoby UI (układ radarowy, motywy Material 3, ikony)
│   ├── app/src/test/java/             # Zestaw 22 testów jednostkowych potoku DSP i logiki
│   └── build.gradle.kts               # Konfiguracja projektu Gradle 8.4
│
├── data/raw/                          # Baza surowych nagrań audio (288 plików, rejestr MD5)
├── dataset_encoded/                   # 5 187 tensorów .npy (Train, Validate, Test – Zero Data Leakage)
├── training_output/                   # Produkcyjne wagi PyTorch, macierze pomyłek i raporty ewaluacji
├── exported_models/                   # Wyeksportowane modele ONNX FP32 / INT8 i TorchScript
└── docx/                              # Specyfikacje i decyzje architektoniczne (ADR-001, ADR-002)
```

---

## 🤖 Współpraca z Antigravity (AGY)
Projekt jest w pełni przystosowany do pracy w trybie pair programming z agentami **Antigravity CLI (AGY)**:
- Niezmienniki techniczne oraz reguły architektoniczne opisano w dokumencie [AGENTS.md](AGENTS.md).
- Decyzje architektoniczne utrwalono w [docx/adr/adr.md](docx/adr/adr.md) (ADR-001: DSP & PyTorch) oraz [docx/adr/adr-002-mobile-architecture.md](docx/adr/adr-002-mobile-architecture.md) (ADR-002: Android & ONNX Mobile).
- Zautomatyzowany most LLM [scripts/local_llm_bridge.py](scripts/local_llm_bridge.py) wspiera lokalną architekturę modeli open-weights (Qwen 3.8 27B + Qwen 2.5 Coder 14B) z zachowaniem pełnego okna kontekstu 128k i zwalnianiem pamięci VRAM przed treningiem.

---

## 📄 Licencja
Projekt rozwijany w celach badawczych, inżynierskich i diagnostycznych. System przeznaczony do wczesnego ostrzegania przed awariami mechanicznymi nośników magnetycznych.
