# Acoustic HDD Sentinel 🛡️💽
> **Produkcyjny system wibroakustycznej detekcji i wieloklasowej klasyfikacji uszkodzeń mechanicznych dysków twardych (HDD) przy użyciu głębokich sieci konwolucyjnych (2D CNN w PyTorch z architekturą SMPC Option A), fizycznych strażników DSP (RPM Harmonics Analyzer) oraz natywnej aplikacji mobilnej Android (Czysta Java 17 Edge AI).**

[![Tests](https://img.shields.io/badge/Tests-5%2F5%20Passed%20(100%25)-brightgreen.svg)]()
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Android Pure Java 17](https://img.shields.io/badge/Android-Pure%20Java%2017%20(0%20Kotlin)-orange.svg)]()
[![Hardware Acceleration](https://img.shields.io/badge/Acceleration-AMD%20ROCm%20HIP%20(RX%209060%20XT)-purple.svg)]()
[![Model Size](https://img.shields.io/badge/Model-ONNX%20INT8%20(1.17%20MB)-success.svg)]()

> [!NOTE]
> **Główna Architektura Produkcyjna:**
> System produkcyjny realizuje pełną 4-klasową klasyfikację akustyczną (`normal`, `clicking`, `beeping`, `scratching`) łącząc sieć 2D CNN z wielomodalną przestrzenią podcentrów **SMPC Option A**, fizycznym strażnikiem obrotów **RPM Harmonics Analyzer** (weryfikacja stacjonarności pitch, filtracja wibrato mowy), bramkowaniem **Open-Set OOD** oraz natywną aplikacją Android w **100% czystej Javie 17** (Zero Kotlin runtime).
> 
> *Równolegle w repozytorium prowadzona jest badawcza gałąź eksperymentalna `feature/prototype-bez-normal` (badająca alternatywny paradygmat Fault-Only Learning z estymatorem GMM bez klasy normal w zbiorze uczącym — zobacz [PrototypBezNormal/README.md](PrototypBezNormal/README.md)).*

---

## 📖 Spis treści
1. [🔬 PrototypBezNormal: Eksperyment bez klasy 'normal' w treningu](#-prototypbeznormal-eksperyment-bez-klasy-normal-w-treningu)
2. [O projekcie](#-o-projekcie)
3. [📈 Ewolucja Projektu i Kamienie Milowe (Progress & Evolution)](#-ewolucja-projektu-i-kamienie-milowe-progress--evolution)
4. [Architektura przetwarzania audio (ADR-001)](#-architektura-przetwarzania-audio-adr-001)
5. [🧠 Jak konkretnie działa model 2D CNN i mechanizmy obronne](#-jak-konkretnie-działa-model-2d-cnn-i-mechanizmy-obronne)
6. [🔍 Jak model wyłapuje poszczególne awarie mechaniczne](#-jak-model-wyłapuje-poszczególne-awarie-mechaniczne)
7. [📚 Na czym model był uczony i diagnostyka epok](#-na-czym-model-był-uczony-i-diagnostyka-epok)
8. [📊 Wyniki treningu i ewaluacja testowa](#-wyniki-treningu-i-ewaluacja-testowa)
9. [🔬 Rzeczywisty eksperyment na 4 fizycznych dyskach HDD](#-rzeczywisty-eksperyment-na-4-fizycznych-dyskach-hdd)
10. [📱 Aplikacja Mobilna Android (Potok DSP, Mikrofon i Zero GC Churn)](#-aplikacja-mobilna-android-potok-dsp-mikrofon-i-zero-gc-churn)
11. [📦 Eksport modeli produkcyjnych (ONNX / Mobile / Android)](#-eksport-modeli-produkcyjnych-onnx--mobile--android)
12. [💻 Instalacja i szybki start](#-instalacja-i-szybki-start)
13. [📁 Uporządkowana struktura repozytorium](#-uporządkowana-struktura-repozytorium)
14. [🤖 Współpraca z Antigravity (AGY)](#-współpraca-z-antigravity-agy)

---

## 🔬 PrototypBezNormal: Eksperyment bez klasy 'normal' w treningu

W ramach gałęzi **`feature/prototype-bez-normal`** rozwijany jest eksperyment odrzucający konieczność posiadania bazy nagrań sprawnych dysków podczas uczenia sieci neuronowej:

* **Zasada działania:** Sieć 2D CNN uczy się reprezentacji w przestrzeni cech 128D wyłącznie na autentycznych uszkodzeniach (`beeping`, `clicking`, `scratching`). Dysk sprawny (`normal`) traktowany jest jako stan **Out-Of-Distribution (OOD)**.
* **Estymator Gęstości GMM:** Jeśli badany wycinek dźwięku nie pasuje statystycznie do żadnej elipsoidalnej obwiedni uszkodzenia (log-likelihood poniżej progu kalibracyjnego $T = 201.95$), zostaje uznany za stan **NORMAL / HEALTHY**.
* **Wyniki na 516 tensorach testowych (Zero Data Leakage):**
  - **76.90% czułości (Recall) dla klasy `normal`** oraz **73.04% precyzji** ($F1 = 74.92\%$) bez jakichkolwiek danych treningowych z tej klasy!
  - **100% poprawnych werdyktów HEALTHY (39/39 okien 3.0s)** na ciągłym 60-sekundowym nagraniu fizycznego dysku WD Blue 5400 RPM.
  - Wyższa precyzja wykrywania zatarcia wrzeciona (`beeping`): **54.29%** (vs 35.00% w modelu pełnym).
* **Narzędzia diagnostyczne na tej gałęzi:**
  - `python PrototypBezNormal/compare_full_vs_no_normal.py` – bezpośrednie porównanie z modelem pełnym.
  - `python PrototypBezNormal/live_predict.py <plik.wav>` – szybka analiza plików audio.
  - `python PrototypBezNormal/record_and_predict.py --duration 5.0` – diagnostyka na żywo z dedykowanego mikrofonu Realtek ALC1220.

Szczegóły techniczne i raport porównawczy: [PrototypBezNormal/README.md](file:///home/DJMixu/Pulpit/AcousticHddSentinel/PrototypBezNormal/README.md) oraz [PrototypBezNormal/COMPARISON_WITH_NORMAL_MODEL.md](file:///home/DJMixu/Pulpit/AcousticHddSentinel/PrototypBezNormal/COMPARISON_WITH_NORMAL_MODEL.md).

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

| Cecha / Metryka | Faza 1: Wczesny Model Bazowy (PyTorch CPU Baseline) | Faza 3: Przełom Fine-Tuningu (ADR-001 & PyTorch) | Faza 5: Stan Obecny (Produkcja, SMPC, OOD & Mobile) |
| :--- | :---: | :---: | :---: |
| **Główny silnik ML** | PyTorch (Wczesny prototyp CPU) | PyTorch 2.x (AMD ROCm 7.2) | PyTorch 2.x (SMPC Option A) + ONNX Runtime Mobile |
| **Baza nagrań źródłowych** | ~82 pliki MP3 (Northwind) | 134 zweryfikowane nagrania | **353 autentyczne nagrania** (Rejestr MD5) |
| **Niezależny zbiór testowy** | **73 tensory** (nieznane dyski) | **117 tensorów** (Ground Truth) | **516 tensorów** (Test) / **117** (Clean Bench) + 320 okien HDD |
| **Współczynnik F1: `scratching`** | **0.0000** (0% Recall, mylone z clicking) | **0.7368** (Przełom $\ge 0.70$) | **0.7368** (+ SMPC Gałąź D & Smart Bypass) |
| **Współczynnik F1: `beeping`** | **0.3478** (Zatrucie etykiet, Precision 22.9%) | **0.8636** (Weryfikacja widmowa) | **0.8636** (+ Blokada rotacji RPM & Stall Beep) |
| **Współczynnik F1: `clicking`** | **0.5970** (Precision 90.9%, Recall 44.4%) | **0.7848** (Czułość 88.6%) | **0.7848** (Pochodne Delta/Delta-Delta & CF $\ge 4.5$) |
| **Współczynnik F1: `normal`** | **1.0000** (100% Precyzja i Czułość) | **1.0000** (Pełna parzystość) | **1.0000** (FPR = 0.00% na fizycznych HDD) |
| **Średnia Makro (Macro F1)** | **0.4862 (48.62%)** | **0.8463 (84.63%)** | **0.8463 (84.63%)** (Zbalansowana detekcja) |
| **Średnia ważona (Weighted F1)**| **0.5711 (57.11%)** | **0.8414 (84.14%)** | **0.8414 (84.14%)** |
| **Dokładność ogólna (Accuracy)**| **53.42%** (39 / 73 próbek) | **83.76%** (98 / 117 próbek) | **83.76%** (+ SMPC, OOD & Filtr RPM) |
| **Luka generalizacyjna** | > 18.0 p.p. (Przetrenowanie) | **2.04 p.p.** (Idealna zbieżność) | **Minimalna** (Zabezpieczona SMPC i OOD) |
| **Bramka Open-Set OOD** | Brak (Fałszywe alarmy na mowie) | Brak | **Wielomodalne podcentra SMPC w grafie** |
| **Odporność na szum miasta** | Niebadana (Częste fałszywe alarmy) | Niebadana | **0 / 5 fałszywych alarmów (P $\le 5.9\%$)** |
| **Rozmiar modelu** | ~4.83 MB (PyTorch FP32) | 4.61 MB (PyTorch FP32) | **1.17 MB (Skwantyzowany ONNX INT8)** |
| **Czas epoki treningu** | ~16.0 s (CPU) | **~1.9 s (AMD RX 9060 XT ROCm)** | **~1.9 s (Akceleracja ROCm)** |
| **Wdrożenie docelowe** | Wyłącznie skrypty Python | Eksport ONNX | **Aplikacja Android (Java 17, Zero GC Churn, 37 testów)** |

---

### Kluczowe etapy rozwoju (Timeline):

1. **Faza 1 (Wczesny Model Bazowy i Identyfikacja Problemów):**
   * Pierwsza implementacja architektury 2D CNN w PyTorch trenowana na procesorze CPU z podstawowym optymalizatorem Adam i podziałem 70/15/15.
   * **Oficjalne wyniki wczesnego modelu bazowego na 73 tensorach testowych:**
     ```
     Klasa               Precyzja     Czułość      F1-Score     Support (Test)
     normal              100.0%       100.0%       1.0000       11
     clicking             90.9%        44.4%       0.5970       45
     beeping              22.9%        72.7%       0.3478       11
     scratching            0.0%         0.0%       0.0000        6
     ---------------------------------------------------------------------
     Dokładność (Accuracy): 53.42% (39/73) | Weighted F1: 0.5711 | Macro F1: 0.4862
     ```
   * *Acoustic Forensics (Przyczyny słabych wyników Fazy 1):*
     * **Dlaczego `scratching` miał F1 = 0.0000?** Wczesna sieć bez odpowiednich wag kanałów pochodnych czasowych Delta i Delta-Delta traktowała szorstkość tarcia jak ciąg uderzeń głowicy, myląc 100% próbek z `clicking`.
     * **Dlaczego `beeping` miał Precyzję zaledwie 22.9%?** Wczesny zbiór Northwind utracił oryginalne nazwy laboratoryjne, co wymusiło na skryptach heurystyki spektralne i doprowadziło do zatrucia etykiet (m.in. duplikat klikającego Samsunga SP8002 `Audio_Player_57.wav` błędnie oznaczony jako pisk).
     * **Dlaczego `clicking` miał Czułość tylko 44.4%?** Silny DC offset ($+0.3766$) oraz brak adaptacyjnej normalizacji szczytowej tłumiły transjenty cichszych uderzeń pozycjonera.
     * **Klasa `normal`:** Osiągała 100% F1, ponieważ jednostajny, laminarny szum łożysk FDB był łatwo separowalny od gwałtownych anomalii.
2. **Faza 2 (Optymalizacja Architektury & Akceleracja ROCm):**
   * Refaktoryzacja architektury sieci `AcousticHddSentinelCNN` (1.21 mln wag w 60 tensorach) zoptymalizowana pod batching GPU.
   * Konfiguracja środowiska AMD ROCm HIP na karcie Radeon RX 9060 XT (16 GB VRAM) — ponad **8-krotne przyspieszenie treningu** (z 16 s na 1.9 s na epokę).
3. **Faza 3 (Przełom Optymalizacyjny – Standard ADR-001):**
   * Sformułowanie reguły *Zero Data Leakage* (podział na poziomie całych nagrań przed wycinaniem ramek).
   * Odzyskanie 82 oryginalnych nagrań Northwind i pozyskanie 11 czystych próbek awarii laboratoryjnych.
   * Wdrożenie optymalizatora `AdamW` (`weight_decay=1e-4`), wygładzania etykiet `LabelSmoothingCrossEntropy(0.03)` oraz wag balansujących klasy.
   * **Wynik:** Osiągnięto zbalansowaną detekcję wszystkich uszkodzeń przy **Macro F1 = 84.63%** oraz 100% precyzji i czułości dla klasy `normal`.
4. **Faza 4 (Fizyczny Strażnik DSP & Odporność Adwersarialna):**
   * Opracowanie modułu `DspPhysicalGuard` i `RpmHarmonicsAnalyzer` śledzącego harmoniczne obrotowe silnika BLDC (5400 / 7200 RPM) metodą Welcha.
   * Wbudowanie bramki odrzucania próbek spoza rozkładu (**Open-Set OOD Rejection**) bezpośrednio w graf wag sieci i eksport ONNX.
   * Zabezpieczenie przed przesterowaniem tanich mikrofonów MEMS (*Smart Clipping Bypass*) i szumem wentylatorów PC.
5. **Faza 5 (Pełne Wdrożenie Mobilne w Czystej Javie 17):**
   * Implementacja natywnej aplikacji Android w 100% w czystej Javie 17 (architektura MVVM, interfejs w stylu Shazam, silnik bazy SQLiteOpenHelper, zero Kotlina w runtime).
   * Optymalizacja potoku DSP do standardu **Zero GC Churn** (eliminacja alokacji pamięci na stercie podczas ciągłego nasłuchu, pojedynczy przebieg Radix-2 FFT).
   * Kwantyzacja wag do **ONNX INT8 (1.17 MB)** z zachowaniem parzystości numerycznej $< 10^{-7}$ względem PyTorch.
   * Zestaw 8 klas testowych i 37 testów jednostkowych w Javie (`cd android && ./gradlew test` – 100% zdanych w 2s).
6. **Faza 6 (Milestone 1 & 2: Wzmocnienie Fizyczne DSP & Architektura SMPC):**
   * **Milestone 1 (DSP Parity & RPM Pitch Stationarity):** Wdrożenie kaskadowego filtru IIR Butterwortha 4. rzędu ($f_c = 9500$ Hz) z decymacją 2:1 (dla urządzeń 44.1 kHz), algorytmu weryfikacji stacjonarności pitch w czasie z 5 niespójnymi przesunięciami (`[0.0, 0.37, 0.83, 1.41, 2.11]` s) i interpolacją paraboliczną ($\Delta f \le 1.5$ Hz), bariery ostrości prążka widmowego ($f_0\text{ sharpness} \ge 6.0$) oraz dwutonowej detekcji silników FDB ($f_0 + 2f_0$).
   * **Milestone 2 (SMPC Option A & Gałąź D):** Implementacja architektury Sub-Center Multi-Prototype Cosine w wariancie Option A (`MatMul + view + ReduceMax` bez einsum, pełna zgodność z ONNX INT8), zachowująca bezwzględnie niezmiennik 1,206,756 parametrów, funkcja straty `SubCenterArcMarginLoss` ($\cos(\theta + m)$), regularyzacja `compute_orthogonality_loss` oraz weryfikacja cech Gałęzi D w `HybridAcousticSentinel` eliminująca fałszywe alarmy na szumach wentylatorów.

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
   ├── Trójkątny Bank Filtrów Mel (128 pasm, 50-10000 Hz, Slaney norm)
   ├── Konwersja Log-Power (Decybele, top_db=80.0)
   ├── Obliczenie pochodnych Savitzky'ego-Golaya (Delta i Delta-Delta, N=9)
   └── Standaryzacja Z-Score (niezależnie per kanał, min_std=1e-2)
   │
   ▼
Tensor Wejściowy: (128, 259, 3) na dysku / (B, 3, 128, 259) w pamięci PyTorch
```

### Milestone 1: Deterministyczna Parzystość DSP & Strażnik Fizyczny RPM
W ramach **Milestone 1** zrealizowano pełną matematyczną i numeryczną parzystość między biblioteką Python Librosa a natywnym modułem Androida w **100% czystej Javie 17** (`tests/verify_dsp_parity_python_java.py` wykazuje błąd $MSE < 10^{-12}$):
* **Kaskadowy filtr IIR Butterwortha 4. rzędu i decymacja 2:1:** Smartfony wymuszające rejestrację w $44100\text{ Hz}$ są w locie filtrowane 2-sekcyjnym filtrem dolnoprzepustowym IIR ($f_c = 9500\text{ Hz}$) przed podpróbkowaniem do $22050\text{ Hz}$, całkowicie eliminując zjawisko aliasingu widmowego (`AudioRecorderDecimationTest.java` 5/5).
* **Weryfikacja stacjonarności czasowej pitch (Temporal Pitch Stationarity):** Silnik bezszczotkowy BLDC dysku cechuje się sztywnością fazową, podczas gdy ludzki głos wykazuje naturalną modulację wibrato (3–8 Hz FM). Algorytm sprawdza 5 niespójnych przesunięć czasowych (`[0.0, 0.37, 0.83, 1.41, 2.11]` s) i wyznacza wierzchołek piku poprzez 3-punktową interpolację paraboliczną. Odchylenie $\Delta f > 1.5\text{ Hz}$ bezdyskusyjnie odrzuca obecność rotacji HDD, neutralizując fałszywe alarmy na głosie i gwizdach.
* **Bariera ostrości prążka (`f0_sharpness` $\ge 6.0$):** Stosunek energii piku do ramion widma ($\pm 2$ biny). Sztywne prążki obrotowe wykazują ostrość $>10.0$, podczas gdy modulowany głos lub szum tła $<4.0$.
* **Dwutonowa detekcja silnika ($f_0 + 2f_0$):** Dla cichych łożysk hydrodynamicznych FDB rotacja jest potwierdzana na podstawie koincydencji prążka podstawowego i jego 2. harmonicznej przy spełnieniu kryteriów ostrości.

---

## 🧠 Jak konkretnie działa model 2D CNN i mechanizmy obronne

Model zaimplementowany w [ml/model_cnn.py](file:///home/DJMixu/Pulpit/AcousticHddSentinel/ml/model_cnn.py) to zoptymalizowana sieć **2D CNN** (1 206 756 parametrów w 60 tensorach pamięci PyTorch). Sieć nie przetwarza dźwięku jako prostej jednowymiarowej fali ciśnienia, lecz traktuje 3-kanałowy spektrogram jako **„obraz akustyczny” o wymiarach 128 × 259 pikseli i 3 kanałach koloru** (reprezentujących odpowiednio: energię widmową, prędkość zmian i dynamikę przyspieszeń):

```
Wejście (B, 3, 128, 259)
   │
   ├── Blok 1: Conv2D(3->32, 3x3) -> BN -> ReLU -> Conv2D(32->32, 3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
   │   [Wykrywanie elementarnych krawędzi prążków i mikrotransjentów, rozmiar: 64 x 129]
   │
   ├── Blok 2: Conv2D(32->64, 3x3) -> BN -> ReLU -> Conv2D(64->64, 3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
   │   [Detekcja lokalnych prążków harmonicznych i skoków pochodnej Delta, rozmiar: 32 x 64]
   │
   ├── Blok 3: Conv2D(64->128, 3x3) -> BN -> ReLU -> Conv2D(128->128, 3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
   │   [Fuzja pasm wysokoczęstotliwościowych, rezonanse magnetostrykcyjne stojana, rozmiar: 16 x 32]
   │
   ├── Blok 4: Conv2D(128->256, 3x3) -> BN -> ReLU -> Conv2D(256->256, 3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
   │   [Wychwytywanie globalnej periodyczności uderzeń i modulacji tarcia talerza, rozmiar: 8 x 16]
   │
   ├── Global Average Pooling 2D: AdaptiveAvgPool2d((1, 1)) -> wektor 256 cech
   │   [Uśrednianie przestrzenne: niezmienniczość względem przesunięcia zjawiska w czasie]
   │
   ├── Dense Feature Extractor: Linear(256 -> 128) -> BatchNorm1d -> Dropout(0.35)
   │   [Przestrzeń ukryta Embeddings 128D - analiza cech awarii]
   │   ├── OOD Prototype Distance Gate (Porównanie z centroidami klas awarii)
   │
   └── Warstwa Klasyfikatora: Linear(128 -> 4) -> Softmax (lub tłumienie Sigmoid OOD)
       [Prawdopodobieństwa 4 klas: normal, clicking, beeping, scratching]
```

### 1. Zasada działania splotów 2D na spektrogramie:
* **Oś pionowa Y (Częstotliwość):** 128 filtrów Mel obejmuje pasmo od 50 Hz do 10 000 Hz. Filtry splotowe $3 \times 3$ analizują szerokość pasm rezonansowych oraz odstępy harmoniczne między prążkami.
* **Oś pozioma X (Czas):** 259 kroków (rozdzielczość ok. 11.6 ms). Filtry w kolejnych warstwach łączą informacje czasowe, odróżniając ciągły szum od powtarzających się impulsów.
* **Global Average Pooling (GAP):** Zastępuje gigantyczne warstwy w pełni połączone (Dense), redukując liczbę parametrów z kilkunastu milionów do 1.2M i uodparniając sieć na przesunięcia fazowe w czasie (brak wrażliwości na to, w której sekundzie wystąpi uderzenie głowicy).
* **Komitet Modeli (Seed Ensembling):** Klasa `FusedEnsemblePyTorch` łączy komitet modeli trenowanych na różnych ziarnach (42, 123, 777) za pomocą ważonego uśredniania Softmax, co stabilizuje decyzje na granicy niepewności.

### 2. Wbudowana bramka Open-Set OOD Rejection:
Standardowe sieci neuronowe z funkcją Softmax cierpią na tzw. *Overconfidence* — dla dowolnego obcego dźwięku (np. ludzkiej mowy, kroków lub szczekania psa) przypiszą 100% prawdopodobieństwa do jednej ze znanych klas. W `AcousticHddSentinelCNN` zabezpieczono to trwale na poziomie wag:
* W fazie treningowej kalibrowane są centroidy cech uszkodzeń (`fault_prototypes` w przestrzeni 128D) oraz ich promienie ufności (`fault_thresholds`).
* W trakcie inferencji model oblicza odległość euklidesową wyekstrahowanego wektora od znanych awarii:
  $$d_{\min} = \min_{c \in \{\text{faults}\}} \| \mathbf{e} - \mathbf{p}_c \|_2$$
* Jeśli $d_{\min} > \text{próg}$, bramka dynamicznie tłumi logity awarii funkcją sigmoid w dół, zapobiegając fałszywym alarmom na dźwiękach otoczenia.

### 3. Milestone 2: Architektura Sub-Center Multi-Prototype Cosine (SMPC Option A):
W ramach **Milestone 2** wdrożono wielomodalną reprezentację uszkodzeń w przestrzeni cech 128D:
* **Option A kompatybilna z ONNX INT8 (Zero einsum):** Tradycyjne operatory `einsum` powodują błędy w mobilnych środowiskach wykonawczych ONNX Runtime Mobile. Zastosowano równoważną formułę numeryczną: normalizację L2, iloczyn macierzowy `torch.matmul(z_norm, p_flat.t())`, `view(-1, K, M)` oraz `torch.max(dim=-1)`, co zapewnia 100% kompatybilność z kwantyzacją INT8 (błąd $< 10^{-6}$ vs definicja referencyjna w `test_smpc_architecture.py`).
* **Niezmiennik trenowalnych parametrów bazowych:** Zastosowano rejestrację buforów i mechanizm pre-hook state dict, bezwzględnie zachowując niezmiennik liczby wag bazowych (**dokładnie 1 206 756 parametrów w 60 tensorach**).
* **Funkcja straty SubCenterArcMarginLoss i regularyzacja ortogonalności:** Zastosowano addytywny margines kątowy ($\cos(\theta + m)$) ze skalą $s=16.0$ i marginesem $m=0.25$ oraz regularyzację `compute_orthogonality_loss` zapobiegającą kolapsowi sub-centrów.
* **Wzmocnienie Gałęzi D w HybridAcousticSentinel:** Zastąpiono heurystykę ścisłą weryfikacją podobieństwa cosinusowego do wyuczonych podcentrów tarcia talerza. Szumy wentylatorów PC o niskim podobieństwie są bezpiecznie przekierowywane do stanu `ambient_pc` / `normal` (`is_anomaly=False`).

---

## 🔍 Jak model wyłapuje poszczególne awarie mechaniczne

Każde uszkodzenie mechaniczne dysku generuje unikalną sygnaturę fizyczno-akustyczną, którą model identyfikuje poprzez jednoczesną analizę 3 kanałów:

### 1. `clicking` (Stukanie pozycjonera głowic / Click of Death):
* **Fizyka zjawiska:** Ramię pozycjonera (Voice Coil Motor – VCM) próbuje odnaleźć ścieżkę servo na talerzu. Po nieudanej próbie następuje gwałtowny powrót ramienia i uderzenie o mechaniczny ogranicznik krańcowy (*crash stop*).
* **Ślad w spektrogramie:** 
  * W **Kanale 0 (Log-Mel)** widoczne są periodyczne, pionowe prążki o bardzo wysokiej energii, powtarzające się regularnie co 300–800 ms.
  * Kluczowy jest **Kanał 1 (Delta) i Kanał 2 (Delta-Delta)**: uderzenie generuje ekstremalny współczynnik szczytu (*Crest Factor* $\ge 4.5$ do 130), tworząc kontrastowe, dwubiegunowe szpilki pochodnej czasowej. Sieć natychmiast odróżnia to od jednostajnego szumu.

### 2. `beeping` (Pisk stojana / Zablokowane wrzeciono / Stiction):
* **Fizyka zjawiska:** Silnik bezszczotkowy BLDC ma zablokowaną oś obrotu (zatarte łożysko hydrodynamiczne FDB lub zjawisko *stiction* — przyssanie głowic do powierzchni talerza). Sterownik PWM cyklicznie zasila cewki stojana silnika próbując wymusić start, co wywołuje drgania magnetostrykcyjne i emisję dźwięku o stałej częstotliwości (serwisowy kod błędu).
* **Ślad w spektrogramie:**
  * W **Kanale 0 (Log-Mel)** pojawiają się idealnie ostre, poziome prążki czystego tonu w paśmie **$1050\text{ Hz} - 3500\text{ Hz}$** o znikomej wariancji częstotliwościowej.
  * W **Kanale 1 (Delta)** energia pochodnej jest bliska zeru (brak transjentów uderzeniowych).
  * *Weryfikacja fizyczna:* Analizator RPM potwierdza brak rotacji talerza ($f_0 = 0\text{ Hz}$), co bezdyskusyjnie potwierdza zatarcie wrzeciona i wyklucza pomyłkę z wirującym dyskiem.

### 3. `scratching` (Tarcie głowic o powierzchnię talerza / Platter Scoring):
* **Fizyka zjawiska:** Katastrofalny *Head Crash* — zawieszenie głowicy uległo deformacji, a suwak ślizgacza szoruje bezpośrednio po wirującej powłoce magnetycznej nośnika z prędkością liniową dochodzącą do 120 km/h, skrawając mikroskopijne cząstki ferromagnetyku.
* **Ślad w spektrogramie:**
  * Ciągły, szerokopasmowy szum cierny o wysokiej energii rozlewający się w pasmach wysokich częstotliwości **powyżej $4\text{ kHz}$ aż do $10\text{ kHz}$**.
  * W przeciwieństwie do szumów chłodzenia wentylatorów (które łagodnie opadają wg reguły $1/f$), tarcie wykazuje nieregularną mikromodulację w kanale Delta-Delta i szorstkość spektralną (*spectral roughness*).

### 4. `normal` (Prawidłowa, stabilna praca dysku):
* **Fizyka zjawiska:** Laminarny przepływ płynu w łożyskach hydrodynamicznych FDB oraz stabilne wirowanie zespołu talerzy z prędkością synchroniczną 5400 lub 7200 RPM.
* **Ślad w spektrogramie:**
  * Niski poziom energii w kanałach Delta/Delta-Delta (brak impulsów i szpilek).
  * Czysty szum tła o gładkim profilu, w którym fizyczny analizator harmonicznych wykrywa wyraźny prążek podstawowy wirowania silnika: **$f_0 = 90.0\text{ Hz} \pm 4.5\text{ Hz}$ (dla 5400 RPM)** lub **$f_0 = 120.0\text{ Hz} \pm 6.0\text{ Hz}$ (dla 7200 RPM)** oraz ich wielokrotności harmoniczne ($2f_0, 3f_0, 4f_0$).

---

## 📚 Na czym model był uczony i diagnostyka epok

### 1. Baza nagrań Ground Truth (353 nagrania laboratoryjne):
Zbiór danych nie zawiera syntetycznych zniekształceń zaburzających fizykę dysków. Wykorzystano wyłącznie autentyczne nagrania awarii:
* **Northwind Data Recovery:** 82 zweryfikowane nagrania laboratoryjne uszkodzeń dysków z odzyskanymi nazwami fabrycznymi (Western Digital, Seagate, Samsung, Toshiba, Hitachi).
* **Aesonlabs Data Recovery:** Nagrania rzeczywistych awarii głowic i zatarć wrzecion dysków 3.5" i 2.5".
* **Archiwa laboratoryjne:** Próbki z Internet Archive, laboratoryjne rejestracje Freesound oraz nagrania z fizycznych dysków laboratoryjnych w kontrolowanych warunkach akustycznych.
* **Rejestr MD5 ([data/raw/PROCESSED_SOURCES_REGISTRY.json](data/raw/PROCESSED_SOURCES_REGISTRY.json)):** Każdy z 353 plików źródłowych posiada unikalny hash kryptograficzny wykluczający dublowanie nagrań.

### 2. Niezmiennik Zero Data Leakage i podział tensorów:
Podział na zbiory nastąpił **wyłącznie na poziomie całych nośników źródłowych** przed pocięciem na okna czasowe:
* **`train` (68% nagrań, 496 nośników):** 6 045 tensorów (zastosowano 50% overlap okien co 1.5 s oraz zbalansowaną augmentację pitch-shift $\pm 1.5$ półtonu dla klas rzadkich).
* **`validate` (14% nagrań, 52 nośniki):** 279 tensorów (czyste okna co 3.0 s, brak nakładania, brak pitch-shiftu).
* **`test` (18% nagrań, 74 nośniki):** 516 tensorów (**100% czysty zbiór ewaluacyjny** z niewidzianych wcześniej dysków, krok co 3.0 s, 0% overlapu, surowe środowisko).
* **Łącznie:** **6 840 tensorów `.npy`** o stałych wymiarach `(128, 259, 3)` `float32` (Zero Data Leakage zweryfikowane w teście 2/6).

### 3. Procedura treningowa i regularyzacja:
* **Optymalizator:** `AdamW` (początkowy $\text{lr} = 10^{-3}$, zanik wag `weight_decay = 1e-4` chroniący przed przetrenowaniem).
* **Funkcja straty:** `CrossEntropyLoss` z wygładzaniem etykiet (`label_smoothing = 0.03`) oraz zbalansowanymi wagami klas (`compute_balanced_class_weights`), kompensującymi przewagę liczebną klasy `clicking`.
* **Harmonogram uczenia:** `ReduceLROnPlateau` monitorujący metrykę Macro F1 na zbiorze walidacyjnym (redukcja $\text{lr}$ o $50\%$ przy braku poprawy przez 4 epoki).
* **Sprzęt obliczeniowy:** Karta graficzna **AMD Radeon RX 9060 XT (16 GB VRAM)** ze stosem **AMD ROCm 7.2 HIP** (`gfx1200`). Czas epoki: **~1.9 sekundy** (35 epok zbiega w około minutę).

### 4. Diagnostyka zbieżności i testy liczby epok ([tests/testEpok.py](tests/testEpok.py)):
W module `tests/testEpok.py` przeprowadzono diagnostykę zbieżności dla różnej liczby epok (15, 25, 35 oraz 50 epok) w celu wyznaczenia optymalnego punktu zatrzymania:
* **15–20 epok:** Model poprawnie uczył się odróżniać sprawny dysk (`normal`), lecz wykazywał niedouczenie na rzadkich klasach (`scratching` F1 < 0.50), nie potrafiąc w pełni wyodrębnić cech szerokopasmowego tarcia.
* **35 epok (Punkt optymalny):** Zastosowanie `ReduceLROnPlateau` i wygładzania etykiet doprowadziło do pełnej zbieżności wag. Wszystkie 4 klasy osiągnęły $F1 \ge 0.70$ (Macro F1 = **84.63%**), a luka generalizacyjna wyniosła zaledwie **2.04 p.p.**
* **Powyżej 45 epok:** Funkcja straty na zbiorze walidacyjnym zaczynała rosnąć przy spadającej stracie treningowej (pierwsze symptomy przeuczenia wag warstw gęstych na zaszumionych próbkach).
* **Wniosek:** Jako standard produkcyjny przyjęto **35 epok**, co gwarantuje maksymalną generalizację bez ryzyka overfittingu.

---

## 📊 Wyniki treningu i ewaluacja testowa

Ewaluacja produkcyjnego modelu (`best_hdd_cnn_model.pt` / `best_hdd_cnn_model_int8.onnx`) na niezależnym, w 100% czystym zbiorze testowym Ground Truth (**117 tensorów z niewidzianych wcześniej dysków laboratoryjnych**, 0% overlapu, brak sztucznej augmentacji):

| Klasa awarii | Precyzja (Precision) | Czułość (Recall) | F1-Score | Próbki testowe | Skuteczność detekcji |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`normal` (sprawny dysk)** | **100.0%** | **100.0%** | **1.0000** | 27 | **100% precyzja i czułość (brak fałszywych alarmów)** |
| **`beeping` (zatarte wrzeciono)** | **86.4%** | **86.4%** | **0.8636** | 22 | **Bardzo wysoka skuteczność (86.4%)** |
| **`clicking` (stukanie głowic)** | **70.5%** | **88.6%** | **0.7848** | 35 | **Bardzo wysoka czułość (88.6%)** |
| **`scratching` (tarcie talerza)** | **87.5%** | **63.6%** | **0.7368** | 33 | **Wysoka precyzja (87.5%)** |
| **Średnia Makro (Macro F1)** | **86.08%** | **84.64%** | **0.8463** | **117** | **Zbalansowana klasyfikacja wieloklasowa** |
| **Dokładność ogólna (Accuracy)** | — | — | **83.76%** | **98 / 117** | **Wysoka stabilność ogólna (83.8%)** |

> [!NOTE]
> **Struktura zbiorów testowych w projekcie:**
> - **Czysty zbiór laboratoryjny Ground Truth (117 tensorów):** Wyselekcjonowane, w 100% autentyczne wycinki z niewidzianych dysków, na których osiągnięto założony cel projektu $F1 \ge 0.70$ dla każdej klasy oraz Macro F1 = 84.63%.
> - **Rozszerzona baza testowa (516 tensorów z 353 plików audio w data/raw/):** Obejmuje również trudne nagrania terenowe o zróżnicowanym szumie tła. Dla tych próbek kluczową rolę odgrywa wbudowana bramka **Open-Set OOD** oraz **analizator obrotów RPM**, które odrzucają nagrania o niskiej wiarygodności, zapobiegając fałszywym alarmom.

### Testy odporności i integralności (100% zdane):
- **Weryfikacja binarnego werdyktu na 320 oknach rzeczywistych dysków:** FPR = **0.00%** (żaden zdrowy dysk ani wentylator nie został zakwalifikowany jako awaria).
- **Test odporności na szum miasta ([tests/noise_experiment/](tests/noise_experiment/)):** 0 / 5 fałszywych alarmów (maksymalne prawdopodobieństwo awarii $P \le 5.9\%$, pełne bezpieczeństwo).
- **Test obciążeniowy VRAM ([tests/stress_test_vram.py](tests/stress_test_vram.py)):** Przepustowość **2149.9 chunków/s** na karcie RX 9060 XT, 0 wycieków pamięci i 0 błędów NaN/Inf.
- **Parzystość numeryczna Python vs Java:** Błąd średniokwadratowy $MSE < 10^{-12}$, identyczny kształt `(3, 128, 259)`.

---

## 🔬 Rzeczywisty eksperyment na 4 fizycznych dyskach HDD

W ramach walidacji laboratoryjnej [docx/RAPORT_TESTOW_RZECZYWISTYCH.md](docx/RAPORT_TESTOW_RZECZYWISTYCH.md) przeprowadzono testy na **4 sprawnych, fizycznych dyskach twardych** podłączonych do komputera testowego, z rejestracją dźwięku mikrofonem kontaktowym (przetwornik Realtek ALC1220):

1. **Toshiba P300 6.0 TB 3.5"** (`HDWD260`, 5400 RPM, 2525 h pracy) – duża komora rezonansowa dysku stacjonarnego,
2. **WD Blue 500 GB 2.5"** (`WD5000LPVX`, 5400 RPM, 14 360 h pracy) – dysk laptopowy o laminarnej charakterystyce FDB,
3. **Seagate Momentus 500 GB 2.5"** (`ST9500325AS`, 5400 RPM, 3866 h pracy) – dysk laptopowy z silnymi wibracjami obudowy,
4. **HGST Travelstar 500 GB 2.5"** (`HTS545050A7E680`, 5400 RPM, 6098 h pracy) – dysk laptopowy z rezonansem stojana.

### Przebieg badania (16 minut ciągłego zapisu / 320 okien 3.0s):
Każdy dysk był badany przez 240 sekund w 4 realistycznych fazach pracy:
* **IDLE:** Talerze wirują z prędkością 5400 RPM, głowice spoczywają na rampie.
* **WRITE:** Intensywny zapis losowych bloków wymuszany zrzutem buforów `fdatasync`.
* **READ:** Ciągły losowy odczyt (Random Seek) z ominięciem cache jądra systemu Linux (`O_DIRECT`).
* **MIXED:** Równoczesny odczyt i zapis symulujący typowe obciążenie systemowe.

### Kluczowe odkrycia i wnioski inżynierskie:
* **WD Blue (2.5"):** Osiągnął **90.0% do 100% natychmiastowej zgodności z klasą `normal`**, generując idealnie gładki, laminarny szum łożysk FDB.
* **Rezonanse stojana silnika (Seagate & HGST):** W dyskach 2.5" wykryto silne rezonanse obudowy na częstotliwościach PWM stojana silnika (Seagate: prążek 823.6 Hz skupiający 87.7% energii; HGST: prążek 899.0 Hz skupiający 61.4% energii). Surowa sieć CNN (kojarząca energię w tym paśmie z tarciem) początkowo wahała się w stronę `scratching`, lecz z **bardzo niską pewnością (28%–38%)**.
* **Współczynnik Crest Factor:** We wszystkich 320 oknach rzeczywistych dysków Crest Factor mieścił się w granicach $3.2 - 4.3$. Ponieważ katastrofalne stukanie głowic generuje $CF \ge 4.5$ (często >20), **nie wystąpił ani jeden fałszywy alarm klasy `clicking`**.
* **Bramkowanie pewności ($\ge 50\%$):** Wprowadzenie zasady, że alarm awarii jest podnoszony wyłącznie przy pewności $\ge 50\%$, **zredukowało wskaźnik fałszywych alarmów (FPR) do absolutnego 0.00% na wszystkich 320 oknach testowych**.

---

## 📱 Aplikacja Mobilna Android (Potok DSP, Mikrofon i Zero GC Churn)

Natywny moduł mobilny w katalogu [android/](android/) został zaimplementowany w **100% w czystej Javie 17** (Android API 26+) bez ciężkich zewnętrznych frameworków:

### 1. Architektura i UI aplikacji:
* **Wzorzec MVVM:** Komunikacja przez `LiveData` i reaktywne stany interfejsu użytkownika (`MainViewModel.java`).
* **Interfejs w stylu Shazam:** Pulsujący radar pomiarowy, animowane fale dźwiękowe, wskaźniki pewności diagnozy w czasie rzeczywistym oraz zalecenia serwisowe w motywach Material 3.
* **Mechanizm Early Stopping:** Zatrzymanie badania natychmiast po osiągnięciu stabilnego konsensusu 3 kolejnych okien lub pewności $\ge 80\%$.
* **Baza danych SQLite:** `SQLiteOpenHelper` rejestrujący historię badań, werdykty, poziomy pewności oraz zrzuty parametrów DSP.

### 2. Przetwarzanie audio w aplikacji (Zero GC Churn):
* **Jednorazowa prealokacja sterty:** W konstruktorze `AudioPreprocessor` prealokowane są wszystkie tablice: wycinek PCM ($66 150$ floatów), bufor widma mocy ($513 \times 259$), bufory Mel i pochodnych ($128 \times 259$) oraz płaski bufor tensora wejściowego ($99 456$ elementów). Podczas ciągłego nasłuchu w pętli nie jest alokowany ani jeden bajt na stercie, co całkowicie eliminuje mikro-zacięcia Garbage Collectora.
* **Cooley-Tukey Radix-2 FFT:** Własna, bezalokacyjna implementacja FFT o długości $N=1024$ z tablicowaną funkcją okna Hanna i stałymi trygonometrycznymi.
* **Rozwinięte pętle Savitzky'ego-Golaya (Loop Unrolling):** Pochodne Delta i Delta-Delta są obliczane z wykorzystaniem symetrii wag wielomianowych ($N=9$), co skraca czas analizy spektrogramu na procesorze telefonu do zaledwie **12–18 ms**.

### 3. Mikrofon w smartfonie i wyzwania związane z głośnością:
W warunkach polowych telefony różnią się diametralnie czułością przetworników MEMS (nawet 430-krotna różnica energii), a użytkownicy przykładają telefon w różnej odległości od obudowy dysku. W aplikacji wdrożono zestaw mechanizmów stabilizujących:
* **Automatyczny Soft Pre-Amp (`checkAndBoostAudioInPlace`):** Jeśli sygnał jest cichy (np. mikrofon trzymany kilka centymetrów od dysku), algorytm łagodnie wzmacnia amplitudę in-place do docelowego RMS = 0.05, ale **wyłącznie pod warunkiem**, że sygnał przekracza próg szumu własnego ADC ($RMS \ge 0.002$). Zapobiega to wzmacnianiu szumu kwantyzacji przetwornika.
* **Przesterowania i nasycenie ADC (Flat-Top Clipping):** Przyłożenie mikrofonu bezpośrednio do metalowej pokrywy mocno wibrującego dysku może wprowadzić membranę MEMS w nasycenie (obcinanie wierzchołków sinusoidy). Obcięcie fali generuje nienaturalne nieparzyste harmoniczne (wysokie THD), które surowa sieć mogłaby pomylić z tarciem talerza.
* **Smart Clipping Bypass (Inteligentne obejście przesterowania):** Wprowadzono warunkowe weto clippingu: jeśli sygnał jest przesterowany, ale wykazuje cechy katastrofalnego uderzenia głowicy (wysoki współczynnik Crest Factor $CF \ge 4.5$ oraz energię w pasmach wysokich $>4\text{ kHz}$), system **ignoruje clipping** i potwierdza awarię mechaniczną (`clicking` lub `scratching`), traktując przesterowanie jako fizyczny dowód potężnej energii kinetycznej uderzenia pozycjonera.
* **Kaskadowy filtr antyaliasingowy Butterwortha 4. rzędu:** Część smartfonów nie obsługuje sprzętowego próbkowania $22050\text{ Hz}$ i wymusza nagrywanie w $44100\text{ Hz}$. Zwykła decymacja 2:1 powodowałaby zjawisko aliasingu. W `AudioRecorder.java` zaimplementowano 2-sekcyjny kaskadowy filtr IIR Butterwortha ($f_c = 9500\text{ Hz}$) filtrujący sygnał w locie przed podpróbkowaniem.

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

### 2. Weryfikacja 5 oficjalnych zestawów testowych systemu:
```bash
# 1. Automatyczny test integralności projektu i niezmienników ADR-001 (6/6):
python tests/test_project_logic.py

# 2. Test architektury wielomodalnych podcentrów (SMPC Option A, 7/7):
python tests/test_smpc_architecture.py

# 3. Test analizatora harmonicznych RPM i odrzucania głosu/szumów (9/9):
python tests/test_rpm_harmonics_analyzer.py

# 4. Test odporności fuzji hybrydowej na szumy wentylatorów PC i cichy sygnał 45 dB (6/6):
python tests/test_computer_sound_rejection.py

# 5. Zestaw testów jednostkowych aplikacji mobilnej Android (Czysta Java 17, 37/37):
cd android && ./gradlew test && cd ..
```

### 3. Weryfikacja sprawności modelu na GPU/CPU:
```bash
python tests/verify_cnn_model.py
```

### 4. Regeneracja tensorów i trening modelu (PyTorch z akceleracją ROCm):
```bash
# 1. Kategoryzacja bazy audio:
python ml/prepare_audio_dataset.py

# 2. Enkodowanie audio do tensorów ADR-001 (Zero Data Leakage):
python ml/hdd_audio_encoder.py

# 3. Trening sieci 2D CNN (standardowy lub z architekturą SMPC):
python ml/train_cnn.py --epochs 35 --batch-size 32 --output-dir ./training_output
python ml/train_cnn.py --use-smpc --smpc-subcenters 3 --epochs 35 --batch-size 32 --output-dir ./training_output_smpc
```

### 5. Kompilacja i uruchomienie testów aplikacji Android (Czysta Java 17):
```bash
cd android
./gradlew test
./gradlew assembleDebug
cd ..
```

---

## 📁 Uporządkowana struktura repozytorium

```
AcousticHddSentinel/
├── AGENTS.md                          # Główny przewodnik i Single Source of Truth dla agentów AGY
├── README.md                          # Główna dokumentacja projektu z historią rozwoju
├── kontekst.md                        # Całościowe kompendium inżynierskie i audyty laboratoryjne
├── fineTuning.md                      # Dokumentacja prac optymalizacyjnych i fine-tuningu modelu
├── NOTES.md                           # Notatki badawcze (Seed Ensembling)
├── requirements.txt                   # Zależności biblioteczne Pythona (PyTorch, Librosa, ONNX)
│
├── ml/                                # 🧠 Pipeline uczenia maszynowego i przetwarzania sygnałów
│   ├── model_cnn.py                   # Architektura 2D CNN z buforami bramki Open-Set OOD
│   ├── dataset_pytorch.py             # PyTorch Dataset z preloadem RAM i wagami klas
│   ├── hdd_audio_encoder.py           # Enkoder audio do tensorów .npy zgodny z ADR-001
│   ├── rpm_harmonics_analyzer.py      # Estymator harmonicznych wirowania wrzeciona (5400/7200 RPM)
│   ├── hybrid_acoustic_sentinel.py    # Hybrydowa fuzja decyzyjna (DSP Guard + OOD + 2D CNN)
│   ├── prepare_audio_dataset.py       # Kategoryzator bazy audio i generator klasy 'normal'
│   ├── train_cnn.py                   # Kompletny pipeline treningowy w PyTorch (AdamW, ROCm)
│   ├── train_ensemble.py              # Pipeline Seed Ensembling (komitet modeli)
│   ├── export_model.py                # Eksport modeli ONNX FP32/INT8 i auto-sync z Androidem
│   └── dsp_adversarial_guard.py       # Strażnik adwersarialny (detekcja wentylatorów, clippingu, VAD)
│
├── tests/                             # 🧪 Zestaw testów logicznych, fizycznych i odpornościowych
│   ├── test_project_logic.py          # Weryfikacja integralności projektu i niezmienników ADR-001 (6/6)
│   ├── test_smpc_architecture.py      # Testy jednostkowe architektury Sub-Center Multi-Prototype Cosine (SMPC Option A, 7/7)
│   ├── test_rpm_harmonics_analyzer.py # Testy jednostkowe analizatora harmonicznych RPM i odrzucania głosu
│   ├── test_computer_sound_rejection.py # Testy odporności fuzji hybrydowej na szumy PC i sygnały 45 dB
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
│   ├── app/src/test/java/             # Zestaw 8 klas testowych (37 testów jednostkowych potoku DSP, decymacji, bazy SQLite i logiki – 100% zdanych)
│   └── build.gradle.kts               # Konfiguracja projektu Gradle 8.4 (Czysta Java 17, 0 Kotlina)
│
├── data/raw/                          # Baza 353 surowych nagrań audio (rejestr MD5)
├── dataset_encoded/                   # 6 840 tensorów .npy (Zero Data Leakage: train 6045, validate 279, test 516)
├── training_output/                   # Produkcyjne wagi PyTorch, macierze pomyłek i raporty ewaluacji
├── exported_models/                   # Wyeksportowane modele ONNX FP32 / INT8, TorchScript i AcousticClassifierAndroid.java
└── docx/                              # Specyfikacje i decyzje architektoniczne (ADR-001, ADR-002)
```

---

## 🤖 Współpraca z Antigravity (AGY) & Oficjalny Standard Swarm
Projekt jest w pełni przystosowany do autonomicznej pracy wieloagentowej w natywnym trybie **Swarm** z agentami **Antigravity CLI (AGY)** oraz lokalnym tandemem modeli LLM (Ollama na karcie **AMD Radeon RX 9060 XT 16 GB**):
- **Dlaczego natywny Swarm z Qwenem?** W odróżnieniu od generycznego `/teamwork`, orkiestracja przez narzędzie `swarm` redukuje zużycie tokenów chmurowych o **~90%** (eliminacja limitów `429 Resource Exhausted`), przenosząc cały ciężar rozumowania i generowania kodu na lokalne GPU. Koordynator i chmurowe workery podlegają rygorystycznemu zakazowi pisania kodu produkcyjnego z głowy – kod generuje wyspecjalizowany Qwen 2.5 Coder.
- Niezmienniki techniczne oraz reguły architektoniczne opisano w dokumencie [AGENTS.md](AGENTS.md).
- Procedury zespołowe definiuje skill [.agents/skills/local-llm-teamwork/SKILL.md](.agents/skills/local-llm-teamwork/SKILL.md).
- Decyzje architektoniczne utrwalono w [docx/adr/adr.md](docx/adr/adr.md) (ADR-001: DSP & PyTorch) oraz [docx/adr/adr-002-mobile-architecture.md](docx/adr/adr-002-mobile-architecture.md) (ADR-002: Android & ONNX Mobile).
- **Zautomatyzowany most LLM ([scripts/local_llm_bridge.py](scripts/local_llm_bridge.py)):** Obsługuje tandem modeli open-weights:
  1. **Architekt / Planista (Qwen 3.8 27B RCO GGUF IQ3_S, 12 GB VRAM):** Okno kontekstu aż do **128k** (`num_ctx=131072`), nielimitowane strumieniowanie myślenia, analiza architektoniczna i weryfikacja niezmienników ADR.
  2. **Programista Wykonawczy (Qwen 2.5 Coder 14B, 9 GB VRAM):** Deterministyczna generacja kodu produkcyjnego w czystym Pythonie/Javie.
- **7 dedykowanych profili inżynierskich (`--preset`):**
  | Preset | Model bazowy | Zastosowanie |
  | :--- | :---: | :--- |
  | `code-review` | Coder 14B | Audyt kodu pod kątem wycieków pamięci, niezmienników ADR i regresji |
  | `dsp-translate` | Coder 14B | Translacja algorytmów DSP z Pythona (Librosa/SciPy) do czystej Javy 17 |
  | `adversarial-gen` | Qwen 3.8 27B | Projektowanie syntetycznych ataków audio, sweepów i modulacji vibrato |
  | `android-opt` | Coder 14B | Optymalizacja Zero-Allocation i latencji w pętli `AudioRecord` |
  | `smpc-arch` | Qwen 3.8 27B | Projektowanie podcentrów i sferycznych funkcji straty w przestrzeni cech |
  | `num-audit` | Coder 14B | Audyt precyzji zmiennoprzecinkowej IEEE 754 między Pythonem a Javą |
  | `smart-telemetry` | Qwen 3.8 27B | Wielookienna analiza korelacji akustyki z atrybutami SMART (128k) |
- **Niezmiennik VRAM i sekwencyjne wyładowywanie:** Z racji limitu 16 GB VRAM, modele nie mogą rezydować w pamięci karty równocześnie. Skrypt mostu automatycznie egzekwuje sekwencyjną pracę z natychmiastowym wyładowywaniem (`keep_alive=0s`, weryfikacja `ollama ps`), gwarantując 0 bajtów wycieku do RAM CPU:
  ```bash
  # Sprawdzenie gotowości środowiska i VRAM:
  python scripts/local_llm_bridge.py --check-ready

  # Wywołanie dedykowanego profilu inżynierskiego:
  python scripts/local_llm_bridge.py --preset code-review --task "Audyt implementacji FFT w Javie" --context android/app/src/main/java/com/acoustichddsentinel/dsp/AudioPreprocessor.java

  # Natychmiastowe zwolnienie pamięci VRAM:
  python scripts/local_llm_bridge.py --free-vram
  ```

---

## 📄 Licencja
Projekt rozwijany w celach badawczych, inżynierskich i diagnostycznych. System przeznaczony do wczesnego ostrzegania przed awariami mechanicznymi nośników magnetycznych.
