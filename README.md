# Acoustic HDD Sentinel 🛡️💽
> **System wibroakustycznej detekcji i wieloklasowej klasyfikacji uszkodzeń mechanicznych dysków twardych (HDD) przy użyciu głębokich sieci konwolucyjnych (2D CNN w PyTorch), fizycznych strażników DSP oraz aplikacji mobilnej Android (Edge AI).**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Android 8.0+](https://img.shields.io/badge/Android-Java%2017%20%7C%20API%2026%2B-green.svg)](https://developer.android.com/)
[![ONNX Runtime Mobile](https://img.shields.io/badge/ONNX%20Runtime-INT8%20(1.17%20MB)-orange.svg)](https://onnxruntime.ai/)
[![Status](https://img.shields.io/badge/Status-Macro%20F1%2084.6%25%20(Production)-brightgreen.svg)]()
[![Hardware Acceleration](https://img.shields.io/badge/Acceleration-AMD%20ROCm%20HIP%20(RX%209060%20XT)-purple.svg)]()

---

## 📖 Spis treści
1. [O projekcie](#-o-projekcie)
2. [📈 Ewolucja Projektu i Kamienie Milowe (Progress & Evolution)](#-ewolucja-projektu-i-kamienie-milowe-progress--evolution)
3. [Architektura przetwarzania audio (ADR-001)](#-architektura-przetwarzania-audio-adr-001)
4. [🧠 Jak konkretnie działa model 2D CNN i mechanizmy obronne](#-jak-konkretnie-działa-model-2d-cnn-i-mechanizmy-obronne)
5. [🔍 Jak model wyłapuje poszczególne awarie mechaniczne](#-jak-model-wyłapuje-poszczególne-awarie-mechaniczne)
6. [📚 Na czym model był uczony i diagnostyka epok](#-na-czym-model-był-uczony-i-diagnostyka-epok)
7. [📊 Wyniki treningu i ewaluacja testowa](#-wyniki-treningu-i-ewaluacja-testowa)
8. [🔬 Rzeczywisty eksperyment na 4 fizycznych dyskach HDD](#-rzeczywisty-eksperyment-na-4-fizycznych-dyskach-hdd)
9. [📱 Aplikacja Mobilna Android (Potok DSP, Mikrofon i Zero GC Churn)](#-aplikacja-mobilna-android-potok-dsp-mikrofon-i-zero-gc-churn)
10. [📦 Eksport modeli produkcyjnych (ONNX / Mobile / Android)](#-eksport-modeli-produkcyjnych-onnx--mobile--android)
11. [💻 Instalacja i szybki start](#-instalacja-i-szybki-start)
12. [📁 Uporządkowana struktura repozytorium](#-uporządkowana-struktura-repozytorium)
13. [🤖 Współpraca z Antigravity (AGY)](#-współpraca-z-antigravity-agy)

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

| Cecha / Metryka | Faza 1: Wczesny Model Bazowy (PyTorch CPU Baseline) | Faza 3: Przełom Fine-Tuningu (ADR-001 & PyTorch) | Faza 5: Stan Obecny (Produkcja, OOD & Mobile) |
| :--- | :---: | :---: | :---: |
| **Główny silnik ML** | PyTorch (Wczesny model CPU) | PyTorch 2.x (AMD ROCm 7.2) | PyTorch 2.x + ONNX Runtime Mobile |
| **Baza próbek audio** | ~82 zanieczyszczone pliki MP3 | 134 zweryfikowane nagrania | **288 autentycznych nagrań** (Rejestr MD5) |
| **Tensory `.npy`** | ~800 tensorów (wyciek danych) | 1 598 tensorów (Zero Data Leakage) | **5 187 tensorów** (Train: 4633, Val: 196, Test: 358) |
| **Współczynnik F1: `scratching`** | **0.00** (100% mylone z clicking) | **0.7368** (Wysoka czułość) | **0.7368** (+ Smart Clipping Bypass) |
| **Współczynnik F1: `beeping`** | **0.20** (Zatrucie etykiet Northwind) | **0.8636** (Weryfikacja widmowa) | **0.8636** (+ Blokada rotacji RPM) |
| **Współczynnik F1: `clicking`** | 0.62 | **0.7848** | **0.7848** |
| **Współczynnik F1: `normal`** | 0.81 (Shortcut learning < 2.4 kHz) | **1.0000** (Pełna parzystość) | **1.0000** (100% Precyzja i Czułość) |
| **Średnia Makro (Macro F1)** | ~0.41 | **84.63%** | **84.63%** (Zbalansowana detekcja) |
| **Luka generalizacyjna** | > 18.0 p.p. (Przetrenowanie) | **2.04 p.p.** (Idealna zbieżność) | **Minimalna** (Zabezpieczona OOD) |
| **Bramka Open-Set OOD** | Brak (Fałszywe alarmy na mowie) | Brak | **Wbudowane prototypy widmowe w grafie** |
| **Odporność na szum miasta** | Niebadana (Częste alarmy) | Niebadana | **0 / 5 fałszywych alarmów (P $\le 5.9\%$)** |
| **Rozmiar modelu** | ~15.2 MB | 4.61 MB (FP32) | **1.17 MB (Skwantyzowany ONNX INT8)** |
| **Czas epoki treningu** | ~16.0 s (CPU) | **~1.9 s (AMD RX 9060 XT ROCm)** | **~1.9 s (Akceleracja ROCm)** |
| **Wdrożenie docelowe** | Wyłącznie skrypty Python | Eksport ONNX | **Aplikacja Android (Java 17, Zero GC Churn)** |

---

### Kluczowe etapy rozwoju (Timeline):

1. **Faza 1 (Wczesny Model Bazowy i Identyfikacja Problemów):**
   * Pierwsza bazowa implementacja architektury 2D CNN w PyTorch (trening CPU z podstawowym optymalizatorem Adam).
   * *Acoustic Forensics:* Zidentyfikowano krytyczne wady wczesnego potoku: silny DC offset ($+0.3766$), artefakty zapętlania `np.tile` generujące sztuczne stukanie, utratę oryginalnych nazw plików w bazie Northwind oraz zafałszowanie etykiety `Audio_Player_57.wav` (duplikat klikającego Samsunga SP8002 oznaczony jako beeping).
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

### 1. Baza nagrań Ground Truth (288 nagrań laboratoryjnych):
Zbiór danych nie zawiera syntetycznych zniekształceń zaburzających fizykę dysków. Wykorzystano wyłącznie autentyczne nagrania awarii:
* **Northwind Data Recovery:** 82 zweryfikowane nagrania laboratoryjne uszkodzeń dysków z odzyskanymi nazwami fabrycznymi (Western Digital, Seagate, Samsung, Toshiba, Hitachi).
* **Aesonlabs Data Recovery:** Nagrania rzeczywistych awarii głowic i zatarć wrzecion dysków 3.5" i 2.5".
* **Archiwa laboratoryjne:** Próbki z Internet Archive, laboratoryjne rejestracje Freesound oraz nagrania z fizycznych dysków laboratoryjnych w kontrolowanych warunkach akustycznych.
* **Rejestr MD5 ([data/raw/PROCESSED_SOURCES_REGISTRY.json](data/raw/PROCESSED_SOURCES_REGISTRY.json)):** Każdy plik źródłowy posiada unikalny hash kryptograficzny wykluczający dublowanie nagrań.

### 2. Niezmiennik Zero Data Leakage i podział tensorów:
Podział na zbiory nastąpił **wyłącznie na poziomie całych nośników źródłowych** przed pocięciem na okna czasowe:
* **`train` (68% nagrań):** 4 633 tensory (zastosowano 50% overlap okien co 1.5 s oraz zbalansowaną augmentację pitch-shift $\pm 1.5$ półtonu dla klas rzadkich).
* **`validate` (14% nagrań):** 196 tensorów (czyste okna co 3.0 s, brak nakładania, brak pitch-shiftu).
* **`test` (18% nagrań):** 358 tensorów (**100% czysty zbiór ewaluacyjny** z niewidzianych wcześniej dysków, krok co 3.0 s, 0% overlapu, surowe środowisko).

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

Ewaluacja produkcyjnego modelu na niezależnym, w 100% czystym zbiorze testowym (`test` – 358 tensorów z niewidzianych wcześniej dysków, 0% overlapu, brak sztucznej augmentacji):

| Klasa awarii | Precyzja (Precision) | Czułość (Recall) | F1-Score | Skuteczność detekcji |
| :--- | :---: | :---: | :---: | :---: |
| **`normal` (sprawny dysk)** | **100.0%** | **100.0%** | **1.0000** | **100% precyzja i czułość (brak fałszywych alarmów)** |
| **`beeping` (zatarte wrzeciono)** | **86.4%** | **86.4%** | **0.8636** | **Bardzo wysoka skuteczność (86.4%)** |
| **`clicking` (stukanie głowic)** | **70.5%** | **88.6%** | **0.7848** | **Bardzo wysoka czułość (88.6%)** |
| **`scratching` (tarcie talerza)** | **87.5%** | **63.6%** | **0.7368** | **Wysoka precyzja (87.5%)** |
| **Średnia Makro (Macro F1)** | **86.08%** | **84.64%** | **0.8463** | **Zbalansowana klasyfikacja wieloklasowa** |
| **Dokładność ogólna (Accuracy)** | — | — | **83.76%** | **Wysoka stabilność ogólna** |

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
├── fineTuning.md                      # Dokumentacja prac optymalizacyjnych i fine-tuningu modelu
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
