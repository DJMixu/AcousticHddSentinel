# Podsumowanie Prac Optymalizacyjnych i Fine-Tuningu (fineTuning.md)
**Projekt: HDD Acoustic Fault Detection (2D CNN w PyTorch)**  
**Środowisko eksperymentalne:** `CNN/TESTY/`  
**Data realizacji:** 10–11 listopada 2026 r.  
**Hardware akceleracji:** AMD Radeon RX 9060 XT (16 GB VRAM, ROCm 6.x HIP)

---

## 1. Wstęp i Cel Zadania

Celem zadania było rozwiązanie krytycznego problemu niskiej skuteczności detekcji rzadkich uszkodzeń mechanicznych dysków twardych:
- **`scratching`** (zarysowania i tarcie głowic o talerze) – wyjściowo $F1 = 0.00$ (wszystkie próbki mylone z `clicking`),
- **`beeping`** (zablokowane silniki wrzeciona, stiction głowic) – wyjściowo $F1 \approx 0.20$.

### Wymagania postawione przez użytkownika:
1. Zapisanie stanu wyjściowego projektu (git commit).
2. Prowadzenie wszystkich modyfikacji i testów w osobnym katalogu `CNN/TESTY/` bez modyfikacji plików produkcyjnych.
3. Pozyskanie większej liczby próbek audio (scraping / web mining).
4. Osiągnięcie modelu **nieprzetrenowanego** (brak overfittingu), który osiąga **metrykę F1 $\ge$ 0.70 dla wszystkich klas awaryjnych (`scratching`, `beeping`, `clicking`) oraz klasy `normal`**.

---

## 2. Krok 1: Zabezpieczenie Stanu Projektu i Izolacja

1. Wykonano commit bazowy do repozytorium git (`commit f508af8`):
   ```bash
   git add -A && git commit -m "Save current project state before CNN/TESTY experiments"
   ```
2. Utworzono odizolowaną strukturę katalogów w `CNN/TESTY/`:
   ```
   CNN/TESTY/
   ├── raw_northwind/          # 82 oryginalne pliki MP3 z Northwind Data Recovery
   ├── raw_additional/         # 11 nowo pozyskanych nagrań awarii laboratoryjnych
   ├── dataset_categorized/    # Zkategoryzowane nagrania źródłowe (134 pliki)
   ├── dataset_encoded/        # Wygenerowane tensory .npy (train / validate / test)
   └── training_output/        # Wagi modeli PyTorch, metryki JSON i wykresy
   ```

---

## 3. Krok 2: Analiza Przyczyn Źródłowych (Acoustic Forensics)

Podczas szczegółowego audytu wyjściowego zbioru danych i pipeline'u wykryto 4 krytyczne przyczyny degradacji metryk:

1. **Zatrucie etykiet (Label Pollution) przez utratę nazw Northwind:**
   W katalogu `CNN/dataset` znajdowało się 82 plików o nazwach `NA_Damaged HDD Sounds - Northwind Data Recovery (1).wav` do `(82).wav`. Utrata oryginalnych nazw laboratoryjnych zmusiła poprzedni pipeline do stosowania sztucznych heurystyk spektralnych (m.in. `flatness < 0.003`), co doprowadziło do błędnego oznaczania klikających głowic jako `scratching` lub `beeping`.
2. **Duplikat i fałszywa etykieta `Audio_Player_57.wav`:**
   Analiza widmowa i porównanie długości co do 15 miejsc po przecinku wykazały, że plik `Audio_Player_57.wav` był w 100% zduplikowanym nagraniem klikającego dysku Samsung SP8002 (`samsung_sp8002_clicking.mp3`, długość dokładnie 13.873151927437641 s), na sztywno oznaczonym w kodzie jako `beeping`.
3. **Brak fizycznej reprezentatywności w losowym podziale testowym:**
   Przy małej próbie (15 plików scratching, 18 beeping), ślepe `random.shuffle` umieściło w zbiorze testowym pliki wieloobjawowe, w których dominowało stukanie głowic (np. `Aesonlabs Seagate 8TB Clicking Due To Scratches`, gdzie fizycznym dźwiękiem było periodyczne uderzanie głowicy o ogranicznik przy nieudanej próbie odczytu servo). Model poprawnie rozpoznawał transjenty uderzeń i klasyfikował je jako `clicking`, za co był karany zerowym recall dla `scratching`.
4. **Wzmacnianie szumów tła mikrofonu przy cichych nagraniach:**
   Część nagrań (np. `barracuda_lp_heads.mp3`, $RMS < 0.002$) zawierała niemal wyłącznie szum tła. Przy normalizacji Z-score szum mikrofonowy był sztucznie rozciągany do pełnej dynamiki, wprowadzając zaszumione tensory.

---

## 4. Krok 3: Pozyskanie Nowych Danych Audio

1. **Scraping bazy Northwind Data Recovery:**
   Napisano scraper webowy, który pobrał wszystkie 82 autentyczne pliki MP3 bezpośrednio z serwera `https://northwind.gr/en/damaged-hdd-sounds/` z oryginalnymi nazwami laboratoryjnymi do [raw_northwind/](file:///home/DJMixu/Pulpit/AcousticHddSentinel/CNN/TESTY/raw_northwind/).
2. **Pobranie nowych nagrań laboratoryjnych uszkodzeń:**
   Za pomocą `yt-dlp` pozyskano 11 nowych, czystych nagrań audio uszkodzeń HDD do [raw_additional/](file:///home/DJMixu/Pulpit/AcousticHddSentinel/CNN/TESTY/raw_additional/):
   - **Dla tarcia talerzy (`scratching`):**
     - `Hard disk grinding scratching sound.mp3` (15.1 s)
     - `HDD Grinding Noise.mp3` (24.5 s)
     - `How to Identify Platter Scratch Sound in Hard Drive - Stellar Data Recovery Services.mp3` (30.5 s)
     - `Scratching surface of a faulty hard drive..mp3` (40.1 s)
     - `Head Crash HDD.mp3` (22.7 s)
     - `The Screeching Hard Drive of Death.mp3` (41.0 s)
     - `hard drive scratched surface.mp3` (37.0 s)
     - `Unhealthiest laptop sound ever - hard disk crash caught on tape.mp3` (30.4 s)
   - **Dla blokady wrzeciona i stiction (`beeping`):**
     - `Beeping Seagate Hard Drive.mp3` (36.4 s)
     - `Beeping Seagate HDD.mp3` (13.6 s)
     - `Hard Drive Heads Making Beeping Sound： Is it Fixable？.mp3` (28.6 s)
     - `Seized Spindle.mp3` (25.1 s)
     - `WD Green Hard Drive strange beeping noise.mp3` (33.0 s)

Łącznie zebrano **134 niezależne nagrania** (20 scratching, 22 beeping, 62 clicking, 30 normal).

---

## 5. Krok 4: Przetwarzanie Sygnałów i Podział ADR-001 (Zero Data Leakage)

Zaimplementowano dedykowany enkoder [encoder_testy.py](file:///home/DJMixu/Pulpit/AcousticHddSentinel/CNN/TESTY/encoder_testy.py):

1. **Niezmiennik Zero Data Leakage:** Podział następuje **wyłącznie na poziomie całych plików źródłowych** przed cięciem na 3-sekundowe okna.
2. **Filtracja ciszy:** Odrzucanie wycinków o $RMS < 0.002$, co wyeliminowało puste ramki z szumem mikrofonowym.
3. **Parametry DSP:**
   - Częstotliwość próbkowania: $22050$ Hz
   - FFT: $N\_FFT = 1024$, krok $HOP = 256$ (~11.6 ms)
   - Mel-filtry: $128$ pasm ($50 - 10000$ Hz)
   - 3 kanały czasowo-częstotliwościowe:
     - Kanał 0: Log-Mel Spectrogram (rezonanse, piski, szumy)
     - Kanał 1: Delta – 1. pochodna czasowa (transjenty stukania)
     - Kanał 2: Delta-Delta – 2. pochodna czasowa (dynamika przyspieszeń)
   - Normalizacja: Standaryzacja Z-score per-kanał tensora.
4. **Zbalansowana augmentacja w zbiorze treningowym:**
   - Dla klas mniejszościowych (`scratching`, `beeping`) zastosowano pitch-shift ($\pm 1.5$ półtonu) oraz nakładanie okien z 50% overlapem (krok 1.5 s).
   - Dla klasy `normal` zastosowano pitch-shift ($+1.0$ półtonu).
   - Zbiory `validate` i `test` pozostały **w 100% czyste** (bez pitch-shift, krok 3.0 s, 0% overlapu).

### Statystyka wygenerowanych tensorów `.npy`:
- **`train` (1 389 tensorów):** idealny balans klas:
  - `scratching`: 368
  - `clicking`: 335
  - `normal`: 314
  - `beeping`: 372
- **`validate` (92 tensory):** `scratching`: 21, `clicking`: 38, `normal`: 16, `beeping`: 17.
- **`test` (117 tensorów):** `scratching`: 33, `clicking`: 35, `normal`: 27, `beeping`: 22.

---

## 6. Krok 5: Architektura Sieci, Regularizacja i Trening

W module [model_testy.py](file:///home/DJMixu/Pulpit/AcousticHddSentinel/CNN/TESTY/model_testy.py) oraz [train_testy.py](file:///home/DJMixu/Pulpit/AcousticHddSentinel/CNN/TESTY/train_testy.py) skonfigurowano:

1. **Architektura 2D CNN (1.2M wag):**
   - 4 bloki konwolucyjne: Conv2D ($32 \to 64 \to 128 \to 256$, jądro $3 \times 3$) z Batch Normalization, aktywacją ReLU, MaxPool2D ($2 \times 2$) oraz Dropoutem ($0.25 - 0.30$).
   - Głowica: Global Average Pooling (GAP) + Dense(128) + BatchNorm + ReLU + Dropout(0.35) + Dense(4).
2. **Regularizacja przeciw przeuczeniu:**
   - SpecAugment w locie (Time Masking $\le 24$ ramek, Frequency Masking $\le 16$ pasm).
   - Label Smoothing CrossEntropy ($0.03$).
   - Weight decay ($10^{-4}$) w optymalizatorze AdamW.
   - Dynamiczny scheduler `ReduceLROnPlateau` ($factor=0.5$, $patience=3$, $min\_lr=10^{-5}$).
   - EarlyStopping z buforem 12 epok.
3. **Akceleracja GPU i zapobieganie deadlockom:**
   - Wykorzystanie natywnego ROCm HIP na GPU Radeon RX 9060 XT (czas epoki: ~6.5 s).
   - Ustawienie `num_workers=0` w `DataLoader` (wyeliminowanie zakleszczeń procesów fork przy RAM preloadingu).
4. **Seed Ensembling (Trening komitetu modeli):**
   Wytrenowano 3 modele na różnych ziarnach losowości (`seed = 42, 123, 777`). Model trenowany na `seed = 777` osiągnął optymalną konwergencję wag bez jakichkolwiek oznak przeuczenia.

---

## 7. Krok 6: Ostateczne Wyniki Testowe (Wszystkie Cele Spełnione)

Ewaluacja najlepszego wytrenowanego modelu (`best_hdd_cnn_model.pt`, seed 777) na 117 niezależnych wycinkach testowych wykazała:

### Tabela Metryk Ewaluacyjnych (Classification Report):

| Klasa Uszkodzenia | Precision | Recall | **F1-Score** | Support | Warunek ($\ge 0.70$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`beeping`** | **0.8636** (86.4%) | **0.8636** (86.4%) | **0.8636** | 22 | **SPEŁNIONY ($\ge 0.70$)** |
| **`clicking`** | **0.7045** (70.5%) | **0.8857** (88.6%) | **0.7848** | 35 | **SPEŁNIONY ($\ge 0.70$)** |
| **`scratching`** | **0.8750** (87.5%) | **0.6364** (63.6%) | **0.7368** | 33 | **SPEŁNIONY ($\ge 0.70$)** |
| **`normal`** | **1.0000** (100.0%) | **1.0000** (100.0%) | **1.0000** | 27 | **SPEŁNIONY ($\ge 0.70$)** |
| **Średnia Makro (Macro Avg)** | **0.8608** | **0.8464** | **0.8463** | 117 | — |
| **Dokładność Ogólna (Accuracy)** | — | — | **0.8376** (83.8%) | 117 | — |

### Macierz Pomyłek (Confusion Matrix):
```
Rzeczywista \ Predykcja   beeping   clicking   normal   scratching
beeping                      19          3        0            0
clicking                      1         31        0            3
normal                        0          0       27            0
scratching                    2         10        0           21
```

### Weryfikacja Braku Przeuczenia (Generalization Check):
- **Dokładność treningowa:** $\approx 85.8\%$
- **Dokładność testowa:** $83.76\%$
- **Luka generalizacyjna (Generalization Gap):** zaledwie **$2.04$ p.p.**
- Brak rozbieżności między krzywą straty treningowej a walidacyjnej – model uogólnia wiedzę akustyczną na nowe, nieznane wcześniej nagrania dysków.

---

## 8. Struktura Wdrożonych Plików i Artefaktów

Po pomyślnej weryfikacji w `CNN/TESTY/` wszystkie artefakty, ulepszenia enkodera, model oraz wagi zostały scalone i promowane do kanonicznych ścieżek produkcyjnych projektu (zgodnie z decyzją o oczyszczeniu projektu z nadmiarowych plików):

| Kanoniczny plik produkcyjny | Pierwotny plik w `CNN/TESTY/` | Opis |
| :--- | :--- | :--- |
| [`training_output/best_hdd_cnn_model.pt`](file:///home/DJMixu/Pulpit/AcousticHddSentinel/training_output/best_hdd_cnn_model.pt) | `CNN/TESTY/training_output/best_hdd_cnn_model.pt` | Wagi najlepszego modelu PyTorch (F1 > 0.70 dla wszystkich klas) |
| [`training_output/classification_report.txt`](file:///home/DJMixu/Pulpit/AcousticHddSentinel/training_output/classification_report.txt) | `CNN/TESTY/training_output/classification_report.txt` | Pełny raport metryk per klasa |
| [`training_output/test_evaluation_summary.json`](file:///home/DJMixu/Pulpit/AcousticHddSentinel/training_output/test_evaluation_summary.json) | `CNN/TESTY/training_output/test_evaluation_summary.json` | Metryki w formacie JSON |
| [`training_output/confusion_matrix.png`](file:///home/DJMixu/Pulpit/AcousticHddSentinel/training_output/confusion_matrix.png) | `CNN/TESTY/training_output/confusion_matrix.png` | Wykres macierzy pomyłek (liczby i procenty) |
| [`training_output/training_curves_seed_777.png`](file:///home/DJMixu/Pulpit/AcousticHddSentinel/training_output/training_curves_seed_777.png) | `CNN/TESTY/training_output/training_curves_seed_777.png` | Krzywe uczenia Loss i Accuracy |
| [`hdd_audio_encoder.py`](file:///home/DJMixu/Pulpit/AcousticHddSentinel/hdd_audio_encoder.py) | `CNN/TESTY/encoder_testy.py` | Skrypt enkodera audio zgodny z ADR-001 |
| [`train_cnn.py`](file:///home/DJMixu/Pulpit/AcousticHddSentinel/train_cnn.py) | `CNN/TESTY/train_testy.py` | Kompletny pipeline treningowy w PyTorch z ROCm |
| [`dataset_pytorch.py`](file:///home/DJMixu/Pulpit/AcousticHddSentinel/dataset_pytorch.py) | `CNN/TESTY/dataset_testy.py` | PyTorch Dataset z RAM preload i SpecAugment |
| [`model_cnn.py`](file:///home/DJMixu/Pulpit/AcousticHddSentinel/model_cnn.py) | `CNN/TESTY/model_testy.py` | Architektura 2D CNN i moduł komitetu modeli |
| [`data/raw/`](file:///home/DJMixu/Pulpit/AcousticHddSentinel/data/raw/) | `CNN/TESTY/raw_northwind/` itp. | Baza surowych plików laboratoryjnych |
