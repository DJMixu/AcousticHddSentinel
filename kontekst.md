# Kontekst Projektu i Podsumowanie Wszystkich Rozmów — Acoustic HDD Sentinel

> **Data sporządzenia:** 11 listopada 2026 r.  
> **Cel dokumentu:** Pełne, usystematyzowane podsumowanie wszystkich dotychczasowych etapów, rozmów konwersacyjnych, audytów, decyzji architektonicznych oraz wyników optymalizacji modelu uczenia maszynowego w projekcie **Acoustic HDD Sentinel**.

---

## 1. Cel i Wymagania Główne Projektu

Projekt **Acoustic HDD Sentinel** to system akustycznej detekcji i wieloklasowej klasyfikacji uszkodzeń mechanicznych dysków twardych (HDD) przy użyciu głębokich sieci konwolucyjnych (2D CNN). System klasyfikuje stan dysku na podstawie krótkich, 3-sekundowych próbek dźwiękowych do jednej z 4 klas:
1. **`normal`** – sprawny dysk (szum łożyska hydrodynamicznego, obroty talerzy, miękkie pozycjonowanie głowic),
2. **`clicking`** – uderzanie głowic o ogranicznik (*Click of Death*, awaria przedwzmacniacza lub uszkodzenie powierzchni servo),
3. **`scratching`** – fizyczne tarcie i żłobienie powierzchni talerzy magnetycznych przez zniszczone głowice (*head crash*),
4. **`beeping`** – zablokowanie osi silnika wrzeciona (*seized spindle motor*) lub przylepienie głowic do powierzchni talerza (*stiction*).

---

## 2. Chronologiczna Historia Rozwoju i Podsumowanie Rozmów

### Faza 1: Architektura Bazowa i Pipeline DSP (TensorFlow / Keras)
*(Identyfikatory sesji: `d7cdbb65`, `cb4488fd`)*

1. **Ustanowienie niezmienników technicznych ([ADR-001](docx/adr/adr.md)):**
   - Zdefiniowano reprezentację czasowo-częstotliwościową w postaci 3-kanałowego tensora `(128, 259, 3)` (Channel-Last na dysku w plikach `.npy`):
     - **Kanał 0:** Log-Mel Spektrogram (rezonanse, harmoniczne silnika wrzeciona, piski),
     - **Kanał 1:** Delta – 1. pochodna czasowa (transjenty, impulsowe klikanie głowic),
     - **Kanał 2:** Delta-Delta – 2. pochodna czasowa (dynamika przyspieszeń i zmian kątowych).
   - Parametry DSP: częstotliwość próbkowania $22050$ Hz, okno $N\_FFT = 1024$, krok $HOP\_LENGTH = 256$ (~11.6 ms – kluczowy dla wysokiego *Crest Factor*), $128$ filtrów Mel w pasmie $50 - 10000$ Hz.
   - **Niezmiennik Zero Data Leakage:** Podział na zbiory `train` (70%), `validate` (15%) i `test` (15%) **musi** odbywać się wyłącznie na poziomie całych nagrań źródłowych przed ich podziałem na 3-sekundowe okna czasowe.
2. **Moduły bazowe:**
   - [hdd_audio_encoder.py](hdd_audio_encoder.py) – enkoder generujący tensory `.npy` z normalizacją dwustopniową (Peak Normalization fali + Z-score per kanał tensora).
   - [prepare_audio_dataset.py](prepare_audio_dataset.py) – kategoryzator i generator próbek audio.
   - [profile_normalization_quality.py](profile_normalization_quality.py) – profiler statystyk i poprawności normalizacji.
   - Pierwszy model 2D CNN zaimplementowany w TensorFlow/Keras.

---

### Faza 2: Rygorystyczny Audyt Zewnętrzny, Stress-Testy i Analiza Logiki
*(Identyfikator sesji: `ad8cb723` — 44 interakcje)*

1. **Pakiet audytowy ([AUDIT_REVIEW_BUNDLE.md](AUDIT_REVIEW_BUNDLE.md)):**
   - Przygotowano całościowy plik (~100k znaków) zawierający architekturę, założenia ADR-001, logi i kody źródłowe w celu poddania projektu niezależnemu audytowi zewnętrznego modelu LLM (Claude Opus).
2. **Wykryte wady i anomalie (Acoustic & Engineering Forensics):**
   - **DC Offset:** W surowych nagraniach występowało silne przesunięcie składowej stałej (`dc_offset_max_abs = 0.3766`), które zaburzało wyliczanie energii RMS i normalizację szczytową.
   - **Artefakty zapętlania (`np.tile`):** Powielanie krótkich nagrań w pętli powodowało skoki fazowe na stykach ramek. Sieć uczyła się rozpoznawać te sztuczne trzaski jako klikanie (`clicking`).
   - **Shortcut Learning w klasie `normal`:** Syntetyczny generator zdrowych dysków filtrował szum powyżej 2400 Hz, przez co sieć uczyła się odróżniać klasę `normal` po braku wysokich częstotliwości, a nie po cechach akustycznych.
   - **Spójność etykiet:** Wprowadzono zabezpieczenia uniemożliwiające przesunięcie indeksów klas w małych splitach.
3. **Długodystansowe testy obciążeniowe:**
   - Opracowano [stress_test.py](stress_test.py) oraz [stress_test_vram.py](stress_test_vram.py) do monitorowania zużycia pamięci, latencji p95 oraz błędów NaN/Inf w trakcie ciągłej inferencji przez 1h i 24h.
   - Opracowano [testEpok.py](testEpok.py) do weryfikacji wpływu liczby epok (20–60) na stabilność wag.
   - Sformułowano w [NOTES.md](NOTES.md) i wdrożono w [train_ensemble.py](train_ensemble.py) technikę **Seed Ensembling** (komitet modeli o różnych ziarnach losowości z uśrednianiem Softmax).

---

### Faza 3: Pełna Migracja na PyTorch i Akceleracja AMD ROCm (RX 9060 XT)
*(Identyfikator sesji: `75303e14`)*

1. **Przepisanie całego stosu na PyTorch (Commit `c45b9ff`):**
   - Opracowano klasę sieci konwolucyjnej `AcousticHddSentinelCNN` w [model_cnn.py](model_cnn.py) (1 206 756 parametrów), wspierającą zarówno Channel-Last, jak i Channel-First.
   - Zaimplementowano loader [dataset_pytorch.py](dataset_pytorch.py) (`HddNpyDataset`) oraz moduł treningowy [train_cnn.py](train_cnn.py).
   - **Purge TensorFlow:** Całkowicie odinstalowano pakiety TensorFlow i Keras ze środowiska, usuwając zbędne zależności i zwalniając pamięć dyskową.
   - Oczyszczono zbiór danych ze zduplikowanych nagrań (usunięto 21 plików przypisanych jednocześnie do `clicking` i `scratching`).
2. **Akceleracja sprzętowa AMD Radeon RX 9060 XT (RDNA 4 / `gfx1200`):**
   - Rozwiązano problem niekompatybilności standardowego ROCm 6.2 z nową architekturą ISA `gfx1200` (`invalid device function`).
   - Zainstalowano dedykowany moduł `amd-torch-device-gfx1200` w środowisku [`.venv_rocm`](.venv_rocm) ze sterownikami ROCm 7.2 / HIP.
   - **Wynik wydajnościowy:** Czas epoki treningowej skrócił się z ~16 s na CPU do **~1.9 s na GPU (ponad 8-krotne przyspieszenie)** przy wykorzystaniu 16 GB VRAM.

---

### Faza 4: Rozwiązanie Problemu Scratching/Beeping i Fine-Tuning w `CNN/TESTY/`
*(Identyfikatory sesji: `0c62a67c`, `3c821c7b`)*

1. **Wymóg izolacji:**
   - Stan bazowy zabezpieczono commitem `f508af8`. Wszystkie eksperymenty prowadzono w wydzielonym katalogu [CNN/TESTY/](CNN/TESTY/).
2. **Identyfikacja przyczyn słabej wykrywalności `scratching` ($F1 = 0.00$) i `beeping` ($F1 \approx 0.20$):**
   - *Label Pollution:* 82 pliki z Northwind miały utracone oryginalne nazwy (pliki `(1).wav`–`(82).wav`). Heurystyki spektralne błędnie oznaczały klikające głowice jako tarcie talerza lub piski.
   - *Zafałszowany plik:* `Audio_Player_57.wav` okazał się być identycznym duplikatem klikającego dysku Samsung SP8002, oznaczonym na sztywno jako `beeping`.
   - *Z-score na szumie:* Bardzo ciche nagrania ($RMS < 0.002$) powodowały sztuczne rozciąganie szumu mikrofonu na cały zakres dynamiczny.
3. **Wykonane usprawnienia:**
   - **Scraping bazy Northwind:** Pobrano z serwera Northwind wszystkie 82 autentyczne pliki MP3 z oryginalnymi nazwami laboratoryjnymi.
   - **Nowe próbki audio:** Za pomocą `yt-dlp` pozyskano 11 nowych, laboratoryjnych nagrań tarcia talerzy i blokady wrzeciona (m.in. Stellar Data Recovery, Head Crash HDD, Seized Spindle). Baza wzrosła do **134 unikalnych nagrań**.
   - **Enkoder [CNN/TESTY/encoder_testy.py](CNN/TESTY/encoder_testy.py):**
     - Odrzucanie ramek o $RMS < 0.002$ (eliminacja szumu tła),
     - Pitch-shift ($\pm 1.5$ półtonu) i 50% overlap **wyłącznie w zbiorze `train`** (**1 389 idealnie zbalansowanych tensorów**: ~350 per klasa),
     - **Zbiór `test` w 100% czysty** (117 tensorów, krok 3.0s, 0% overlapu, bez sztucznej augmentacji).
   - **Trening [CNN/TESTY/train_testy.py](CNN/TESTY/train_testy.py):**
     - SpecAugment w locie (Time Masking $\le 24$, Freq Masking $\le 16$),
     - Label Smoothing CrossEntropy ($0.03$),
     - Optymalizator AdamW z `weight_decay = 1e-4` i dynamiczny scheduler `ReduceLROnPlateau`,
     - Seed Ensembling – optymalny model wyłoniony na ziarnie `seed = 777`.

---

## 3. Zestawienie Wyników Ewaluacji (Przed i Po)

### Porównanie Metryki F1-Score na Niezależnym Zbiorze Testowym:

| Klasa Uszkodzenia | Stan Początkowy (TF/PyTorch Base) | **Stan Końcowy (CNN/TESTY/)** | Wzrost | Warunek ($\ge 0.70$) |
| :--- | :---: | :---: | :---: | :---: |
| **`beeping`** (stiction / zablokowane wrzeciono) | 0.2000 (20.0%) | **0.8636 (86.4%)** | **+66.4 p.p.** | **SPEŁNIONY** |
| **`clicking`** (uderzanie głowic / servo failure) | 0.7200 (72.0%) | **0.7848 (78.5%)** | **+6.5 p.p.** | **SPEŁNIONY** |
| **`scratching`** (tarcie głowic o talerze) | 0.0000 (0.0%) | **0.7368 (73.7%)** | **+73.7 p.p.** | **SPEŁNIONY** |
| **`normal`** (dysk sprawny) | 1.0000 (100.0%) | **1.0000 (100.0%)** | 0.0 p.p. | **SPEŁNIONY** |
| **Średnia Makro (Macro F1)** | **0.4800** | **0.8463** | **+36.6 p.p.** | **SPEŁNIONY** |
| **Dokładność ogólna (Accuracy)** | **~67.0%** | **83.76%** | **+16.8 p.p.** | — |

### Weryfikacja Braku Przeuczenia (Generalization Check):
- **Dokładność treningowa:** $\approx 85.8\%$
- **Dokładność testowa:** $83.76\%$
- **Luka generalizacyjna (Generalization Gap):** zaledwie **$2.04$ p.p.** – model posiada wysoką zdolność generalizacji i nie wykazuje przeuczenia pod zbiór treningowy.

---

## 4. Mapa Kluczowych Plików i Modułów

```
AcousticHddSentinel/
├── kontekst.md                        # Niniejsze całościowe podsumowanie projektu
├── fineTuning.md                      # Szczegółowa dokumentacja prac i metryk w CNN/TESTY/
├── AUDIT_REVIEW_BUNDLE.md             # Historyczny pakiet audytowy dla zewnętrznych modeli LLM
├── NOTES.md                           # Notatki badawcze (koncepcja komitetu Seed Ensembling)
├── AGENTS.md                          # Wytyczne i niezmienniki techniczne dla agentów AI
├── README.md                          # Główna dokumentacja projektu
├── requirements.txt                   # Zależności Python (PyTorch, Librosa, Scikit-learn)
│
├── model_cnn.py                       # Architektura 2D CNN w PyTorch (AcousticHddSentinelCNN)
├── dataset_pytorch.py                 # PyTorch Dataset i DataLoader dla tensorów .npy
├── train_cnn.py                       # Główny pipeline treningowy PyTorch z akceleracją ROCm
├── train_ensemble.py                  # Trening komitetu modeli o różnych seedach losowości
├── hdd_audio_encoder.py               # Enkoder audio do tensorów zgodny z ADR-001
├── prepare_audio_dataset.py           # Narzędzie kategoryzacji nagrań audio
├── profile_normalization_quality.py   # Profiler jakości normalizacji DSP
│
├── stress_test.py                     # Skrypt 1h/24h stress testu inferencji
├── stress_test_vram.py                # Skrypt monitoringu pamięci VRAM GPU
├── testEpok.py                        # Eksperyment porównawczy dla 20-60 epok
├── verify_cnn_model.py                # Testy weryfikacyjne poprawności modelu PyTorch
│
├── training_output/                   # Poprzedni bazowy model produkcyjny
│   └── best_hdd_cnn_model.pt          # Wagi modelu bazowego
│
├── CNN/TESTY/                         # Nowe, zoptymalizowane środowisko eksperymentalne
│   ├── raw_northwind/                 # 82 oryginalne pliki MP3 z serwera Northwind
│   ├── raw_additional/                # 11 nowo pozyskanych nagrań awarii (yt-dlp)
│   ├── dataset_categorized/           # 134 unikalne nagrania źródłowe podzielone na klasy
│   ├── dataset_encoded/               # Wygenerowane tensory (1389 train, 92 val, 117 test)
│   ├── encoder_testy.py               # Enkoder z filtracją ciszy i zbalansowaną augmentacją
│   ├── dataset_testy.py               # Dataset PyTorch z RAM-preloading i SpecAugment
│   ├── model_testy.py                 # Architektura CNN i moduł komitetu modeli
│   ├── train_testy.py                 # Pipeline treningowy dla komitetu modeli
│   └── training_output/               # Wyjściowe wagi i raporty nowego modelu:
│       ├── best_hdd_cnn_model.pt      # Wagi najlepszego modelu (F1 > 0.70 dla wszystkich klas)
│       ├── classification_report.txt  # Pełny raport metryk per klasa
│       ├── confusion_matrix.png       # Wykres macierzy pomyłek
│       └── test_evaluation_summary.json # Podsumowanie metryk w formacie JSON
│
└── docx/adr/
    └── adr.md                         # ADR-001: Architektura DSP i niezmienniki techniczne
```

---

## 5. Historia Istotnych Commitów Git

- `f508af8` — *Save current project state before CNN/TESTY experiments* (zabezpieczenie kodu przed optymalizacją w `CNN/TESTY/`).
- `c45b9ff` — *feat: complete migration from TensorFlow to PyTorch, eliminate data leakage and purge TF* (pełne usunięcie TensorFlow, wdrożenie PyTorch, eliminacja zdublowanych plików audio, konfiguracja ROCm dla RX 9060 XT).
- `16031f4` — *Zapis stanu projektu: bazowy model CNN, stress test i skrypty weryfikacyjne*.
- `876d65a` — *feat(cnn): implement 2D CNN architecture, training pipeline, and AGY workspace setup*.
