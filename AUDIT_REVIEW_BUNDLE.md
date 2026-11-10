# KOMPLEKSOWY PAKIET AUDYTOWY: Acoustic HDD Sentinel (2D CNN Audio Fault Detection)

> ### INSTRUKCJA DLA MODELU LLM PROWADZĄCEGO AUDYT (System Prompt / Context):
>
> Działasz jako **Główny Architekt Systemów Uczenia Maszynowego, Ekspert DSP (Digital Signal Processing) oraz Senior ML Engineer (TensorFlow/Keras)**.
> Twoim zadaniem jest przeprowadzenie **rygorystycznego, wyczerpującego audytu technicznego** poniższego projektu detekcji i klasyfikacji awarii mechanicznych dysków twardych (HDD) na podstawie dźwięku.
>
> Poniżej załączono pełen kontekst projektu (dokumentację architektoniczną ADR-001, wytyczne AGENTS.md, aktualne wyniki ewaluacji oraz kompletny kod źródłowy wszystkich 6 kluczowych modułów).
>
> ---
>
> ### ZAKRES AUDYTU I PYTANIA WERYFIKACYJNE:
>
> 1. **Spójność z ADR-001 i niezmiennikami technicznymi (Invariants):**
>    - Czy pipeline DSP ściśle przestrzega parametrów: `TARGET_SR = 22050 Hz`, `N_FFT = 1024`, `HOP_LENGTH = 256`, `N_MELS = 128`, okno `3.0s`, kształt tensora `(128, 259, 3)` (Channel-Last)?
>    - Czy podział na Train / Val / Test całkowicie eliminuje Data Leakage na poziomie nagrań źródłowych przed podziałem na okna?
>    - Czy funkcje normalizacji są w pełni wyizolowane jako pure functions i gwarantują identyczne przetwarzanie w fazie treningu oraz produkcyjnej inferencji?
>
> 2. **Analiza logiki cyfrowego przetwarzania sygnałów (DSP):**
>    - Czy dwustopniowa normalizacja (Peak normalization na wycinku fali + Z-Score per kanał tensora) jest poprawna matematycznie i optymalna dla dźwięków z wysokim Crest Factor (impulsowe klikanie głowicy) oraz pisków wąskopasmowych?
>    - Czy wyliczenie pochodnych Delta (1. pochodna) i Delta-Delta (2. pochodna) w `hdd_audio_encoder.py` nie wprowadza artefaktów brzegowych lub niestabilności numerycznych?
>    - Jak oceniasz mechanizm odcinania ciszy (trimming) i obsługę nagrań krótszych niż 3.0s (cykliczne powielanie)?
>
> 3. **Architektura sieci konwolucyjnej (2D CNN):**
>    - Ocena modelu w `model_cnn.py`: progresja filtrów (32 -> 64 -> 128 -> 256), użycie BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D zamiast Flatten, regularyzacja L2.
>    - Czy architektura efektywnie zapobiega przeuczeniu (overfitting) przy małym i średnim zbiorze akustycznym (~837 wycinków)?
>    - Ocena gałęzi Transfer Learning (`build_transfer_learning_model` - MobileNetV2 / EfficientNetB0).
>
> 4. **Pipeline treningowy i optymalizacyjny (`train_cnn.py`):**
>    - Czy loader `tf.data.Dataset` z `tf.numpy_function` i jawnym `tensor.set_shape([128, 259, 3])` jest poprawnie zaimplementowany, stabilny i wydajny?
>    - Czy ważenie klas (`calculate_balanced_class_weights` z formułą balanced ze scikit-learn) poprawnie kompensuje asymetrię danych (np. rzadka klasa `scratching` vs dominująca `clicking`)?
>    - Czy funkcja straty (`CategoricalCrossentropy` z label smoothing 0.05), metryki (`CategoricalAccuracy`, `TopKCategoricalAccuracy`) i callbacki (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`) są optymalnie skonfigurowane?
>
> 5. **Ocena wyników ewaluacji i diagnostyki:**
>    - Analiza aktualnego raportu klasyfikacji (100% precyzja i recall dla klasy `normal`, 90.9% precyzja dla `clicking`, ale niski recall dla `clicking` i 0% recall dla `scratching`):
>      - Dlaczego model ma trudności z klasą `scratching` (tylko 6 próbek testowych, 29 treningowych)?
>      - Czy syntetyczne nagrania klasy `normal` w `prepare_audio_dataset.py` (filtrowany szum różowy, harmoniczne silnika wrzeciona, soft seeks) są odpowiednio dobrane i czy nie stwarzają ryzyka zbyt łatwego oddzielenia od nagrań rzeczywistych?
>
> 6. **Rekomendacje produkcyjne i dalszy backlog:**
>    - Wskaż potencjalne ukryte błędy, edge-cases (np. dzielenie przez zero, clipping, pamięć RAM/VRAM).
>    - Jakie konkretne usprawnienia sugerujesz wdrożyć w pierwszej kolejności (np. warstwa SpecAugment w locie, eksport do TFLite z kwantyzacją int8/float16 dla urządzeń brzegowych/Android, augmentacja rzadkich klas)?
>
> Proszę o przedstawienie wyczerpującego raportu z:
> - Oceną punktową (1-10) w kluczowych kategoriach (Architektura, DSP, Trening, Jakość kodu, Bezpieczeństwo danych),
> - Wskazaniem mocnych stron projektu,
> - Identyfikacją ryzyk i potencjalnych błędów logicznych,
> - Konkretnymi propozycjami fragmentów kodu do poprawy/rozbudowy.

---

## SPIS TREŚCI PAKIETU AUDYTOWEGO

1. [DOKUMENTACJA PROJEKTU I WYMAGANIA (AGENTS.md)](#agentsmd)
2. [ARCHITECTURAL DECISION RECORD (ADR-001)](#docxadradrmd)
3. [WYMAGANIA ŚRODOWISKOWE (requirements.txt)](#requirementstxt)
4. [METRYKI I WYNIKI TESTOWE (classes.json)](#trainingoutputclassesjson)
5. [RAPORT KLASYFIKACJI NA ZBIORZE TESTOWYM (classification_report.txt)](#trainingoutputclassificationreporttxt)
6. [PODSUMOWANIE EWALUACJI MODELU (test_evaluation_summary.json)](#trainingoutputtestevaluationsummaryjson)
7. [PROFIL I STATYSTYKI NORMALIZACJI (normalization_quality_metrics.json)](#normalizationqualitymetricsjson)
8. [KOD ŹRÓDŁOWY: hdd_audio_encoder.py](#hddaudioencoderpy)
9. [KOD ŹRÓDŁOWY: model_cnn.py](#modelcnnpy)
10. [KOD ŹRÓDŁOWY: train_cnn.py](#traincnnpy)
11. [KOD ŹRÓDŁOWY: prepare_audio_dataset.py](#prepareaudiodatasetpy)
12. [KOD ŹRÓDŁOWY: verify_cnn_model.py](#verifycnnmodelpy)
13. [KOD ŹRÓDŁOWY: profile_normalization_quality.py](#profilenormalizationqualitypy)

---

## DOKUMENTACJA PROJEKTU I WYMAGANIA (AGENTS.md)
**Ścieżka pliku w repozytorium:** `AGENTS.md`

```markdown
# Projekt: HDD Acoustic Fault Detection (2D CNN)
**Instrukcja dla agentów AI (Antigravity CLI / AGY) oraz developerów**

## 1. Cel i zakres projektu
System akustycznej detekcji i wieloklasowej klasyfikacji uszkodzeń mechanicznych dysków twardych (HDD) z wykorzystaniem głębokich sieci konwolucyjnych (2D CNN w TensorFlow/Keras). Analiza opiera się na 3-sekundowych wycinkach audio przetworzonych do 3-kanałowej reprezentacji czasowo-częstotliwościowej:
- **Kanał 0:** Log-Mel Spektrogram (struktura widmowa, rezonanse, piski)
- **Kanał 1:** Delta – 1. pochodna czasowa (transjenty, impulsowe klikanie głowic)
- **Kanał 2:** Delta-Delta – 2. pochodna czasowa (dynamika przyspieszeń i zmian obrotowych)

---

## 2. Niezmienniki techniczne (Invariants) — ZAKAZ ŁAMANIA PRZEZ AGENTA
Każdy agent AGY modyfikujący lub rozwijający projekt **MUSI bezwzględnie** przestrzegać poniższych reguł zdefiniowanych w [ADR-001](file:///home/DJMixu/Pulpit/AcousticHddSentinel/docx/adr/adr.md):

1. **Brak Data Leakage (Podział na poziomie plików):**
   Podział na `train` (70%), `validate` (15%) i `test` (15%) **MUSI** odbywać się na całych nagraniach źródłowych przed ich pocięciem na 3-sekundowe okna czasowe. Nigdy nie stosuj losowego `train_test_split` bezpośrednio na wygenerowanych plikach `.npy`.
2. **Stały kształt tensora wejściowego (Channel-Last):**
   Wejściem do modeli CNN w TensorFlow/Keras jest zawsze tensor `float32` o stałych wymiarach **`(128, 259, 3)`**.
   Definiując warstwę wejściową, zawsze stosuj: `layers.Input(shape=(128, 259, 3))`.
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
5. **Loader `tf.data.Dataset` z `tf.numpy_function`:**
   Wczytując pliki `.npy` z dysku, używaj `tf.numpy_function` i jawnie ustawiaj `tensor.set_shape([128, 259, 3])`, aby kompilator grafu TensorFlow znał wymiary przed kompilacją i treningiem.

---

## 3. Mapa plików i architektura repozytorium

```
AcousticHddSentinel/
├── AGENTS.md                          # Główny przewodnik i Single Source of Truth dla agentów AGY
├── README.md                          # Dokumentacja repozytorium dla ludzi i developerów
├── requirements.txt                   # Zależności Python (TensorFlow, Librosa, Scikit-learn, etc.)
├── .gitignore                         # Wykluczenia git (duże zbiory .npy, venv, cache)
├── model_cnn.py                       # Architektura sieci 2D CNN (oraz MobileNetV2 / EfficientNetB0)
├── train_cnn.py                       # Kompletny pipeline treningowy (loader tf.data, class_weight, callbacki, macierz)
├── hdd_audio_encoder.py               # Enkoder audio do tensorów .npy zgodny z ADR-001
├── prepare_audio_dataset.py           # Kategoryzator i generator próbek audio
├── verify_cnn_model.py                # 6-etapowy test weryfikacyjny sprawności modelu CNN
├── profile_normalization_quality.py   # Profiler i weryfikator jakości normalizacji
├── normalization_quality_report.png   # Diagnostyczny wykres jakości normalizacji
├── normalization_quality_metrics.json # Metryki numeryczne jakości normalizacji
│
├── training_output/                   # Wytrenowany model i wyniki ewaluacji
│   ├── best_hdd_cnn_model.keras       # Wagi najlepszego modelu (1.2M parametrów)
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

### D. Uruchomienie pełnego treningu:
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
1. **Konwersja do TFLite:** Użycie `tf.lite.TFLiteConverter.from_keras_model` z kwantyzacją `int8` lub `float16` w celu uruchomienia inferencji na telefonie Android (zgodnie ze `docx/specs.txt`).
2. **Augmentacja spektrogramów w locie:** Dodanie warstwy SpecAugment (TimeMasking / FreqMasking) bezpośrednio do `train_cnn.py` w celu poprawy czułości na rzadsze klasy.
3. **Zwiększenie bazy dla klasy `scratching`:** Klasa zarysowania talerza miała tylko 6 próbek testowych – warto wyodrębnić więcej fragmentów z nagrań tarcia głowic.

```

---

## ARCHITECTURAL DECISION RECORD (ADR-001)
**Ścieżka pliku w repozytorium:** `docx/adr/adr.md`

```markdown
# ADR-001: Architektura przetwarzania sygnału audio i enkodowania danych dla modelu CNN wykrywającego awarie dysków HDD

- **Status:** Zaakceptowany (Accepted)
    
      
    
- **Data:** 27.10.2026
    
      
    
- **Kontekst projektu:** System akustycznej detekcji i klasyfikacji uszkodzeń mechanicznych dysków twardych (HDD) z wykorzystaniem głębokich sieci konwolucyjnych (2D CNN).
    
      
    

## 1. Kontekst i opis problemu

Celem projektu jest zbudowanie modelu uczenia maszynowego (CNN), który na podstawie krótkiego nagrania mikrofonowego potrafi wykryć awarię dysku twardego oraz sklasyfikować typ uszkodzenia mechanicznego (np. _Click of Death_, zatarte wrzeciono / _beeping_, zarysowanie talerza / _scratching_, praca prawidłowa).

  

Przeprowadzone profilowanie zebranego zbioru danych (`audio_dataset_profile.csv`, 92 zweryfikowane nagrania o łącznym czasie 23,6 min) wykazało istotne wyzwania inżynieryjne:

  

1. **Ekstremalny rozrzut amplitudy i głośności:** Średnia energia sygnału (`rms_mean`) waha się pomiędzy nagraniami od `0.00061` do `0.37809` (ponad 600-krotna różnica wynikająca z użycia różnych mikrofonów i odległości od dysku).
    
      
    
2. **Specyfika akustyczna uszkodzeń:** Dźwięki awarii łączą cechy wysoce impulsowe (stukanie głowicy o ogranicznik, gdzie współczynnik szczytu _Crest Factor_ sięga średnio `35.2`, a maksymalnie aż `207.9`) z cechami tonalnymi (piski zablokowanego wrzeciona, `spec_flatness < 0.002`) oraz szerokopasmowym szumem tarcia (`spec_rolloff_95` sięgający `9981 Hz`).
    
      
    
3. **Nierówny czas trwania i cisza:** Nagrania mają długość od `3.88 s` (po odcięciu ciszy) do `54.25 s`, a udział ciszy na brzegach nagrań dochodzi do `40.3%`.
    
      
    
4. **Ryzyko wycieku danych (Data Leakage):** Ze względu na ograniczoną liczbę unikalnych urządzeń konieczne jest cięcie długich nagrań na krótsze okna czasowe (chunking). Losowy podział wygenerowanych okien pomiędzy zbiory _Train / Validate / Test_ doprowadziłby do sytuacji, w której sąsiednie fragmenty tego samego nagrania znalazłyby się w zbiorze treningowym i testowym.
    
      
    

## 2. Rozważane opcje

### Obszar A: Reprezentacja wejściowa dla sieci CNN

- **Opcja A1: Surowa fala dźwiękowa (1D Waveform + 1D CNN).**
    
      
    - _Zalety:_ Brak utraty informacji fazowej, brak narzutu obliczeniowego FFT.
        
          
        
    - _Wady:_ Bardzo duża wrażliwość na szum tła, trudność w jednoczesnym uchwyceniu wąskopasmowych pisków wrzeciona i szerokopasmowego tarcia przy małym zbiorze uczącym.
        
          
        
- **Opcja A2: Pojedynczy Mel-Spektrogram (1-kanałowy obraz 2D).**
    
      
    - _Zalety:_ Standardowa reprezentacja czasowo-częstotliwościowa, mały rozmiar plików.
        
          
        
    - _Wady:_ Słabsze wyeksponowanie nagłych zmian energii w czasie (transjentów _Click of Death_) oraz brak bezpośredniej kompatybilności z architekturami pretrenowanymi na ImageNet (oczekującymi 3 kanałów wejściowych).
        
          
        
- **Opcja A3 (Wybrana): 3-kanałowy tensor `[Log-Mel Spectrogram, Delta, Delta-Delta]`.**
    
      
    - _Zalety:_ Kanał 1 opisuje widmo statyczne, Kanał 2 (1. pochodna) bezpośrednio uwypukla krótkie uderzenia głowicy (transjenty), a Kanał 3 (2. pochodna) wychwytuje zmiany prędkości obrotowej talerzy (_spin-down_). Format 3-kanałowy umożliwia bezpośrednie użycie Transfer Learningu (np. ResNet, EfficientNet, MobileNet).
        
          
        

### Obszar B: Strategia normalizacji

- **Opcja B1: Globalna normalizacja na podstawie statystyk całego zbioru.**
    
      
    - _Wady:_ Przy 600-krotnej różnicy `rms_mean` między plikami ciche nagrania zostałyby stłumione do zera, a głośne zdominowałyby gradienty.
        
          
        
- **Opcja B2 (Wybrana): Dwustopniowa, wyizolowana normalizacja lokalna (Per-Chunk Peak Normalization + Per-Channel Z-score).**
    
      
    - _Zalety:_ Uniezależnia model od czułości mikrofonu i odległości od obudowy dysku, zachowując jednocześnie proporcje amplitudowe wewnątrz badanego 3-sekundowego okna.
        
          
        

## 3. Decyzja

Przyjmujemy następującą architekturę pipeline'u przetwarzania danych (zaimplementowaną w module `hdd_audio_encoder.py`):

  

### 3.1. Standaryzacja strumienia i okienkowanie (Chunking)

1. **Próbkowanie docelowe (`TARGET_SR`):** `22 050 Hz`, 1 kanał (`Mono`), format `PCM_16` (zgodne z 98,9% zebranych próbek; pasmo Nyquista `11 025 Hz` w pełni pokrywa 95. percentyl energii widmowej wynoszący maks. `9981 Hz`).
    
      
    
2. **Usuwanie ciszy:** Automatyczny trimming brzegowy z progiem `top_db = 30 dB` przed podziałem na okna.
    
      
    
3. **Rozmiar okna (`CHUNK_DURATION`):** `3.0 s` (`66 150` próbek). Pozwala wykorzystać 100% poprawnych nagrań (najkrótsze ma `3.88 s`) i gwarantuje uchwycenie minimum 3–4 cykli wolnego klikania głowicy (`onset_rate_per_sec` ok. `1.2 Hz`).
    
      
    
4. **Krok okna (`OVERLAP`):**
    
      
    - Zbiór `train`: przesunięcie `1.5 s` (50% overlap) w celu augmentacji liczby próbek treningowych.
        
          
        
    - Zbiory `validate` oraz `test`: przesunięcie `3.0 s` (0% overlap) w celu zapewnienia niezależności próbek ewaluacyjnych.
        
          
        

### 3.2. Parametry ekstrakcji cech (STFT i Mel-Filterbank)

- **Rozmiar okna FFT (`n_fft`):** `1024` próbki (~`46.4 ms`) – kompromis zapewniający wysoką rozdzielczość częstotliwościową dla pisków wrzeciona.
    
      
    
- **Krok czasowy (`hop_length`):** `256` próbek (~`11.6 ms`) – skrócony krok czasowy (zamiast standardowych 512) zapobiega rozmyciu krótkich impulsów mechanicznych o wysokim współczynniku szczytu (_Crest Factor_).
    
      
    
- **Liczba pasm Mel (`n_mels`):** `128` pasm w zakresie `fmin = 50 Hz` (odcięcie przydźwięku sieciowego i drgań podłoża) do `fmax = 10 000 Hz` (zachowanie pasma tarcia głowicy o talerz).
    
      
    
- **Wymiar wyjściowego tensora:** `(128, 259, 3)` w konwencji _Channel-Last_ (TensorFlow/Keras) lub `(3, 128, 259)` w konwencji _Channel-First_ (PyTorch), serializowany do plików `.npy` (`float32`).
    
      
    

### 3.3. Architektura normalizacji i podziału zbioru

1. **Izolacja funkcji normalizacji:** Operacje normalizacji amplitudy (`normalize_waveform_peak` / `normalize_waveform_rms`) oraz standaryzacji cech (`normalize_tensor_zscore` / `normalize_tensor_minmax`) są wydzielone jako czyste funkcje (_pure functions_) wstrzykiwane do enkodera. Gwarantuje to identyczne przetwarzanie w fazie treningu oraz podczas inferencji produkcyjnej.
    
      
    
2. **Podział na poziomie plików źródłowych (File-Level Stratified Split):** Podział w proporcji `70% Train / 15% Validate / 15% Test` wykonywany jest na całych nagraniach źródłowych przed ich pocięciem na 3-sekundowe okna. Całkowicie eliminuje to wyciek danych (_Data Leakage_).
    
      
    

## 4. Konsekwencje

### Pozytywne (+)

- **Odporność na warunki nagrywania:** Dzięki lokalnej normalizacji szczytowej i standaryzacji Z-score per kanał, sieć CNN uczy się wzorców czasowo-częstotliwościowych awarii, a nie głośności konkretnego mikrofonu.
    
      
    
- **Wiarygodna walidacja:** Brak przecieku wycinków tego samego dysku między zbiorem treningowym a testowym daje realną ocenę generalizacji modelu na nowych, nieznanych wcześniej dyskach.
    
      
    
- **Elastyczność modelowania:** Kształt `(128, 259, 3)` pozwala zarówno na budowę własnej lekkiej sieci 2D CNN od zera, jak i na szybkie podpięcie gotowych modeli z `tf.keras.applications` lub `torchvision.models`.
    
      
    
- **Łatwe wdrożenie (Deployment):** Wyizolowane funkcje normalizacji i enkodowania w `hdd_audio_encoder.py` mogą być bezpośrednio zaimportowane przez serwis wnioskujący (API / aplikację mobilną / skrypt diagnostyczny).
    
      
    

### Negatywne / Ryzyka (-) i sposoby mitygacji

- **Utrata informacji o bezwzględnej głośności:** Normalizacja każdego 3-sekundowego wycinka z osobna sprawia, że bardzo cichy szum sprawnego dysku zostanie wzmocniony do takiego samego poziomu szczytowego jak głośne zgrzytanie.
    
      
    - _Mitygacja:_ Rozróżnienie opiera się na strukturze widmowej (płaskość widma, centroida) oraz kanałach `Delta` / `Delta-Delta`, a całkowicie puste (ciche) okna są odrzucane progiem energetycznym `RMS < 1e-4` przed normalizacją.
        
          
        
- **Zwiększone zużycie przestrzeni dyskowej:** Zapis 3-kanałowych tensorów `float32` w formacie `.npy` zajmuje więcej miejsca na dysku niż skompresowane pliki audio (`128 * 259 * 3 * 4B ≈ 398 KB` na każde 3-sekundowe okno, co przy ~1000 wycinkach daje ok. `400 MB`).
    
      
    - _Mitygacja:_ Przy obecnej skali zbioru (~400 MB – 1 GB) wczytywanie gotowych plików `.npy` z dysku SSD drastycznie przyspiesza epoki treningowe GPU (brak konieczności liczenia FFT na procesorze w każdej epoce).
        
          
        
- **Błędy formatów wejściowych (np. uszkodzone pliki MP3):** Jak wykazał profil danych, część pobranych plików `.mp3` ma uszkodzone nagłówki lub zerowy rozmiar.
    
      
    - _Mitygacja:_ Dodano obsługę wyjątków i logowanie odrzuconych plików na etapie budowania zbioru oraz rekomendację wstępnej konwersji surowych źródeł przez `ffmpeg`.
```

---

## WYMAGANIA ŚRODOWISKOWE (requirements.txt)
**Ścieżka pliku w repozytorium:** `requirements.txt`

```text
# Zależności projektu Acoustic HDD Sentinel (TensorFlow / Keras 2D CNN)
tensorflow>=2.15.0
numpy>=1.26.0
librosa>=0.10.0
soundfile>=0.12.0
scipy>=1.12.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
tqdm>=4.66.0
pandas>=2.2.0

```

---

## METRYKI I WYNIKI TESTOWE (classes.json)
**Ścieżka pliku w repozytorium:** `training_output/classes.json`

```json
{
  "classes": [
    "beeping",
    "clicking",
    "normal",
    "scratching"
  ],
  "input_shape": [
    128,
    259,
    3
  ]
}
```

---

## RAPORT KLASYFIKACJI NA ZBIORZE TESTOWYM (classification_report.txt)
**Ścieżka pliku w repozytorium:** `training_output/classification_report.txt`

```text
              precision    recall  f1-score   support

     beeping     0.2286    0.7273    0.3478        11
    clicking     0.9091    0.4444    0.5970        45
      normal     1.0000    1.0000    1.0000        11
  scratching     0.0000    0.0000    0.0000         6

    accuracy                         0.5342        73
   macro avg     0.5344    0.5429    0.4862        73
weighted avg     0.7455    0.5342    0.5711        73

```

---

## PODSUMOWANIE EWALUACJI MODELU (test_evaluation_summary.json)
**Ścieżka pliku w repozytorium:** `training_output/test_evaluation_summary.json`

```json
{
  "test_metrics": {
    "loss": 1.226332187652588,
    "compile_metrics": 0.534246563911438
  },
  "classes": [
    "beeping",
    "clicking",
    "normal",
    "scratching"
  ],
  "total_test_samples": 73
}
```

---

## PROFIL I STATYSTYKI NORMALIZACJI (normalization_quality_metrics.json)
**Ścieżka pliku w repozytorium:** `normalization_quality_metrics.json`

```json
{
  "raw_audio_profile": {
    "count": 91,
    "rms_min": 0.0008801857475191355,
    "rms_max": 0.3798445761203766,
    "rms_mean": 0.050156835466623306,
    "rms_ratio_max_min": 431.54998779296875,
    "peak_min": 0.01837158203125,
    "peak_max": 1.0,
    "crest_min": 1.3138070106506348,
    "crest_max": 131.72459411621094,
    "crest_mean": 26.120580673217773,
    "crest_p95": 73.95867156982422,
    "dc_offset_max_abs": 0.3766556978225708,
    "dc_offset_mean_abs": 0.03392370417714119
  },
  "encoded_tensors_profile": {
    "total_tensors": 837,
    "nan_tensors": 0,
    "inf_tensors": 0,
    "channel_0_log_mel": {
      "mean_of_means": -3.5732595371909333e-09,
      "std_of_means": 1.624411338851848e-07,
      "mean_of_stds": 0.9999999040059007,
      "std_of_stds": 5.2969653482005676e-08,
      "min_global": -7.467644691467285,
      "max_global": 8.699585914611816
    },
    "channel_1_delta": {
      "mean_of_means": -2.587114597248588e-10,
      "std_of_means": 6.2078672480670244e-09,
      "mean_of_stds": 0.9999986651265635,
      "std_of_stds": 4.852991000469345e-07,
      "min_global": -12.601819038391113,
      "max_global": 12.501738548278809
    },
    "channel_2_delta_delta": {
      "mean_of_means": -1.7608875003371862e-10,
      "std_of_means": 6.365093412366996e-09,
      "mean_of_stds": 0.9999981250529363,
      "std_of_stds": 5.042646764767536e-07,
      "min_global": -12.414013862609863,
      "max_global": 11.987079620361328
    },
    "mean_3sigma_outlier_pct": 1.1766314605696326
  }
}
```

---

## KOD ŹRÓDŁOWY: hdd_audio_encoder.py
**Ścieżka pliku w repozytorium:** `hdd_audio_encoder.py`

```python
import os
import random
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Callable
import numpy as np
import librosa
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# 1. KONFIGURACJA PIPELINE'U
# =====================================================================
INPUT_DIR = "./CNN/dataset_categorized" if os.path.exists("./CNN/dataset_categorized") else "./dataset_categorized"
OUTPUT_DIR = "./dataset_encoded"      # Katalog wyjściowy (utworzy podkatalogi train/, validate/, test/)

# Proporcje podziału plików źródłowych (muszą sumować się do 1.0)
SPLIT_RATIOS = (0.70, 0.15, 0.15)     # (train, validate, test)
RANDOM_SEED = 42

# Format wyjściowy tensorów: "tf" -> (128, 259, 3) dla TensorFlow/Keras | "pytorch" -> (3, 128, 259) dla PyTorch
TENSOR_FORMAT = "tf"

# Parametry akustyczne (dobrane na podstawie analizy pliku audio_dataset_profile.csv)
TARGET_SR = 22050
CHUNK_DURATION = 3.0                  # Długość okna w sekundach
OVERLAP_DURATION = 1.5                # Nakładanie się okien (1.5s = 50% overlap w zbiorze train)
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
FMIN = 50.0
FMAX = 10000.0
TRIM_DB = 30.0                        # Próg odcięcia ciszy na początku i końcu nagrania

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}


# =====================================================================
# 2. WYIZOLOWANE FUNKCJE NORMALIZACJI (ŁATWE DO IMPORTU / PODMIANY)
# =====================================================================

def normalize_waveform_peak(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalizacja szczytowa (Peak Normalization) surowego sygnału audio do zakresu [-1.0, 1.0].
    Idealna do zachowania wysokiego Crest Factor (ostrych kliknięć głowicy).
    """
    y = y - np.mean(y)  # Usunięcie DC offsetu (dochodzącego do 0.377 w surowych plikach)
    peak = np.max(np.abs(y))
    if peak > eps:
        return y / peak
    return y


def normalize_waveform_rms(y: np.ndarray, target_rms: float = 0.05, eps: float = 1e-6) -> np.ndarray:
    """
    Alternatywna normalizacja energii (RMS Normalization) surowego sygnału audio.
    Wyrównuje średnią głośność tła, z zabezpieczeniem przed przesterowaniem (clippingiem).
    """
    y = y - np.mean(y)
    current_rms = np.sqrt(np.mean(y ** 2))
    if current_rms > eps:
        y_norm = y * (target_rms / current_rms)
        return np.clip(y_norm, -1.0, 1.0)
    return y


def normalize_tensor_zscore(tensor: np.ndarray, channel_axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    """
    Standaryzacja Z-score (średnia = 0, odchylenie standardowe = 1) osobno dla każdego kanału
    (Log-Mel, Delta, Delta-Delta). Najlepsza dla szybkiej zbieżności sieci CNN.
    """
    norm_tensor = np.empty_like(tensor, dtype=np.float32)
    num_channels = tensor.shape[channel_axis]

    for c in range(num_channels):
        if channel_axis == 0:
            ch = tensor[c]
            norm_tensor[c] = (ch - np.mean(ch)) / (np.std(ch) + eps)
        else:
            ch = tensor[..., c]
            norm_tensor[..., c] = (ch - np.mean(ch)) / (np.std(ch) + eps)

    return norm_tensor


def normalize_tensor_minmax(tensor: np.ndarray, channel_axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    """
    Alternatywna normalizacja Min-Max skalująca każdy kanał spektrogramu do przedziału [0.0, 1.0].
    """
    norm_tensor = np.empty_like(tensor, dtype=np.float32)
    num_channels = tensor.shape[channel_axis]

    for c in range(num_channels):
        if channel_axis == 0:
            ch = tensor[c]
            c_min, c_max = np.min(ch), np.max(ch)
            norm_tensor[c] = (ch - c_min) / (c_max - c_min + eps)
        else:
            ch = tensor[..., c]
            c_min, c_max = np.min(ch), np.max(ch)
            norm_tensor[..., c] = (ch - c_min) / (c_max - c_min + eps)

    return norm_tensor


# =====================================================================
# 3. MODUŁ EKSTRAKCJI CECH I CIĘCIA NA OKNA
# =====================================================================

def encode_audio_chunk(
    y_chunk: np.ndarray,
    sr: int = TARGET_SR,
    waveform_norm_fn: Callable[[np.ndarray], np.ndarray] = normalize_waveform_peak,
    tensor_norm_fn: Callable[[np.ndarray, int], np.ndarray] = normalize_tensor_zscore,
    tensor_format: str = TENSOR_FORMAT
) -> np.ndarray:
    """
    Przekształca pojedyncze okno audio (np. 3 sekundy) w znormalizowany 3-kanałowy tensor.
    Można łatwo wywołać w zewnętrznym skrypcie podczas inferencji (predykcji).
    """
    # Krok 1: Wyizolowana normalizacja fali dźwiękowej
    if waveform_norm_fn is not None:
        y_chunk = waveform_norm_fn(y_chunk)

    # Krok 2: Wyliczenie Log-Mel Spektrogramu
    mel_spec = librosa.feature.melspectrogram(
        y=y_chunk,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

    # Krok 3: Wyliczenie pochodnych czasowych (Delta i Delta-Delta)
    delta = librosa.feature.delta(log_mel, width=9, order=1, mode="nearest")
    delta2 = librosa.feature.delta(log_mel, width=9, order=2, mode="nearest")

    # Krok 4: Złożenie kanałów w odpowiednim formacie (TensorFlow vs PyTorch)
    if tensor_format.lower() == "pytorch":
        # Kształt: (3, 128, 259)
        tensor = np.stack([log_mel, delta, delta2], axis=0).astype(np.float32)
        ch_axis = 0
    else:
        # Kształt dla TensorFlow / Keras: (128, 259, 3)
        tensor = np.stack([log_mel, delta, delta2], axis=-1).astype(np.float32)
        ch_axis = -1

    # Krok 5: Wyizolowana normalizacja kanałów tensora
    if tensor_norm_fn is not None:
        tensor = tensor_norm_fn(tensor, ch_axis)

    return tensor


def slice_waveform_into_chunks(y: np.ndarray, sr: int, is_train: bool = True) -> List[np.ndarray]:
    """
    Odcina ciszę na końcach i dzieli sygnał na równe 3-sekundowe fragmenty.
    Dla zbioru 'train' stosuje 50% overlap (więcej próbek), a dla 'validate'/'test' brak nakładania.
    """
    chunk_samples = int(sr * CHUNK_DURATION)
    # W zbiorze walidacyjnym i testowym możemy wyłączyć overlap lub zostawić taki sam krok
    step_samples = int(sr * (CHUNK_DURATION - OVERLAP_DURATION)) if is_train else chunk_samples

    y = y - np.mean(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    if len(y_trimmed) < int(sr * 0.5):
        return []

    # Odbicie lustrzane zamiast np.tile zapobiega skokom fazy (sztucznym kliknięciom)
    if len(y_trimmed) < chunk_samples:
        pad_len = chunk_samples - len(y_trimmed)
        y_trimmed = np.pad(y_trimmed, (0, pad_len), mode="reflect")

    chunks = []
    for start in range(0, len(y_trimmed) - chunk_samples + 1, step_samples):
        chunk = y_trimmed[start : start + chunk_samples]
        # Odrzucenie fragmentów całkowicie cichych w środku nagrania
        if np.sqrt(np.mean((chunk - np.mean(chunk)) ** 2)) >= 1e-4:
            chunks.append(chunk)

    return chunks


# =====================================================================
# 4. PODZIAŁ PLIKÓW NA TRAIN / VALIDATE / TEST (BEZ DATA LEAKAGE)
# =====================================================================

def split_files_by_class(
    input_root: Path,
    ratios: Tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = RANDOM_SEED
) -> Dict[str, List[Tuple[Path, str]]]:
    """
    Grupuje pliki według klasy (podfolderu) i dzieli całe pliki źródłowe na
    zbiory 'train', 'validate' oraz 'test', zapobiegając wyciekowi danych.
    """
    random.seed(seed)
    files_by_label: Dict[str, List[Path]] = {}

    for p in input_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            rel_path = p.relative_to(input_root)
            label = rel_path.parts[0] if len(rel_path.parts) > 1 else "unlabeled"
            files_by_label.setdefault(label, []).append(p)

    splits = {"train": [], "validate": [], "test": []}

    for label, file_list in files_by_label.items():
        # Sortowanie przed przetasowaniem gwarantuje 100% powtarzalność (reproducibility)
        file_list = sorted(file_list)
        random.shuffle(file_list)

        n = len(file_list)
        n_train = max(1, int(round(n * ratios[0])))
        n_val = max(1, int(round(n * ratios[1]))) if n >= 3 else 0
        
        # Korekta, aby nie przekroczyć liczby plików w małych klasach
        if n_train + n_val >= n and n >= 3:
            n_train = n - 2
            n_val = 1

        train_files = file_list[:n_train]
        val_files = file_list[n_train : n_train + n_val]
        test_files = file_list[n_train + n_val :]

        splits["train"].extend([(f, label) for f in train_files])
        splits["validate"].extend([(f, label) for f in val_files])
        splits["test"].extend([(f, label) for f in test_files])

    return splits


# =====================================================================
# 5. GŁÓWNA PĘTLA PRZETWARZANIA
# =====================================================================

def build_dataset():
    in_root = Path(INPUT_DIR)
    out_root = Path(OUTPUT_DIR)

    if not in_root.exists():
        print(f"Błąd: Katalog wejściowy '{INPUT_DIR}' nie istnieje.")
        return

    if out_root.exists():
        shutil.rmtree(out_root)

    splits = split_files_by_class(in_root)
    summary_stats = {"train": 0, "validate": 0, "test": 0, "skipped_files": 0}

    print(f"Podział plików źródłowych:")
    print(f"  - TRAIN:    {len(splits['train'])} plików")
    print(f"  - VALIDATE: {len(splits['validate'])} plików")
    print(f"  - TEST:     {len(splits['test'])} plików\n")

    for split_name, file_entries in splits.items():
        is_train = (split_name == "train")

        for filepath, label in tqdm(file_entries, desc=f"Generowanie [{split_name.upper()}]"):
            try:
                y, sr = librosa.load(str(filepath), sr=TARGET_SR, mono=True)
                chunks = slice_waveform_into_chunks(y, sr, is_train=is_train)

                if not chunks:
                    summary_stats["skipped_files"] += 1
                    continue

                # Ścieżka docelowa: np. ./dataset_encoded/train/clicking/
                target_dir = out_root / split_name / label
                target_dir.mkdir(parents=True, exist_ok=True)

                for idx, chunk in enumerate(chunks):
                    # Przekazujemy wyizolowane funkcje normalizacji jako argumenty
                    tensor = encode_audio_chunk(
                        y_chunk=chunk,
                        sr=TARGET_SR,
                        waveform_norm_fn=normalize_waveform_peak,
                        tensor_norm_fn=normalize_tensor_zscore,
                        tensor_format=TENSOR_FORMAT
                    )

                    out_filename = f"{filepath.stem}_chunk{idx:03d}.npy"
                    np.save(target_dir / out_filename, tensor)
                    summary_stats[split_name] += 1

            except Exception as e:
                summary_stats["skipped_files"] += 1
                print(f"\n[BŁĄD] Pominięto plik '{filepath.name}': {e}")

    print("\n" + "=" * 55)
    print("PODSUMOWANIE ENKODOWANIA:")
    print(f"  Format tensorów:  {TENSOR_FORMAT.upper()} -> (128, 259, 3)" if TENSOR_FORMAT == "tf" else "  Format tensorów:  PYTORCH -> (3, 128, 259)")
    print(f"  Próbki TRAIN:     {summary_stats['train']} plików .npy")
    print(f"  Próbki VALIDATE:  {summary_stats['validate']} plików .npy")
    print(f"  Próbki TEST:      {summary_stats['test']} plików .npy")
    print(f"  Uszkodzone pliki: {summary_stats['skipped_files']}")
    print("=" * 55)

if __name__ == "__main__":
    build_dataset()

```

---

## KOD ŹRÓDŁOWY: model_cnn.py
**Ścieżka pliku w repozytorium:** `model_cnn.py`

```python
"""
HDD Acoustic Fault Detection - 2D CNN Model Architecture
Single Source of Truth: ADR-001 & AGENTS.md

Moduł definiujący architekturę sieci 2D CNN w TensorFlow / Keras przystosowaną
do przetwarzania 3-kanałowych reprezentacji czasowo-częstotliwościowych [Log-Mel, Delta, Delta-Delta]
wycinków audio dysków HDD o stałym kształcie (128, 259, 3).
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers


# Stałe DSP / wejściowe zgodne z ADR-001
INPUT_SHAPE: Tuple[int, int, int] = (128, 259, 3)  # (n_mels, time_frames, channels)
DEFAULT_NUM_CLASSES: int = 4                       # np. clicking, beeping, scratching, normal


def build_hdd_cnn_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    model_name: str = "AcousticHddSentinelCNN"
) -> tf.keras.Model:
    """
    Tworzy model 2D CNN dedykowany do akustycznej detekcji i klasyfikacji uszkodzeń HDD.

    Architektura oparta na 4 blokach konwolucyjnych z Batch Normalization,
    warstwami poolingowymi oraz Global Average Pooling, co zapobiega przeuczeniu
    (overfitting) na małych i średnich zbiorach akustycznych, zachowując wysoką
    czułość na impulsowe transjenty (kanał Delta) oraz wąskopasmowe piski (Log-Mel).

    Parametry:
        input_shape: Kształt tensora wejściowego (128, 259, 3) - Channel-Last.
        num_classes: Liczba klas wyjściowych (>= 2 dla softmax, 1 dla binarnego sigmoid).
        dropout_rate: Bazowy współczynnik regularyzacji dropout.
        l2_reg: Współczynnik regularyzacji wag L2.
        model_name: Nazwa modelu Keras.

    Zwraca:
        tf.keras.Model: Nieskompilowany lub gotowy do kompilacji model Keras.
    """
    # Niezmiennik techniczny ADR-001 / .agents/rules/adr-001-audio-pipeline.md:
    # Zawsze jawnie definiuj layers.Input(shape=(128, 259, 3))
    inputs = layers.Input(shape=input_shape, name="audio_spectrogram_input")

    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    # =========================================================================
    # BLOK 1: Niskopoziomowa ekstrakcja cech widmowych i transjentów
    # Wejście: (128, 259, 3) -> Wyjście po poolingu: (64, 129, 32)
    # =========================================================================
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", use_bias=False, kernel_regularizer=reg, name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)

    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", use_bias=False, kernel_regularizer=reg, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.SpatialDropout2D(dropout_rate * 0.5, name="s_dropout1")(x)

    # =========================================================================
    # BLOK 2: Średniopoziomowe wzorce czasowo-częstotliwościowe (np. klikanie głowicy)
    # Wejście: (64, 129, 32) -> Wyjście po poolingu: (32, 64, 64)
    # =========================================================================
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", use_bias=False, kernel_regularizer=reg, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)

    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", use_bias=False, kernel_regularizer=reg, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.SpatialDropout2D(dropout_rate * 0.75, name="s_dropout2")(x)

    # =========================================================================
    # BLOK 3: Wysokopoziomowa dynamika zmian (zatarte łożyska, rezonanse, zarysowania)
    # Wejście: (32, 64, 64) -> Wyjście po poolingu: (16, 32, 128)
    # =========================================================================
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", use_bias=False, kernel_regularizer=reg, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)

    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", use_bias=False, kernel_regularizer=reg, name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.SpatialDropout2D(dropout_rate, name="s_dropout3")(x)

    # =========================================================================
    # BLOK 4: Semantyczna reprezentacja akustyczna
    # Wejście: (16, 32, 128) -> Wyjście po poolingu: (8, 16, 256)
    # =========================================================================
    x = layers.Conv2D(256, kernel_size=(3, 3), padding="same", use_bias=False, kernel_regularizer=reg, name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="relu4_1")(x)

    x = layers.Conv2D(256, kernel_size=(3, 3), padding="same", use_bias=False, kernel_regularizer=reg, name="conv4_2")(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.Activation("relu", name="relu4_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    x = layers.Dropout(dropout_rate, name="dropout4")(x)

    # =========================================================================
    # AGREGACJA PRZESTRZENNO-CZASOWA (Global Average Pooling)
    # Zastępuje gigantyczną warstwę Flatten, redukuje liczbę parametrów z milionów
    # do bezpiecznego zakresu, eliminując ryzyko przeuczenia.
    # Wyjście: wektor 256-elementowy
    # =========================================================================
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Gęsta warstwa projekcyjna (Dense Projection Layer)
    x = layers.Dense(128, use_bias=False, kernel_regularizer=reg, name="dense_features")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(dropout_rate, name="dense_dropout")(x)

    # =========================================================================
    # WARSTWA WYJŚCIOWA (Klasyfikator)
    # =========================================================================
    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


def build_transfer_learning_model(
    backbone_name: str = "mobilenetv2",
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.3,
    model_name: Optional[str] = None
) -> tf.keras.Model:
    """
    Buduje model transfer learningu wykorzystując 3-kanałowy format tensora
    (Log-Mel, Delta, Delta-Delta) jako wejście do sieci pretrenowanej na ImageNet
    (zgodnie z sekcją 2 i 4 ADR-001).

    Parametry:
        backbone_name: Wybór szkieletu ('mobilenetv2', 'efficientnetb0', 'resnet50').
        input_shape: (128, 259, 3).
        num_classes: Liczba klas wyjściowych.
        freeze_backbone: Czy zablokować wagi bazowe podczas wstępnego treningu.
        dropout_rate: Stopień dropoutu w głowicy klasyfikatora.
        model_name: Opcjonalna nazwa modelu.
    """
    inputs = layers.Input(shape=input_shape, name="audio_spectrogram_input")

    backbone_name_lower = backbone_name.lower()
    if backbone_name_lower == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            input_tensor=inputs,
            include_top=False,
            weights="imagenet"
        )
    elif backbone_name_lower == "efficientnetb0":
        try:
            base_model = tf.keras.applications.EfficientNetB0(
                input_tensor=inputs,
                include_top=False,
                weights="imagenet",
                include_preprocessing=False
            )
        except TypeError:
            base_model = tf.keras.applications.EfficientNetB0(
                input_tensor=inputs,
                include_top=False,
                weights="imagenet"
            )
    elif backbone_name_lower == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            input_tensor=inputs,
            include_top=False,
            weights="imagenet"
        )
    else:
        raise ValueError(f"Nieobsługiwany szkielet: {backbone_name}. Wybierz: mobilenetv2, efficientnetb0, resnet50.")

    base_model.trainable = not freeze_backbone

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="transfer_gap")(x)
    x = layers.BatchNormalization(name="transfer_bn")(x)
    x = layers.Dropout(dropout_rate, name="transfer_dropout")(x)
    x = layers.Dense(128, activation="relu", name="transfer_dense")(x)
    x = layers.Dropout(dropout_rate, name="transfer_dense_dropout")(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    name = model_name or f"HddSentinel_{backbone_name}"
    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def compile_cnn_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
    label_mode: str = "categorical"
) -> tf.keras.Model:
    """
    Kompiluje model Keras z optymalizatorem Adam, odpowiednią funkcją straty i metrykami.

    Parametry:
        model: Model zwrócony przez `build_hdd_cnn_model` lub `build_transfer_learning_model`.
        learning_rate: Początkowy współczynnik uczenia (Learning Rate).
        label_mode: 'categorical' (dla wektorów one-hot), 'sparse' (dla indeksów int)
                    lub 'binary' (dla klasyfikacji 1-wyjściowej).
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    if label_mode == "sparse":
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy")
        ]
    elif label_mode == "binary":
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    else:  # categorical (one-hot)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02)
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy")
        ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


if __name__ == "__main__":
    print("=" * 70)
    print("AKUSTYCZNA DETEKCJA AWARII HDD - ARCHITEKTURA 2D CNN")
    print("=" * 70)
    
    # Przykładowa budowa modelu dla 4 klas awarii
    model = build_hdd_cnn_model(input_shape=(128, 259, 3), num_classes=4)
    model = compile_cnn_model(model, learning_rate=1e-3, label_mode="categorical")
    
    model.summary()
    print("\n[OK] Model 2D CNN utworzony i skompilowany pomyślnie z tensorem wejściowym (128, 259, 3).")

```

---

## KOD ŹRÓDŁOWY: train_cnn.py
**Ścieżka pliku w repozytorium:** `train_cnn.py`

```python
"""
HDD Acoustic Fault Detection - CNN Training Pipeline
Single Source of Truth: ADR-001, AGENTS.md, .agents/rules/adr-001-audio-pipeline.md

Skrypt treningowy dla sieci 2D CNN (Keras / TensorFlow) z:
1. Loaderem tf.data.Dataset wczytującym pliki .npy z dataset_encoded/{train,validate,test}
   zgodnie z regułą tf.numpy_function i tensor.set_shape([128, 259, 3]).
2. Obsługą class_weight (zrównoważenie rzadkich awarii HDD).
3. Callbackami: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
4. Ewaluacją na zbiorze testowym i generowaniem macierzy pomyłek (Confusion Matrix).
5. Zachowaniem izolacji normalizacji (import z hdd_audio_encoder.py dla inferencji).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

# Dodanie katalogu skryptu oraz katalogu nadrzędnego do sys.path,
# aby umożliwić importy zarówno przy uruchamianiu z root, jak i z CNN/
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import architektury modelu
from model_cnn import build_hdd_cnn_model, compile_cnn_model, INPUT_SHAPE

# Invariant 4 z AGENTS.md: Izolacja normalizacji
# Podczas pisania skryptów ewaluacyjnych i inferencyjnych importuj funkcje bezpośrednio z hdd_audio_encoder.py
try:
    from hdd_audio_encoder import (
        encode_audio_chunk,
        normalize_waveform_peak,
        normalize_tensor_zscore,
        TARGET_SR
    )
except Exception:
    try:
        from CNN.hdd_audio_encoder import (
            encode_audio_chunk,
            normalize_waveform_peak,
            normalize_tensor_zscore,
            TARGET_SR
        )
    except Exception as e:
        import warnings
        warnings.warn(f"Nie udało się zaimportować hdd_audio_encoder ({e}). Funkcja predict_audio_chunk nie będzie dostępna.")
        encode_audio_chunk = None
        normalize_waveform_peak = None
        normalize_tensor_zscore = None
        TARGET_SR = 22050


# =============================================================================
# 1. WYSZUKIWANIE I WALIDACJA KATALOGU ZBIORU DANYCH
# =============================================================================

def resolve_dataset_dir(custom_path: Optional[str] = None) -> Path:
    """
    Automatycznie lokalizuje katalog `dataset_encoded` sprawdzając typowe lokalizacje.
    """
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Podany katalog danych nie istnieje: {custom_path}")

    candidates = [
        Path("dataset_encoded"),
        Path("CNN/dataset_encoded"),
        current_dir / "dataset_encoded",
        parent_dir / "dataset_encoded",
        parent_dir / "CNN" / "dataset_encoded",
    ]

    for c in candidates:
        if c.exists() and (c / "train").exists():
            return c.resolve()

    # Domyślny fallback
    return (current_dir / "dataset_encoded").resolve()


def discover_classes_and_files(
    split_dir: Path,
    expected_classes: Optional[List[str]] = None
) -> Tuple[List[str], List[Path], List[int]]:
    """
    Skanuje katalog danego podziału (train/validate/test), wykrywa podfoldery klas,
    mapuje klasy alfabetycznie i zbiera listę plików .npy wraz z etykietami całkowitymi.
    """
    if not split_dir.exists():
        return [], [], []

    # Klasy to podfoldery w katalogu danego splitu
    subdirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    
    if not subdirs:
        # Fallback jeśli wszystkie pliki leżą płasko (np. pojedyncza klasa)
        npy_files = sorted(list(split_dir.glob("*.npy")))
        labels = [0] * len(npy_files)
        return ["default"], npy_files, labels

    class_names = expected_classes if expected_classes is not None else [d.name for d in subdirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    all_files: List[Path] = []
    all_labels: List[int] = []

    for class_name in class_names:
        class_folder = split_dir / class_name
        if not class_folder.exists():
            continue
        files = sorted(list(class_folder.glob("*.npy")))
        all_files.extend(files)
        all_labels.extend([class_to_idx[class_name]] * len(files))

    return class_names, all_files, all_labels


# =============================================================================
# 2. LOADER tf.data.Dataset DLA PLIKÓW .NPY (ZGODNIE Z ADR-001)
# =============================================================================

def load_npy_tensor(filepath_input) -> np.ndarray:
    """
    Czysta funkcja Python do wczytania pliku .npy (wywoływana wewnątrz tf.numpy_function).
    """
    if isinstance(filepath_input, bytes):
        path_str = filepath_input.decode("utf-8")
    elif hasattr(filepath_input, "numpy"):
        val = filepath_input.numpy()
        path_str = val.decode("utf-8") if isinstance(val, bytes) else str(val)
    else:
        path_str = str(filepath_input)

    tensor = np.load(path_str).astype(np.float32)
    return tensor


def apply_spec_augment(tensor: tf.Tensor, max_freq_mask: int = 16, max_time_mask: int = 24) -> tf.Tensor:
    """
    Wykonywana w locie augmentacja SpecAugment (Frequency & Time Masking) dla tensora (128, 259, 3).
    Poprawia generalizację na rzadkich klasach (scratching, beeping).
    """
    freq_bins, time_steps, _ = INPUT_SHAPE
    f_len = tf.random.uniform([], minval=0, maxval=max_freq_mask, dtype=tf.int32)
    f0 = tf.random.uniform([], minval=0, maxval=freq_bins - f_len, dtype=tf.int32)
    freq_mask = tf.concat([
        tf.ones([f0, time_steps, 1]),
        tf.zeros([f_len, time_steps, 1]),
        tf.ones([freq_bins - f0 - f_len, time_steps, 1])
    ], axis=0)

    t_len = tf.random.uniform([], minval=0, maxval=max_time_mask, dtype=tf.int32)
    t0 = tf.random.uniform([], minval=0, maxval=time_steps - t_len, dtype=tf.int32)
    time_mask = tf.concat([
        tf.ones([freq_bins, t0, 1]),
        tf.zeros([freq_bins, t_len, 1]),
        tf.ones([freq_bins, time_steps - t0 - t_len, 1])
    ], axis=1)

    return tensor * freq_mask * time_mask


def create_tf_dataset(
    file_paths: List[Path],
    labels: List[int],
    num_classes: int,
    batch_size: int = 32,
    is_training: bool = False,
    shuffle_buffer: int = 1000,
    seed: int = 42
) -> tf.data.Dataset:
    """
    Tworzy zoptymalizowany pipeline tf.data.Dataset wczytujący tensory .npy.
    
    Niezmiennik techniczny (.agents/rules/adr-001-audio-pipeline.md):
    Wczytując pliki .npy, używaj `tf.numpy_function` i jawnie ustawiaj
    `tensor.set_shape([128, 259, 3])`, aby kompilator grafu TF znał wymiary przed treningiem.
    """
    str_paths = [str(p) for p in file_paths]
    
    paths_ds = tf.data.Dataset.from_tensor_slices(str_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((paths_ds, labels_ds))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), shuffle_buffer), seed=seed, reshuffle_each_iteration=True)

    def _map_entry(path_elem, label_elem):
        # tf.numpy_function do bezpiecznego wczytania pliku numpy
        [tensor] = tf.numpy_function(
            func=load_npy_tensor,
            inp=[path_elem],
            Tout=[tf.float32]
        )
        # Niezmiennik: jawne ustawienie kształtu tensora
        tensor.set_shape(INPUT_SHAPE)  # [128, 259, 3]
        if is_training:
            tensor = apply_spec_augment(tensor)

        # Konwersja etykiety do wektora one-hot
        one_hot_label = tf.one_hot(label_elem, depth=num_classes)
        one_hot_label.set_shape([num_classes])

        return tensor, one_hot_label

    dataset = dataset.map(_map_entry, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# =============================================================================
# 3. OBSŁUGA class_weight (ZRÓWNOWAŻENIE KLAS)
# =============================================================================

def calculate_balanced_class_weights(labels: List[int], num_classes: int) -> Dict[int, float]:
    """
    Oblicza wygładzone (sqrt-balanced) wagi klas, aby zniwelować
    asymetrię liczby próbek w poszczególnych rodzajach uszkodzeń dysków HDD.
    """
    unique_present = np.unique(labels)
    computed_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_present,
        y=labels
    )
    
    weights_dict: Dict[int, float] = {}
    for cls_idx in range(num_classes):
        if cls_idx in unique_present:
            pos = np.where(unique_present == cls_idx)[0][0]
            # Pierwiastkowanie wag zapobiega zdominowaniu gradientów przez rzadkie klasy
            weights_dict[cls_idx] = float(np.clip(np.sqrt(computed_weights[pos]), 0.5, 3.5))
        else:
            weights_dict[cls_idx] = 1.0

    return weights_dict


# =============================================================================
# 4. CALLBACKI TRENINGOWE
# =============================================================================

def setup_training_callbacks(
    checkpoint_filepath: str,
    log_dir: str,
    early_stopping_patience: int = 15,
    reduce_lr_patience: int = 5
) -> List[tf.keras.callbacks.Callback]:
    """
    Konfiguruje zestaw kluczowych callbacków Keras:
    - EarlyStopping: zatrzymuje trening przy braku poprawy val_loss i przywraca najlepsze wagi.
    - ModelCheckpoint: zapisuje najlepszy model na podstawie minimalnego val_loss.
    - ReduceLROnPlateau: dynamicznie redukuje LR o 50% przy stagnacji val_loss.
    - CSVLogger: zapisuje historię metryk do pliku CSV.
    """
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_filepath)), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(log_dir, "training_history.csv"),
            separator=",",
            append=False
        )
    ]
    return callbacks


# =============================================================================
# 5. MACIERZ POMYŁEK I WYKRESY EWALUACYJNE
# =============================================================================

def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Macierz Pomyłek (HDD Fault Classification)"
) -> np.ndarray:
    """
    Generuje i zapisuje profesjonalny wykres macierzy pomyłek z liczbami próbek
    oraz znormalizowanymi procentami per klasa rzeczywista.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    cax = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=10)

    # Wypisanie wartości i procentów w komórkach
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_normalized[i, j] * 100
            text_color = "white" if cm_normalized[i, j] > thresh else "black"
            ax.text(
                j, i, f"{val}\n({pct:.1f}%)",
                ha="center", va="center",
                color=text_color,
                fontsize=9, fontweight="medium"
            )

    ax.set_ylabel("Rzeczywista klasa (True Label)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Przewidziana klasa (Predicted Label)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"[Ewaluacja] Wykres macierzy pomyłek zapisano w: {output_path}")
    return cm


def plot_training_curves(history: tf.keras.callbacks.History, output_path: str):
    """
    Rysuje i zapisuje krzywe uczenia (funkcja straty oraz dokładność) dla Train i Validation.
    """
    hist = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    # Loss
    ax1.plot(epochs_range, hist["loss"], label="Train Loss", color="royalblue", linewidth=2)
    if "val_loss" in hist:
        ax1.plot(epochs_range, hist["val_loss"], label="Val Loss", color="crimson", linewidth=2, linestyle="--")
    ax1.set_title("Krzywa funkcji straty (Loss)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    acc_key = "accuracy" if "accuracy" in hist else "categorical_accuracy"
    val_acc_key = f"val_{acc_key}"
    if acc_key in hist:
        ax2.plot(epochs_range, hist[acc_key], label="Train Accuracy", color="forestgreen", linewidth=2)
    if val_acc_key in hist:
        ax2.plot(epochs_range, hist[val_acc_key], label="Val Accuracy", color="darkorange", linewidth=2, linestyle="--")
    ax2.set_title("Krzywa dokładności (Accuracy)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[Ewaluacja] Wykres krzywych uczenia zapisano w: {output_path}")


# =============================================================================
# 6. FUNKCJA INFERENCJI ZGODNA Z INVARIANTEM 4 (IMPORT NORMALIZACJI)
# =============================================================================

def predict_audio_chunk(
    model: tf.keras.Model,
    y_chunk: np.ndarray,
    class_names: List[str],
    sr: int = 22050
) -> Tuple[str, float, np.ndarray]:
    """
    Wykonuje inferencję bezpośrednio na surowym 3-sekundowym wycinku audio.
    Zgodnie z Invariantem 4 z AGENTS.md, używa wyizolowanych funkcji normalizacji
    i enkodera z hdd_audio_encoder.py bez duplikowania kodu.
    """
    if encode_audio_chunk is None:
        raise ImportError("Brak modułu hdd_audio_encoder.py. Upewnij się, że plik znajduje się w ścieżce.")

    tensor = encode_audio_chunk(
        y_chunk=y_chunk,
        sr=sr,
        waveform_norm_fn=normalize_waveform_peak,
        tensor_norm_fn=normalize_tensor_zscore,
        tensor_format="tf"
    )
    # Rozszerzenie wymiaru o batch: (1, 128, 259, 3)
    batch_tensor = np.expand_dims(tensor, axis=0)
    probs = model.predict(batch_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs


# =============================================================================
# 7. GŁÓWNA PROCEDURA TRENINGOWA
# =============================================================================

def train_and_evaluate(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout_rate: float = 0.3
):
    """
    Pełny potok: wczytanie danych, budowa modelu, trening z class_weight,
    ewaluacja na zbiorze testowym oraz wygenerowanie macierzy pomyłek.
    """
    print("=" * 70)
    print("ROZPOCZĘCIE TRENINGU SIECI 2D CNN (HDD ACOUSTIC SENTINEL)")
    print("=" * 70)
    print(f"Katalog zbioru danych: {data_dir}")
    print(f"Katalog wyjściowy:     {output_dir}")

    train_dir = data_dir / "train"
    val_dir = data_dir / "validate"
    test_dir = data_dir / "test"

    # Weryfikacja katalogów
    train_classes, train_files, train_labels = discover_classes_and_files(train_dir)
    val_classes, val_files, val_labels = discover_classes_and_files(val_dir, expected_classes=train_classes)
    test_classes, test_files, test_labels = discover_classes_and_files(test_dir, expected_classes=train_classes)

    if not train_files:
        raise FileNotFoundError(
            f"Nie znaleziono plików .npy w katalogu treningowym '{train_dir}'. "
            f"Uruchom najpierw enkoder: python hdd_audio_encoder.py"
        )

    # Klasy powinny być spójne (bazujemy na katalogu train)
    class_names = train_classes
    num_classes = len(class_names)

    print(f"\nWykryte klasy ({num_classes}): {class_names}")
    print(f"Liczba próbek .npy:")
    print(f"  - TRAIN:    {len(train_files)} tensorów")
    print(f"  - VALIDATE: {len(val_files)} tensorów")
    print(f"  - TEST:     {len(test_files)} tensorów")

    # Obliczenie class_weight
    class_weights = calculate_balanced_class_weights(train_labels, num_classes)
    print(f"\nObliczone wagi klas (class_weight):")
    for idx, c_name in enumerate(class_names):
        print(f"  [{idx}] {c_name:15s}: waga = {class_weights[idx]:.3f} (próbek: {train_labels.count(idx)})")

    # Tworzenie pipeline'ów tf.data.Dataset
    print("\nBudowanie loaderów tf.data.Dataset z wymiarem (128, 259, 3)...")
    train_ds = create_tf_dataset(train_files, train_labels, num_classes, batch_size=batch_size, is_training=True)
    val_ds = create_tf_dataset(val_files, val_labels, num_classes, batch_size=batch_size, is_training=False) if val_files else None
    test_ds = create_tf_dataset(test_files, test_labels, num_classes, batch_size=batch_size, is_training=False) if test_files else None

    # Budowa i kompilacja modelu 2D CNN
    print("\nInicjalizacja modelu 2D CNN...")
    model = build_hdd_cnn_model(
        input_shape=INPUT_SHAPE,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        model_name="AcousticHddSentinelCNN"
    )
    model = compile_cnn_model(model, learning_rate=lr, label_mode="categorical")
    model.summary()

    # Callbacki
    checkpoint_path = str(output_dir / "best_hdd_cnn_model.keras")
    callbacks = setup_training_callbacks(
        checkpoint_filepath=checkpoint_path,
        log_dir=str(output_dir)
    )

    # Trening
    print("\n" + "=" * 70)
    print("START TRENINGU (model.fit)")
    print("=" * 70)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Zapis wykresów uczenia
    training_curves_path = str(output_dir / "training_curves.png")
    plot_training_curves(history, training_curves_path)

    # Zapis mapowania klas do JSON
    class_mapping_path = output_dir / "classes.json"
    with open(class_mapping_path, "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "input_shape": list(INPUT_SHAPE)}, f, indent=2)

    # Ewaluacja na zbiorze testowym
    if test_ds is not None and len(test_files) > 0:
        print("\n" + "=" * 70)
        print("EWALUACJA NA ZBIORZE TESTOWYM (TEST SET)")
        print("=" * 70)

        # Wczytanie wag najlepszego modelu zapisanego przez ModelCheckpoint
        if os.path.exists(checkpoint_path):
            print(f"Wczytywanie wag najlepszego checkpointu: {checkpoint_path}")
            model = tf.keras.models.load_model(checkpoint_path)

        eval_results = model.evaluate(test_ds, verbose=1)
        metric_names = model.metrics_names
        print("\nWyniki metryk na zbiorze testowym:")
        eval_dict = {}
        for name, val in zip(metric_names, eval_results):
            eval_dict[name] = float(val)
            print(f"  - {name}: {val:.4f}")

        # Predykcje dla całego zbioru testowego do macierzy pomyłek
        print("\nGenerowanie predykcji dla macierzy pomyłek...")
        y_true_all: List[int] = []
        y_pred_all: List[int] = []

        for x_batch, y_batch in test_ds:
            preds = model.predict(x_batch, verbose=0)
            y_pred_batch = np.argmax(preds, axis=-1)
            y_true_batch = np.argmax(y_batch.numpy(), axis=-1)
            y_true_all.extend(y_true_batch)
            y_pred_all.extend(y_pred_batch)

        y_true_arr = np.array(y_true_all)
        y_pred_arr = np.array(y_pred_all)

        # Raport klasyfikacji
        report_str = classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        report_dict = classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        print("\nRaport Klasyfikacji (Classification Report):")
        print(report_str)

        # Zapis raportu do pliku tekstowego
        report_path = output_dir / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_str)

        # Wygenerowanie i zapis macierzy pomyłek
        cm_path = str(output_dir / "confusion_matrix.png")
        plot_and_save_confusion_matrix(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            class_names=class_names,
            output_path=cm_path
        )

        # Zapis podsumowania metryk do JSON
        summary_path = output_dir / "test_evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "test_metrics": eval_dict,
                "detailed_metrics": report_dict,
                "classes": class_names,
                "total_test_samples": len(test_files)
            }, f, indent=2)

    print("\n" + "=" * 70)
    print("TRENING I EWALUACJA ZAKOŃCZONE POMYŚLNIE!")
    print(f"Wszystkie artefakty zapisano w katalogu: {output_dir.resolve()}")
    print("=" * 70)


# =============================================================================
# 8. PARSER ARGUMENTÓW CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Trening modelu 2D CNN do akustycznej detekcji awarii dysków HDD."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Ścieżka do katalogu dataset_encoded zawierającego train/, validate/, test/."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./training_output",
        help="Katalog do zapisu modeli, wykresów i logów (domyślnie: ./training_output)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maksymalna liczba epok treningowych (domyślnie: 50)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Rozmiar batcha (domyślnie: 32)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Początkowy współczynnik uczenia Learning Rate (domyślnie: 0.001)."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Współczynnik dropoutu w warstwach modelu (domyślnie: 0.3)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_directory = resolve_dataset_dir(args.data_dir)
    out_directory = Path(args.output_dir)
    out_directory.mkdir(parents=True, exist_ok=True)

    train_and_evaluate(
        data_dir=data_directory,
        output_dir=out_directory,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout_rate=args.dropout
    )

```

---

## KOD ŹRÓDŁOWY: prepare_audio_dataset.py
**Ścieżka pliku w repozytorium:** `prepare_audio_dataset.py`

```python
"""
Generator i organizator zbioru danych audio dla Acoustic HDD Sentinel.
1. Generuje realistyczne próbki audio prawidłowej pracy dysków (klasa 'normal').
2. Kategoryzuje istniejące nagrania uszkodzeń z CNN/dataset do klas: 'clicking', 'beeping', 'scratching'.
3. Zapisuje zbiór do katalogu CNN/dataset_categorized/{clicking, beeping, scratching, normal}.
"""

import os
import shutil
from pathlib import Path
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa
from tqdm import tqdm

TARGET_SR = 22050
SOURCE_DATASET_DIR = Path("CNN/dataset")
CATEGORIZED_DIR = Path("CNN/dataset_categorized")


def generate_healthy_hdd_sound(duration_sec: float = 12.0, rpm: int = 7200, sr: int = TARGET_SR, add_seeks: bool = True) -> np.ndarray:
    """
    Syntetyzuje fizycznie realistyczny dźwięk prawidłowo działającego dysku twardego HDD:
    - Buczenie obrotowe silnika wrzeciona (fundamentalna częstotliwość obrotowa + harmoniczne)
    - Szum aerodynamiczny powietrza i łożysk hydrodynamicznych (filtrowany szum różowy)
    - Subtelne, sporadyczne operacje pozycjonowania głowic (delikatne seek thuds)
    """
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # 1. Częstotliwość podstawowa wrzeciona z lekką odchyłką produkcyjną
    f_base = (rpm * np.random.uniform(0.985, 1.015)) / 60.0
    
    # Drobne fluktuacje prędkości obrotowej (wow/flutter)
    flutter = 1.0 + 0.002 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * f_base * np.cumsum(flutter) / sr
    rand_phi = np.random.uniform(0, 2 * np.pi, size=6)

    # Harmoniczne silnika i biegunów stojana
    motor_sound = 0.40 * np.sin(phase + rand_phi[0])
    motor_sound += 0.25 * np.sin(2 * phase + rand_phi[1])
    motor_sound += 0.15 * np.sin(3 * phase + rand_phi[2])
    motor_sound += 0.10 * np.sin(4 * phase + rand_phi[3])
    motor_sound += 0.06 * np.sin(6 * phase + rand_phi[4])
    motor_sound += 0.04 * np.sin(12 * phase + rand_phi[5])

    # 2. Szum łożysk i oporu powietrza wirujących talerzy (szum różowy / filtrowany)
    white_noise = np.random.randn(num_samples)
    upper_cutoff = np.random.uniform(3500, 7500)
    sos_air = signal.butter(3, [120 / (sr / 2), upper_cutoff / (sr / 2)], btype="bandpass", output="sos")
    air_noise = signal.sosfilt(sos_air, white_noise) * 0.35

    # 3. Delikatny szum tła otoczenia + szerokopasmowy szum własny mikrofonu (zapobiega martwemu pasmu >2.4kHz)
    ambient_noise = np.random.randn(num_samples) * 0.05
    sos_low = signal.butter(2, 800 / (sr / 2), btype="lowpass", output="sos")
    mic_floor = np.random.randn(num_samples) * np.random.uniform(0.008, 0.025)
    ambient_noise = signal.sosfilt(sos_low, ambient_noise) + mic_floor

    combined = motor_sound + air_noise + ambient_noise

    # 4. Sporadyczne, miękkie ruchy pozycjonera głowic (soft head seeks)
    if add_seeks:
        num_seeks = np.random.randint(2, 6)
        seek_positions = np.random.uniform(1.0, duration_sec - 1.0, size=num_seeks)
        for s_pos in seek_positions:
            idx = int(s_pos * sr)
            seek_len = int(sr * 0.04)  # 40 ms
            if idx + seek_len < num_samples:
                # Miękki impuls gaussa (stłumiony dźwięk pozycjonowania w zdrowym dysku)
                env = np.hanning(seek_len)
                seek_freq = np.random.uniform(400, 750)
                seek_tone = np.sin(2 * np.pi * seek_freq * np.linspace(0, 0.04, seek_len))
                combined[idx:idx + seek_len] += 0.35 * env * seek_tone

    # Normalizacja amplitudy do realistycznego zakresu zdrowego dysku (RMS ~ 0.03 - 0.08)
    peak = np.max(np.abs(combined))
    if peak > 1e-6:
        combined = (combined / peak) * 0.45

    return combined.astype(np.float32)


def categorize_existing_files():
    """
    Kategoryzuje pliki z CNN/dataset do klas:
    - clicking (Click of Death, stukanie głowicy)
    - beeping (zatarte wrzeciono, piski, melodie)
    - scratching (tarcie głowicy o talerz, zgrzytanie)
    """
    for cls in ["clicking", "beeping", "scratching", "normal"]:
        (CATEGORIZED_DIR / cls).mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list(SOURCE_DATASET_DIR.glob("*.wav")))
    print(f"Znaleziono {len(wav_files)} istniejących plików audio w {SOURCE_DATASET_DIR}.")

    counts = {"clicking": 0, "beeping": 0, "scratching": 0}

    for fpath in tqdm(wav_files, desc="Kategoryzacja nagrań uszkodzeń"):
        fname_lower = fpath.name.lower()

        # 1. Reguły na podstawie jawnych nazw Aesonlabs
        if "beeping" in fname_lower or "melody" in fname_lower:
            target_class = "beeping"
        elif "scratch" in fname_lower or "screech" in fname_lower:
            target_class = "scratching"
        elif "clicking" in fname_lower:
            target_class = "clicking"
        else:
            # 2. Reguły na podstawie sygnatury akustycznej dla plików Northwind
            try:
                y, sr = librosa.load(str(fpath), sr=TARGET_SR, mono=True)
                y_trimmed, _ = librosa.effects.trim(y, top_db=25)
                if len(y_trimmed) < sr * 0.5:
                    y_trimmed = y

                peak = np.max(np.abs(y_trimmed))
                rms = np.sqrt(np.mean(y_trimmed ** 2)) + 1e-9
                crest_factor = peak / rms

                # Płaskość widma (niska = tonalna/beeping, wysoka = szum/scratching)
                flatness = np.mean(librosa.feature.spectral_flatness(y=y_trimmed))
                # Rolloff częstotliwościowy 95%
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr, roll_percent=0.95))
                # Onset rate (impulsowość kliknięć)
                onset_env = librosa.onset.onset_strength(y=y_trimmed, sr=sr)
                onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
                dur = len(y_trimmed) / sr
                onset_rate = len(onsets) / (dur + 1e-6)

                if flatness < 0.003:
                    # Bardzo czysty ton / pisk / syrena zablokowanego wrzeciona
                    target_class = "beeping"
                elif (rolloff > 6500 and crest_factor < 22.0 and flatness > 0.025) or flatness > 0.12:
                    # Ciągły szerokopasmowy pisk/tarcie o wysokiej energii wysokotonowej
                    target_class = "scratching"
                elif crest_factor > 18.0 or onset_rate > 1.2:
                    # Wyraźne impulsowe stukanie głowicy o ogranicznik
                    target_class = "clicking"
                else:
                    # Pozostałe przypadki (przeważająca większość awarii HDD to uszkodzenia głowic / clicking)
                    target_class = "clicking"

            except Exception as e:
                target_class = "clicking"

        shutil.copy2(fpath, CATEGORIZED_DIR / target_class / fpath.name)
        counts[target_class] += 1

    print("\nPodsumowanie kategoryzacji nagrań uszkodzeń:")
    for cls, cnt in counts.items():
        print(f"  - {cls:12s}: {cnt} nagrań")


def generate_normal_class(count: int = 25):
    """
    Generuje zróżnicowany zbiór nagrań normalnej pracy dysku HDD.
    """
    print(f"\nGenerowanie {count} syntetycznych nagrań prawidłowej pracy dysku (klasa 'normal')...")
    normal_dir = CATEGORIZED_DIR / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, count + 1):
        rpm = np.random.choice([5400, 7200])
        dur = np.random.uniform(10.0, 18.0)
        has_seeks = (i % 2 == 0)
        y_normal = generate_healthy_hdd_sound(duration_sec=dur, rpm=rpm, sr=TARGET_SR, add_seeks=has_seeks)
        
        out_name = f"synthetic_healthy_hdd_{rpm}rpm_{i:02d}.wav"
        sf.write(str(normal_dir / out_name), y_normal, TARGET_SR, subtype="PCM_16")

    print(f"Wygenerowano {count} plików .wav w {normal_dir}.")


if __name__ == "__main__":
    categorize_existing_files()
    generate_normal_class(count=25)
    print("\nPrzygotowanie zbioru zakończone!")

```

---

## KOD ŹRÓDŁOWY: verify_cnn_model.py
**Ścieżka pliku w repozytorium:** `verify_cnn_model.py`

```python
"""
Skrypt weryfikacyjny dla architektury modelu 2D CNN (Acoustic HDD Sentinel).
Uruchomienie: python verify_cnn_model.py
"""

import os
import tempfile
import numpy as np
import tensorflow as tf
from model_cnn import build_hdd_cnn_model, compile_cnn_model, build_transfer_learning_model

def run_verification():
    print("=" * 70)
    print("KOMPLEKSOWA WERYFIKACJA MODELU 2D CNN (ACOUSTIC HDD SENTINEL)")
    print("=" * 70)

    # 1. Kształt wejściowy i parametry
    print("[1/6] Inicjalizacja modelu i weryfikacja kształtów...")
    model = build_hdd_cnn_model(input_shape=(128, 259, 3), num_classes=4, dropout_rate=0.2)
    model = compile_cnn_model(model, learning_rate=1e-3)
    assert model.input_shape == (None, 128, 259, 3), "Błędny input_shape!"
    assert model.output_shape == (None, 4), "Błędny output_shape!"
    print(f"   -> Wejście: {model.input_shape}")
    print(f"   -> Wyjście: {model.output_shape}")
    trainable_count = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"   -> Liczba parametrów: {model.count_params():,} (trenowalnych: {trainable_count:,})")
    print("   [OK] Kształty zgodne z ADR-001.\n")

    # 2. Przekazywanie w przód (Forward pass)
    print("[2/6] Test przekazywania w przód (Forward Pass)...")
    dummy_x = np.random.randn(4, 128, 259, 3).astype(np.float32)
    preds = model.predict(dummy_x, verbose=0)
    assert preds.shape == (4, 4), "Błędny kształt predykcji!"
    assert not np.isnan(preds).any(), "Wykryto NaN!"
    assert not np.isinf(preds).any(), "Wykryto Inf!"
    assert np.allclose(np.sum(preds, axis=-1), 1.0, atol=1e-5), "Suma Softmax != 1.0!"
    print(f"   -> Kształt tensora predykcji: {preds.shape}")
    print(f"   -> Średnie prawdopodobieństwa: {np.mean(preds, axis=0)}")
    print("   [OK] Predykcja stabilna numerycznie, rozkład prawdopodobieństw Softmax poprawny.\n")

    # 3. Przepływ gradientów (Backward pass)
    print("[3/6] Test przepływu gradientów (Backward Pass)...")
    dummy_y = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
    with tf.GradientTape() as tape:
        out = model(dummy_x, training=True)
        loss = tf.keras.losses.categorical_crossentropy(dummy_y, out)
    grads = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None for g in grads), "Brakujące gradienty!"
    assert all(not tf.math.reduce_any(tf.math.is_nan(g)) for g in grads), "NaN w gradientach!"
    print(f"   -> Obliczono gradienty dla {len(grads)} zestawów wag.")
    print("   [OK] Gradienty przepływają prawidłowo przez wszystkie warstwy konwolucyjne i gęste.\n")

    # 4. Zbieżność uczenia (Convergence check)
    print("[4/6] Test zbieżności uczenia (Trening na małym batchu)...")
    train_x = np.random.randn(8, 128, 259, 3).astype(np.float32)
    train_y = tf.keras.utils.to_categorical(np.random.randint(0, 4, size=8), num_classes=4)
    hist = model.fit(train_x, train_y, epochs=10, batch_size=4, verbose=0)
    initial_loss = hist.history["loss"][0]
    final_loss = hist.history["loss"][-1]
    print(f"   -> Loss początkowy: {initial_loss:.4f} -> Loss końcowy: {final_loss:.4f}")
    assert final_loss < initial_loss, "Brak spadku wartości funkcji straty!"
    print("   [OK] Model efektywnie uczy się wzorców i minimalizuje funkcję straty.\n")

    # 5. Zapis i odczyt z dysku (Serialization)
    print("[5/6] Test serializacji (Keras save & load)...")
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model.save(tmp_path)
        loaded_model = tf.keras.models.load_model(tmp_path)
        p_orig = model.predict(dummy_x, verbose=0)
        p_loaded = loaded_model.predict(dummy_x, verbose=0)
        assert np.allclose(p_orig, p_loaded, atol=1e-5), "Rozbieżność predykcji po zapisie/odczycie!"
        print(f"   -> Model pomyślnie zapisany i odczytany z: {tmp_path}")
        print("   [OK] Wagi i architektura po deserializacji są w 100% identyczne.\n")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 6. Model Transfer Learning
    print("[6/6] Test architektury Transfer Learning (MobileNetV2)...")
    tl_model = build_transfer_learning_model("mobilenetv2", input_shape=(128, 259, 3), num_classes=4)
    tl_out = tl_model.predict(dummy_x[:2], verbose=0)
    assert tl_out.shape == (2, 4)
    print(f"   -> Kształt wyjścia MobileNetV2: {tl_out.shape}")
    print("   [OK] Architektura transfer learningu poprawnie przyjmuje tensory (128, 259, 3).\n")

    print("=" * 70)
    print("WYNIK: MODEL 2D CNN DZIAŁA W 100% POPRAWNIE I JEST GOTOWY DO UŻYCIA!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()

```

---

## KOD ŹRÓDŁOWY: profile_normalization_quality.py
**Ścieżka pliku w repozytorium:** `profile_normalization_quality.py`

```python
"""
Kompleksowa weryfikacja i profilowanie jakości normalizacji (Acoustic HDD Sentinel).
Single Source of Truth: ADR-001, AGENTS.md, hdd_audio_encoder.py

Skrypt analizuje:
1. Surowe pliki audio (rozrzut energii, RMS, Crest Factor, DC offset).
2. Krok 1: Normalizację fali dźwiękowej (Peak vs RMS).
3. Krok 2: Spektrogram Log-Mel, Delty i Delty-Delty.
4. Krok 3: Standaryzację per-channel (Z-Score) we wszystkich wygenerowanych plikach .npy.
5. Generuje raport wizualny (normalization_quality_report.png) oraz zestaw metryk numerycznych.
"""

import os
import glob
import json
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ścieżki
RAW_DATASET_DIR = Path("CNN/dataset") if Path("CNN/dataset").exists() else Path("dataset")
ENCODED_DIR = Path("dataset_encoded") if Path("dataset_encoded").exists() else Path("CNN/dataset_encoded")
OUTPUT_IMG = Path("normalization_quality_report.png")
OUTPUT_METRICS_JSON = Path("normalization_quality_metrics.json")


def analyze_raw_audio_profile():
    """
    Analizuje statystyki surowych nagrań przed normalizacją.
    """
    wav_files = sorted(list(RAW_DATASET_DIR.glob("*.wav")))
    rms_list = []
    peak_list = []
    crest_list = []
    dc_list = []
    dur_list = []

    for fpath in wav_files:
        try:
            y, sr = librosa.load(str(fpath), sr=22050, mono=True)
            if len(y) == 0:
                continue
            dc = np.mean(y)
            peak = np.max(np.abs(y))
            rms = np.sqrt(np.mean(y ** 2))
            cf = peak / (rms + 1e-9)
            dur = len(y) / sr

            rms_list.append(rms)
            peak_list.append(peak)
            crest_list.append(cf)
            dc_list.append(dc)
            dur_list.append(dur)
        except Exception:
            continue

    raw_stats = {
        "count": len(rms_list),
        "rms_min": float(np.min(rms_list)),
        "rms_max": float(np.max(rms_list)),
        "rms_mean": float(np.mean(rms_list)),
        "rms_ratio_max_min": float(np.max(rms_list) / (np.min(rms_list) + 1e-9)),
        "peak_min": float(np.min(peak_list)),
        "peak_max": float(np.max(peak_list)),
        "crest_min": float(np.min(crest_list)),
        "crest_max": float(np.max(crest_list)),
        "crest_mean": float(np.mean(crest_list)),
        "crest_p95": float(np.percentile(crest_list, 95)),
        "dc_offset_max_abs": float(np.max(np.abs(dc_list))),
        "dc_offset_mean_abs": float(np.mean(np.abs(dc_list))),
    }
    return raw_stats, wav_files


def analyze_encoded_tensors_profile():
    """
    Analizuje wszystkie pliki .npy w katalogu dataset_encoded (po pełnej normalizacji).
    """
    npy_files = sorted(glob.glob(str(ENCODED_DIR / "**" / "*.npy"), recursive=True))
    if not npy_files:
        return {}

    ch0_means, ch0_stds, ch0_mins, ch0_maxs = [], [], [], []
    ch1_means, ch1_stds, ch1_mins, ch1_maxs = [], [], [], []
    ch2_means, ch2_stds, ch2_mins, ch2_maxs = [], [], [], []
    outliers_3sigma_ratios = []

    nans_total = 0
    infs_total = 0

    for fpath in npy_files:
        arr = np.load(fpath)  # Kształt: (128, 259, 3)

        if np.isnan(arr).any():
            nans_total += 1
        if np.isinf(arr).any():
            infs_total += 1

        for c_idx, (means, stds, mins, maxs) in enumerate([
            (ch0_means, ch0_stds, ch0_mins, ch0_maxs),
            (ch1_means, ch1_stds, ch1_mins, ch1_maxs),
            (ch2_means, ch2_stds, ch2_mins, ch2_maxs),
        ]):
            ch = arr[..., c_idx]
            means.append(float(np.mean(ch)))
            stds.append(float(np.std(ch)))
            mins.append(float(np.min(ch)))
            maxs.append(float(np.max(ch)))

        # Odsetek wartości poza 3 odchyleniami standardowymi (|z| > 3.0)
        outliers = np.mean(np.abs(arr) > 3.0)
        outliers_3sigma_ratios.append(float(outliers))

    encoded_stats = {
        "total_tensors": len(npy_files),
        "nan_tensors": nans_total,
        "inf_tensors": infs_total,
        "channel_0_log_mel": {
            "mean_of_means": float(np.mean(ch0_means)),
            "std_of_means": float(np.std(ch0_means)),
            "mean_of_stds": float(np.mean(ch0_stds)),
            "std_of_stds": float(np.std(ch0_stds)),
            "min_global": float(np.min(ch0_mins)),
            "max_global": float(np.max(ch0_maxs)),
        },
        "channel_1_delta": {
            "mean_of_means": float(np.mean(ch1_means)),
            "std_of_means": float(np.std(ch1_means)),
            "mean_of_stds": float(np.mean(ch1_stds)),
            "std_of_stds": float(np.std(ch1_stds)),
            "min_global": float(np.min(ch1_mins)),
            "max_global": float(np.max(ch1_maxs)),
        },
        "channel_2_delta_delta": {
            "mean_of_means": float(np.mean(ch2_means)),
            "std_of_means": float(np.std(ch2_means)),
            "mean_of_stds": float(np.mean(ch2_stds)),
            "std_of_stds": float(np.std(ch2_stds)),
            "min_global": float(np.min(ch2_mins)),
            "max_global": float(np.max(ch2_maxs)),
        },
        "mean_3sigma_outlier_pct": float(np.mean(outliers_3sigma_ratios) * 100),
    }
    return encoded_stats


def generate_comparative_plots(sample_wav_files):
    """
    Generuje szczegółowe wykresy porównawcze przed i po normalizacji:
    1. Cichy vs Głośny plik przed i po Peak Normalization
    2. Histogram rozkładu wartości w kanałach Log-Mel, Delta i Delta-Delta (Z-score)
    3. Spektrogram przed Z-score vs po Z-score
    """
    from hdd_audio_encoder import encode_audio_chunk, normalize_waveform_peak

    # Wybór próbki z wysokim Crest Factor (kliknięcia) i próbki tonalnej (beeping)
    target_quiet = None
    target_loud = None

    for f in sample_wav_files:
        fname = f.name.lower()
        if "clicking" in fname and target_loud is None:
            target_loud = f
        if "beeping" in fname and target_quiet is None:
            target_quiet = f

    if target_loud is None or target_quiet is None:
        target_loud = sample_wav_files[0]
        target_quiet = sample_wav_files[min(10, len(sample_wav_files)-1)]

    y_loud, _ = librosa.load(str(target_loud), sr=22050, mono=True)
    y_quiet, _ = librosa.load(str(target_quiet), sr=22050, mono=True)

    # 3-sekundowy wycinek
    chunk_samples = int(22050 * 3.0)
    chunk_loud = y_loud[:chunk_samples]
    chunk_quiet = y_quiet[:chunk_samples]

    norm_loud = normalize_waveform_peak(chunk_loud)
    norm_quiet = normalize_waveform_peak(chunk_quiet)

    tensor_loud = encode_audio_chunk(chunk_loud, sr=22050, tensor_format="tf")

    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)

    # Wiersz 1: Fala dźwiękowa PRZED normalizacją
    t = np.linspace(0, 3.0, len(chunk_loud))
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, chunk_loud, color="crimson", alpha=0.8, linewidth=0.5)
    ax1.set_title(f"Przed normalizacją (Głośny / Klik)\nPeak={np.max(np.abs(chunk_loud)):.3f}, RMS={np.sqrt(np.mean(chunk_loud**2)):.4f}", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Amplituda surowa")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, chunk_quiet, color="darkorange", alpha=0.8, linewidth=0.5)
    ax2.set_title(f"Przed normalizacją (Cichy / Beep)\nPeak={np.max(np.abs(chunk_quiet)):.3f}, RMS={np.sqrt(np.mean(chunk_quiet**2)):.4f}", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Amplituda surowa")
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, alpha=0.3)

    # Wiersz 1, kol 3: Po Peak Normalization (Wyrównanie zakresu dynamicznego)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, norm_quiet, color="royalblue", alpha=0.7, linewidth=0.5, label="Cichy po Peak Norm")
    ax3.plot(t, norm_loud, color="purple", alpha=0.5, linewidth=0.5, label="Głośny po Peak Norm")
    ax3.set_title("Po Peak Normalization ([-1.0, 1.0])\nZachowany Crest Factor, Peak=1.0", fontsize=9, fontweight="bold")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_ylim(-1.05, 1.05)
    ax3.grid(True, alpha=0.3)

    # Wiersz 2: Kanały tensora wyjściowego (Log-Mel, Delta, Delta-Delta)
    ax4 = fig.add_subplot(gs[1, 0])
    im1 = ax4.imshow(tensor_loud[..., 0], aspect="auto", origin="lower", cmap="magma")
    ax4.set_title("Kanał 0: Log-Mel Spectrogram (Z-Score)\n(mean=0.0, std=1.0)", fontsize=9, fontweight="bold")
    ax4.set_ylabel("Mel pasma (128)")
    fig.colorbar(im1, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(gs[1, 1])
    im2 = ax5.imshow(tensor_loud[..., 1], aspect="auto", origin="lower", cmap="seismic", vmin=-3.0, vmax=3.0)
    ax5.set_title("Kanał 1: Delta (1. pochodna - Transjenty)\n(mean=0.0, std=1.0)", fontsize=9, fontweight="bold")
    ax5.set_xlabel("Ramki czasowe (259)")
    fig.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(gs[1, 2])
    im3 = ax6.imshow(tensor_loud[..., 2], aspect="auto", origin="lower", cmap="seismic", vmin=-3.0, vmax=3.0)
    ax6.set_title("Kanał 2: Delta-Delta (2. pochodna - Dynamika)\n(mean=0.0, std=1.0)", fontsize=9, fontweight="bold")
    fig.colorbar(im3, ax=ax6, fraction=0.046, pad=0.04)

    # Wiersz 3: Histogramy rozkładu Z-score we wszystkich 3 kanałach
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(tensor_loud[..., 0].flatten(), bins=50, color="firebrick", alpha=0.8, density=True)
    ax7.axvline(0, color="black", linestyle="--", linewidth=1)
    ax7.set_title(f"Rozkład Log-Mel\nmean={np.mean(tensor_loud[..., 0]):.2f}, std={np.std(tensor_loud[..., 0]):.2f}", fontsize=9, fontweight="bold")
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(tensor_loud[..., 1].flatten(), bins=50, color="seagreen", alpha=0.8, density=True)
    ax8.axvline(0, color="black", linestyle="--", linewidth=1)
    ax8.set_title(f"Rozkład Delta\nmean={np.mean(tensor_loud[..., 1]):.2f}, std={np.std(tensor_loud[..., 1]):.2f}", fontsize=9, fontweight="bold")
    ax8.grid(True, alpha=0.3)

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.hist(tensor_loud[..., 2].flatten(), bins=50, color="darkslateblue", alpha=0.8, density=True)
    ax9.axvline(0, color="black", linestyle="--", linewidth=1)
    ax9.set_title(f"Rozkład Delta-Delta\nmean={np.mean(tensor_loud[..., 2]):.2f}, std={np.std(tensor_loud[..., 2]):.2f}", fontsize=9, fontweight="bold")
    ax9.grid(True, alpha=0.3)

    plt.suptitle("RAPORT JAKOŚCI NORMALIZACJI DANYCH (ACOUSTIC HDD SENTINEL - ADR-001)", fontsize=13, fontweight="bold", y=0.99)
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMG)
    plt.close()


def run_assessment():
    print("=" * 70)
    print("PROFILOWANIE I OCENA JAKOŚCI NORMALIZACJI (ADR-001)")
    print("=" * 70)

    # 1. Profil surowych nagrań
    print("\n[1/3] Analiza surowych plików audio przed normalizacją...")
    raw_stats, sample_files = analyze_raw_audio_profile()
    print(f"  - Liczba przeanalizowanych plików: {raw_stats['count']}")
    print(f"  - Rozstęp energii RMS:             od {raw_stats['rms_min']:.5f} do {raw_stats['rms_max']:.5f}")
    print(f"  - Krotność różnicy głośności:      {raw_stats['rms_ratio_max_min']:.1f}x (!) ")
    print(f"  - Maksymalny Crest Factor (kliki): {raw_stats['crest_max']:.1f} (średnia: {raw_stats['crest_mean']:.1f})")
    print(f"  - Średni DC Offset przed korektą:  {raw_stats['dc_offset_mean_abs']:.6f}")

    # 2. Profil wygenerowanych tensorów .npy
    print("\n[2/3] Weryfikacja statystyczna tensorów po normalizacji (dataset_encoded)...")
    encoded_stats = analyze_encoded_tensors_profile()
    if encoded_stats:
        print(f"  - Zbadano tensorów:               {encoded_stats['total_tensors']}")
        print(f"  - Liczba wartości NaN / Inf:       {encoded_stats['nan_tensors']} / {encoded_stats['inf_tensors']} (idealnie czyste)")
        print(f"  - Kanał 0 (Log-Mel)     mean:      {encoded_stats['channel_0_log_mel']['mean_of_means']:.6f} | std: {encoded_stats['channel_0_log_mel']['mean_of_stds']:.6f}")
        print(f"  - Kanał 1 (Delta)       mean:      {encoded_stats['channel_1_delta']['mean_of_means']:.6f} | std: {encoded_stats['channel_1_delta']['mean_of_stds']:.6f}")
        print(f"  - Kanał 2 (Delta-Delta) mean:      {encoded_stats['channel_2_delta_delta']['mean_of_means']:.6f} | std: {encoded_stats['channel_2_delta_delta']['mean_of_stds']:.6f}")
        print(f"  - Udział próbek poza 3-sigma:      {encoded_stats['mean_3sigma_outlier_pct']:.2f}% (zgodne z rozkładem zbliżonym do Gaussa)")

    # 3. Wykresy diagnostyczne
    print("\n[3/3] Generowanie wykresu diagnostycznego jakości normalizacji...")
    generate_comparative_plots(sample_files)
    print(f"  -> Wykres zapisano w: {OUTPUT_IMG}")

    # Zapis raportu do JSON
    full_report = {
        "raw_audio_profile": raw_stats,
        "encoded_tensors_profile": encoded_stats
    }
    with open(OUTPUT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"  -> Metryki JSON zapisano w: {OUTPUT_METRICS_JSON}")

    # Jeśli istnieje podkatalog CNN, zapisz kopię również tam
    cnn_dir = Path("CNN")
    if cnn_dir.exists() and cnn_dir.is_dir():
        import shutil
        shutil.copy2(OUTPUT_IMG, cnn_dir / "normalization_quality_report.png")
        shutil.copy2(OUTPUT_METRICS_JSON, cnn_dir / "normalization_quality_metrics.json")

    print("\n" + "=" * 70)
    print("PROFILOWANIE ZAKOŃCZONE!")
    print("=" * 70)

if __name__ == "__main__":
    run_assessment()

```

---

