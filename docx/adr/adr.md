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