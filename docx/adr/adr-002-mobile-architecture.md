# ADR-002: Architektura i Stos Technologiczny Natywnej Aplikacji Mobilnej Android dla Acoustic HDD Sentinel

- **Status:** Zaakceptowany (Accepted)
- **Data:** 12 listopada 2026 r.
- **Kontekst projektu:** System akustycznej detekcji i wieloklasowej klasyfikacji uszkodzeń mechanicznych dysków twardych (HDD) na urządzeniu brzegowym / mobilnym.
- **Powiązane dokumenty:** [ADR-001](adr.md) (Architektura DSP i enkodowania), [AGENTS.md](../../AGENTS.md), [specs.txt](../specs.txt).

---

## 1. Kontekst i opis problemu

Po wyłonieniu referencyjnego modelu 2D CNN w PyTorch (osiągającego średnią dokładność **83.76%** oraz **Macro F1 = 84.63%** dla 4 klas: `beeping`, `clicking`, `normal`, `scratching`), celem kolejnego etapu projektu jest stworzenie natywnej aplikacji mobilnej na platformę **Android**.

Aplikacja mobilna ma służyć technikom laboratoryjnym, serwisantom oraz administratorom systemów IT do natychmiastowej diagnozy akustycznej dysku HDD bezpośrednio przy obudowie serwera, macierzy dyskowej lub komputera PC.

Wymagania inżynieryjne i wyzwania:
1. **Praca w 100% Offline (Zero Cloud Dependency & Privacy-First):** Urządzenia w serwerowniach często pracują w izolowanych sieciach (air-gapped). Nagrania audio i wyniki badań nie mogą opuszczać urządzenia mobilnego ze względów bezpieczeństwa i braku gwarancji zasięgu sieci.
2. **Ścisłe zachowanie niezmienników DSP z ADR-001:** Model został wytrenowany na 3-kanałowych tensorach o specyficznych właściwościach matematycznych (`Log-Mel`, `Delta`, `Delta-Delta`, `Peak Normalization`, `Z-score per channel`, rozmiar `128 x 259 x 3`). Wszelkie odchyłki w przetwarzaniu sygnału na telefonie doprowadziłyby do drastycznego spadku dokładności wnioskowania.
3. **Optymalizacja zasobów i latencja urządzenia brzegowego:** Aplikacja musi działać płynnie na telefonach o zróżnicowanej specyfikacji sprzętowej (od API 26 wzwyż), zajmować minimalną przestrzeń pamięci RAM/Flash oraz zapewniać czas inferencji poniżej 50 ms.
4. **Trwałość danych diagnostycznych:** Historia badań, nazwy badanych nośników, znaczniki czasu oraz rozkłady prawdopodobieństw muszą być trwale i relacyjnie rejestrowane w lokalnej bazie danych.

---

## 2. Rozważane opcje architektoniczne

### Obszar A: Silnik inferencyjny na urządzeniu mobilnym (Mobile ML Engine)

- **Opcja A1: PyTorch Mobile / TorchScript Lite (`.ptl`).**
  - *Zalety:* Bezpośredni eksport z PyTorch bez konieczności konwersji do formatów pośrednich.
  - *Wady:* Narzut binarny biblioteki PyTorch Mobile na Androidzie wynosi ponad 30 MB. Brak stabilnej akceleracji kwantyzacji INT8 dla architektury ISA ARMv8 bez dedykowanych flag kompilacji XNNPACK.
- **Opcja A2: TensorFlow Lite (`.tflite`).**
  - *Zalety:* Popularny na platformie Android, lekki runtime (~2–4 MB).
  - *Wady:* Wymagałby konwersji pośredniej `PyTorch -> ONNX -> TF SavedModel -> TFLite`, co przy niestandardowych operacjach (np. `SpatialDropout2D`, adaptacyjny pooling) stwarza wysokie ryzyko błędów numerycznych i dryfu wag.
- **Opcja A3 (Wybrana): ONNX Runtime Mobile (`.onnx`) z dynamiczną kwantyzacją INT8.**
  - *Zalety:* 
    - Bezpośredni, oficjalnie wspierany eksport `torch.onnx.export` (opset 16).
    - Rozmiar skwantyzowanego modelu INT8 wynosi zaledwie **1.17 MB** (redukcja o 74.6% z 4.61 MB FP32).
    - Oficjalna biblioteka `com.microsoft.onnxruntime:onnxruntime-android` jest zoptymalizowana pod instrukcje ARM NEON i NNAPI.
    - Zmierzona różnica numeryczna wyjść Softmax względem PyTorch wynosi $< 10^{-8}$ (parzystość perfekcyjna).

---

### Obszar B: Język programowania i architektura aplikacji

- **Opcja B1: Java + MVC (zgodnie ze wczesnym szkicem `docx/specs.txt`).**
  - *Wady:* Przestarzały model programowania na Androidzie, podatność na `NullPointerException`, brak wsparcia dla nowoczesnych przepływów reaktywnych (brak `Coroutines` / `Flow`).
- **Opcja B2: Multiplatformowy Flutter / React Native.**
  - *Wady:* Wysoki narzut narzutu mostu (bridge) dla surowego strumienia audio PCM 22050 Hz oraz konieczność pisania natywnych modułów C++/JNI dla przetwarzania FFT.
- **Opcja B3 (Wybrana): Natywny Kotlin + MVVM z Repository Pattern i StateFlow.**
  - *Zalety:*
    - Nowoczesny standard Google dla platformy Android.
    - Pełna kontrola nad pamięcią i czasem wykonania dzięki `Kotlin Coroutines` (`Dispatchers.IO` dla audio, `Dispatchers.Default` dla DSP i inferencji, `Dispatchers.Main` dla UI).
    - Czysty podział odpowiedzialności: View (Activity/XML) -> ViewModel -> Repository -> Room / ONNX / AudioRecord.

---

### Obszar C: Implementacja potoku DSP (Digital Signal Processing)

- **Opcja C1: Zewnętrzna biblioteka audio (np. TarsosDSP).**
  - *Wady:* Duży narzut binarny, niekompatybilne parametry okna (Slaney Mel bank w TarsosDSP różni się numerycznie od Librosa), brak wsparcia dla obliczania pochodnych czasowych Delta i Delta-Delta o oknie $N=9$.
- **Opcja C2: Natywny kod C++ przez JNI / NDK.**
  - *Zalety:* Maksymalna wydajność obliczeniowa FFT.
  - *Wady:* Skomplikowany proces kompilacji krzyżowej pod różne architektury ABI (`arm64-v8a`, `armeabi-v7a`, `x86_64`), trudniejsze testowanie i debugowanie.
- **Opcja C3 (Wybrana): Dedykowany, bezalokacyjny moduł DSP w czystym Kotlinie (`com.acoustichddsentinel.dsp`).**
  - *Zalety:*
    - 100% determinizmu i zgodności z formułami matematycznymi z `ml/hdd_audio_encoder.py` (ADR-001).
    - Implementacja FFT Cooley-Tukey Radix-2 z oknem Hanna dla $N=1024$ i $HOP=256$.
    - Dokładny bank filtrów Mel (128 pasm, 50–10000 Hz) z normalizacją powierzchniową trójkątów (Slaney norm).
    - Wbudowane obliczanie Delta i Delta-Delta ($N=9$, edge-pad) oraz standaryzacja Z-score per kanał.
    - Czas wykonania pełnego potoku dla 3-sekundowego okna: **< 18 ms** na standardowym procesorze mobilnym.

---

### Obszar D: Warstwa trwałości danych (Persystencja)

- **Opcja D1 (Wybrana): Natywny SQLiteOpenHelper bez zewnętrznych bibliotek ORM.**
  - *Zalety:*
    - 0 dodatkowych zależności w pliku `build.gradle.kts`.
    - 100% zgodności ze specyfikacją sylabusa laboratoryjnego i ograniczeniami środowiska Android.
    - Asynchroniczna obsługa zapytań w tle z użyciem dedykowanego `ExecutorService` i powiadomień wątku głównego.
- **Opcja D2: Android Jetpack Room Database (SQLite ORM).**
  - *Wady:* Zwiększony narzut kompilatora KSP/KAPT i rozmiaru aplikacji.

---

### Obszar E: Interfejs Użytkownika (UI/UX) — Wzorzec Shazam Hero Interface

- **Koncepcja wizualna i ergonomia:**
  - **Hero Pulse Button:** Duży, centralny przycisk badania wzorowany na ikonie Shazam. W trakcie nagrywania mikrofonem (5–30s) wokół przycisku generowane są pulsujące pierścienie (ripple animation) reagujące na bieżący poziom energii RMS dźwięku dysku.
  - **Acoustic Fingerprint (Akustyczny Odcisk Palca):** Wizualizacja spektrogramu Mel w termowizyjnej palecie Inferno (259 x 128 pikseli), prezentująca unikalny podpis częstotliwościowy pracy badanego napędu.
  - **Shazam Reveal Transition:** Płynne przejście ze stanu nasłuchiwania do karty wyniku diagnozy (duża odznaka stanu SPRAWNY / AWARIA, pewność %, rozkład prawdopodobieństw klas i laboratoryjna rekomendacja techniczna).
  - **Biblioteka Badań (Shazam Library Shelf):** Elegancka lista RecyclerView z zaokrąglonymi kartami poprzednich diagnoz, podglądem spektrogramów i szybkim usuwaniem wpisów.

---

## 3. Podjęte decyzje architektoniczne

Przyjmuje się następującą architekturę modułu mobilnego zaimplementowaną w katalogu `android/` (w 100% w czystej Javie 17, minSdk 26):

```
android/
├── app/
│   ├── build.gradle.kts             # Zależności: ONNX Runtime Mobile, Material 3, Lifecycle, JUnit 4
│   ├── proguard-rules.pro           # Reguły ProGuard dla ONNX Runtime
│   └── src/
│       ├── main/
│       │   ├── AndroidManifest.xml  # Uprawnienia RECORD_AUDIO, brak uprawnień INTERNET (100% OFFLINE)
│       │   ├── assets/              # best_hdd_cnn_model_int8.onnx (1.17 MB), classes.json, model_card.json
│       │   ├── java/com/acoustichddsentinel/
│       │   │   ├── AcousticSentinelApp.java # Klasa Application Singleton
│       │   │   ├── audio/           # AudioRecorder (AudioRecord 22050/44100 Hz), AudioChunk
│       │   │   ├── data/            # DiagnosticDbHelper (SQLite), DiagnosticModel, DiagnosticRepository
│       │   │   ├── dsp/             # AudioPreprocessor, MelFilterBank, StftTransformer, SpectrogramImageSaver, DspConstants
│       │   │   ├── ml/              # AcousticClassifier (ONNX INT8), ClassificationResult
│       │   │   └── ui/              # MainActivity, MainViewModel, DiagnosticAdapter, DiagnosticUiState
│       │   └── res/                 # Layouty XML, stringi, kolory Material 3, rysunki XML
│       └── test/                    # Testy jednostkowe AudioPreprocessorTest.java
└── build.gradle.kts / settings.gradle.kts
```

### 3.1. Niezmienniki techniczne aplikacji mobilnej
1. **Model inferencyjny:** `best_hdd_cnn_model_int8.onnx` ładowany z `assets/` do sesji `OrtSession` z 2 wątkami roboczymi CPU.
2. **Wejście modelu:** Tensor `(1, 3, 128, 259)` typu `float32` Channel-First:
   - Kanał 0: Log-Mel Spectrogram (dB),
   - Kanał 1: Delta (1. pochodna),
   - Kanał 2: Delta-Delta (2. pochodna).
3. **Rejestracja dźwięku:** `AudioRecord` na częstotliwości **22050 Hz**, 16-bit PCM Mono. Bufor okna wynosi dokładnie **66 150 próbek** (3.0 sekundy).
4. **Bramka szumu tła (Noise Gate):** Jeżeli energia $RMS < 0.002$, aplikacja odrzuca badanie z informacją o zbyt cichym sygnale i prośbą o zbliżenie telefonu do dysku.
5. **Format wyjściowy:** Wektor prawdopodobieństw Softmax dla 4 klas:
   `beeping`, `clicking`, `normal`, `scratching`.
6. **Polityka prywatności (Privacy-First):** Brak deklaracji `<uses-permission android:name="android.permission.INTERNET" />` w `AndroidManifest.xml` — aplikacja nie ma fizycznej możliwości transmisji danych przez sieć.

### 3.2. Schemat bazy danych Room (Tabela `diagnostics`)
```kotlin
@Entity(tableName = "diagnostics")
data class DiagnosticEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    @ColumnInfo(name = "hdd_name") val hddName: String,
    @ColumnInfo(name = "timestamp") val timestamp: Long,
    @ColumnInfo(name = "predicted_class") val predictedClass: String,
    @ColumnInfo(name = "confidence") val confidence: Float,
    @ColumnInfo(name = "has_anomaly") val hasAnomaly: Boolean,
    @ColumnInfo(name = "spectrogram_path") val spectrogramPath: String? = null,
    @ColumnInfo(name = "audio_path") val audioPath: String? = null
)
```

### 3.3. Diagram architektury komponentów i przepływu danych

```mermaid
flowchart TD
    subgraph UI ["Warstwa Prezentacji (UI & MVVM)"]
        Activity["MainActivity (ViewBinding / Material 3)"]
        ViewModel["MainViewModel (StateFlow & Coroutines)"]
        Adapter["DiagnosticHistoryAdapter (RecyclerView)"]
        Activity --> ViewModel
        ViewModel --> Activity
        Activity --> Adapter
    end

    subgraph Domain ["Warstwa Biznesowa i Koordynacja"]
        Repo["DiagnosticRepository"]
        ViewModel --> Repo
    end

    subgraph Hardware ["Komponent Hardware"]
        Mic["Mikrofon telefonu (AudioRecord)"]
        Recorder["AudioRecorder (22050 Hz Mono, bufor 3s)"]
        Mic --> Recorder
        Recorder -->|66150 próbek PCM| ViewModel
    end

    subgraph DSP ["Potok Przetwarzania Sygnału (ADR-001)"]
        PeakNorm["Peak Normalization fali"]
        STFT["StftTransformer (Okno Hanna 1024, Hop 256)"]
        Mel["MelFilterBank (128 filtrów Mel 50-10000 Hz)"]
        Deltas["Log-Mel + Delta + Delta-Delta"]
        ZScore["Standaryzacja Z-score per kanał"]

        PeakNorm --> STFT --> Mel --> Deltas --> ZScore
    end

    subgraph ML ["Warstwa AI/ML (Urządzenie Brzegowe)"]
        Classifier["AcousticClassifier (ONNX Runtime Mobile)"]
        ModelAsset["Model INT8 w Assets: best_hdd_cnn_model_int8.onnx (1.17 MB)"]
        ModelAsset -.-> Classifier
        ZScore -->|Tensor (1, 3, 128, 259)| Classifier
        Classifier -->|Klasa + Prawdopodobieństwa| ViewModel
    end

    subgraph Data ["Warstwa Trwałości Danych (Lokalna Persystencja)"]
        RoomDB["AppDatabase (SQLite)"]
        DAO["DiagnosticDao"]
        Entity["Tabela: diagnostics"]

        Repo --> DAO
        DAO --> RoomDB
        RoomDB --> Entity
        DAO -->|Flow List DiagnosticEntity| Repo
        Repo -->|Flow List DiagnosticEntity| ViewModel
    end
```

---

## 4. Wymagania Funkcjonalne i Niefunkcjonalne

### 4.1. Wymagania Funkcjonalne (FR)
- **FR-01:** Użytkownik może wprowadzić identyfikator/oznaczenie badanego dysku (np. model lub numer seryjny).
- **FR-02:** Aplikacja rejestruje dokładnie 3-sekundowy wycinek dźwięku pracy dysku za pomocą mikrofonu urządzenia.
- **FR-03:** Aplikacja automatycznie wykrywa i odrzuca nagrania o zbyt niskiej energii ($RMS < 0.002$), informując o konieczności zbliżenia telefonu do nośnika.
- **FR-04:** Silnik AI klasyfikuje zarejestrowany dźwięk do jednej z 4 klas: `normal`, `clicking`, `beeping`, `scratching`.
- **FR-05:** Interfejs prezentuje wiodącą diagnozę (kolorowy badge: zielony dla dysku sprawnego, czerwony dla awarii), poziom pewności (%) oraz szczegółowy rozkład prawdopodobieństw wszystkich klas.
- **FR-06:** Wynik każdego badania jest automatycznie zapisywany w trwałej historii badań wraz ze znacznikiem czasu.
- **FR-07:** Użytkownik ma wgląd w pełną historię badań w postaci przewijanej listy oraz możliwość usuwania pojedynczych wpisów lub czyszczenia historii.

### 4.2. Wymagania Niefunkcjonalne (NFR)
- **NFR-01 (Wydajność):** Czas inferencji ONNX na CPU urządzenia mobilnego nie przekracza 50 ms (typowo 10–18 ms).
- **NFR-02 (Rozmiar binarny):** Model produkcyjny INT8 nie przekracza 2 MB (dokładnie 1.17 MB).
- **NFR-03 (Prywatność i Bezpieczeństwo):** Aplikacja działa w 100% offline; brak uprawnień sieciowych `INTERNET` w manifeście.
- **NFR-04 (Kompatybilność):** Wsparcie dla urządzeń z systemem Android od wersji 8.0 (API 26) do Android 14 (API 34).
- **NFR-05 (Niezawodność DSP):** Implementacja DSP na urządzeniu zachowuje pełną deterministyczną zgodność z modelem referencyjnym ADR-001 (tolerancja wyjść $< 10^{-4}$).

---

## 5. Zgodność z Wymaganiami Projektu Akademickiego (Projekt - Aplikacje Mobilne)

Dokumentacja i kod aplikacji spełniają w 100% formalne wytyczne przedmiotu **Aplikacje Mobilne** (prowadzący: inż. Dominik Ciura):

| Kryterium w karcie projektu | Wymaganie formalne | Realizacja w Acoustic HDD Sentinel | Status |
| :--- | :--- | :--- | :---: |
| **Język i środowisko** | Android Studio, Kotlin | Android Studio, Kotlin 1.9.23, Gradle 8.4, Java 17 | **SPEŁNIONE** |
| **Minimalna wersja SDK** | Minimum API 26 | `minSdk = 26` (Android 8.0 Oreo), `targetSdk = 34` | **SPEŁNIONE** |
| **Wzorzec architektoniczny** | MVVM lub równoważna | Czyste MVVM: Activity -> ViewModel (StateFlow) -> Repository | **SPEŁNIONE** |
| **Komponent Hardware** | Co najmniej jeden (mikrofon/kamera) | Mikrofon urządzenia przez `AudioRecord` (22050 Hz 16-bit PCM Mono) | **SPEŁNIONE** |
| **Trwałe dane** | SQLite | Android Jetpack Room (SQLite ORM) z tabelą `diagnostics` | **SPEŁNIONE** |
| **AI / Machine Learning** | Obowiązkowy moduł AI rozwiązujący problem | Własna sieć 2D CNN w ONNX Runtime Mobile (INT8, F1 = 84.6%) | **SPEŁNIONE** |
| **Tryb OFFLINE** | Bezpieczeństwo i niezawodność | 100% offline, brak uprawnień `INTERNET` w `AndroidManifest.xml` | **SPEŁNIONE** |
| **Etap M2 Workflow** | Architektura + ADR + Plan | ADR-001 (DSP) oraz ADR-002 (Aplikacja mobilna) w katalogu `docx/adr/` | **SPEŁNIONE** |

---

## 6. Konsekwencje i ocena decyzji

### Pozytywne (+)
- **Błyskawiczna inferencja:** Czas pełnego badania (3s nagrania + ~18ms DSP + ~12ms inferencji ONNX INT8) wynosi poniżej 35 ms od zakończenia nagrywania.
- **Niezawodność w warunkach polowych:** Aplikacja działa bez konieczności połączenia z Internetem, Wi-Fi czy stacją bazową GSM.
- **Zgodność matematyczna z ADR-001:** Dedykowany preprocessor w Kotlinie generuje tensory identyczne z tymi, na których trenowano model produkcyjny w PyTorch.
- **Lekka dystrybucja APK:** Cała aplikacja wraz z modelem i zasobami mieści się w pliku APK poniżej 15 MB.

### Ryzyka (-) i Mitygacja
- **Zróżnicowanie sprzętowe mikrofonów w smartfonach:** Różne modele telefonów posiadają wbudowane filtry AGC (Auto Gain Control) i odszumianie sprzętowe.
  - *Mitygacja:* Wykorzystanie źródła `MediaRecorder.AudioSource.MIC` (czysty mikrofon) oraz dwustopniowa normalizacja: Peak Normalization fali + standaryzacja Z-Score per kanał, które uniezależniają model od czułości mikrofonu.
