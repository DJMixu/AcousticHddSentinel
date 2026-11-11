# 📱 Acoustic HDD Sentinel — Moduł Aplikacji Mobilnej (Android Java)

Moduł natywnej aplikacji mobilnej na platformę **Android** zaimplementowany w czystej **Javie** (OpenJDK 17 / API 26+), służący do akustycznej detekcji i wieloklasowej klasyfikacji uszkodzeń mechanicznych dysków twardych (HDD) w czasie rzeczywistym.

Projekt zrealizowany w ramach przedmiotu **Projekt - Aplikacje Mobilne** (prowadzący: inż. Dominik Ciura) w oparciu o specyfikację `docx/ADR APLIKACJI MOBILNEJ.md` oraz `docx/adr/adr.md` (ADR-001).

---

## 🛠️ Architektura i Stos Technologiczny

- **Język i platforma:** Java (OpenJDK 17), Android SDK (Min API 26 / Android 8.0 Oreo, Target API 34 / Android 14).
- **Wzorzec architektoniczny:** MVVM (Model-View-ViewModel) z Repository Pattern, `LiveData` oraz ViewBinding.
- **Komponent Hardware:** Wbudowany mikrofon urządzenia (`android.media.AudioRecord`, 16-bit PCM Mono, sampling 22050 Hz z automatycznym fallbackiem do 44100 Hz i decymacją 2:1).
- **Zmienny czas nagrania:** Płynna regulacja suwakiem od **5 do 30 sekund** (domyślnie 10s) z podziałem na okna analityczne 3.0s (66 150 próbek).
- **Decyzja wielookienkowa:** Jeżeli w **którymkolwiek** oknie zostanie wykryta anomalia mechaniczna (`clicking`, `beeping`, `scratching`), cały test klasyfikowany jest jako uszkodzenie (`has_anomaly = 1`).
- **Silnik uczenia maszynowego (Edge AI):** **ONNX Runtime Mobile dla Java** (`ai.onnxruntime.*`).
- **Skwantyzowany model ONNX:** `best_hdd_cnn_model_int8.onnx` (1.17 MB) zoptymalizowany pod urządzenia brzegowe (lokalizacja: `app/src/main/assets/`). Czas inferencji okna: 10–14 ms.
- **Baza Danych:** Natywny **`SQLiteOpenHelper`** (tabela `Diagnostics`) z asynchronicznym repozytorium wielowątkowym (`ExecutorService`).
- **Magazyn Obrazów:** Zapis wygenerowanych spektrogramów Mel jako plików `.png` w pamięci wewnętrznej aplikacji (`context.getFilesDir() + "/spectrograms/"`), w bazie zapisywana jest wyłącznie ścieżka.
- **Tryb 100% OFFLINE:** Całkowity brak uprawnień sieciowych (`INTERNET`), pełna prywatność i praca w warunkach laboratoryjnych/serwerowych.

---

## 📂 Struktura Katalogów Modułu Android

```
android/
├── app/
│   ├── build.gradle.kts             # Konfiguracja modułu w Java (ViewBinding, ONNX Runtime Mobile, AndroidX)
│   ├── proguard-rules.pro           # Reguły ProGuard / R8
│   └── src/
│       ├── main/
│       │   ├── AndroidManifest.xml  # Uprawnienia RECORD_AUDIO i deklaracja aktywności
│       │   ├── assets/              # Artefakty modelu ONNX i specyfikacji
│       │   │   ├── best_hdd_cnn_model_int8.onnx   # Skwantyzowany model 2D CNN (1.17 MB)
│       │   │   ├── classes.json                   # Mapowanie indeksów klas
│       │   │   └── model_card.json                # Karta modelu i specyfikacja DSP
│       │   ├── java/com/acoustichddsentinel/
│       │   │   ├── AcousticSentinelApp.java       # Klasa aplikacji (Application Singleton)
│       │   │   ├── audio/                         # Moduł rejestracji AudioRecord ze sprzętu
│       │   │   │   ├── AudioChunk.java            # Kontener wycinka audio i metryk RMS
│       │   │   │   └── AudioRecorder.java         # Wątek nagrywania 5-30s, decymacja i chunking
│       │   │   ├── data/                          # Baza danych SQLite i repozytorium
│       │   │   │   ├── DiagnosticModel.java       # Model encji wpisu diagnostycznego
│       │   │   │   ├── DiagnosticDbHelper.java    # Implementacja SQLiteOpenHelper (tabela Diagnostics)
│       │   │   │   └── DiagnosticRepository.java  # Asynchroniczne CRUD na puli wątków
│       │   │   ├── dsp/                           # Pełny potok DSP zgodny z ADR-001 w czystej Javie
│       │   │   │   ├── DspConstants.java          # Niezmienniki: SR=22050, FFT=1024, Hop=256, Mel=128
│       │   │   │   ├── MelFilterBank.java         # Bank 128 filtrów Mel z normalizacją Slaneya (< 4e-9 diff)
│       │   │   │   ├── StftTransformer.java       # Cooley-Tukey Radix-2 FFT, okno Hanna i reflect pad
│       │   │   │   ├── AudioPreprocessor.java     # Peak Norm, STFT, Mel, Delta/Delta-Delta, Z-Score
│       │   │   │   └── SpectrogramImageSaver.java # Renderowanie mapy cieplnej Bitmap i zapis do PNG
│       │   │   ├── ml/                            # Inferencja Edge AI
│       │   │   │   ├── AcousticClassifier.java    # Wrapper sesji ONNX Runtime Mobile
│       │   │   │   └── ClassificationResult.java  # Wyniki, prawdopodobieństwa Softmax i zalecenia
│       │   │   └── ui/                            # Warstwa prezentacji i architektura MVVM
│       │   │       ├── MainActivity.java          # Główny ekran z podglądem spektrogramu i wyników
│       │   │       ├── MainViewModel.java         # ViewModel koordynujący sprzęt, ML i bazę
│       │   │       ├── DiagnosticAdapter.java     # Adapter historii RecyclerView
│       │   │       └── DiagnosticUiState.java     # Reaktywny stan interfejsu (UI State)
│       │   └── res/                               # Zasoby XML (layouty Material 3, kolory, ikony)
│       └── test/                                  # Testy jednostkowe warstwy DSP w Java
│           └── java/com/acoustichddsentinel/dsp/AudioPreprocessorTest.java
├── gradle/
│   └── wrapper/
│       ├── gradle-wrapper.jar
│       └── gradle-wrapper.properties
├── build.gradle.kts                 # Główny plik Gradle projektu
├── settings.gradle.kts              # Konfiguracja modułów
├── gradle.properties                # Konfiguracja pamięci JVM
├── local.properties                 # Ścieżka do Android SDK (sdk.dir)
└── README.md                        # Niniejsza dokumentacja
```

---

## 🚀 Jak Otworzyć i Uruchomić w Android Studio

1. Uruchom **Android Studio** (Koala, Hedgehog, Iguana lub nowsze).
2. Wybierz opcję **Open...** i wskaż katalog:
   ```
   /home/DJMixu/Pulpit/AcousticHddSentinel/android
   ```
3. Poczekaj na zakończenie synchronizacji Gradle (**Gradle Sync**).
4. Podłącz fizyczny telefon z Androidem przez USB (z włączonym debugowaniem USB) lub uruchom emulator (API $\ge 26$).
5. Kliknij zielony przycisk **Run 'app'** (`Shift + F10`).
6. Po uruchomieniu nadaj uprawnienie do mikrofonu, ustaw czas rejestracji suwakiem (np. 10s) i kliknij **Rozpocznij Badanie**.

---

## 🔄 Synchronizacja Wag z Pipeline Treningowego

Gdy wytrenujesz nowy model za pomocą PyTorch (`ml/train_cnn.py` lub `ml/train_ensemble.py`), wyeksportuj zoptymalizowany model ONNX bezpośrednio do katalogu assets aplikacji:
```bash
python ml/export_model.py --weights training_output/best_hdd_cnn_model.pt --output-dir exported_models
```
Skrypt `export_model.py` automatycznie synchronizuje pliki `best_hdd_cnn_model_int8.onnx`, `classes.json` oraz `model_card.json` z katalogiem `android/app/src/main/assets/`.
