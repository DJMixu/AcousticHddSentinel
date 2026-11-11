# 📱 Acoustic HDD Sentinel — Moduł Aplikacji Mobilnej (Android)

Moduł natywnej aplikacji mobilnej na platformę **Android** (Kotlin), służący do akustycznej detekcji i klasyfikacji uszkodzeń mechanicznych dysków twardych (HDD) w czasie rzeczywistym.

---

## 🛠️ Architektura i Stos Technologiczny

- **Język i platforma:** Kotlin, Android SDK (Min API 26 / Android 8.0 Oreo, Target API 34 / Android 14).
- **Wzorzec architektoniczny:** MVVM (Model-View-ViewModel) z Repository Pattern i StateFlow.
- **Silnik uczenia maszynowego:** **ONNX Runtime Mobile** (`com.microsoft.onnxruntime:onnxruntime-android`).
- **Skwantyzowany model ONNX:** `best_hdd_cnn_model_int8.onnx` (1.17 MB) zoptymalizowany pod urządzenia brzegowe (lokalizacja: `app/src/main/assets/`).
- **Lokalna baza danych:** Android Jetpack **Room / SQLite** do trwałego zapisu historii diagnoz i metryk akustycznych.
- **Rejestracja audio:** `AudioRecord` (22050 Hz, 16-bit PCM Mono, bufor okna 3.0s — 66 150 próbek).
- **Potok DSP (Digital Signal Processing):**
  - Normalizacja szczytowa fali (Peak Normalization do [-1.0, 1.0]),
  - Okno Hanna + Transformata STFT (`n_fft = 1024`, `hop_length = 256`),
  - 128 filtrów Mel w paśmie 50–10000 Hz z kompresją dB ($10 \cdot \log_{10}$),
  - Wyliczanie pierwszej (Delta) i drugiej (Delta-Delta) pochodnej czasowej w oknie $N = 9$,
  - Standaryzacja Z-Score per kanał ($\mu = 0, \sigma = 1$),
  - Wygenerowanie tensora wejściowego o kształcie **`(1, 3, 128, 259)`**.

---

## 📂 Struktura Katalogów Modułu

```
android/
├── app/
│   ├── build.gradle.kts             # Zależności modułu aplikacji (ONNX Runtime, Room, Coroutines)
│   ├── proguard-rules.pro           # Reguły ProGuard / R8
│   └── src/
│       ├── main/
│       │   ├── AndroidManifest.xml  # Uprawnienia RECORD_AUDIO i deklaracje
│       │   ├── assets/              # Artefakty modelu ONNX i specyfikacji
│       │   │   ├── best_hdd_cnn_model_int8.onnx
│       │   │   ├── classes.json
│       │   │   └── model_card.json
│       │   ├── java/com/acoustichddsentinel/
│       │   │   ├── AcousticSentinelApp.kt   # Klasa Application
│       │   │   ├── audio/                   # Moduł AudioRecord i obsługa buforów
│       │   │   │   ├── AudioChunk.kt
│       │   │   │   ├── AudioRecorder.kt
│       │   │   │   └── AudioRecordState.kt
│       │   │   ├── data/                    # Baza danych Room SQLite
│       │   │   │   ├── local/
│       │   │   │   │   ├── AppDatabase.kt
│       │   │   │   │   ├── DiagnosticDao.kt
│       │   │   │   │   └── DiagnosticEntity.kt
│       │   │   │   └── repository/
│       │   │   │       └── DiagnosticRepository.kt
│       │   │   ├── dsp/                     # Implementacja potoku DSP zgodna z ADR-001
│       │   │   │   ├── DspConstants.kt
│       │   │   │   ├── MelFilterBank.kt
│       │   │   │   ├── StftTransformer.kt
│       │   │   │   └── AudioPreprocessor.kt
│       │   │   ├── ml/                      # Inferencja ONNX Runtime
│       │   │   │   ├── AcousticClassifier.kt
│       │   │   │   └── ClassificationResult.kt
│       │   │   └── ui/                      # Warstwa prezentacji i MVVM
│       │   │       ├── MainActivity.kt
│       │   │       ├── DiagnosticHistoryAdapter.kt
│       │   │       └── viewmodel/
│       │   │           ├── DiagnosticUiState.kt
│       │   │           ├── MainViewModel.kt
│       │   │           └── MainViewModelFactory.kt
│       │   └── res/                         # Zasoby (layouty XML, kolory, motywy, ikony)
│       └── test/                            # Testy jednostkowe DSP i logiki
│           └── java/com/acoustichddsentinel/dsp/AudioPreprocessorTest.kt
├── gradle/
│   └── wrapper/
│       ├── gradle-wrapper.jar
│       └── gradle-wrapper.properties
├── build.gradle.kts                 # Konfiguracja buildscriptu projektu
├── settings.gradle.kts              # Konfiguracja modułów Gradle
├── gradle.properties                # Flagi JVM i AndroidX
└── gradlew / gradlew.bat            # Wrapper Gradle
```

---

## 🚀 Jak Otworzyć i Uruchomić w Android Studio

1. Uruchom **Android Studio** (Hedgehog, Iguana, Jellyfish lub nowsze).
2. Wybierz opcję **Open...** i wskaż katalog:
   ```
   /home/DJMixu/Pulpit/AcousticHddSentinel/android
   ```
3. Poczekaj na zakończenie synchronizacji Gradle (**Gradle Sync**).
4. Podłącz fizyczny telefon z Androidem przez USB (z włączonym debugowaniem USB) lub uruchom emulator (API $\ge 26$).
5. Kliknij zielony przycisk **Run 'app'** (`Shift + F10`).
6. Po uruchomieniu nadaj uprawnienie do nagrywania dźwięku z mikrofonu.

---

## 🔄 Synchronizacja Modeli z Pipeline Treningowego

Gdy wytrenujesz nowy model za pomocą `train_cnn.py` lub `train_ensemble.py`, uruchom w katalogu głównym:
```bash
python export_model.py
```
Skrypt wyeksportuje zoptymalizowany model INT8 i automatycznie zaktualizuje pliki `best_hdd_cnn_model_int8.onnx`, `classes.json` oraz `model_card.json` w katalogu `android/app/src/main/assets/`.
