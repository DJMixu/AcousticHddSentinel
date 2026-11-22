---
name: dsp-adversarial-sentinel
description: >-
  Standard analizy cyfrowego przetwarzania sygnałów (DSP), fizycznych strażników obrotów RPM,
  odporności adwersarialnej oraz testowania potoku hybrydowego (Python & Android Java 17).
  Aktywuj ten skill, gdy zadanie dotyczy: analizy FFT/STFT, banku filtrów Mel, stacjonarności pitch,
  filtracji szumu wentylatorów PC, odrzucania mowy/wibrato, kalibracji czułości mikrofonu (RAW Audio)
  lub weryfikacji całego datasetu audio.
---

# DSP & Adversarial Audio Sentinel — Przewodnik Inżynierski

## 1. Cel i Zakres Skilla
System Acoustic HDD Sentinel łączy głębokie sieci 2D CNN z deterministycznym cyfrowym przetwarzaniem sygnałów (DSP).
Niniejszy skill definiuje procedury inżynierskie dla:
1. **Analizy i weryfikacji prążków RPM:** Metoda Welcha, interpolacja paraboliczna w 5 oknach czasowych, ostrość prążka $\ge 6.0$.
2. **Strażników fizycznych i filtracji adwersarialnej (`DSPAdversarialGuard` & `DspPhysicalGuard`):**
   - Aerodynamiczny szum wentylatorów komputerowych: `slope <= -1.80`, `kurtosis < 4.5`, `crest < 4.2`.
   - Modulacja vibrato głosu ludzkiego: modulacja częstotliwości 3–8 Hz, odchylenie $\Delta f > 1.5\text{ Hz}$.
   - Szum ruchu ulicznego / miasta: dominacja pasma $< 300\text{ Hz}$, weto dla stukania gdy `crest < 4.0`.
   - Wykrywanie zatarcia silnika (*Stall Beep Tone*): prążki rezonansowe w paśmie 1000–4500 Hz.
3. **Kalibracji czułości toru mikrofonowego i normalizacji:**
   - Czysta cisza cyfrowa: odrzucenie ramki gdy $RMS < 0.0003$.
   - Cichy sygnał pomiarowy: dynamiczne wzmocnienie z zakresu $0.0003 \le RMS < 0.025$ do poziomu analitycznego $RMS = 0.025$.
   - Tryb RAW Audio: sprzętowy pre-gain $+18.06\text{ dB}$ (`UNPROCESSED_HARDWARE_PREGAIN = 8.0f`) kompensujący odcięcie przedwzmacniacza w Androidzie.
4. **Parzystości numerycznej Python $\leftrightarrow$ Java 17:** Identyczne wyniki DSP na desktopie i telefonie (Zero GC Churn).

---

## 2. Niezmienniki Parametrów DSP (ADR-001)

| Parametr | Wartość | Uzasadnienie inżynierskie |
|:---|:---|:---|
| `TARGET_SR` | **22050 Hz** | Pokrywa pasmo akustyczne HDD do 10 kHz; decymacja 2:1 z 44.1 kHz. |
| `CHUNK_DURATION` | **3.0 s** | 66 150 próbek; minimalny czas na stabilną estymację obrotów RPM (5400/7200 RPM). |
| `N_FFT` | **1024** | Rozdzielczość częstotliwościowa $\Delta f \approx 21.53\text{ Hz}$. |
| `HOP_LENGTH` | **256** | Krok czasowy $\approx 11.61\text{ ms}$; wysoka rozdzielczość transjentów (Crest Factor). |
| `N_MELS` | **128** | Percepcyjna skala Mel w paśmie 50–10000 Hz (norma Slaney). |
| `OVERLAP` | **1.5 s (train), 0.0 s (val/test)** | 50% augmentacji wyłącznie w zbiorze uczącym (Zero Data Leakage). |
| `RMS_MIN_BOOSTABLE` | **0.0003** | Dolny próg słyszalnego sygnału; poniżej uznawany za martwą ciszę. |
| `RMS_TARGET_LEVEL` | **0.025** | Poziom analityczny po automatycznym podbiciu (Auto-Gain Boost). |

---

## 3. Kompletny Zestaw Narzędzi Walidacyjnych DSP

Gdy wprowadzasz zmiany w modułach DSP (`ml/rpm_harmonics_analyzer.py`, `ml/dsp_adversarial_guard.py`, `android/.../dsp/*`):

### Krok 1: Weryfikacja logiczna i fizyczna w Pythonie
```bash
# 1. Test analizatora obrotów RPM i odrzucania głosu/szumów (9/9):
python tests/test_rpm_harmonics_analyzer.py

# 2. Test odporności na szumy wentylatorów PC, GPU i cichy sygnał 45 dB (6/6):
python tests/test_computer_sound_rejection.py

# 3. Profiler jakości normalizacji Z-Score (weryfikacja braku NaN, log(0)):
python tests/profile_normalization_quality.py
```

### Krok 2: Weryfikacja na całym zbiorze danych (Zero Regresji)
```bash
# 1. Test wykrywalności awarii na 100% całego datasetu (6 840 tensorów):
python tests/test_full_dataset_fault_detection_rate.py

# 2. Ewaluacja potoku aplikacji (z DSP vs bez DSP, macierze pomyłek, 0% FPR na dyskach):
python tests/evaluate_app_pipeline_on_dataset.py
```

### Krok 3: Weryfikacja modułu mobilnego Android (Czysta Java 17)
```bash
# 13 klas testowych (74 testy jednostkowe: DSP edge cases, dataset pipeline, ONNX inference):
cd android && ./gradlew test && cd ..
```

---

## 4. Wytyczne dla Modułu Mobilnego Android (ADR-002)

1. **Zero Garbage Collection Churn w pętli audio:**
   Wszystkie tablice pomocnicze (`fftBuffer`, `melEnergies`, `windowBuffer`, `magnitudes`) muszą być prealokowane w konstruktorze `AudioPreprocessor` lub `RpmHarmonicsAnalyzer`. Nigdy nie twórz tablic (`new float[...]`) wewnątrz metod wywoływanych co 3 sekundy.
2. **Brak Kotlina:** Moduł aplikacji mobilnej Android musi pozostać w 100% w czystej Javie 17 (`0% Kotlin runtime`).
3. **Zabezpieczenie zmiennoprzecinkowe IEEE 754:**
   - Zabezpieczenie przed dzieleniem przez zero: $\max(\sigma, 10^{-6})$.
   - Zabezpieczenie logarytmu: $\max(E, 10^{-12})$ przed konwersją do dB.
   - Wartości Z-score przycinane do zakresu $[-5.0, 5.0]$.
