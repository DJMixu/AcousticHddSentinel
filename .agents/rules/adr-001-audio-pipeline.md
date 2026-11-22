---
trigger: always_on
globs: ["**/*.py", "**/*.java"]
---

# Reguły inżynierii audio, DSP i modelu 2D CNN (ADR-001 & ADR-002)

1. **Brak Data Leakage (Podział na poziomie plików):**
   Podział na `train` (68%), `validate` (14%) i `test` (18%) MUSI odbywać się na całych nagraniach źródłowych przed ich podziałem na 3-sekundowe okna czasowe. Nigdy nie stosuj losowego podziału bezpośrednio na wygenerowanych plikach `.npy`.

2. **Kształt tensora wejściowego:**
   - Na dysku w plikach `.npy`: tensor `float32` o stałych wymiarach **`(128, 259, 3)`** (Channel-Last).
   - W pamięci modelu PyTorch/ONNX: tensor **`(B, 3, 128, 259)`** (Channel-First).
   - Klasa `AcousticHddSentinelCNN` w `ml/model_cnn.py` transparentnie obsługuje oba układy wymiarów.

3. **Parametry cyfrowego przetwarzania sygnałów (DSP):**
   - `TARGET_SR = 22050` Hz, decymacja 2:1 z 44.1 kHz filtrem Butterwortha 4. rzędu ($f_c = 9500\text{ Hz}$).
   - `N_FFT = 1024`, `HOP_LENGTH = 256`, `N_MELS = 128` (pasmo 50–10000 Hz, norma Slaney).
   - `CHUNK_DURATION = 3.0` s (66 150 próbek), `OVERLAP = 1.5` s (train), `0.0` s (val/test).
   - Progi RMS: Martwa cisza $RMS < 0.0003$. Dynamic Auto-Gain dla cichego sygnału $0.0003 \le RMS < 0.025$ do poziomu analitycznego $RMS = 0.025$.
   - Tryb RAW Audio: sprzętowy pre-gain $+18.06\text{ dB}$ (`UNPROCESSED_HARDWARE_PREGAIN = 8.0f`) dla źródła `AudioSource.UNPROCESSED`.

4. **Strażnicy Fizyczni i Odporność Adwersarialna:**
   - Kryterium szumu wentylatorów PC: `slope <= -1.80`, `kurtosis < 4.5`, `crest < 4.2`.
   - Modulacja vibrato mowy: badanie stacjonarności pitch w 5 oknach, odrzucenie gdy $\Delta f > 1.5\text{ Hz}$.
   - Weryfikacja obrotów wrzeciona: ostrość prążka `f0_sharpness >= 6.0` oraz prążki 5400/7200 RPM.
   - Szum uliczny: dominacja pasma $< 300\text{ Hz}$, weto dla stukania gdy `crest < 4.0`.

5. **Natywny Moduł Android (100% Czysta Java 17):**
   - Bezwzględny zakaz kodu w Kotlinie (`0% Kotlin runtime`).
   - Zero GC Churn w pętli audio: wszystkie bufory i tablice pomocnicze prealokowane w konstruktorach.
   - Wzorzec Strategii z podwójnym modelem (`LegacyCnnDetector` vs `AlternativeCnnDetector`).

6. **Obsługa uszkodzonych nagrań źródłowych:**
   W zbiorze mogą występować pliki uszkodzone (0 bajtów lub błędy kontenera). Każdy skrypt przetwarzający surowe audio musi stosować bloki `try/except` lub normalizować przez `ffmpeg -y -ac 1 -ar 22050`.

7. **Niezmiennik Sprzętowy Mikrofonu Fizycznego:**
   Jedynym właściwym źródłem w systemie Linux jest `alsa_input.pci-0000_7b_00.6.analog-stereo` (Realtek ALC1220 Analog). Nigdy nie używać dongla USB C-Media bez mikrofonu!
