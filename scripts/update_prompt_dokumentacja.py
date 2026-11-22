#!/usr/bin/env python3
"""
Skrypt aktualizujący plik PROMPT_DOKUMENTACJA_DLA_PROWADZACEGO.md
Wstrzykuje najnowszy kod źródłowy klas DSP, Androida oraz testów szumu,
a także aktualizuje sekcję wprowadzenia i pytań audytowych dla modelu ARGON.
"""

import os
from pathlib import Path

ROOT = Path("/home/DJMixu/Pulpit/AcousticHddSentinel")
PROMPT_FILE = ROOT / "PROMPT_DOKUMENTACJA_DLA_PROWADZACEGO.md"

def read_file_content(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def main():
    print(f"Aktualizacja pliku: {PROMPT_FILE}")
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Odczyt aktualnych plików źródłowych
    dsp_guard_py = read_file_content(ROOT / "ml/dsp_adversarial_guard.py")
    rpm_analyzer_py = read_file_content(ROOT / "ml/rpm_harmonics_analyzer.py")
    dsp_guard_java = read_file_content(ROOT / "android/app/src/main/java/com/acoustichddsentinel/dsp/DspPhysicalGuard.java")
    rpm_analyzer_java = read_file_content(ROOT / "android/app/src/main/java/com/acoustichddsentinel/dsp/RpmHarmonicsAnalyzer.java")
    main_viewmodel_java = read_file_content(ROOT / "android/app/src/main/java/com/acoustichddsentinel/ui/MainViewModel.java")
    test_city_noise_py = read_file_content(ROOT / "tests/noise_experiment/test_city_noise_immunity.py")
    test_binary_verdict_py = read_file_content(ROOT / "tests/verify_binary_verdict_and_real_drives.py")

    # 2. Aktualizacja części wstępnej i zakresu audytu (Część A i Część B)
    old_intro_marker = "### CZĘŚĆ A: SKOMPRESOWANA DOKUMENTACJA PROCESU WYTWARZANIA I ZASAD DZIAŁANIA"
    new_intro_section = """### CZĘŚĆ A: SKOMPRESOWANA DOKUMENTACJA PROCESU WYTWARZANIA I ZASAD DZIAŁANIA
1. **Istota i mechanika fizyczna uszkodzeń HDD:**
   - 4 klasy: `normal` (szum hydrodynamiczny FDB, laminarny przepływ, rotacja 5400/7200 RPM), `clicking` (uderzenia pozycjonera o Crash Stop, awaria servo/preampu), `scratching` (head crash, mikro-skrawanie powłoki magnetycznej), `beeping` (stiction głowic, zablokowane wrzeciono BLDC).
2. **Niezmiennik potoku DSP (ADR-001):**
   - Wejście: 3.0s PCM 16-bit mono ($66150$ próbek przy $22050\\text{ Hz}$).
   - STFT (okno Hanna, $N=1024$, krok $H=256$, $\\Delta f=21.53\\text{ Hz}$), 128 filtrów Mel ($50 - 10000\\text{ Hz}$), kompresja decybelowa Log-Mel (dynamika $80\\text{ dB}$).
   - Różniczkowanie Savitzky'ego-Golaya dla kanałów dynamicznych: Delta (1. pochodna) i Delta-Delta (2. pochodna).
   - Dwuetapowa normalizacja: odcięcie DC offsetu + Peak na fali $\\to$ Z-Score per-kanał tensora $\\to$ stały kształt `(128, 259, 3)` / `(3, 128, 259)`. Noise gate: odrzucanie $RMS < 0.002$.
3. **Pozyskiwanie i higiena danych:**
   - Archiwa laboratoryjne (Northwind 82 pliki z odzyskanymi nazwami, Aesonlabs, Stellar Data Recovery, Internet Archive, Freesound — łącznie 288 autentycznych nagrań audio).
   - Zastosowanie głębokiego modelu **Silero VAD (16 kHz)** do detekcji i usuwania zanieczyszczeń mową ludzką.
   - Zasada **Zero Data Leakage**: podział na poziomie całych plików przed okienkowaniem (408 train, 42 val, 60 test -> 5187 tensorów .npy, zbiór testowy 358 tensorów w 100% czysty).
4. **Ewolucja modeli — od Keras do Hybrydowego ONNX INT8 i Strażnika DSP v2:**
   - *Model v1 (Keras/TF z commitu 16031f4):* wady (artefakty zapętlania `np.tile` fałszujące clicking, DC offset $+0.3766$, shortcut learning w szumie filtrem $<2400\\text{ Hz}$).
   - *Model v2 (Wczesny PyTorch ROCm):* załamanie metryk (scratching F1=0.0, beeping F1=0.22, macro F1=0.48), śledztwo kryminalistyczne (*Acoustic Forensics* — zatrucie etykiet Northwind przez naiwną heurystykę spektralną, fałszywy duplikat Samsunga SP8002, Z-Score na szumie).
   - *Model v3 (Fine-Tuning i Seed Ensembling):* regeneracja bazy, SpecAugment, AdamW, Label Smoothing 0.03, komitet modeli o ziarnach 42, 123, 777. Wynik: Macro F1 = 84.63%, F1 dla scratching = 0.737, beeping = 0.864, clicking = 0.785, normal = 1.000.
   - *Model v4 (Adversarial Lab):* ochrona przed szumem wentylatorów PC ($1/f$, BPF) oraz przesterowaniami mikrofonów smartfona MEMS (Flat-Top Clipping i harmoniczne THD fałszujące scratching).
   - *Model v5 (Silnik Hybrydowy):* analizator harmonicznych RPM metodą Welcha (prążki 90 Hz dla 5400 RPM i 120 Hz dla 7200 RPM) jako fizyczne potwierdzenie obracającego się dysku.
   - *Model v6 (Android Edge AI):* kwantyzacja wag ONNX INT8 (1.18 MB, redukcja o 74.6%), inferencja w 18–32 ms, implementacja w czystej Javie 17 bez NDK, pętla adaptacyjna.
   - *Model v7 (Strażnik Fizyczny v2 & Odporność na Głośny Szum Miasta i Przesterowanie):*
     * **Pełne weto przesterowania (`isClipped`):** Obcięcie wierzchołków fali mikrofonu (`clipRatio > 2%`) bezwzględnie wetuje wszystkie klasy uszkodzeń (`clicking`, `beeping`, `scratching`), eliminując sztuczne transjenty i harmoniczne THD.
     * **Filtr głośnego szumu miejskiego i ruchu ulicznego (`isCityTrafficNoise`):** Dominacja pasma basowego $<300\\text{ Hz}$ ($>75\\%$) przy braku energii tarcia $>4\\text{ kHz}$ ($<3.5\\%$) skutkuje automatycznym wetem dla awarii mechanicznych.
     * **Wyrazistość pisków stojana silnika (`hasStallBeep`):** Przesunięcie dolnego progu do $1050\\text{ Hz}$, wymóg wyrazistości piku $\\ge 3.0\\times$ ponad lokalne tło $\\pm 65\\text{ Hz}$ oraz minimalny udział energii pasma $\\ge 0.5\\%$ (ochrona przed piskami cewek zasilaczy).
     * **Paradygmat werdyktu binarnego:** Agregacja $P(\\text{fault}) = P(\\text{beeping}) + P(\\text{clicking}) + P(\\text{scratching}) \\ge 70\\%$ z twardym wymogiem potwierdzenia w **minimum 2 oknach pomiarowych**.
     * **Przebudowa UI badania:** Czas okien, status, wskaźnik poziomu wejściowego mikrofonu (**VU Meter na żywo**) oraz kolorowe paski klas.
5. **Kompaktowy bilans inżynierski (Mocne i słabe strony):**
   - Sukcesy vs rzeczywiste ograniczenia fizyczne (wpływ sprzętowego usuwania szumu/AGC w smartfonach, wieloobjawowe awarie, ograniczenie wielkości bazy laboratoryjnej).

---

### CZĘŚĆ B: GŁĘBOKI AUDYT TECHNICZNY I POLOWANIE NA BŁĘDY (DEEP AUDIT & ERROR HUNTING)
Przeprowadź dogłębną inspekcję załączonego poniżej kodu pod kątem:
1. **Zgodność numeryczna Python (Librosa/PyTorch) vs Czysta Java 17:**
   - Czy implementacja Radix-2 FFT, okna Hanna, trójkątnego banku Mel i wag Savitzky'ego-Golaya w `AudioPreprocessor.java` daje identyczne tensory co Librosa?
   - Czy standaryzacja Z-score w Javie przy cichych sygnałach ma zabezpieczenie $\\epsilon$ oraz minimalne progi wariancji (`min_std`) zapobiegające dzieleniu przez zero i wzmacnianiu szumu wentylatorów?
2. **Stabilność potoku audio na Androidzie:**
   - Czy rejestrator `AudioRecorder.java` korzystający z `UNPROCESSED` / `VOICE_RECOGNITION` poprawnie omija systemowe filtry odszumiania AGC smartfona?
   - Czy przeniesienie callbacku `onWindowReady` do `backgroundExecutor` oraz prealokacja tablic roboczych DSP w 100% eliminuje zacięcia GC (Garbage Collection)?
3. **Logika bezpieczników fizycznych i odporność na fałszywe alarmy:**
   - Czy twarde weto przesterowania `isClipped` w `DspPhysicalGuard.java` (blokujące `clicking`, `beeping` i `scratching`) nie zamaskuje katastrofalnego uszkodzenia, jeśli głośność uderzeń dysku doprowadzi do clippingu przetwornika ADC?
   - Czy reguła dominacji basu miejskiego (`lowFreqRatio > 0.75f && highFreqFraction < 0.035f`) nie wyciszy uszkodzonego dysku 3.5 cala wibrującego silnie w obudowie komputera?
   - Czy analizator harmonicznych `RpmHarmonicsAnalyzer.java` z wymogiem `energyRatio >= 0.005f` oraz wyrazistości piku $\\ge 3.0\\times$ jest odporny na zasilacze impulsowe i cewki GPU (Coil Whine)?
   - Czy paradygmat werdyktu binarnego $P(\\text{fault}) \\ge 70\\%$ z wymogiem potwierdzenia w $\\ge 2$ oknach całkowicie eliminuje fałszywe alarmy od przypadkowego stuknięcia w obudowę telefonu?
4. **Architektura sieci i kwantyzacja INT8:**
   - Czy bufor OOD (`register_buffer("use_open_set")`) w `model_cnn.py` oraz kolejność obliczeń logitów przed `return_logits` gwarantują stabilność eksportu ONNX?
   - Czy model ONNX INT8 (1.18 MB) w `android/app/src/main/assets/` nie wykazuje degradacji czułości w porównaniu z FP32 na skrajnych klasach awarii?"""

    # Zamiana sekcji A i B
    pos_a = content.find("### CZĘŚĆ A: SKOMPRESOWANA DOKUMENTACJA PROCESU WYTWARZANIA I ZASAD DZIAŁANIA")
    pos_komp1 = content.find("## KOMPENDIUM KODU 1: CYFROWE PRZETWARZANIE SYGNAŁÓW (DSP)")
    if pos_a != -1 and pos_komp1 != -1:
        content = content[:pos_a] + new_intro_section + "\n\n---\n\n" + content[pos_komp1:]

    # 3. Aktualizacja ml/dsp_adversarial_guard.py
    guard_py_header = "### Strażnik Fizyczny DSP (Wentylatory 1/f i Clipping) (`ml/dsp_adversarial_guard.py`)\n```python\n"
    pos_guard_py = content.find(guard_py_header)
    if pos_guard_py != -1:
        end_guard_py = content.find("```\n\n### Benchmark wydajnościowy ONNX vs Guard", pos_guard_py)
        if end_guard_py != -1:
            content = content[:pos_guard_py + len(guard_py_header)] + dsp_guard_py + "\n" + content[end_guard_py:]

    # 4. Dodanie/Aktualizacja ml/rpm_harmonics_analyzer.py w Kompendium 5 lub 6
    rpm_py_section = f"""### Analizator harmonicznych RPM i pisków stojana (Python) (`ml/rpm_harmonics_analyzer.py`)
```python
{rpm_analyzer_py}
```

### Test odporności na głośny szum miasta i przesterowanie (`tests/noise_experiment/test_city_noise_immunity.py`)
```python
{test_city_noise_py}
```

### Test weryfikacji binarnego werdyktu i 320 okien rzeczywistych dysków (`tests/verify_binary_verdict_and_real_drives.py`)
```python
{test_binary_verdict_py}
```
"""
    # Wstrzyknięcie po benchmarku DSP Guard
    pos_bench = content.find("## KOMPENDIUM KODU 6: PRODUKCYJNY MODEL PYTORCH I SEED ENSEMBLING")
    if pos_bench != -1:
        content = content[:pos_bench] + rpm_py_section + "\n\n" + content[pos_bench:]

    # 5. Aktualizacja RpmHarmonicsAnalyzer.java
    rpm_java_header = "### Analizator obrotów RPM metodą Welcha (5400/7200 RPM) (`RpmHarmonicsAnalyzer.java`)\n```java\n"
    pos_rpm_java = content.find(rpm_java_header)
    if pos_rpm_java != -1:
        end_rpm_java = content.find("```\n\n### Rejestrator mikrofonowy AudioRecord", pos_rpm_java)
        if end_rpm_java != -1:
            content = content[:pos_rpm_java + len(rpm_java_header)] + rpm_analyzer_java + "\n" + content[end_rpm_java:]

    # 6. Dodanie DspPhysicalGuard.java w Kompendium 7
    dsp_guard_java_section = f"""### Fizyczny Strażnik DSP w Javie (Weto wentylatorów, clippingu i szumu miasta) (`DspPhysicalGuard.java`)
```java
{dsp_guard_java_java_code if 'dsp_guard_java_java_code' in locals() else dsp_guard_java}
```

"""
    pos_recorder = content.find("### Rejestrator mikrofonowy AudioRecord z buforem kołowym (`AudioRecorder.java`)")
    if pos_recorder != -1:
        content = content[:pos_recorder] + dsp_guard_java_section + content[pos_recorder:]

    # 7. Aktualizacja MainViewModel.java
    vm_java_header = "### Pętla Adaptacyjna, Werdykt Binarny i Konsensus Okien (`MainViewModel.java`)\n```java\n"
    pos_vm_java = content.find(vm_java_header)
    if pos_vm_java == -1:
        vm_java_header = "### Pętla Adaptacyjna z Early Stopping i Crest Factor Guard (`MainViewModel.java`)\n```java\n"
        pos_vm_java = content.find(vm_java_header)
    if pos_vm_java != -1:
        end_vm_java = content.find("```\n\n## KOMPENDIUM KODU 8: AUTOMATYCZNY TESTER NIEZMIENNIKÓW", pos_vm_java)
        if end_vm_java != -1:
            new_vm_header = "### Pętla Adaptacyjna, Werdykt Binarny i Konsensus Okien (`MainViewModel.java`)\n```java\n"
            content = content[:pos_vm_java] + new_vm_header + main_viewmodel_java + "\n" + content[end_vm_java:]

    with open(PROMPT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Pomyślnie zaktualizowano {PROMPT_FILE} (rozmiar: {len(content)} bajtów, {content.count(chr(10))} linii).")

if __name__ == "__main__":
    main()
