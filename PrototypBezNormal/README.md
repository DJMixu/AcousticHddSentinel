# Eksperyment Badawczy: PrototypBezNormal v2 (OOD Density Learning) 🛡️💽
> **System detekcji uszkodzeń HDD trenowany WYŁĄCZNIE na awariach mechanicznych (Fault-Only Learning) z detekcją stanu `normal` za pomocą estymacji gęstości prawdopodobieństwa GMM.**

---

## 🎯 1. Cel i Założenia Badawcze

Eksperyment **PrototypBezNormal** bada alternatywny paradygmat diagnostyki wibroakustycznej:
* **Trening (Train / Validate):** Model 2D CNN uczy się **WYŁĄCZNIE na 3 klasach uszkodzeń mechanicznych**:
  1. `beeping` (zatarte wrzeciono, stiction głowic, pisk stojana PWM)
  2. `clicking` (Click of Death, uderzenia pozycjonera o ogranicznik VCM)
  3. `scratching` (tarcie głowic o powierzchnię talerza, head crash)
  * **Klasa `normal` (dysk sprawny) została w 100% odfiltrowana i jest całkowicie nieobecna w procesie uczenia wag!**
* **Test i Eksploatacja (Test / Real-Time):** Model jest poddawany rygorystycznym testom na zbiorze **ZAWIERAJĄCYM 303 tensory klasy `normal`** oraz na autentycznych fizycznych nagraniach sprawnych dysków.

---

## 🧠 2. Architektura i Ewolucja Metody (v1 vs v2)

| Cecha / Moduł | Prototyp v1 (Baseline) | Prototyp v2 (GMM OOD Engine) |
|---|---|---|
| **Dane Treningowe Awarii** | 2 065 tensorów .npy | **3 421 tensorów .npy** (+33 nowe nagrania Archive/Lab) |
| **Metoda Detekcji `normal`** | Prosta odległość euklidesowa $D_{\min}$ | **Gaussian Mixture Model (GMM)** w przestrzeni cech 128D |
| **Miara Przynależności** | Odległość geometryczna od centroidu | Logarytm gęstości prawdopodobieństwa $\log p(\mathbf{z} \mid \text{faults})$ |
| **Kalibracja Progu OOD** | Ręczny mnożnik $\sigma$ | Kalibracja percentylowa na zbiorze walidacyjnym awarii |
| **Dokładność Ogólna (Test 4 klasy)** | 45.64% | **61.82%** (+16.18 pp) |
| **Wykrywanie `normal` (Recall)** | 72.8% (206/283) | **76.90% (233/303)** |
| **Wykrywanie `normal` (Precision)** | 58.69% | **73.04%** (+14.35 pp) |
| **Wykrywanie `normal` (F1-Score)** | 0.6498 (65.0%) | **0.7492 (74.92%)** (+9.94 pp) |
| **AUROC (Normal vs Uszkodzenia)** | ~64.0% | **72.33%** |
| **AUPR (Precision-Recall Curve)** | ~62.5% | **70.50%** |

---

## 📊 3. Wyniki Empiryczne na Zbiorze Testowym (516 tensorów)

Model v2 został poddany ewaluacji na zbiorze testowym zgodnie z zasadą **Zero Data Leakage** (74 niezależne nagrania, 516 wycinków 3.0s bez nakładania okien):

```
              precision    recall  f1-score   support

     beeping     0.5429    0.3016    0.3878        63
    clicking     0.4151    0.6168    0.4962       107
      normal     0.7304    0.7690    0.7492       303
  scratching     0.3333    0.0233    0.0435        43

    accuracy                         0.6182       516
   macro avg     0.5054    0.4277    0.4192       516
weighted avg     0.6090    0.6182    0.5938       516
```

### Kluczowe Wnioski Badawcze:
1. **Wysoka Zdolność Odrzucania Stanu Normalnego:** Model, który **nigdy w życiu nie widział sprawnego dysku**, rozpoznaje **76.90% okien zdrowych dysków** jako `normal` z precyzją **73.04%** ($F1 = 0.7492$).
2. **Dlaczego GMM działa lepiej niż odległość euklidesowa?**
   Przestrzeń cech 128D jest silnie anizotropowa – wariancje w poszczególnych wymiarach różnią się o rzędy wielkości. Prosta odległość euklidesowa traktuje przestrzeń jak sferę, co powoduje wyciek próbek normalnych do klastrów awarii. Model GMM estymuje diagonalną macierz kowariancji, precyzyjnie opisując elipsoidalne powłoki uszkodzeń.

---

## 🔬 4. Weryfikacja na Rzeczywistych Dyskach Fizycznych

Przeprowadzono serię testów weryfikacyjnych za pomocą skryptu [live_predict.py](file:///home/DJMixu/Pulpit/AcousticHddSentinel/PrototypBezNormal/live_predict.py):

### A. Fizyczny Dysk Sprawny (WD Blue 5400 RPM - 60 sekund ciągłego zapisu):
* **Liczba okien 3.0s:** 39
* **Wynik:** **39 na 39 okien (100.0%) zaklasyfikowane jako `NORMAL`**
* **Średnia pewność:** **81.46%**
* **Log-Likelihood:** 130.8 – 162.6 (stabilnie poniżej progu awarii 201.9)
* **Ogólny werdykt:** **`HEALTHY / NORMAL`**

### B. Awaria Pozycjonera: Click of Death (`Aesonlabs Data Recovery_Clicking Hard Drive.wav`):
* **Liczba okien 3.0s:** 8
* **Wynik:** **8 na 8 okien (100.0%) zaklasyfikowane jako `CLICKING`**
* **Średnia pewność:** **98.33%**
* **Log-Likelihood:** 265.7 – 280.2 (daleko powyżej progu awarii 201.9)
* **Ogólny werdykt:** **`FAULT DETECTED: CLICKING`**

### C. Zatarte Wrzeciono / Stiction (`Aesonlabs Data Recovery_1.5TB Seagate Beeping.wav`):
* **Liczba okien 3.0s:** 12
* **Wynik:** **12 na 12 okien (100.0%) zaklasyfikowane jako `BEEPING`**
* **Średnia pewność:** **98.96%**
* **Log-Likelihood:** 228.5 – 241.5 (wyraźnie powyżej progu 201.9)
* **Ogólny werdykt:** **`FAULT DETECTED: BEEPING`**

---

## 🚀 5. Narzędzia i Instrukcja Obsługi

### 1. Analiza Gotowego Pliku Audio (.wav, .mp3, .ogg):
```bash
./.venv_rocm/bin/python PrototypBezNormal/live_predict.py <ścieżka_do_audio.wav>
```

### 2. Interaktywne Nagrywanie z Fizycznego Mikrofonu (Realtek ALC1220):
```bash
./.venv_rocm/bin/python PrototypBezNormal/record_and_predict.py --duration 5.0
```
*Skrypt automatycznie używa dedykowanego złącza PCI Realtek ALC1220 (`alsa_input.pci-0000_7b_00.6.analog-stereo`), nagrywa 5 sekund audio, dzieli na okna i wyświetla raport diagnostyczny.*

### 3. Uruchomienie Pełnego Benchmarku OOD:
```bash
./.venv_rocm/bin/python PrototypBezNormal/evaluate_gmm_ood.py
```

### 4. Uruchomienie Testów Jednostkowych Prototypu:
```bash
./.venv_rocm/bin/python PrototypBezNormal/test_metric_prototype.py
```
*(6/6 testów zdanych pomyślnie)*

---

## 📁 6. Struktura Katalogu `PrototypBezNormal/`

```
PrototypBezNormal/
├── README.md                      # Kompletna dokumentacja eksperymentu v2
├── ood_gmm_engine.py              # Zintegrowany silnik detektora OOD (CNN + GMM)
├── live_predict.py                # Skrypt CLI do analizy plików audio w oknach 3.0s
├── record_and_predict.py          # Narzędzie nagrywania mikrofonem Realtek ALC1220
├── train_robust_fault_cnn.py      # Trening 2D CNN wyłącznie na 3 klasach awarii
├── train_gmm_ood.py               # Trening i kalibracja estymatora gęstości GMM
├── evaluate_gmm_ood.py            # Pełna ewaluacja OOD na 516 tensorach testowych
├── test_metric_prototype.py       # Zestaw 6 testów jednostkowych (100% PASS)
│
└── output_v2/                     # Wytrenowane wagi, modele GMM i raporty ewaluacji
    ├── best_fault_cnn_v2.pt       # Wagi sieci 2D CNN (Val F1 69.7%, Tr Acc 94.0%)
    ├── gmm_density_model.joblib   # Skalibrowany model estymacji gęstości GMM
    ├── ood_gmm_config.json        # Parametry progowe i statystyki gęstości
    ├── confusion_matrix_gmm_4x4.png # Wykres macierzy pomyłek 4x4
    ├── gmm_log_likelihood_histogram.png # Wykres rozkładu gęstości prawdopodobieństwa
    └── gmm_test_evaluation_summary.json # Numeryczne wyniki benchmarku testowego
```
