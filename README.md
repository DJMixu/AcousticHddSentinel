# Acoustic HDD Sentinel 🛡️💽
> **System akustycznej detekcji i klasyfikacji awarii mechanicznych dysków twardych (HDD) przy użyciu głębokich sieci konwolucyjnych (2D CNN w PyTorch).**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Model%20Trained%20%26%20Verified-brightgreen.svg)]()

---

## 📖 Spis treści
1. [O projekcie](#-o-projekcie)
2. [Architektura przetwarzania audio (ADR-001)](#-architektura-przetwarzania-audio-adr-001)
3. [Architektura sieci 2D CNN](#-architektura-sieci-2d-cnn)
4. [Instalacja i wymagania](#-instalacja-i-wymagania)
5. [Szybki start i instrukcja obsługi](#-szybki-start-i-instrukcja-obsługi)
6. [Wyniki treningu i ewaluacja testowa](#-wyniki-treningu-i-ewaluacja-testowa)
7. [Struktura projektu](#-struktura-projektu)
8. [Współpraca z Antigravity (AGY)](#-współpraca-z-antigravity-agy)

---

## 🎯 O projekcie
Acoustic HDD Sentinel analizuje krótkie, 3-sekundowe nagrania dźwiękowe pracy dysków twardych, wykrywając uszkodzenia mechaniczne i klasyfikując ich typ przed katastrofalną utratą danych:
- **`clicking`** – stukanie głowicy o ogranicznik (*Click of Death*, uszkodzenie pozycjonera / zespołu głowic).
- **`beeping`** – pisk / melodia zatartego silnika wrzeciona (*stuck spindle*, brak możliwości rozpędzenia talerzy).
- **`scratching`** – szerokopasmowe tarcie głowicy po magnetycznej powierzchni talerza (*head crash / platter scoring*).
- **`normal`** – prawidłowa, stabilna praca dysku twardego.

---

## 🔬 Architektura przetwarzania audio (ADR-001)
Zgodnie z decyzją architektoniczną **ADR-001** (`docx/adr/adr.md`), każde 3-sekundowe okno audio jest transformowane do **3-kanałowego tensora `(128, 259, 3)`**:
1. **Kanał 0:** Log-Mel Spektrogram (128 pasm Mel, skala decybelowa)
2. **Kanał 1:** Delta – 1. pochodna czasowa (detekcja ostrych transjentów i uderzeń)
3. **Kanał 2:** Delta-Delta – 2. pochodna czasowa (dynamika przyspieszeń obrotowych)

### Normalizacja:
- **Peak Normalization:** Surowy wycinek audio jest skalowany szczytowo do `[-1.0, 1.0]`, co zachowuje wysoki współczynnik szczytu (*Crest Factor* sięgający w nagraniach awarii aż `131.7`).
- **Per-Channel Z-score:** Każdy z 3 kanałów tensora jest niezależnie standaryzowany ($\mu = 0.0$, $\sigma = 1.0$), uniezależniając model od 430-krotnego rozrzutu głośności mikrofonów.

---

## 🧠 Architektura sieci 2D CNN
Model w `model_cnn.py` został zaprojektowany z myślą o specyfice danych akustycznych:
- **Format tensora:** Natywny `(B, 3, 128, 259)` (Channel-First) z przezroczystą obsługą `(B, 128, 259, 3)` (Channel-Last).
- **4 Bloki splotowe:** `Conv2D` (32 -> 64 -> 128 -> 256 filtrów $3 \times 3$), `BatchNorm2d`, `ReLU`, `MaxPool2d(2, 2)`, `Dropout2d` oraz regularyzacja wag.
- **Agregacja:** `AdaptiveAvgPool2d((1, 1))` (Global Average Pooling redukujący wymiar do 256 bez eksplozji parametrów, całkowita liczba wag: **1.21 mln**).
- **Klasyfikator:** `Linear(256, 128, bias=False)` + `BatchNorm1d` + `Dropout(0.3)` + `Linear(128, num_classes)`.

---

## 💻 Instalacja i wymagania

Wymagany Python 3.10+ oraz wirtualne środowisko:

```bash
git clone git@github.com:DJMixu/AcousticHddSentinel.git
cd AcousticHddSentinel

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Szybki start i instrukcja obsługi

### 1. Weryfikacja działania modelu
Sprawdź poprawność modelu PyTorch, przejście w przód, przepływ gradientów i zapis wag:
```bash
python verify_cnn_model.py
```

### 2. Sprawdzenie jakości normalizacji danych
Przeanalizuj statystyki audio przed i po normalizacji oraz wygeneruj raport:
```bash
python profile_normalization_quality.py
```
*(Wykres diagnostyczny zostanie zapisany w `normalization_quality_report.png`)*.

### 3. Przygotowanie danych i enkodowanie
```bash
# 1. Kategoryzacja nagrań i przygotowanie próbek
python prepare_audio_dataset.py

# 2. Enkodowanie do tensorów .npy z podziałem 70/15/15 bez wycieku danych
python hdd_audio_encoder.py
```

### 4. Uruchomienie treningu
```bash
python train_cnn.py --epochs 30 --batch-size 32 --output-dir ./training_output
```
W trakcie treningu automatycznie działają:
- `EarlyStopping` (przywraca najlepsze wagi)
- `ModelCheckpoint` (zapisuje `best_hdd_cnn_model.pt`)
- `ReduceLROnPlateau` (dynamicznie redukuje LR o 50%)
- Obsługa wag klas (`CrossEntropyLoss(weight=...)`)

---

## 📊 Wyniki treningu i ewaluacja testowa

Ewaluacja najlepszego modelu na niezależnym zbiorze testowym (`TEST` – 92 tensory z niewidzianych wcześniej dysków):

| Klasa awarii | Precyzja (Precision) | Czułość (Recall) | F1-Score | Próbki testowe |
| :--- | :---: | :---: | :---: | :---: |
| **`normal` (sprawny dysk)** | **96.0%** | **100.0%** | **0.9796** | 24 |
| **`clicking` (stukanie głowic)** | **77.4%** | **87.2%** | **0.8200** | 47 |
| **`beeping` (zatarte wrzeciono)** | **80.0%** | **57.1%** | **0.6667** | 7 |
| **`scratching` (tarcie talerza)** | **44.4%** | **28.6%** | **0.3478** | 14 |
| **Dokładność (Accuracy)** | — | — | **79.35%** | **92** |

Wygenerowane pliki i wizualizacje znajdują się w katalogu [training_output/](training_output/):
- `confusion_matrix.png` – macierz pomyłek z wartościami liczbowymi i procentami
- `training_curves.png` – wykresy straty i dokładności z procesu uczenia
- `best_hdd_cnn_model.pt` – zapisany, sprawdzony model do inferencji w PyTorch

---

## 🤖 Współpraca z Antigravity (AGY)
Projekt jest w pełni przystosowany do pracy w trybie pair programming z agentami **Antigravity CLI (AGY)**:
- Szczegółowe niezmienniki i wytyczne dla agenta znajdują się w [AGENTS.md](AGENTS.md).
- Reguły automatyczne zdefiniowano w [.agents/rules/adr-001-audio-pipeline.md](.agents/rules/adr-001-audio-pipeline.md).
- Decyzje architektoniczne opisano w [docx/adr/adr.md](docx/adr/adr.md).

---

## 📄 Licencja
Projekt rozwijany w celach badawczych i diagnostycznych.
