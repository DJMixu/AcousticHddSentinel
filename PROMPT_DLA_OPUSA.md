# PROMPT DLA MODELU

> **Instrukcja dla użytkownika:**  
> Skopiuj poniższą treść (wszystko poniżej poziomej linii) i wklej bezpośrednio w oknie czatu z Claude Opus w przeglądarce (np. claude.ai lub Open WebUI).

---

```markdown
Jesteś starszym inżynierem uczenia maszynowego (Senior MLOps / Deep Learning Engineer) specjalizującym się w TensorFlow/Keras oraz cyfrowym przetwarzaniu sygnałów audio (DSP).

Twoim zadaniem jest napisanie kompletnego, produkcyjnego modułu `train_ensemble.py` dla projektu **HDD Acoustic Fault Detection (Acoustic HDD Sentinel)**.

### 1. Kontekst projektu i niezmienniki techniczne (ADR-001)
System wykrywa uszkodzenia mechaniczne dysków HDD na podstawie 3-sekundowych wycinków audio przetworzonych do 3-kanałowego tensora czasowo-częstotliwościowego:
- **Kanał 0:** Log-Mel Spektrogram (n_mels = 128)
- **Kanał 1:** Delta (1. pochodna czasowa — klikanie głowic)
- **Kanał 2:** Delta-Delta (2. pochodna czasowa — przyspieszenia obrotowe)

**Niezmienniki, których nie wolno łamać:**
1. **Kształt tensora wejściowego (Channel-Last):** Zawsze dokładnie `(128, 259, 3)` typu `float32`. Warstwa wejściowa: `layers.Input(shape=(128, 259, 3))`.
2. **Brak Data Leakage:** Podział train/val/test znajduje się na dysku w `dataset_encoded/{train,validate,test}`.
3. **Loader danych:** Używa `tf.data.Dataset` z `tf.numpy_function` i jawnym `tensor.set_shape([128, 259, 3])`.
4. **Ważenie klas:** Zastosowanie wag `class_weight` (sqrt-balanced: `np.clip(np.sqrt(w), 0.5, 3.5)`).

---

### 2. Wytyczne z dokumentu NOTES.md (Seed Ensembling)
Zamiast szukać pojedynczego "szczęśliwego" seeda (co grozi overfittingiem do zbioru testowego), wdrażamy **Seed Ensembling** (komitet modeli):
1. **Trening N modeli:** Trenujemy $N$ modeli o identycznej architekturze 2D CNN (np. $N=3$ z `seeds = [42, 123, 777]`), każdy z innym seedem (`tf.random.set_seed(seed)`, `np.random.seed(seed)`).
   - Każdy model otrzymuje inną inicjalizację wag, inną kolejność mieszania batchy (`shuffle`) oraz inne losowe maskowania w locie (`SpecAugment`, `SpatialDropout2D`).
2. **Inferencja komitetowa (Soft Voting / Probability Averaging):**
   Dla każdej próbki ze zbioru testowego każdy model generuje wektor prawdopodobieństw Softmax. Prawdopodobieństwa są uśredniane:
   $$P_{\text{ensemble}}(c) = \frac{1}{N} \sum_{i=1}^{N} P_{\text{model}_i}(c)$$
   Klasa końcowa to:
   $$\hat{y} = \arg\max_{c} P_{\text{ensemble}}(c)$$
3. **Fuzja komitetu (Fused Keras Model):**
   Po wytrenowaniu modeli skrypt powinien opcjonalnie połączyć je w jeden nadrzędny graf Keras:
   `tf.keras.layers.Average()([model_1(inputs), model_2(inputs), ...])`
   i zapisać jako `fused_ensemble_model.keras` gotowy do pojedynczego wdrożenia i późniejszej konwersji do TFLite na urządzenia mobilne/Android.

---

### 3. Wymagania wobec kodu `train_ensemble.py`
Napisz kompletny, samodzielny skrypt Python `train_ensemble.py`, który:
1. **Importuje istniejące moduły projektu:**
   - Z `model_cnn.py`: `build_hdd_cnn_model`, `compile_cnn_model`, `INPUT_SHAPE`
   - Z `train_cnn.py`: `resolve_dataset_dir`, `discover_classes_and_files`, `create_tf_dataset`, `calculate_balanced_class_weights`, `plot_and_save_confusion_matrix`
2. **Wspiera akcelerację GPU (AMD ROCm / NVIDIA):**
   - Bezpieczna obsługa pamięci VRAM: `tf.config.experimental.set_memory_growth(gpu, True)` aby zapobiec alokacji 100% VRAM na raz.
3. **Pętla treningowa komitetu:**
   - Dla każdego seeda z listy (domyślnie `--seeds 42 123 777`) trenuje model przez `--epochs` (domyślnie: 30) z callbackami `ModelCheckpoint` i `ReduceLROnPlateau`.
   - Zapisuje wagi każdego modelu do `output_dir/models/ensemble_model_seed_{seed}.keras`.
4. **Ewaluacja i porównanie:**
   - Ocenia każdy model indywidualnie na zbiorze testowym (`test`).
   - Oblicza predykcję komitetu (Soft Voting) dla całego zbioru testowego.
   - Wylicza metryki: Accuracy, Weighted F1-Score oraz F1-Score per klasa (`clicking`, `scratching`, `beeping`, `normal`).
   - Drukuje w konsoli czytelną tabelę ASCII zestawiającą: Modele indywidualne vs Średnia indywidualna vs 🏆 Komitet (Ensemble) z podaniem zysku $\Delta$ F1 i $\Delta$ Accuracy.
5. **Generowanie artefaktów:**
   - Macierz pomyłek komitetu: `ensemble_confusion_matrix.png`
   - Wykres słupkowy porównujący F1 per klasa: `ensemble_comparison_bars.png`
   - Raport numeryczny JSON: `ensemble_summary.json`
   - Połączony model Keras: `fused_ensemble_model.keras`
6. **CLI Parser (argparse):**
   - `--seeds` (lista intów, np. 42 123 777)
   - `--epochs` (domyślnie: 30 lub 40)
   - `--data-dir` (ścieżka do `dataset_encoded`)
   - `--output-dir` (domyślnie: `./ensemble_output`)
   - `--batch-size` (domyślnie: 32)
   - `--lr` (domyślnie: 0.001)
   - `--dropout` (domyślnie: 0.3)
   - `--no-fuse` (opcjonalne pominięcie fuzji)

Kod ma być czysty, w pełni otypowany (Type Hints), zoptymalizowany i gotowy do natychmiastowego uruchomienia bez żadnych brakujących zmiennych czy błędów.
```
