---
trigger: always_on
globs: ["**/*.py"]
---

# Reguły inżynierii audio i modelu 2D CNN (ADR-001 - PyTorch)

1. **Obsługa uszkodzonych plików MP3:** W zbiorze występują uszkodzone pliki `.mp3` (0 bajtów lub błędne nagłówki). Każdy skrypt przetwarzający surowe audio musi używać bloków `try/except` lub wcześniej konwertować pliki przez `ffmpeg -ac 1 -ar 22050`.
2. **Kształt wejścia PyTorch:** Tensory w formacie Channel-First w PyTorch mają wymiary `(B, 3, 128, 259)`. Model `AcousticHddSentinelCNN` w `model_cnn.py` transparentnie obsługuje również tensory Channel-Last `(B, 128, 259, 3)` wczytywane z plików `.npy`.
3. **Loader danych (`HddNpyDataset` / `DataLoader`):** Wczytując pliki `.npy` z katalogów `dataset_encoded/{train,validate,test}`, używaj PyTorch DataLoader z `HddNpyDataset` z `dataset_pytorch.py`, zachowując podział bez wycieku danych (No Data Leakage).