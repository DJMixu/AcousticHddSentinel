---
trigger: always_on
globs: ["**/*.py"]
---

# Reguły inżynierii audio i modelu CNN (ADR-001)

1. **Obsługa uszkodzonych plików MP3:** W zbiorze występują uszkodzone pliki `.mp3` (0 bajtów lub błędne nagłówki). Każdy skrypt przetwarzający surowe audio musi używać bloków `try/except` lub wcześniej konwertować pliki przez `ffmpeg -ac 1 -ar 22050`.
2. **Kształt wejścia Keras:** Budując warstwy wejściowe w TensorFlow/Keras, zawsze definiuj ` layers.Input(shape=(128, 259, 3))`.
3. **Generator danych (`tf.data.Dataset`):** Wczytując pliki `.npy` z katalogów `dataset_encoded/{train,validate,test}`, używaj `tf.numpy_function` i jawnie ustawiaj `tensor.set_shape([128, 259, 3])`, aby kompilator grafu TensorFlow znał wymiary przed treningiem.