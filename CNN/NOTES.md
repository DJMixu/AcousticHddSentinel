# Notatki Projektowe — Acoustic HDD Sentinel

## 1. Seed Ensembling (Komitet modeli o różnych ziarnach losowości)
> **Data dodania:** 2026-11-10  
> **Status:** Zaplanowane do wdrożenia (Backlog)

### Cel i motywacja
Przy stosunkowo niedużych zbiorach danych audio (np. 854 próbki treningowe) i rzadszych klasach awarii mechanicznych (`scratching`, `beeping`), pojedynczy model może być podatny na specyficzny układ wag początkowych lub kolejność próbek w batchach. Zamiast poszukiwać jednego „magicznego” seeda (co grozi przeuczeniem pod zbiór testowy), profesjonalnym i wysoce skutecznym podejściem produkcyjnym jest **Seed Ensembling** (zespół / komitet modeli).

### Koncepcja techniczna
1. **Trening $N$ modeli (np. $N = 3$ lub $N = 5$):**
   - Każdy model ma identyczną architekturę 2D CNN (zdefiniowaną w `model_cnn.py`).
   - Każdy model trenowany jest od zera z innym seedem (`seed = 42`, `seed = 123`, `seed = 777` itp.).
   - Modele otrzymują inną losową inicjalizację wag, inną kolejność danych (`shuffle`) oraz inne losowe maskowania w locie (`SpecAugment`, `SpatialDropout2D`).

2. **Inferencja zespołowa (Soft Voting / Probability Averaging):**
   Dla zadanego 3-sekundowego wycinka audio każdy z modeli generuje wektor prawdopodobieństw klas (Softmax). Wynikiem końcowym jest średnia arytmetyczna lub ważona prawdopodobieństw:
   $$P_{\text{ensemble}}(c) = \frac{1}{N} \sum_{i=1}^{N} P_{\text{model}_i}(c)$$
   Klasa wynikowa:
   $$\hat{y} = \arg\max_{c} P_{\text{ensemble}}(c)$$

3. **Zalety:**
   - **Drastyczna redukcja wariancji błędu:** Różne modele popełniają drobne błędy w innych miejscach; ich uśrednienie wygasza fałszywe alarmy.
   - **Wyższa pewność predykcji:** Wartość prawdopodobieństwa odzwierciedla realny konsensus kilku niezależnych modeli.
   - **Podniesienie metryk F1:** W praktyce zespoły modeli niemal zawsze przewyższają pojedyncze modele o 1–4 punkty procentowe F1-Score.

### Plan wdrożenia (Do zrobienia):
- [ ] Utworzenie modułu treningowego `train_ensemble.py` trenującego $N$ modeli i zapisującego je w katalogu `models_ensemble/` (`model_seed_42.keras`, `model_seed_123.keras`, ...).
- [ ] Rozszerzenie pipeline'u ewaluacyjnego o inferencję komitetową (uśrednianie predykcji na zbiorze testowym).
- [ ] Porównanie macierzy pomyłek pojedynczego modelu vs komitetu 3 i 5 modeli.
- [ ] Opcjonalnie: fuzja modeli w pojedynczy graf Keras (`tf.keras.layers.Average()([m1.output, m2.output, m3.output])`) w celu eksportu do jednego pliku TFLite.
