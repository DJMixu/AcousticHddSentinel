# Notatki Projektowe — Acoustic HDD Sentinel

## 1. Seed Ensembling (Komitet modeli o różnych ziarnach losowości)
> **Data dodania:** 2026-11-10  
> **Status:** Wdrożone i zweryfikowane w produkcji (Model FusedEnsemble z F1 = 84.63%)

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

### Plan wdrożenia:
- [x] Utworzenie modułu treningowego `train_ensemble.py` trenującego $N$ modeli w PyTorch i zapisującego je w katalogu `models/` (`ensemble_model_seed_42.pt`, `ensemble_model_seed_123.pt`, ...).
- [x] Rozszerzenie pipeline'u ewaluacyjnego o inferencję komitetową (uśrednianie predykcji na zbiorze testowym w PyTorch).
- [x] Porównanie macierzy pomyłek pojedynczego modelu vs komitetu modeli.
- [x] Fuzja modeli w pojedynczy moduł PyTorch (`FusedEnsemblePyTorch`) w celu bezpośredniej inferencji komitetowej.

---

## 2. Krytyczna Decyzja Architektoniczna: Dlaczego Model Klasyczny (LEGACY), a NIE Model Alternatywny (SMPC / BezNormal / OOD)?
> **Data decyzji:** 2026-11-22  
> **Status:** Ostatecznie zatwierdzone. Model Klasyczny (`LEGACY`) jest jedynym modelem produkcyjnym. Model Alternatywny pełni wyłącznie rolę archiwalną / eksperymentalną z twardym fallbackiem na model klasyczny.

### Kontekst i geneza Modelu Alternatywnego
W fazie badawczej rozwijano alternatywny prototyp detekcji anomalii **SMPC (Sub-Center Multi-Prototype Cosine)**:
- **Koncepcja SMPC:** Sieć trenowana wyłącznie na 3 klasach uszkodzeń mechanicznych (`beeping`, `clicking`, `scratching`). Dźwięki dysków sprawnych oraz szum otoczenia miały być wykrywane jako „anomalia poza rozkładem” (OOD – *Out-Of-Distribution*) na bazie odległości kątowej (metryki cosinusowej) do podcentrów klas awarii w przestrzeni cech wielowymiarowych.
- W kodzie aplikacji mobilnej Android ([AlternativeCnnDetector.java](file:///home/DJMixu/Pulpit/AcousticHddSentinel/android/app/src/main/java/com/acoustichddsentinel/ml/AlternativeCnnDetector.java)) zaimplementowano interfejs wzorca Strategii (ADR-002) oraz przełącznik w UI (`btnModelAlternative`).

### Dlaczego Model Alternatywny został odrzucony na rzecz Modelu Klasycznego?

#### 1. Niestabilność progów OOD w zmiennych warunkach akustycznych mikrofonu
- W teorii laboratoryjnej metryka kątowa SMPC separowała wektory cech.
- W praktyce smartfonowej odległość mikrofonu od obudowy komputera (np. 2 cm vs 10 cm), kąt przyłożenia telefonu oraz zmienne właściwości przetworników MEMS w różnych modelach telefonów powodowały drastyczne przesunięcia odległości kątowej.
- Sztywny próg odrzucenia OOD stawał się niefizyczny: w jednych warunkach cichy, sprawny dysk był fałszywie klasyfikowany jako awaria, a w innych realne, subtelne stukanie głowicy było odrzucane jako „zwykły szum otoczenia”.

#### 2. Brak bezpośredniego uczenia cech dysku sprawnego
- W podejściu SMPC brakowało klasy `normal` w procesie optymalizacji wag (brak nadzoru).
- W Modelu Klasycznym (`LEGACY`) sieć 2D CNN widziała w treningu setki wycinków rzeczywistych, sprawnych dysków (Toshiba, WD, Seagate, HGST). Model nauczył się bezpośrednio identyfikować prawidłowy, stacjonarny szum łożysk tocznych i obracających się talerzy, zamiast definiować go jedynie jako „niepodobny do znanej awarii”.

#### 3. Bezkonkurencyjna stabilność i wydajność kwantyzacji INT8 Modelu Klasycznego
- Model Klasyczny (`best_hdd_cnn_model_int8.onnx`) waży zaledwie **1.17 MB** (redukcja o 74.6% z FP32) i wykonuje inferencję w czasie **15–20 ms** na mobilnym silniku ONNX Runtime, nie drenując baterii smartfona.
- Osiąga znakomite metryki na pełnym datasecie: **Macro F1 = 84.6%**, 100% precyzję na dyskach sprawnych w zbiorze testowym oraz niemal 90% dokładności w fuzji z fizycznym strażnikiem DSP.
- Model SMPC wymagałby dodatkowego, kosztownego narzutu obliczeniowego na liczenie norm i odległości do wielu prototypów po stronie CPU telefonu.

#### 4. Realizacja w kodzie produkcyjnym Androida (Stan Faktyczny)
- W katalogu `android/app/src/main/assets/` celowo **nie umieszczono** eksperymentalnego pliku `best_smpc_model_int8.onnx`.
- W klasie `AlternativeCnnDetector.java` zaimplementowano bezpieczny mechanizm kaskadowy (Graceful Degradation): gdy aplikacja nie znajdzie wag SMPC, automatycznie uruchamia sprawdzony model produkcyjny `best_hdd_cnn_model_int8.onnx`.
- Dzięki temu nawet jeśli użytkownik przestawi przełącznik na „Model Alternatywny”, aplikacja nie ulegnie awarii (Zero Crash) i zachowa pełną stabilność oraz najwyższą jakość predykcji.

### Wniosek końcowy dla zespołu i prowadzących
Model Klasyczny 2D CNN w tandemie z Fizycznym Strażnikiem DSP ([DspPhysicalGuard.java](file:///home/DJMixu/Pulpit/AcousticHddSentinel/android/app/src/main/java/com/acoustichddsentinel/dsp/DspPhysicalGuard.java)) i Analizatorem RPM ([RpmHarmonicsAnalyzer.java](file:///home/DJMixu/Pulpit/AcousticHddSentinel/android/app/src/main/java/com/acoustichddsentinel/dsp/RpmHarmonicsAnalyzer.java)) stanowi jedyną dopuszczoną, stabilną i optymalną ścieżkę produkcyjną. Architektura alternatywna SMPC pozostaje udokumentowanym, zamkniętym eksperymentem badawczym.
