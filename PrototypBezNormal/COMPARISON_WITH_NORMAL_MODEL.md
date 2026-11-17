# Raport Porównawczy: Model z Klasą Normal vs Model Bez Normal 💽🔬

> **Bezpośrednie porównanie (side-by-side) na identycznym zbiorze testowym 516 tensorów `.npy` (Zero Data Leakage).**

---

## 1. Zestawienie Metryk Klasyfikacji 4-Klasowej

| Klasa Audio | Model A (Pełny z Normal) F1 | Model B (Bez Normal OOD) F1 | Model A Recall | Model B Recall | Model A Precision | Model B Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`beeping`** | **34.15%** | **38.78%** | 33.33% | 30.16% | 35.00% | 54.29% |
| **`clicking`** | **54.55%** | **49.62%** | 58.88% | 61.68% | 50.81% | 41.51% |
| **`normal`** | **86.33%** | **74.92%** | 86.47% | 76.90% | 86.18% | 73.04% |
| **`scratching`** | **53.52%** | **4.35%** | 44.19% | 2.33% | 67.86% | 33.33% |
| **DOKŁADNOŚĆ (Accuracy)** | **70.74%** | **61.82%** | - | - | - | - |
| **ŚREDNIA MAKRO (Macro F1)** | **57.13%** | **41.92%** | - | - | - | - |

---

## 2. Zestawienie Metryk Binarnych (Dysk Sprawny vs Uszkodzony)

| Metryka Diagnostyczna | Model A (Pełny z Normal) | Model B (Bez Normal OOD) | Wniosek Inżynierski |
| :--- | :---: | :---: | :--- |
| **Dokładność Binarna (Binary Acc)** | **83.91%** | **69.77%** | Zdolność do binarnego podziału Zdrowy / Uszkodzony |
| **Zrównoważona Dokładność (Balanced Acc)** | **83.38%** | **68.26%** | Niezależna od dysproporcji klas w teście |
| **Czułość na Zdrowy Dysk (TNR / Normal Recall)** | **86.47%** | **76.90%** | Odsetek poprawnie rozpoznanych zdrowych dysków |
| **Czułość na Awarie (TPR / Fault Recall)** | **80.28%** | **59.62%** | Zdolność do wykrycia uszkodzenia mechanicznego |

---

## 3. Kluczowe Wnioski Badawcze i Zastosowania

### Dlaczego Model B (Bez Klasy Normal) jest tak ważny?
1. **Problem Dystrybucji Klasy Normalnej w Świecie Rzeczywistym:**
   - W praktyce przemysłowej i data recovery istnieje nieskończenie wiele odmian dźwięku 'normalnego' (różne serie dysków, obroty 5400/7200/10000/15000 RPM, odmienne wentylatory, rezonanse różnych obudów PC i serwerów NAS).
   - Model uczony z klasą `normal` jest podatny na zjawisko *false alarm* na dyskach o niespotykanej wcześniej charakterystyce hydrodynamicznej.
2. **Zaleta Podejścia Fault-Only + OOD:**
   - Model definiuje ściśle zamknięte obwiednie uszkodzeń w przestrzeni cech (GMM).
   - Osiąga **76.90% wykrywalności zdrowych dysków** ($F1 = 74.92%$) **bez ani jednej próbki zdrowego dysku w treningu!**
   - Na rzeczywistym fizycznym dysku WD Blue 5400 RPM (60s nagrania) uzyskał **100% werdyktów `HEALTHY / NORMAL`**.

### Kiedy stosować Model A, a kiedy Model B?
* **Model A (Pełny):** Najwyższa czułość i dokładność ogólna w kontrolowanych warunkach, gdzie dźwięk tła i modele dysków są reprezentowane w zbiorze treningowym.
* **Model B (PrototypBezNormal):** Środowiska otwarte (Out-Of-Distribution), laboratoria odzyskiwania danych badające nietypowe napędy, diagnostyka wstępna bez konieczności zbierania bazy zdrowych napędów.

---
*(Wykres porównawczy: `PrototypBezNormal/comparison_with_normal_model.png`)*
