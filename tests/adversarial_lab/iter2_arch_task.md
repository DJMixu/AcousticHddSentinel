# ZADANIE DLA ARCHITEKTA QWEN 3.8 (ITERACJA 2: OBRONA PRZED SZUMEM AERODYNAMICZNYM / TURBINĄ PC / ODKURZACZEM)

## 1. Problem badawczy
Model 2D CNN HDD Sentinel może ulec atakowi typu OOD (Out-Of-Distribution) wywołanemu przez głośne wentylatory serwerowe (np. turbiny 1U Delta 6000-12000 RPM, wentylatory obudów PC 120/140mm na pełnych obrotach) oraz domowe urządzenia nadmuchowe (suszarka do włosów, odkurzacz, wentylator biurkowy).
Dźwięki te charakteryzują się:
- potężną energią turbulencji powietrza o widmie opadającym 1/f (różowy/brązowy szum),
- obecnością prążków harmonicznych BPF (Blade Pass Frequency) w pasmie 300 - 3000 Hz,
- brakiem mechanicznych uderzeń impulsowych właściwych dla HDD.

Naiwna klasyfikacja 2D CNN może mylić BPF z zatarciem wrzeciona ("beeping") lub szum powietrza z tarciem głowicy o talerz ("scratching").

## 2. Pytania architektoniczne do Ciebie (Qwen 3.8):
1. Jakie matematyczno-fizyczne niezmienniki sygnałowe (DSP) jednoznacznie odróżniają szum turbulencyjno-aerodynamiczny od:
   a) uszkodzenia `beeping` (blokada wrzeciona, oscylacje BLDC),
   b) uszkodzenia `scratching` (kontakt mechaniczny slider-platter, mikro-skrawanie i pył ferromagnetyczny)?
2. Jakie formalne metryki statystyczne należy obliczyć (np. Spectral Roll-off, Spectral Flatness, Temporal Kurtosis, Crest Factor, Energy Distribution across Octaves / Bark scale)?
3. Podaj konkretne, matematycznie uzasadnione progi i warunki logiczne dla modułu `is_aerodynamic_fan_noise(...)` w `ml/dsp_adversarial_guard.py`.
4. Jak zagwarantować 0% False Positives na prawdziwych uszkodzeniach HDD (żeby autentyczny `scratching` i `beeping` nigdy nie zostały omyłkowo zaklasyfikowane jako wentylator)?
