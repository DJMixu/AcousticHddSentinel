# ZADANIE DLA ARCHITEKTA QWEN 3.8 (ITERACJA 3: OBRONA PRZED PRZESTEROWANIEM MIKROFONU SMARTFONA I DYSTORSJAMI KANAŁU AKUSTYCZNEGO)

## 1. Problem badawczy
Gdy użytkownik zbliża smartfon do dysku twardego, mikrofon MEMS i wbudowany tor analogowo-cyfrowy (ADC / AGC) smartfona mogą ulec przesterowaniu (Mic Clipping / Saturation), a kanał akustyczny (odbicia od obudowy PC, biurka) wprowadza pogłos (reverberation) i zniekształcenia nieliniowe (THD).
Przesterowanie (obcinanie wierzchołków fali) generuje w widmie sztuczne harmoniczne o wysokiej częstotliwości (4-10 kHz), które model 2D CNN może błędnie zaklasyfikować jako mechaniczne tarcie głowic o talerz ("scratching"). Z kolei kompresja dynamiki (AGC) może spłaszczyć impulsy stukania ("clicking"), utrudniając ich detekcję.

## 2. Pytania architektoniczne do Ciebie (Qwen 3.8):
1. Jakie kryteria DSP pozwalają niezawodnie wykryć przesterowanie mikrofonu smartfona (Flat-Top Clipping Detection, odsetek nasyconych próbek, asymetria histogramu)?
2. Jak odróżnić sztuczne harmoniczne THD powstałe w wyniku przesterowania mikrofonu od prawdziwego, autentycznego fizycznego tarcia głowic HDD (`scratching`)?
3. Jak zaimplementować Cepstral Mean Subtraction / Normalization lub kompensację odpowiedzi kanału akustycznego w potoku ADR-001 bez łamania niezmienników kształtu tensora (128, 259, 3)?
4. Podaj formalne progi i logikę modułu `is_microphone_clipped_or_distorted(...)` oraz reguły sanityzacji predykcji.
