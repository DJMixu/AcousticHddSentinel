# ZADANIE DLA PROGRAMISTY QWEN CODER 14B (ITERACJA 2: IMPLEMENTACJA AERODYNAMIC FAN GUARD)

Na podstawie specyfikacji Architekta Qwen 3.8 w `tests/adversarial_lab/results/qwen_arch_iter2.txt`, zaimplementuj klasę `AerodynamicFanGuard` oraz rozszerz moduł `DSPAdversarialGuard` w `ml/dsp_adversarial_guard.py`.

## Wymagania:
1. Zaimplementuj dataclass `GuardMetrics`, `GuardStatus`, `GuardResult` oraz klasę `AerodynamicFanGuard`.
2. Zastosuj niezmienniki ADR-001:
   - `TARGET_SR = 22050`
   - `N_FFT = 1024`
   - `HOP_LENGTH = 256`
   - `CHUNK_SAMPLES = 66150` (3.0 sekundy)
   - Odrzucanie ciszy: RMS < 0.002
3. Metody DSP:
   - STFT Hann window (można użyć numpy/scipy lub librosa bez naruszania ADR-001).
   - Obliczanie `bpf_comb_score` (wyszukiwanie Blade Pass Frequency 80-1000 Hz i harmonicznych).
   - Obliczanie `turbulence_score` (opadanie widma 1/f, log-log slope, r2).
   - Temporal kurtosis, crest factor, spectral flux, high freq ratio.
   - Safety Veto list (Crest > 7.0, Kurtosis > 8.0, HighFreq > 0.22, Flux > 0.40 -> NIE FAN).
4. Napisz gotowy, kompletny, produkcyjny kod w czystym Pythonie bez placeholderów `pass` i `TODO`.
