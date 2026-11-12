# ZADANIE DLA PROGRAMISTY QWEN CODER 14B (ITERACJA 3: IMPLEMENTACJA MICROPHONE CLIPPING GUARD)

Na podstawie specyfikacji Architekta Qwen 3.8 w `tests/adversarial_lab/results/qwen_arch_iter3.txt`, zaimplementuj moduł `ml/mic_clipping_guard.py` oraz zintegruj go z `DSPAdversarialGuard` w `ml/dsp_adversarial_guard.py`.

## Wymagania implementacyjne:
1. Zaimplementuj `ClippingGuardConfig`, `ClippingMetrics`, `ClippingReport` oraz klasę `MicrophoneClippingGuard`.
2. Zaimplementuj metody:
   - `detect_flat_top_runs(waveform)`: wyszukiwanie serii nasyconych próbek (|x| >= 0.98 * peak, min_run >= 4).
   - `compute_thd_comb_metrics(waveform)`: analiza widma pasma 4-10 kHz (comb_ratio, spectral_flatness_high, THD harmonics).
   - `is_microphone_clipped_or_distorted(waveform, prediction)`: zwracająca `is_clipped`, `is_distorted`, `artifact_likelihood`, `authentic_scratching_likely`.
   - `sanitize_prediction(probs, waveform)`: neutralizacja fałszywego `scratching` wywołanego przesterowaniem do `normal`.
3. Zastosuj niezmienniki ADR-001 (SR=22050, N_FFT=1024, HOP=256, CHUNK=66150).
4. Napisz kompletny, w 100% działający kod produkcyjny bez skrótów i placeholderów.
