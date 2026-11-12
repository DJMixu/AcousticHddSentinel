"""
Moduł DSP Guard: Zaawansowana obrona przed atakami adwersarialnymi i fałszywymi alarmami.
Single Source of Truth: ADR-001, AGENTS.md oraz specyfikacja Qwen 3.8.

Wykrywa i neutralizuje specyficzne wzorce Out-Of-Distribution:
1. GPU Coil Whine: wąskopasmowe harmoniczne PWM mylone z tarciem głowic (scratching).
2. Syntetyczny metronom: idealnie regularne impulsy zegarowe mylone z Click of Death (clicking).
"""

import numpy as np
import torch
import scipy.stats
from typing import Dict, Tuple, Union, Optional

CLASSES = ["beeping", "clicking", "normal", "scratching"]
TARGET_SR = 22050
N_FFT = 1024
HOP_LENGTH = 256
CHUNK_SAMPLES = 66150  # 3.0 s


class DSPAdversarialGuard:
    """
    Strażnik DSP analizujący tensor czasowo-częstotliwościowy ADR-001
    w układzie (128, 259, 3) lub (B, 3, 128, 259) oraz opcjonalnie surowy waveform.
    Chroni system przed fałszywymi predykcjami (OOD):
    1. GPU Coil Whine (wąskopasmowy pisk PWM -> fałszywy scratching)
    2. Periodyczny metronom (impulsy zegarowe -> fałszywy clicking/beeping)
    3. Aerodynamiczny szum wentylatorów PC / turbin serwerowych / odkurzaczy (niska kurtoza, spadek 1/f -> fałszywy clicking/scratching)
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def _prepare_mel_matrix(self, tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Sprowadza tensor do 2D macierzy mel-spektrogramu (128, 259).
        Używa kanału 0 (Log-Mel Spektrogram).
        """
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.array(tensor, copy=False)

        if arr.ndim == 4:  # (B, 3, 128, 259) lub (B, 128, 259, 3)
            arr = arr[0]

        if arr.ndim == 3:
            if arr.shape[0] == 3:  # (3, 128, 259)
                mel = arr[0]
            elif arr.shape[-1] == 3:  # (128, 259, 3)
                mel = arr[:, :, 0]
            else:
                mel = arr[0]
        elif arr.ndim == 2:
            mel = arr
        else:
            raise ValueError(f"Nieobsługiwany wymiar tensora: {arr.shape}")

        return mel.astype(np.float32)

    def compute_features(self, tensor: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        Oblicza zestaw cech fizycznych i DSP z tensora wejściowego.
        """
        mel = self._prepare_mel_matrix(tensor)  # (128, 259)
        n_mels, n_frames = mel.shape

        # 1. Prawdopodobieństwo rozkładu energii w każdej klatce (Softmax po osi częstotliwości)
        # Shift invariance dla wartości ujemnych z-score
        mel_shifted = mel - np.max(mel, axis=0, keepdims=True)
        exp_mel = np.exp(mel_shifted)
        p = exp_mel / (np.sum(exp_mel, axis=0, keepdims=True) + self.eps)  # (128, 259)

        # 2. RMS w klatkach
        frame_rms = np.sqrt(np.mean(mel ** 2, axis=0) + self.eps)  # (259,)

        # 3. Koncentracja energii w wąskim paśmie (top 6 binów)
        top_k_indices = np.argsort(p, axis=0)[-6:, :]
        top_k_energy = np.take_along_axis(p, top_k_indices, axis=0)
        narrowband_ratio = float(np.mean(np.sum(top_k_energy, axis=0)))

        # 4. Stabilność czasowa widma (podobieństwo cosinusowe klatek ze średnim widmem)
        mean_spec = np.mean(p, axis=1, keepdims=True)  # (128, 1)
        p_norm = p / (np.linalg.norm(p, axis=0, keepdims=True) + self.eps)
        mean_norm = mean_spec / (np.linalg.norm(mean_spec) + self.eps)
        frame_similarities = np.sum(p_norm * mean_norm, axis=0)
        temporal_stability = float(np.mean(frame_similarities))

        # 5. Entropia spektralna Mel (Shannon)
        entropy_per_frame = -np.sum(p * np.log(p + self.eps), axis=0)
        mel_entropy = float(np.mean(entropy_per_frame))

        # 6. Detekcja impulsów i ich regularność (inter-event interval CV)
        thresh = np.median(frame_rms) + 1.8 * np.std(frame_rms)
        peaks = (frame_rms[1:-1] > frame_rms[:-2]) & (frame_rms[1:-1] > frame_rms[2:]) & (frame_rms[1:-1] > thresh)
        peak_indices = np.where(peaks)[0] + 1

        num_events = len(peak_indices)
        transient_density = float(num_events / n_frames)

        if num_events >= 3:
            intervals = np.diff(peak_indices)
            inter_event_cv = float(np.std(intervals) / (np.mean(intervals) + self.eps))
        else:
            inter_event_cv = 1.0

        # 7. Autokorelacja periodyczności (badanie powtarzalności dla okresów 20 - 70 klatek)
        centered_rms = frame_rms - np.mean(frame_rms)
        denom = np.sum(centered_rms ** 2) + self.eps
        max_autocorr = 0.0
        for lag in range(20, min(80, n_frames // 2)):
            corr = np.sum(centered_rms[:-lag] * centered_rms[lag:]) / denom
            if corr > max_autocorr:
                max_autocorr = float(corr)
        periodicity_score = max(0.0, max_autocorr)

        # 8. Spójność widmowa impulsów (jeśli wykryto impulsy)
        if num_events >= 2:
            peak_spectra = p[:, peak_indices]  # (128, K)
            norm_pk = peak_spectra / (np.linalg.norm(peak_spectra, axis=0, keepdims=True) + self.eps)
            dot_products = [np.sum(norm_pk[:, i] * norm_pk[:, i + 1]) for i in range(num_events - 1)]
            event_spectral_consistency = float(np.mean(dot_products))
        else:
            event_spectral_consistency = 0.0

        return {
            "narrowband_ratio": narrowband_ratio,
            "temporal_stability": temporal_stability,
            "mel_entropy": mel_entropy,
            "transient_density": transient_density,
            "inter_event_cv": inter_event_cv,
            "periodicity_score": periodicity_score,
            "num_events": float(num_events),
            "event_spectral_consistency": event_spectral_consistency,
        }

    def is_coil_whine(self, tensor: Union[np.ndarray, torch.Tensor]) -> Tuple[bool, str]:
        """
        Zwraca True, jeśli sygnał wykazuje cechy wąskopasmowego pisku przetwornicy PWM (GPU Coil Whine).
        """
        feats = self.compute_features(tensor)
        # Empirycznie zweryfikowane progi: coil whine ma brak mikrotransjentów, b. wysoką stabilność czasową i koncentrację
        is_narrowband = feats["narrowband_ratio"] > 0.22
        is_temporally_stable = feats["temporal_stability"] > 0.93
        low_transients = feats["transient_density"] < 0.012

        if is_narrowband and is_temporally_stable and low_transients:
            reason = f"Narrowband PWM tone (narrowband={feats['narrowband_ratio']:.2f}, stability={feats['temporal_stability']:.2f}, transients={feats['transient_density']:.3f})"
            return True, reason
        return False, ""

    def is_periodic_metronome(self, tensor: Union[np.ndarray, torch.Tensor]) -> Tuple[bool, str]:
        """
        Zwraca True, jeśli sygnał to idealnie regularny syntetyczny metronom (CV < 0.005, spójność widmowa > 0.96).
        """
        feats = self.compute_features(tensor)
        has_regular_clicks = (feats["num_events"] >= 3) and (feats["inter_event_cv"] < 0.005)
        has_spectral_consistency = feats["event_spectral_consistency"] > 0.96

        if has_regular_clicks and has_spectral_consistency:
            reason = f"Periodic synthetic metronome (CV={feats['inter_event_cv']:.4f}, consistency={feats['event_spectral_consistency']:.3f})"
            return True, reason
        return False, ""

    def is_aerodynamic_fan(
        self,
        tensor: Union[np.ndarray, torch.Tensor],
        waveform: Optional[np.ndarray] = None,
        sr: int = TARGET_SR,
    ) -> Tuple[bool, str]:
        """
        Zwraca True, jeśli sygnał reprezentuje ciągły szum aerodynamiczny (wentylatory PC, turbiny, odkurzacz)
        o opadającym widmie 1/f i gaussowskiej kurtozie, bez transjentów mechanicznych HDD.
        """
        # Jeśli dostarczono surowy chunk audio 1D (np. 66150 próbek)
        if waveform is not None and len(waveform) >= 1024:
            chunk = np.asarray(waveform, dtype=np.float32)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < 0.002:
                return False, ""

            peak = float(np.max(np.abs(chunk)))
            crest = peak / max(rms, 1e-12)
            kurt = float(scipy.stats.kurtosis(chunk, fisher=False))

            # Analiza widmowa FFT
            fft = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)

            mask_slope = (freqs >= 100) & (freqs <= 4000)
            if np.sum(mask_slope) > 8:
                log_f = np.log(freqs[mask_slope])
                log_p = np.log(fft[mask_slope] ** 2 + 1e-12)
                slope, _, r_val, _, _ = scipy.stats.linregress(log_f, log_p)
            else:
                slope = 0.0

            total_energy = np.sum(fft ** 2) + 1e-12
            high_freq_ratio = float(np.sum(fft[freqs > 4000] ** 2) / total_energy)

            # Safety Veto - jeśli sygnał ma ostre uderzenia lub wysoką energię tarcia >4kHz, to NIE jest wentylator!
            if crest > 6.5 or kurt > 6.0:
                return False, ""
            if high_freq_ratio > 0.15:
                return False, ""
            mid_high_energy_ratio = float(np.sum(fft[freqs >= 1000] ** 2) / total_energy)
            if mid_high_energy_ratio > 0.50:
                return False, ""

            # Pozytywne kryterium szumu aerodynamicznego (spadek 1/f i quasigaussowska kurtoza)
            if slope < -0.70 and kurt < 5.5 and crest < 6.2:
                reason = f"Aerodynamic fan/turbulence (slope={slope:.2f}, kurtosis={kurt:.2f}, crest={crest:.2f}, high_freq={high_freq_ratio*100:.1f}%)"
                return True, reason

        # Proxy fallback z tensora mel (jeśli brak surowego waveformu)
        feats = self.compute_features(tensor)
        mel = self._prepare_mel_matrix(tensor)  # (128, 259)
        # Opadanie widma po osi Mel
        mel_mean = np.mean(mel, axis=1)  # (128,)
        mel_bins = np.arange(len(mel_mean))
        slope_proxy, _, _, _, _ = scipy.stats.linregress(mel_bins[10:90], mel_mean[10:90])

        if feats["transient_density"] < 0.02 and feats["temporal_stability"] > 0.80 and slope_proxy < -0.05:
            reason = f"Aerodynamic mel profile (transient_density={feats['transient_density']:.3f}, mel_slope={slope_proxy:.3f})"
            return True, reason

        return False, ""

    def is_microphone_clipped_or_distorted(
        self,
        tensor: Union[np.ndarray, torch.Tensor],
        waveform: Optional[np.ndarray] = None,
        sr: int = TARGET_SR,
    ) -> Tuple[bool, str]:
        """
        Zwraca True, jeśli sygnał wykazuje nienaturalne przesterowanie toru mikrofonowego (Hard Clipping / Flat-Tops)
        lub silne zniekształcenia harmoniczne (THD comb), które sieć 2D CNN mylnie interpretuje jako scratching.
        Chroni autentyczne uszkodzenia HDD (gdzie Crest Factor > 3.2 i szerokopasmowe tarcie > 4 kHz).
        """
        if waveform is not None and len(waveform) >= 1024:
            chunk = np.asarray(waveform, dtype=np.float32)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < 0.002:
                return False, ""

            peak = float(np.max(np.abs(chunk)))
            crest = peak / max(rms, 1e-12)

            is_rail = np.abs(chunk) >= (0.985 * peak)
            diff = np.diff(np.concatenate(([0], is_rail.astype(np.int8), [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            run_lengths = ends - starts if len(starts) > 0 else [0]
            max_run = int(np.max(run_lengths))
            clip_ratio = float(np.mean(is_rail))

            # Analiza widmowa powyżej 4 kHz (badanie czy wysokie częstotliwości to szum tarcia, czy grzebień harmoniczny THD)
            fft = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)
            mask_high = (freqs >= 4000) & (freqs <= 10000)
            power_high = (fft[mask_high] ** 2) + 1e-12
            geom_mean = np.exp(np.mean(np.log(power_high)))
            arith_mean = np.mean(power_high)
            flatness_high = float(geom_mean / arith_mean)

            # Safety Veto dla autentycznego uszkodzenia HDD:
            # Prawdziwe tarcie głowic o talerz (scratching) lub uderzenia (clicking)
            # posiadają wysoką dynamikę impulsową (Crest > 3.2) oraz szerokopasmowe widmo tarcia (flatness > 0.18).
            if crest >= 3.2 and flatness_high > 0.18 and clip_ratio < 0.05:
                return False, ""

            # Kryterium 1: Twarde obcięcie fali (Hard Clipping)
            if (clip_ratio > 0.02 and max_run >= 3) or (clip_ratio > 0.10):
                if crest < 2.5 or flatness_high < 0.08:
                    reason = f"Microphone hard clipping (clip_ratio={clip_ratio*100:.1f}%, max_run={max_run}, crest={crest:.2f}, flatness_high={flatness_high:.4f})"
                    return True, reason

            # Kryterium 2: Niskodynamiczny ton przesterowany (Crest < 2.0 i grzebień harmoniczny THD)
            if crest < 2.0 and flatness_high < 0.05:
                reason = f"Distorted tone / harmonic comb (crest={crest:.2f}, flatness_high={flatness_high:.4f})"
                return True, reason

        return False, ""

    def is_enterprise_spindle_windage(
        self,
        tensor: Union[np.ndarray, torch.Tensor],
        waveform: Optional[np.ndarray] = None,
        sr: int = TARGET_SR,
    ) -> Tuple[bool, str]:
        """
        Zwraca True, jeśli sygnał reprezentuje ciągły szum obracających się talerzy dysku enterprise (10k / 15k RPM)
        o stabilnych harmonicznych obrotowych (166.7 Hz lub 250 Hz) i wysokiej stabilności czasowej,
        mylonych przez sieć z tarciem głowic (scratching).
        """
        feats = self.compute_features(tensor)
        if feats["temporal_stability"] < 0.80 or feats["transient_density"] > 0.12:
            return False, ""

        if waveform is not None and len(waveform) >= 1024:
            chunk = np.asarray(waveform, dtype=np.float32)
            fft = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)

            # Sprawdzenie harmonicznych 10k RPM (~166.7 Hz) i 15k RPM (~250 Hz)
            for f_cand in [166.7, 250.0]:
                m_f0 = (freqs >= f_cand - 6) & (freqs <= f_cand + 6)
                if np.sum(m_f0) == 0:
                    continue
                f0 = freqs[m_f0][np.argmax(fft[m_f0])]
                harmonics_found = 0
                for h in [2, 3, 4, 5]:
                    target = f0 * h
                    m_h = (freqs >= target - 5) & (freqs <= target + 5)
                    m_bg = (freqs >= target - 25) & (freqs <= target + 25) & ~m_h
                    if np.sum(m_h) > 0 and np.sum(m_bg) > 0:
                        prom = np.max(fft[m_h]) / max(np.mean(fft[m_bg]), 1e-12)
                        if prom > 1.35:
                            harmonics_found += 1
                if harmonics_found >= 2:
                    rpm_est = int(f0 * 60)
                    reason = f"Enterprise {rpm_est} RPM Spindle Windage (f0={f0:.1f}Hz, harmonics={harmonics_found}, stability={feats['temporal_stability']:.2f})"
                    return True, reason

        return False, ""

    def protect_and_sanitize_predictions(
        self,
        probs: Union[np.ndarray, torch.Tensor],
        tensor: Union[np.ndarray, torch.Tensor],
        waveform: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, str]:
        """
        Weryfikuje predykcję modelu. Jeśli wykryto fałszywy alarm adwersarialny,
        bezpiecznie przekierowuje wynik do klasy 'normal' z wysokim zaufaniem.
        """
        if isinstance(probs, torch.Tensor):
            p = probs.detach().cpu().numpy().flatten()
        else:
            p = np.array(probs, dtype=np.float32).flatten()

        is_cw, reason_cw = self.is_coil_whine(tensor)
        is_metro, reason_metro = self.is_periodic_metronome(tensor)
        is_fan, reason_fan = self.is_aerodynamic_fan(tensor, waveform=waveform)
        is_clip, reason_clip = self.is_microphone_clipped_or_distorted(tensor, waveform=waveform)
        is_ent, reason_ent = self.is_enterprise_spindle_windage(tensor, waveform=waveform)
        feats = self.compute_features(tensor)

        normal_idx = CLASSES.index("normal")
        scratching_idx = CLASSES.index("scratching")
        clicking_idx = CLASSES.index("clicking")
        beeping_idx = CLASSES.index("beeping")

        # 1. Neutralizacja Coil Whine klasyfikowanego jako SCRATCHING
        if is_cw and p[scratching_idx] > 0.25:
            new_p = np.copy(p)
            stolen = new_p[scratching_idx]
            new_p[scratching_idx] = 0.02
            new_p[normal_idx] += stolen * 0.98
            new_p = new_p / np.sum(new_p)
            return new_p, True, f"Odrzucono fałszywe drapanie (Coil Whine): {reason_cw}"

        # 2. Neutralizacja Metronomu klasyfikowanego jako CLICKING / BEEPING
        if is_metro and (p[clicking_idx] > 0.20 or p[beeping_idx] > 0.20):
            new_p = np.copy(p)
            stolen = new_p[clicking_idx] + new_p[beeping_idx]
            new_p[clicking_idx] = 0.02
            new_p[beeping_idx] = 0.02
            new_p[normal_idx] += stolen * 0.96
            new_p = new_p / np.sum(new_p)
            return new_p, True, f"Odrzucono fałszywe klikanie (Metronom): {reason_metro}"

        # 3. Neutralizacja Szumu Wentylatorów / Turbin PC klasyfikowanego jako CLICKING / SCRATCHING / BEEPING
        if is_fan and (p[clicking_idx] > 0.25 or p[scratching_idx] > 0.25 or p[beeping_idx] > 0.25):
            new_p = np.copy(p)
            fault_prob = new_p[clicking_idx] + new_p[scratching_idx] + new_p[beeping_idx]
            new_p[clicking_idx] = 0.02
            new_p[scratching_idx] = 0.02
            new_p[beeping_idx] = 0.02
            new_p[normal_idx] += fault_prob * 0.95
            new_p = new_p / np.sum(new_p)
            return new_p, True, f"Odrzucono fałszywy alarm wentylatora/turbiny: {reason_fan}"

        # 4. Neutralizacja Przesterowania Toru Mikrofonu / THD klasyfikowanego jako SCRATCHING / CLICKING
        if is_clip and (p[scratching_idx] > 0.20 or p[clicking_idx] > 0.25):
            new_p = np.copy(p)
            stolen = new_p[scratching_idx] + (new_p[clicking_idx] if new_p[clicking_idx] > 0.25 else 0.0)
            new_p[scratching_idx] = 0.02
            if new_p[clicking_idx] > 0.25:
                new_p[clicking_idx] = 0.02
            new_p[normal_idx] += stolen * 0.98
            new_p = new_p / np.sum(new_p)
            return new_p, True, f"Odrzucono fałszywy alarm przesterowania mikrofonu: {reason_clip}"

        # 5. Neutralizacja fałszywego klikania bez transjentów (Zero-transient false clicking)
        # Fizyczny Click of Death wymaga uderzeń pozycjonera (num_events >= 2).
        if waveform is not None and len(waveform) >= 1024:
            if p[clicking_idx] > 0.20 and feats["num_events"] == 0 and feats["transient_density"] == 0.0:
                peak_w = float(np.max(np.abs(waveform)))
                rms_w = float(np.sqrt(np.mean(waveform ** 2)))
                crest_w = peak_w / max(rms_w, 1e-12)

                # Bezpiecznik maskowania: jeśli w sygnale występuje wysoka dynamika impulsowa (zamaskowany click), zachowaj klikanie!
                if crest_w <= 7.0:
                    fft_t = np.abs(np.fft.rfft(waveform))
                    freqs_t = np.fft.rfftfreq(len(waveform), 1.0 / TARGET_SR)
                    mid_high = float(np.sum(fft_t[freqs_t >= 1000] ** 2) / (np.sum(fft_t ** 2) + 1e-12))
                    if mid_high > 0.50:
                        new_p = np.copy(p)
                        stolen = new_p[clicking_idx]
                        new_p[clicking_idx] = 0.02
                        new_p[scratching_idx] += stolen * 0.98
                        new_p = new_p / np.sum(new_p)
                        return new_p, True, f"Przekierowano do drapania (Tarcie powierzchniowe bez kliknięć: mid_high={mid_high*100:.1f}%)"
                    elif crest_w < 6.5:
                        new_p = np.copy(p)
                        stolen = new_p[clicking_idx]
                        new_p[clicking_idx] = 0.02
                        new_p[normal_idx] += stolen * 0.98
                        new_p = new_p / np.sum(new_p)
                        return new_p, True, "Odrzucono fałszywe klikanie (Brak fizycznych impulsów / transjentów: num_events=0)"

        # 6. Neutralizacja szumu aerodynamicznego obrotu talerzy enterprise 10k/15k RPM
        if is_ent and p[scratching_idx] > max(p[clicking_idx], p[beeping_idx]) and p[scratching_idx] > 0.25:
            new_p = np.copy(p)
            stolen = new_p[scratching_idx]
            new_p[scratching_idx] = 0.02
            new_p[normal_idx] += stolen * 0.98
            new_p = new_p / np.sum(new_p)
            return new_p, True, f"Odrzucono fałszywe drapanie (Enterprise Spindle Windage): {reason_ent}"

        return p, False, ""
