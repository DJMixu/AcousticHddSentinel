# Kod Wygenerowany przez (qwen2.5-coder:14b):
```python
import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import scipy.signal

# Constants from ADR-001
TARGET_SR = 22050
N_FFT = 1024
HOP_LENGTH = 256
CHUNK_SAMPLES = 66150  # 3.0 seconds

@dataclass
class GuardMetrics:
    bpf_comb_score: float
    turbulence_score: float
    temporal_kurtosis: float
    crest_factor: float
    spectral_flux: float
    high_freq_ratio: float

@dataclass
class GuardStatus:
    is_fan: bool
    metrics: GuardMetrics

@dataclass
class GuardResult:
    status: GuardStatus
    chunk_index: int

class AerodynamicFanGuard:
    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        self.audio_data, self.sample_rate = self.load_audio()
        if self.sample_rate != TARGET_SR:
            raise ValueError(f"Sample rate must be {TARGET_SR} Hz, but got {self.sample_rate} Hz.")
        
    def load_audio(self) -> Tuple[np.ndarray, int]:
        try:
            audio_data, sample_rate = librosa.load(self.audio_path, sr=None)
            return audio_data, sample_rate
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}") from e

    def compute_stft(self, audio_chunk: np.ndarray) -> np.ndarray:
        stft = librosa.stft(audio_chunk, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann')
        return np.abs(stft)

    def bpf_comb_score(self, stft: np.ndarray) -> float:
        frequencies = librosa.fft_frequencies(sr=TARGET_SR, n_fft=N_FFT)
        bpf_mask = (frequencies >= 80) & (frequencies <= 1000)
        bpf_stft = stft[bpf_mask]
        return np.sum(bpf_stft)

    def turbulence_score(self, stft: np.ndarray) -> float:
        frequencies = librosa.fft_frequencies(sr=TARGET_SR, n_fft=N_FFT)
        log_frequencies = np.log(frequencies + 1e-6)
        log_spectrogram = np.log(stft + 1e-6)
        slope, _, _, _, _ = scipy.stats.linregress(log_frequencies, log_spectrogram.mean(axis=1))
        return slope

    def temporal_kurtosis(self, audio_chunk: np.ndarray) -> float:
        return scipy.stats.kurtosis(audio_chunk)

    def crest_factor(self, audio_chunk: np.ndarray) -> float:
        peak = np.max(np.abs(audio_chunk))
        rms = np.sqrt(np.mean(audio_chunk**2))
        return peak / rms

    def spectral_flux(self, stft: np.ndarray) -> float:
        return np.sum(np.abs(np.diff(stft, axis=1)), axis=0).mean()

    def high_freq_ratio(self, stft: np.ndarray) -> float:
        frequencies = librosa.fft_frequencies(sr=TARGET_SR, n_fft=N_FFT)
        high_freq_mask = frequencies > 5000
        high_freq_energy = np.sum(stft[high_freq_mask])
        total_energy = np.sum(stft)
        return high_freq_energy / total_energy

    def analyze_chunk(self, audio_chunk: np.ndarray) -> GuardStatus:
        stft = self.compute_stft(audio_chunk)
        bpf_comb_score = self.bpf_comb_score(stft)
        turbulence_score = self.turbulence_score(stft)
        temporal_kurtosis = self.temporal_kurtosis(audio_chunk)
        crest_factor = self.crest_factor(audio_chunk)
        spectral_flux = self.spectral_flux(stft)
        high_freq_ratio = self.high_freq_ratio(stft)

        metrics = GuardMetrics(
            bpf_comb_score=bpf_comb_score,
            turbulence_score=turbulence_score,
            temporal_kurtosis=temporal_kurtosis,
            crest_factor=crest_factor,
            spectral_flux=spectral_flux,
            high_freq_ratio=high_freq_ratio
        )

        is_fan = not (crest_factor > 7.0 or temporal_kurtosis > 8.0 or high_freq_ratio > 0.22 or spectral_flux > 0.40)
        return GuardStatus(is_fan=is_fan, metrics=metrics)

    def process_audio(self) -> List[GuardResult]:
        results = []
        for i in range(0, len(self.audio_data), CHUNK_SAMPLES):
            audio_chunk = self.audio_data[i:i + CHUNK_SAMPLES]
            if np.mean(np.abs(audio_chunk)) < 0.002:
                continue
            status = self.analyze_chunk(audio_chunk)
            result = GuardResult(status=status, chunk_index=i // CHUNK_SAMPLES)
            results.append(result)
        return results

# Example usage:
# guard = AerodynamicFanGuard('path_to_audio_file.wav')
# results = guard.process_audio()
# for result in results:
#     print(f"Chunk {result.chunk_index}: Is Fan? {result.status.is_fan}, Metrics: {result.status.metrics}")
```

This code implements the `AerodynamicFanGuard` class as specified, including all required methods and data classes. It processes audio data in chunks, computes various metrics, and determines if each chunk corresponds to a fan sound based on predefined criteria. The code is designed to be production-ready, with error handling and type hints included.
