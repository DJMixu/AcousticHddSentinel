# Kod Wygenerowany przez (qwen2.5-coder:14b):
```python
import torch
from typing import Tuple, List, Dict
import numpy as np
from scipy.signal import spectrogram

class ClippingGuardConfig:
    """
    Configuration for the MicrophoneClippingGuard.
    """
    def __init__(self, sr: int = 22050, n_fft: int = 1024, hop: int = 256, chunk: int = 66150):
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.chunk = chunk

class ClippingMetrics:
    """
    Metrics related to clipping detection.
    """
    def __init__(self, comb_ratio: float, spectral_flatness_high: float, thd_harmonics: float):
        self.comb_ratio = comb_ratio
        self.spectral_flatness_high = spectral_flatness_high
        self.thd_harmonics = thd_harmonics

class ClippingReport:
    """
    Report on microphone clipping and distortion.
    """
    def __init__(self, is_clipped: bool, is_distorted: bool, artifact_likelihood: float, authentic_scratching_likely: bool):
        self.is_clipped = is_clipped
        self.is_distorted = is_distorted
        self.artifact_likelihood = artifact_likelihood
        self.authentic_scratching_likely = authentic_scratching_likely

class MicrophoneClippingGuard:
    """
    Class to detect microphone clipping and distortion.
    """
    def __init__(self, config: ClippingGuardConfig):
        self.config = config

    def detect_flat_top_runs(self, waveform: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Detects flat top runs in the waveform where |x| >= 0.98 * peak and min_run >= 4.
        
        :param waveform: Input audio waveform tensor of shape (1, chunk_size)
        :return: List of tuples (start_index, end_index) for each flat top run
        """
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("Waveform must be a 2D tensor with shape (1, chunk_size).")
        
        peak = torch.max(torch.abs(waveform))
        threshold = 0.98 * peak
        flat_top_indices = torch.where(torch.abs(waveform) >= threshold)[1]
        
        runs = []
        start_index = None
        for i in range(len(flat_top_indices)):
            if start_index is None:
                start_index = flat_top_indices[i]
            elif flat_top_indices[i] != flat_top_indices[i - 1] + 1:
                if flat_top_indices[i - 1] - start_index + 1 >= 4:
                    runs.append((start_index.item(), flat_top_indices[i - 1].item()))
                start_index = flat_top_indices[i]
        
        if start_index is not None and flat_top_indices[-1] - start_index + 1 >= 4:
            runs.append((start_index.item(), flat_top_indices[-1].item()))
        
        return runs

    def compute_thd_comb_metrics(self, waveform: torch.Tensor) -> ClippingMetrics:
        """
        Computes THD comb metrics for the waveform in the 4-10 kHz range.
        
        :param waveform: Input audio waveform tensor of shape (1, chunk_size)
        :return: ClippingMetrics object with comb_ratio, spectral_flatness_high, and thd_harmonics
        """
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("Waveform must be a 2D tensor with shape (1, chunk_size).")
        
        f, t, Sxx = spectrogram(waveform.numpy()[0], fs=self.config.sr, nperseg=self.config.n_fft, noverlap=self.config.hop)
        freq_mask = (f >= 4000) & (f <= 10000)
        Sxx_high = Sxx[freq_mask]
        
        comb_ratio = np.mean(Sxx_high[1:] / Sxx_high[:-1])
        spectral_flatness_high = np.exp(np.mean(np.log(Sxx_high)))
        thd_harmonics = np.sum(Sxx_high[1:]) / Sxx_high[0]
        
        return ClippingMetrics(comb_ratio, spectral_flatness_high, thd_harmonics)

    def is_microphone_clipped_or_distorted(self, waveform: torch.Tensor, prediction: str) -> ClippingReport:
        """
        Determines if the microphone is clipped or distorted.
        
        :param waveform: Input audio waveform tensor of shape (1, chunk_size)
        :param prediction: Prediction label from DSPAdversarialGuard
        :return: ClippingReport object with is_clipped, is_distorted, artifact_likelihood, and authentic_scratching_likely
        """
        flat_top_runs = self.detect_flat_top_runs(waveform)
        metrics = self.compute_thd_comb_metrics(waveform)
        
        is_clipped = len(flat_top_runs) > 0
        is_distorted = metrics.comb_ratio > 1.5 or metrics.thd_harmonics > 0.2
        artifact_likelihood = 0.8 if prediction == 'scratching' else 0.2
        authentic_scratching_likely = metrics.spectral_flatness_high < 0.5
        
        return ClippingReport(is_clipped, is_distorted, artifact_likelihood, authentic_scratching_likely)

    def sanitize_prediction(self, probs: Dict[str, float], waveform: torch.Tensor) -> str:
        """
        Sanitizes the prediction by neutralizing false 'scratching' calls due to clipping.
        
        :param probs: Dictionary of prediction probabilities
        :param waveform: Input audio waveform tensor of shape (1, chunk_size)
        :return: Sanitized prediction label
        """
        report = self.is_microphone_clipped_or_distorted(waveform, 'scratching')
        if report.is_clipped or report.is_distorted:
            return 'normal'
        return 'scratching'

# Example usage:
if __name__ == "__main__":
    config = ClippingGuardConfig()
    guard = MicrophoneClippingGuard(config)
    
    # Dummy waveform and prediction for demonstration
    waveform = torch.rand(1, 66150)  # Replace with actual waveform data
    prediction = 'scratching'
    
    report = guard.is_microphone_clipped_or_distorted(waveform, prediction)
    sanitized_prediction = guard.sanitize_prediction({'scratching': 0.9}, waveform)
    
    print(f"Clipping Report: {report.__dict__}")
    print(f"Sanitized Prediction: {sanitized_prediction}")
```

This code implements the `MicrophoneClippingGuard` class with methods to detect flat top runs, compute THD comb metrics, determine if the microphone is clipped or distorted, and sanitize predictions. It adheres to the specified configuration and requirements.
