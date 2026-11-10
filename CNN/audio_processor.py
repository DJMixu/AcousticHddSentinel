"""
Acoustic HDD Sentinel - Audio Preprocessing & Normalization Module
Moduł przetwarzania i normalizacji dźwięku pracy dysków HDD przed ekstrakcją cech do CNN.
"""

import math
import wave
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Tuple, List, Optional, Union


class AudioNormalizer:
    """
    Klasa odpowiedzialna za normalizację sygnału audio HDD:
    1. Usunięcie składowej stałej (DC offset removal)
    2. Resampling do zadanej częstotliwości próbkowania (domyślnie 44100 Hz)
    3. Normalizacja amplitudy w dziedzinie czasu (Peak normalization / RMS normalization)
    4. Cięcie na 5-sekundowe okna z ewentualnym paddingiem
    5. Konwersja do Mel-spektrogramu
    6. Normalizacja spektrogramu (Log-mel + Min-Max [0, 1])
    """

    def __init__(
        self,
        target_sample_rate: int = 44100,
        chunk_duration_sec: float = 5.0,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        peak_target: float = 0.95,
        eps: float = 1e-9
    ):
        self.target_sample_rate = target_sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.chunk_samples = int(target_sample_rate * chunk_duration_sec)  # np. 220500 dla 5s @ 44.1kHz
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.peak_target = peak_target
        self.eps = eps

        # MelSpectrogram transformator PyTorch
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )

    # --------------------------------------------------------------------------
    # 1. Odczyt i wczytywanie pliku WAV
    # --------------------------------------------------------------------------
    @staticmethod
    def load_wav(file_path: str) -> Tuple[torch.Tensor, int]:
        """
        Wczytuje plik .wav za pomocą modułu `wave` (niezależnie od zewnętrznych bibliotek).
        Zwraca tensor (channels, samples) o wartościach float32 w zakresie [-1.0, 1.0] oraz sample_rate.
        """
        with wave.open(file_path, 'rb') as w:
            n_channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            sample_rate = w.getframerate()
            n_frames = w.getnframes()
            raw_bytes = w.readframes(n_frames)

        if sampwidth == 2:  # 16-bit PCM
            audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:  # 8-bit PCM (unsigned)
            audio_np = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sampwidth == 4:  # 32-bit PCM
            audio_np = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Nieobsługiwany format próbki: sampwidth={sampwidth}")

        # Jeśli multi-channel -> reshape do (n_channels, n_samples)
        if n_channels > 1:
            audio_np = audio_np.reshape(-1, n_channels).T
            # Konwersja do mono przez uśrednienie kanałów
            audio_np = np.mean(audio_np, axis=0, keepdims=True)
        else:
            audio_np = audio_np.reshape(1, -1)

        tensor = torch.from_numpy(audio_np).float()
        return tensor, sample_rate

    # --------------------------------------------------------------------------
    # 2. Usuwanie DC Offset (Centrowanie sygnału)
    # --------------------------------------------------------------------------
    def remove_dc_offset(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Usuwa składową stałą: x = x - mean(x).
        Kluczowe dla nagrań z mikrofonów i kart dźwiękowych z biasem napięciowym.
        """
        return waveform - torch.mean(waveform, dim=-1, keepdim=True)

    # --------------------------------------------------------------------------
    # 3. Resampling (Dopasowanie częstotliwości próbkowania)
    # --------------------------------------------------------------------------
    def resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Resampluje waveform do docelowego target_sample_rate (np. 22050 -> 44100).
        """
        if orig_sr == self.target_sample_rate:
            return waveform
        resampler = T.Resample(orig_freq=orig_sr, new_freq=self.target_sample_rate)
        return resampler(waveform)

    # --------------------------------------------------------------------------
    # 4. Normalizacja amplitudy (dziedzina czasu)
    # --------------------------------------------------------------------------
    def normalize_peak(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Peak normalization:
        Skaluje sygnał tak, aby maksymalna wartość bezwzględna wynosiła `peak_target` (domyślnie 0.95).
        Pozostawia 5% headroomu zabezpieczającego przed clippingiem.
        """
        max_val = torch.max(torch.abs(waveform))
        if max_val > self.eps:
            return waveform * (self.peak_target / max_val)
        return waveform

    def normalize_rms(self, waveform: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
        """
        RMS normalization:
        Normalizuje moc sygnału do zadanego poziomu RMS, z twardym ograniczeniem (clippingiem) do [-1.0, 1.0].
        """
        current_rms = torch.sqrt(torch.mean(waveform ** 2) + self.eps)
        gain = target_rms / current_rms
        normalized = waveform * gain
        # Zabezpieczenie przed przesterem
        return torch.clamp(normalized, -1.0, 1.0)

    # --------------------------------------------------------------------------
    # 5. Okienkowanie (Chunking na 5-sekundowe fragmenty)
    # --------------------------------------------------------------------------
    def chunk_audio(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        Dzieli sygnał na sztywne 5-sekundowe fragmenty (chunk_samples).
        Ostatni fragment, jeśli jest krótszy niż 5s, dopełniany jest zerami (zero-padding).
        Zwraca listę tensorów o wymiarze [1, chunk_samples].
        """
        total_samples = waveform.shape[-1]
        if total_samples == 0:
            return []

        chunks = []
        num_chunks = math.ceil(total_samples / self.chunk_samples)

        for i in range(num_chunks):
            start = i * self.chunk_samples
            end = min(start + self.chunk_samples, total_samples)
            chunk = waveform[:, start:end]

            # Dopełnienie zerami jeśli za krótki
            if chunk.shape[-1] < self.chunk_samples:
                pad_size = self.chunk_samples - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_size), mode='constant', value=0.0)

            chunks.append(chunk)

        return chunks

    # --------------------------------------------------------------------------
    # 6. Ekstrakcja i normalizacja Mel-spektrogramu
    # --------------------------------------------------------------------------
    def compute_mel_spectrogram(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Generuje Mel-spektrogram z 5-sekundowego fragmentu:
        1. STFT + Mel filterbank
        2. Log-skala: log(mel + eps)
        3. Min-Max normalizacja do [0.0, 1.0]
        Zwraca tensor o kształcie [1, n_mels, time_frames] np. [1, 64, 431].
        """
        # Mel spectrogram (power)
        mel_spec = self.mel_transform(chunk)  # [1, n_mels, time_frames]

        # Log scaling (kompresja zakresu dynamicznego)
        log_mel = torch.log(mel_spec + self.eps)

        # Min-Max normalization do zakresu [0.0, 1.0] per spektrogram
        min_val = torch.min(log_mel)
        max_val = torch.max(log_mel)
        denom = max_val - min_val

        if denom > self.eps:
            norm_mel = (log_mel - min_val) / denom
        else:
            norm_mel = torch.zeros_like(log_mel)

        return norm_mel

    # --------------------------------------------------------------------------
    # 7. Kompletny potok przetwarzania pojedynczego pliku
    # --------------------------------------------------------------------------
    def process_file(
        self,
        file_path: str,
        normalization_method: str = "peak"
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Wykonuje pełny proces od pliku .wav do znormalizowanych spektrogramów:
        1. Load WAV -> float32 [-1, 1], Mono
        2. DC Offset Removal
        3. Resampling do 44.1 kHz
        4. Normalizacja amplitudy (peak / rms)
        5. Podział na okna 5s
        6. Ekstrakcja znormalizowanych Mel-spektrogramów [1, 64, 431]

        Zwraca listę krotek: (chunk_waveform, normalized_mel_spectrogram)
        """
        # 1. Wczytanie
        waveform, sr = self.load_wav(file_path)

        # 2. Usunięcie DC offset
        waveform = self.remove_dc_offset(waveform)

        # 3. Resampling do 44.1 kHz
        waveform = self.resample(waveform, sr)

        # 4. Okienkowanie
        chunks = self.chunk_audio(waveform)

        processed = []
        for chunk in chunks:
            # Normalizacja amplitudy wewnątrz okna
            if normalization_method == "peak":
                chunk_norm = self.normalize_peak(chunk)
            elif normalization_method == "rms":
                chunk_norm = self.normalize_rms(chunk)
            else:
                chunk_norm = chunk

            # Generacja i normalizacja spektrogramu
            mel_spec = self.compute_mel_spectrogram(chunk_norm)
            processed.append((chunk_norm, mel_spec))

        return processed
