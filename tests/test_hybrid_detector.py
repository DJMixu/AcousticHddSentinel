"""
Testy logiczne i numeryczne Hybrydowego Silnika Decyzyjnego Acoustic HDD Sentinel (ml/hybrid_sentinel_engine.py).
Weryfikacja fuzji reguł DSP (RPM piki, Stall Beep, stacjonarność) oraz 2D CNN.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import scipy.signal as signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.hybrid_sentinel_engine import HybridSentinelEngine
from ml.rpm_harmonics_analyzer import TARGET_SR

class TestHybridSentinelEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = HybridSentinelEngine()

    def setUp(self):
        self.sr = TARGET_SR
        self.dur = 3.0
        self.n_samples = int(self.dur * self.sr)
        self.t = np.linspace(0, self.dur, self.n_samples, endpoint=False)

    def test_5400_rpm_healthy_disk(self):
        """Zdrowy dysk 5400 RPM musi być zaklasyfikowany jako NORMAL (z potwierdzeniem 5400 RPM)."""
        f0 = 90.0
        y = 0.25 * np.sin(2 * np.pi * f0 * self.t)
        y += 0.15 * np.sin(2 * np.pi * 2 * f0 * self.t)
        y += 0.10 * np.sin(2 * np.pi * 3 * f0 * self.t)
        y += 0.02 * np.random.randn(self.n_samples)

        res = self.engine.analyze_audio_chunk(y.astype(np.float32), sr=self.sr)
        self.assertEqual(res["decision"], "normal")
        self.assertFalse(res["is_anomaly"])
        self.assertEqual(res["nominal_rpm"], 5400)

    def test_7200_rpm_healthy_disk(self):
        """Zdrowy dysk 7200 RPM musi być zaklasyfikowany jako NORMAL (z potwierdzeniem 7200 RPM)."""
        f0 = 120.0
        y = 0.30 * np.sin(2 * np.pi * f0 * self.t)
        y += 0.18 * np.sin(2 * np.pi * 2 * f0 * self.t)
        y += 0.12 * np.sin(2 * np.pi * 3 * f0 * self.t)
        y += 0.02 * np.random.randn(self.n_samples)

        res = self.engine.analyze_audio_chunk(y.astype(np.float32), sr=self.sr)
        self.assertEqual(res["decision"], "normal")
        self.assertFalse(res["is_anomaly"])
        self.assertEqual(res["nominal_rpm"], 7200)

    def test_pc_fan_noise_not_classified_as_fault(self):
        """Szum wentylatora PC (1200 RPM / BPF 140 Hz) NIE MOŻE być zgłoszony jako awaria (scratching/clicking)!"""
        white = np.random.randn(self.n_samples)
        sos = signal.butter(3, [70 / (self.sr/2), 2000 / (self.sr/2)], btype="bandpass", output="sos")
        fan = signal.sosfilt(sos, white) * 0.15
        fan += 0.05 * np.sin(2 * np.pi * 140.0 * self.t)

        res = self.engine.analyze_audio_chunk(fan.astype(np.float32), sr=self.sr)
        # Nie może być awarią (nie clicking, nie scratching, nie beeping)!
        self.assertFalse(res["is_anomaly"], f"Wentylator PC nie może być awarią! Wynik: {res['decision']}")

    def test_seized_spindle_stall_beep_confirmed(self):
        """Zablokowany silnik (pisk 1500 Hz bez rotacji 90/120 Hz) musi być zgłoszony jako BEEPING."""
        y = 0.25 * np.sin(2 * np.pi * 1500.0 * self.t)
        y += 0.08 * np.sin(2 * np.pi * 3000.0 * self.t)
        y += 0.01 * np.random.randn(self.n_samples)

        res = self.engine.analyze_audio_chunk(y.astype(np.float32), sr=self.sr)
        self.assertEqual(res["decision"], "beeping")
        self.assertTrue(res["is_anomaly"])

    def test_mouth_hum_rejected_from_beeping(self):
        """Buczenie ustami (formanty wokalne z wibrato) nie może być fałszywie zgłoszone jako AWARIA BEEPING."""
        pitch_mod = 130.0 + 15.0 * np.sin(2 * np.pi * 5.0 * self.t)
        phase = 2 * np.pi * np.cumsum(pitch_mod) / self.sr
        vocal = 0.3 * np.sin(phase) + 0.2 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)

        res = self.engine.analyze_audio_chunk(vocal.astype(np.float32), sr=self.sr)
        # System hybrydowy musi odrzucić mowę lub oznaczyć jako uncertain_or_ambient
        self.assertFalse(res["is_anomaly"], f"Głos ludzki nie może być fałszywą awarią! Wynik: {res['decision']}")


if __name__ == "__main__":
    unittest.main()
