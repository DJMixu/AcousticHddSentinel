"""
Test odporności systemu na realistyczne warunki rejestracji smartfonem przy głośności ~45 dB SPL.
Badanie stosunku sygnału do szumu (SNR), obecności szumu wentylatorów w tle
oraz weryfikacja, że cichy dysk HDD (45 dB SPL) nie jest fałszywie klasyfikowany jako awaria.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import scipy.signal as signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from ml.rpm_harmonics_analyzer import analyze_frequency_peaks, TARGET_SR

class TestSoundLevel45dB(unittest.TestCase):

    def setUp(self):
        self.sr = TARGET_SR
        self.dur = 3.0
        self.n_samples = int(self.dur * self.sr)
        self.t = np.linspace(0, self.dur, self.n_samples, endpoint=False)

    def test_quiet_hdd_with_fan_background_snr_6db(self):
        """
        Symulacja cichego dysku 5400 RPM (~45 dB SPL) pracującego w obudowie komputera
        z wentylatorem w tle (~39 dB SPL, SNR = +6 dB).
        """
        # 1. Dysk HDD 5400 RPM (amplituda ~0.020, RMS ~0.012)
        f0 = 90.0
        hdd = 0.020 * np.sin(2 * np.pi * f0 * self.t)
        hdd += 0.012 * np.sin(2 * np.pi * 2 * f0 * self.t)
        hdd += 0.008 * np.sin(2 * np.pi * 3 * f0 * self.t)
        hdd += 0.005 * np.sin(2 * np.pi * 4 * f0 * self.t)
        
        # 2. Szum wentylatora komputera w tle (amplituda ~0.010, RMS ~0.006)
        white = np.random.randn(self.n_samples)
        sos_fan = signal.butter(3, [80 / (self.sr/2), 2000 / (self.sr/2)], btype="bandpass", output="sos")
        fan = signal.sosfilt(sos_fan, white) * 0.010

        combined = (hdd + fan).astype(np.float32)
        rms = np.sqrt(np.mean(combined ** 2))
        
        # Weryfikacja poziomu RMS (musi być powyżej bramki 0.002, ale typowy dla cichego sprzętu ~45 dB)
        self.assertGreater(rms, 0.005)
        self.assertLess(rms, 0.05)

        # Analiza DSP
        res = analyze_frequency_peaks(combined, sr=self.sr)
        self.assertEqual(res["status"], "ANALYZED")
        self.assertTrue(res["has_rotation"], "Detektor musi wyodrębnić rotację 5400 RPM nawet w obecności szumu wentylatora!")
        self.assertEqual(res["detected_rpm"], "5400_rpm")
        self.assertFalse(res["has_stall_beep"])

    def test_quiet_hdd_7200_rpm_at_45db(self):
        """Symulacja dysku 7200 RPM o niskim poziomie głośności (~45 dB SPL)."""
        f0 = 120.0
        hdd = 0.022 * np.sin(2 * np.pi * f0 * self.t)
        hdd += 0.014 * np.sin(2 * np.pi * 2 * f0 * self.t)
        hdd += 0.009 * np.sin(2 * np.pi * 3 * f0 * self.t)
        # Szum własny mikrofonu smartfona
        hdd += 0.002 * np.random.randn(self.n_samples)

        combined = hdd.astype(np.float32)
        res = analyze_frequency_peaks(combined, sr=self.sr)
        self.assertTrue(res["has_rotation"])
        self.assertEqual(res["detected_rpm"], "7200_rpm")

    def test_pure_ambient_45db_does_not_trigger_false_rotation(self):
        """Cichy szum otoczenia 45 dB SPL bez dysku NIE może wywołać fałszywej detekcji rotacji."""
        white = np.random.randn(self.n_samples)
        sos = signal.butter(2, 1000 / (self.sr/2), btype="lowpass", output="sos")
        ambient = signal.sosfilt(sos, white) * 0.015

        res = analyze_frequency_peaks(ambient.astype(np.float32), sr=self.sr)
        self.assertFalse(res["has_rotation"], "Szum tła 45 dB bez dysku nie może zgłosić obecności rotacji HDD!")


if __name__ == "__main__":
    unittest.main()
