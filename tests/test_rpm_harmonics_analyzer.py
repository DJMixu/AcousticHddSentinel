"""
Testy logiczne i numeryczne analizatora pików częstotliwościowych oraz harmonicznych RPM dla dysków HDD.
Weryfikacja implementacji z ml/rpm_harmonics_analyzer.py.

Sprawdza:
1. Prawidłowe rozpoznawanie prędkości 5400 RPM (pik 90 Hz + harmoniczne 180, 270, 360 Hz)
2. Prawidłowe rozpoznawanie prędkości 7200 RPM (pik 120 Hz + harmoniczne 240, 360, 480 Hz)
3. Odrzucanie normalnych dźwięków komputera (szum wentylatora PC 1200 RPM, szum GPU) jako rotacji HDD
4. Odrzucanie mowy ludzkiej / świstów jako rotacji HDD
5. Wykrywanie zatarcia silnika (Stall Beep w paśmie 1 - 3 kHz bez rotacji 90/120 Hz)
6. Działanie przy cichym sygnale ~45 dB SPL (telefon przy obudowie komputera / dysku)
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import scipy.signal as signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from ml.rpm_harmonics_analyzer import analyze_frequency_peaks, TARGET_SR

class TestRpmHarmonicsAnalyzer(unittest.TestCase):

    def setUp(self):
        self.sr = TARGET_SR
        self.duration = 3.0
        self.num_samples = int(self.duration * self.sr)
        self.t = np.linspace(0, self.duration, self.num_samples, endpoint=False)

    def test_5400_rpm_detection(self):
        """Test detekcji dysku 5400 RPM (90 Hz fundamentalna + harmoniczne obrotowe)."""
        f0 = 90.0
        # Sygnał wirującego dysku 5400 RPM
        y = 0.30 * np.sin(2 * np.pi * f0 * self.t)
        y += 0.20 * np.sin(2 * np.pi * 2 * f0 * self.t)
        y += 0.12 * np.sin(2 * np.pi * 3 * f0 * self.t)
        y += 0.08 * np.sin(2 * np.pi * 4 * f0 * self.t)
        y += 0.05 * np.sin(2 * np.pi * 6 * f0 * self.t)
        # Szum tła łożysk hydrodynamicznych FDB
        y += 0.02 * np.random.randn(self.num_samples)

        res = analyze_frequency_peaks(y.astype(np.float32), sr=self.sr)
        self.assertEqual(res["status"], "ANALYZED")
        self.assertEqual(res["detected_rpm"], "5400_rpm")
        self.assertEqual(res["rpm_nominal"], 5400)
        self.assertTrue(res["has_rotation"])
        self.assertGreater(res["rpm_confidence"], 0.5)

    def test_7200_rpm_detection(self):
        """Test detekcji dysku 7200 RPM (120 Hz fundamentalna + harmoniczne obrotowe)."""
        f0 = 120.0
        # Sygnał wirującego dysku 7200 RPM
        y = 0.35 * np.sin(2 * np.pi * f0 * self.t)
        y += 0.22 * np.sin(2 * np.pi * 2 * f0 * self.t)
        y += 0.15 * np.sin(2 * np.pi * 3 * f0 * self.t)
        y += 0.10 * np.sin(2 * np.pi * 4 * f0 * self.t)
        y += 0.06 * np.sin(2 * np.pi * 6 * f0 * self.t)
        y += 0.02 * np.random.randn(self.num_samples)

        res = analyze_frequency_peaks(y.astype(np.float32), sr=self.sr)
        self.assertEqual(res["status"], "ANALYZED")
        self.assertEqual(res["detected_rpm"], "7200_rpm")
        self.assertEqual(res["rpm_nominal"], 7200)
        self.assertTrue(res["has_rotation"])
        self.assertGreater(res["rpm_confidence"], 0.5)

    def test_pc_fan_noise_rejection(self):
        """Test odrzucania szumu wentylatora PC (1200 RPM, łopatki 140 Hz, szum wirowy) jako rotacji HDD."""
        # Wentylator PC nie ma harmonicznych dysku 90 Hz / 120 Hz
        white = np.random.randn(self.num_samples)
        sos_fan = signal.butter(3, [80 / (self.sr/2), 2000 / (self.sr/2)], btype="bandpass", output="sos")
        fan_noise = signal.sosfilt(sos_fan, white) * 0.15
        blade_pass = 0.08 * np.sin(2 * np.pi * 140.0 * self.t)  # BPF wentylatora 140 Hz (nie 90 ani 120!)
        pc_fan = (fan_noise + blade_pass).astype(np.float32)

        res = analyze_frequency_peaks(pc_fan, sr=self.sr)
        self.assertEqual(res["status"], "ANALYZED")
        self.assertFalse(res["has_rotation"], "Wentylator PC nie może być rozpoznany jako wirujący dysk HDD!")
        self.assertIsNone(res["detected_rpm"])

    def test_human_voice_rejection(self):
        """Test odrzucania głosu ludzkiego / mowy jako rotacji HDD."""
        # Głos ludzki ze zmienną wysokością tonu (wibrato 130 Hz z modulacją 5 Hz)
        pitch_mod = 130.0 + 15.0 * np.sin(2 * np.pi * 5.0 * self.t)
        phase = 2 * np.pi * np.cumsum(pitch_mod) / self.sr
        vocal = 0.3 * np.sin(phase) + 0.2 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)

        res = analyze_frequency_peaks(vocal.astype(np.float32), sr=self.sr)
        self.assertEqual(res["status"], "ANALYZED")
        self.assertFalse(res["has_rotation"], "Głos ludzki z modulacją nie może być rozpoznany jako silnik HDD!")
        self.assertIsNone(res["detected_rpm"])

    def test_seized_spindle_stall_beep_detection(self):
        """Test wykrywania zatartego silnika wrzeciona (brak rotacji 90/120 Hz, ale ostry pisk 1.5 kHz)."""
        # Pisk stojana 1500 Hz (PWM beeper stojana)
        stall_beep = 0.25 * np.sin(2 * np.pi * 1500.0 * self.t)
        stall_beep += 0.08 * np.sin(2 * np.pi * 3000.0 * self.t)
        stall_beep += 0.01 * np.random.randn(self.num_samples)

        res = analyze_frequency_peaks(stall_beep.astype(np.float32), sr=self.sr)
        self.assertEqual(res["status"], "ANALYZED")
        self.assertFalse(res["has_rotation"], "Zatarty silnik nie ma rotacji!")
        self.assertTrue(res["has_stall_beep"], "Zatarty silnik musi wykazać ostry pisk stojana (Stall Beep)!")
        self.assertAlmostEqual(res["stall_beep_freq_hz"], 1500.0, delta=10.0)

    def test_quiet_45db_signal_preservation(self):
        """Test poprawności analizy przy cichym sygnale o poziomie ~45 dB SPL (RMS ~0.005)."""
        f0 = 7200 / 60.0 # 120 Hz
        # Skalowanie amplitudy do cichego poziomu 0.015 (odpowiada telefonowi w odległości 10 cm od dysku)
        y = 0.015 * np.sin(2 * np.pi * f0 * self.t)
        y += 0.008 * np.sin(2 * np.pi * 2 * f0 * self.t)
        y += 0.005 * np.sin(2 * np.pi * 3 * f0 * self.t)
        y += 0.001 * np.random.randn(self.num_samples)

        res = analyze_frequency_peaks(y.astype(np.float32), sr=self.sr)
        self.assertEqual(res["status"], "ANALYZED")
        self.assertEqual(res["detected_rpm"], "7200_rpm")
        self.assertTrue(res["has_rotation"])


if __name__ == "__main__":
    unittest.main()
