"""
Test weryfikacyjny: Odrzucanie normalnych dźwięków komputera (wentylatory, szum PC)
oraz prawidłowe rozpoznawanie zdrowych dysków (różne RPM) w systemie Acoustic HDD Sentinel.

Weryfikuje zadanie użytkownika:
1. "program powinien wykrywać że jest pik na danej częstotliwości i na podstawie tego działać"
2. "dyski zdrowe mają charakterystyczne dźwięki różne dla różnych rpm należy to sprawdzić i zaimplementować w folderze /testy"
3. "przetestować czy sieć wykrywa jako złe dyski normalne dźwięki komputera"
4. "należy uwzględnić że telefon będzie normalnie nagrywał dźwięk o głośności 45db (około) bo ogólnie dyski nie są głośne"
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import librosa
import scipy.signal as signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from ml.hybrid_acoustic_sentinel import HybridAcousticSentinel
from ml.hdd_audio_encoder import TARGET_SR

class TestComputerSoundRejectionAndRpm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        weights_path = str(PROJECT_ROOT / "training_output" / "best_hdd_cnn_model.pt")
        cls.sentinel = HybridAcousticSentinel(model_weights_path=weights_path)
        cls.sr = TARGET_SR

    def test_real_healthy_7200_rpm_drive(self):
        """Autentyczny dysk Toshiba N300 7200 RPM musi zostać sklasyfikowany jako ZDROWY (NORMAL, 7200 RPM)."""
        p = PROJECT_ROOT / "data" / "raw" / "verified_real" / "TOSHIBA_HDWR480_7200rpm_healthy_spinup.wav"
        self.assertTrue(p.exists(), f"Plik {p} nie istnieje!")
        y, _ = librosa.load(str(p), sr=self.sr)
        
        # Test na stabilnym segmencie pracy silnika (po rozruchu i ustabilizowaniu obrotów)
        seg = y[int(10 * self.sr) : int(13 * self.sr)]
        
        res = self.sentinel.diagnose_chunk(seg)
        print(f"\n[TEST 1] Toshiba 7200 RPM: Decyzja={res['decision'].upper()}, RPM={res['rpm_detected']}, Anomalia={res['is_anomaly']}")
        self.assertEqual(res["decision"], "normal")
        self.assertFalse(res["is_anomaly"])
        self.assertEqual(res["rpm_detected"], "7200_rpm")

    def test_real_pc_chassis_fan(self):
        """Prawdziwy wentylator PC NIE MOŻE być sklasyfikowany jako awaria HDD (scratching/beeping/clicking)!"""
        p = PROJECT_ROOT / "data" / "raw" / "computer_ambient" / "Sounds_made_by_a_fan.wav"
        self.assertTrue(p.exists(), f"Plik {p} nie istnieje!")
        y, _ = librosa.load(str(p), sr=self.sr)
        
        mid = len(y) // 2
        seg = y[mid : mid + self.sr * 3]
        
        res = self.sentinel.diagnose_chunk(seg)
        print(f"[TEST 2] Wentylator PC: Decyzja={res['decision'].upper()}, Anomalia={res['is_anomaly']}, Mechanizm={res['mechanism']}")
        self.assertFalse(res["is_anomaly"], f"Wentylator komputera został błędnie oznaczony jako awaria: {res}")
        self.assertIn(res["decision"], ["ambient_pc", "normal"])

    def test_simulated_gpu_and_cpu_cooling_fans(self):
        """Symulowany szum wentylatora GPU/CPU (aerodynamiczny szum przepływu powietrza) nie może być awarią."""
        num_samples = int(3.0 * self.sr)
        t = np.linspace(0, 3.0, num_samples, endpoint=False)
        white = np.random.randn(num_samples)
        
        # Filtr pasmowo-przepustowy dla wentylatora karty graficznej (turbina 2000 RPM)
        sos_gpu = signal.butter(4, [300 / (self.sr/2), 3500 / (self.sr/2)], btype="bandpass", output="sos")
        gpu_air_rush = signal.sosfilt(sos_gpu, white) * 0.12
        blade_tone = 0.04 * np.sin(2 * np.pi * 230.0 * t) # BPF 230 Hz
        gpu_sound = (gpu_air_rush + blade_tone).astype(np.float32)

        res = self.sentinel.diagnose_chunk(gpu_sound)
        print(f"[TEST 3] Chłodzenie GPU: Decyzja={res['decision'].upper()}, Anomalia={res['is_anomaly']}")
        self.assertFalse(res["is_anomaly"], "Chłodzenie GPU nie może wywołać alarmu awarii HDD!")

    def test_realistic_45db_sound_level_normal_hdd(self):
        """Weryfikacja przy cichym dźwięku ~45 dB SPL (telefon nagrywający cichy dysk HDD w odległości 10 cm)."""
        num_samples = int(3.0 * self.sr)
        t = np.linspace(0, 3.0, num_samples, endpoint=False)
        f0 = 90.0 # 5400 RPM
        
        # Poziom głośności odpowiadający 45 dB SPL (RMS ~0.008)
        y_45db = 0.015 * np.sin(2 * np.pi * f0 * t)
        y_45db += 0.009 * np.sin(2 * np.pi * 2 * f0 * t)
        y_45db += 0.005 * np.sin(2 * np.pi * 3 * f0 * t)
        y_45db += 0.001 * np.random.randn(num_samples) # Szum tła pokoju
        y_45db = y_45db.astype(np.float32)

        res = self.sentinel.diagnose_chunk(y_45db)
        print(f"[TEST 4] Dysk 5400 RPM przy 45 dB SPL: Decyzja={res['decision'].upper()}, RPM={res['rpm_detected']}, RMS={res['rms']:.5f}")
        self.assertEqual(res["decision"], "normal")
        self.assertEqual(res["rpm_detected"], "5400_rpm")
        self.assertFalse(res["is_anomaly"])

    def test_real_faulty_hdd_clicking_detected(self):
        """Prawdziwa awaria głowic (Aesonlabs Clicking) musi być niezmiennie wykryta jako CLICKING."""
        p = PROJECT_ROOT / "data" / "raw" / "aesonlabs" / "Aesonlabs Data Recovery_Clicking Hard Drive.wav"
        self.assertTrue(p.exists(), f"Plik {p} nie istnieje!")
        y, _ = librosa.load(str(p), sr=self.sr)
        
        # Test pierwszych 3 sekund
        seg = y[: self.sr * 3]
        res = self.sentinel.diagnose_chunk(seg)
        print(f"[TEST 5] Autentyczny Clicking HDD: Decyzja={res['decision'].upper()}, Anomalia={res['is_anomaly']}")
        self.assertTrue(res["is_anomaly"])
        self.assertEqual(res["decision"], "clicking")

    def test_seized_spindle_motor_stall_beep_detected(self):
        """Autentyczne zatarcie silnika (Seized Spindle Beeping) musi być wykryte jako BEEPING."""
        p = PROJECT_ROOT / "data" / "raw" / "additional" / "Seized Spindle.mp3"
        self.assertTrue(p.exists(), f"Plik {p} nie istnieje!")
        y, _ = librosa.load(str(p), sr=self.sr)
        
        seg = y[self.sr : self.sr + self.sr * 3]
        res = self.sentinel.diagnose_chunk(seg)
        print(f"[TEST 6] Zatarty silnik (Seized Spindle): Decyzja={res['decision'].upper()}, Anomalia={res['is_anomaly']}")
        self.assertTrue(res["is_anomaly"])
        self.assertEqual(res["decision"], "beeping")


if __name__ == "__main__":
    unittest.main()
