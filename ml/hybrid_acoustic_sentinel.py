"""
Hybrydowy system diagnostyki akustycznej Acoustic HDD Sentinel.
Łączy:
1. Fizyczny analizator pików częstotliwościowych i grzebienia harmonicznego RPM (ml/rpm_harmonics_analyzer.py)
2. Filtr odrzucania dźwięków otoczenia komputera (wentylatory, mowa, tło PC ~45 dB SPL)
3. Głęboki model 2D CNN dla precyzyjnej klasyfikacji specyficznych uszkodzeń mechanicznych

Zgodność: ADR-001, fizyka akustyczna HDD, wymagania testowe w tests/.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import librosa

from ml.rpm_harmonics_analyzer import analyze_frequency_peaks, TARGET_SR
from ml.hdd_audio_encoder import encode_audio_chunk
from ml.model_cnn import AcousticHddSentinelCNN

class HybridAcousticSentinel:
    def __init__(self, model_weights_path: Optional[str] = None, num_classes: int = 4):
        self.sr = TARGET_SR
        self.cnn_model = AcousticHddSentinelCNN(num_classes=num_classes)
        self.classes = ["beeping", "clicking", "normal", "scratching"]
        
        if model_weights_path is not None:
            cp = torch.load(model_weights_path, map_location="cpu")
            if "state_dict" in cp:
                self.cnn_model.load_state_dict(cp["state_dict"])
            else:
                self.cnn_model.load_state_dict(cp)
            if "classes" in cp:
                self.classes = cp["classes"]
            if "fault_prototypes" in cp.get("state_dict", cp):
                self.cnn_model.use_open_set = True
        self.cnn_model.eval()

    def diagnose_chunk(self, y_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Diagnozuje pojedynczy 3-sekundowy wycinek audio:
        1. Etap DSP: Analiza pików częstotliwościowych (90 Hz / 120 Hz) i harmonicznych RPM
        2. Etap OOD / Fan Rejection: Weryfikacja, czy sygnał to nie jest po prostu szum wentylatora lub głos
        3. Etap CNN: Wnioskowanie siecią neuronową
        4. Fuzja decyzyjna (Hybrid Decision Fusion)
        """
        rms = float(np.sqrt(np.mean(y_chunk ** 2)))
        if rms < 0.002:
            return {
                "decision": "SILENCE",
                "is_anomaly": False,
                "confidence": 1.0,
                "details": "Poziom dźwięku poniżej progu szumu (RMS < 0.002).",
                "rms": rms,
                "rpm_detected": None
            }

        # 1. Analiza częstotliwościowa
        peak_analysis = analyze_frequency_peaks(y_chunk, sr=self.sr)
        
        # 2. Inferencja CNN
        tensor = encode_audio_chunk(y_chunk, sr=self.sr, tensor_format="pytorch")
        t_input = torch.from_numpy(tensor).unsqueeze(0)
        with torch.no_grad():
            cnn_probs = self.cnn_model(t_input)[0].numpy()
            
        cnn_top_idx = int(np.argmax(cnn_probs))
        cnn_top_class = self.classes[cnn_top_idx]
        cnn_top_prob = float(cnn_probs[cnn_top_idx])

        # 3. Analiza transjentów (Crest Factor) dla stukania głowic (Clicking)
        peak_val = float(np.max(np.abs(y_chunk)))
        crest_factor = peak_val / (rms + 1e-9)
        # Klikanie głowicy HDD ma wysoki Crest Factor (> 4.5) oraz ostre impulsy w Delta
        has_mechanical_clicks = (crest_factor > 4.5 and cnn_probs[1] > 0.40)

        # 4. Fuzja decyzyjna:
        # A. Jeśli wykryto stabilną rotację silnika 5400 RPM lub 7200 RPM,
        #    a brak jest silnych impulsów klikania ani zatarcia silnika:
        #    -> DYSK JEST ZDROWY (NORMAL)!
        if peak_analysis["has_rotation"] and not has_mechanical_clicks and not peak_analysis["has_stall_beep"]:
            rpm_str = peak_analysis["detected_rpm"].replace("_", " ").upper()
            return {
                "decision": "normal",
                "is_anomaly": False,
                "confidence": max(peak_analysis["rpm_confidence"], 0.85),
                "rpm_detected": peak_analysis["detected_rpm"],
                "rpm_nominal": peak_analysis["rpm_nominal"],
                "mechanism": f"Wykryto prawidłową, stabilną rotację silnika wrzeciona ({rpm_str}, f0={peak_analysis['rpm_details']['f0_found']:.1f} Hz).",
                "cnn_probabilities": {self.classes[i]: float(cnn_probs[i]) for i in range(len(self.classes))},
                "crest_factor": round(crest_factor, 2),
                "rms": round(rms, 5)
            }

        # B. Jeśli występują silne mechaniczne uderzenia głowicy (wysoki Crest Factor):
        #    -> AWARIA POZYCJONERA GŁOWIC (CLICKING)!
        if has_mechanical_clicks:
            return {
                "decision": "clicking",
                "is_anomaly": True,
                "confidence": float(cnn_probs[1]),
                "rpm_detected": peak_analysis["detected_rpm"],
                "mechanism": f"Wykryto periodyczne uderzenia pozycjonera głowic w odbojnik (Crest Factor={crest_factor:.1f}).",
                "cnn_probabilities": {self.classes[i]: float(cnn_probs[i]) for i in range(len(self.classes))},
                "crest_factor": round(crest_factor, 2),
                "rms": round(rms, 5)
            }

        # C. Jeśli wykryto ostry pisk stojana (Stall Beep) przy braku rotacji i braku uderzeń głowic,
        #    a model CNN nie wskazuje z wysoką pewnością (>= 80%) na dźwięk normalny/wentylatora:
        #    -> ZATARCIE SILNIKA (BEEPING / SEIZED SPINDLE)!
        if peak_analysis["has_stall_beep"] and not peak_analysis["has_rotation"] and cnn_probs[2] < 0.80:
            return {
                "decision": "beeping",
                "is_anomaly": True,
                "confidence": 0.95,
                "rpm_detected": None,
                "mechanism": f"Wykryto ostry pisk stojana zatartego silnika wrzeciona (f={peak_analysis['stall_beep_freq_hz']} Hz). Brak rotacji talerzy!",
                "cnn_probabilities": {self.classes[i]: float(cnn_probs[i]) for i in range(len(self.classes))},
                "crest_factor": round(crest_factor, 2),
                "rms": round(rms, 5)
            }

        # D. Weryfikacja dźwięków otoczenia komputera (Wentylatory PC, Szum tła ~45 dB SPL):
        # Jeśli brak rotacji HDD, brak Stall Beep, brak klikania, a spektrum to szerokopasmowy szum:
        # -> Odrzucenie jako AMBIENT / PC_FAN / OOD!
        if not peak_analysis["has_rotation"] and not has_mechanical_clicks and not peak_analysis["has_stall_beep"]:
            # Sprawdź, czy CNN nie zgaduje z niską pewnością (< 0.70)
            if cnn_top_prob < 0.70 or cnn_top_class in ["scratching", "beeping"]:
                return {
                    "decision": "ambient_pc",
                    "is_anomaly": False,
                    "confidence": 0.90,
                    "rpm_detected": None,
                    "mechanism": "Dźwięk tła komputera / wentylator chłodzenia. Brak charakterystycznych sygnatur fizycznych dysku HDD.",
                    "cnn_probabilities": {self.classes[i]: float(cnn_probs[i]) for i in range(len(self.classes))},
                    "crest_factor": round(crest_factor, 2),
                    "rms": round(rms, 5)
                }

        # Domyślna decyzja z CNN
        return {
            "decision": cnn_top_class,
            "is_anomaly": (cnn_top_class != "normal"),
            "confidence": cnn_top_prob,
            "rpm_detected": peak_analysis["detected_rpm"],
            "mechanism": f"Klasyfikacja sieci neuronowej 2D CNN ({cnn_top_class}).",
            "cnn_probabilities": {self.classes[i]: float(cnn_probs[i]) for i in range(len(self.classes))},
            "crest_factor": round(crest_factor, 2),
            "rms": round(rms, 5)
        }
