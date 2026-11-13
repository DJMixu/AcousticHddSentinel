"""
Hybrydowy silnik decyzyjny Acoustic HDD Sentinel (DSP Harmonic Combs + 2D CNN).
Single Source of Truth: ADR-001, ADR-002, AGENTS.md

Łączy deterministyczną analizę fizyki akustycznej dysków twardych:
1. Piki częstotliwościowe i grzebienie harmoniczne prędkości obrotowej (90 Hz = 5400 RPM, 120 Hz = 7200 RPM)
2. Wykrywanie pisku stojana zatartego silnika (Stall Beep 1 - 3.5 kHz)
3. Wykrywanie transjentów uderzeniowych (Crest Factor)
4. Głęboki klasyfikator 2D CNN (Log-Mel + Delta + Delta-Delta)
5. Filtr dźwięków spoza rozkładu (OOD / Wentylatory PC / Mowa ludzka)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.rpm_harmonics_analyzer import analyze_frequency_peaks, TARGET_SR
from ml.hdd_audio_encoder import encode_audio_chunk
from ml.model_cnn import AcousticHddSentinelCNN

class HybridSentinelEngine:

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = AcousticHddSentinelCNN(num_classes=4).to(self.device)
        
        if model_path is None:
            model_path = str(PROJECT_ROOT / "training_output" / "best_hdd_cnn_model.pt")
            
        cp = torch.load(model_path, map_location=self.device)
        if "state_dict" in cp:
            self.model.load_state_dict(cp["state_dict"], strict=False)
            self.classes = cp["classes"]
        else:
            self.model.load_state_dict(cp)
            self.classes = ["beeping", "clicking", "normal", "scratching"]
            
        self.model.eval()

    def analyze_audio_chunk(self, y: np.ndarray, sr: int = TARGET_SR) -> Dict[str, Any]:
        """
        Wykonuje pełną hybrydową diagnozę 3-sekundowego wycinka audio:
        1. DSP Spectral & RPM Analysis
        2. 2D CNN Inference
        3. Fuzja regułowa i dyskryminacja OOD
        """
        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms < 0.002:
            return {
                "decision": "SILENCE",
                "is_anomaly": False,
                "confidence": 1.0,
                "detected_rpm": None,
                "recommendation": "Sygnał audio zbyt cichy. Przyłóż mikrofon do obudowy dysku HDD.",
                "details": {"rms": rms}
            }

        # 1. Analiza DSP
        dsp_res = analyze_frequency_peaks(y, sr=sr)
        
        # 2. Inferencja CNN
        tensor = encode_audio_chunk(y, sr=sr, tensor_format="pytorch")
        tensor_t = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cnn_probs = self.model(tensor_t)[0].cpu().numpy()
            
        prob_dict = {self.classes[i]: float(cnn_probs[i]) for i in range(len(self.classes))}
        cnn_top_idx = int(np.argmax(cnn_probs))
        cnn_top_class = self.classes[cnn_top_idx]
        cnn_confidence = float(cnn_probs[cnn_top_idx])

        # 3. Logika fuzji i zabezpieczeń OOD
        has_rotation = dsp_res["has_rotation"]
        has_stall_beep = dsp_res["has_stall_beep"]
        detected_rpm = dsp_res["detected_rpm"]
        rpm_nominal = dsp_res["rpm_nominal"]

        final_decision = cnn_top_class
        final_confidence = cnn_confidence
        is_anomaly = (cnn_top_class != "normal")
        status_note = ""

        # REGUŁA 1: Wykryto stabilną rotację silnika 5400/7200 RPM
        # Jeśli fizyka potwierdza stabilną rotację silnika i brak pisku zablokowania:
        if has_rotation and not has_stall_beep:
            if cnn_top_class != "clicking" and cnn_top_class != "scratching":
                final_decision = "normal"
                is_anomaly = False
                final_confidence = max(cnn_confidence, dsp_res["rpm_confidence"])
                status_note = f"Potwierdzono obroty silnika wrzeciona: {rpm_nominal} RPM (stabilna praca łożysk)."
            elif cnn_top_class == "clicking" and cnn_confidence < 0.65:
                # Słabe kliknięcie przy stabilnej rotacji może być normalnym seekiem
                final_decision = "normal"
                is_anomaly = False
                status_note = f"Dysk sprawny ({rpm_nominal} RPM). Normalne pozycjonowanie głowic (seek)."

        # REGUŁA 2: Wykryto pisk stojana zatartego silnika (Stall Beep) bez rotacji
        elif has_stall_beep and not has_rotation:
            final_decision = "beeping"
            is_anomaly = True
            final_confidence = max(cnn_confidence, 0.85)
            status_note = f"KRYTYCZNY: Wykryto pisk stojana na {dsp_res['stall_beep_freq_hz']} Hz bez rotacji silnika (Seized Spindle)!"

        # REGUŁA 3: Brak rotacji HDD i rozmyte prawdopodobieństwa (Wentylator PC / Mowa / Dźwięk ustami)
        elif not has_rotation and not has_stall_beep:
            # Sprawdź czy pewność CNN jest niska (< 70%) lub czy to szum otoczenia
            if cnn_confidence < 0.70 or (cnn_top_class == "beeping" and not has_stall_beep):
                final_decision = "uncertain_or_ambient"
                is_anomaly = False
                final_confidence = cnn_confidence
                status_note = "Dźwięk nierozpoznany lub brak fizycznych cech dysku HDD (brak rotacji 90/120 Hz). Upewnij się, że badasz pracujący dysk."

        return {
            "decision": final_decision,
            "is_anomaly": is_anomaly,
            "confidence": round(final_confidence, 3),
            "detected_rpm": detected_rpm,
            "nominal_rpm": rpm_nominal,
            "status_note": status_note,
            "probabilities": prob_dict,
            "dsp_analysis": dsp_res
        }
