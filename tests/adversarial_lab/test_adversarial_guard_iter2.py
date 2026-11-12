#!/usr/bin/env python3
"""
Test Walidacyjny Iteracji 2: DSP Adversarial Guard (Obrona przed szumem wentylatorów PC i turbin).
Sprawdza:
1. Neutralizację fałszywych alarmów dla wentylatora PC, szafy serwerowej, suszarki i odkurzacza.
2. 0% False Positives (brak błędnej klasyfikacji) na autentycznych uszkodzeniach HDD (Hitachi Deathstar, Seagate Spindle Beep oraz próbkach ze zbioru testowego).
"""

import os
import sys
import glob
sys.path.insert(0, os.path.abspath("."))
import soundfile as sf
import numpy as np
import onnxruntime as ort
from ml.hdd_audio_encoder import encode_audio_chunk
from ml.dsp_adversarial_guard import DSPAdversarialGuard, CLASSES

def evaluate_wav(session, guard, path, desc=""):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = np.mean(data, axis=-1)
    chunk = data[:66150]
    tensor = encode_audio_chunk(chunk, sr=sr)
    tensor_cf = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: tensor_cf})[0]
    exp = np.exp(out - np.max(out))
    probs = (exp / np.sum(exp))[0]
    
    raw_class = CLASSES[np.argmax(probs)]
    raw_p = probs[np.argmax(probs)]
    
    clean_p, is_san, reason = guard.protect_and_sanitize_predictions(probs, tensor, waveform=chunk)
    clean_class = CLASSES[np.argmax(clean_p)]
    clean_conf = clean_p[np.argmax(clean_p)]
    
    print(f"\n[Test: {desc}]")
    print(f"  Plik:             {os.path.basename(path)}")
    print(f"  Surowy ONNX:      {raw_class:12s} | P({raw_class}) = {raw_p*100:.1f}% (normal: {probs[CLASSES.index('normal')]*100:.1f}%)")
    print(f"  Po DSP Guard:     {clean_class:12s} | P({clean_class}) = {clean_conf*100:.1f}% (normal: {clean_p[CLASSES.index('normal')]*100:.1f}%)")
    if is_san:
        print(f"  🛡️ OBRONA:         {reason}")
    else:
        print(f"  ℹ️ Bez zmian")
        
    return clean_class, is_san

def main():
    session = ort.InferenceSession("exported_models/best_hdd_cnn_model.onnx")
    guard = DSPAdversarialGuard()
    
    print("="*75)
    print("🧪 TESTY WALIDACYJNE ITERACJI 2: OBRONA PRZED WENTYLATORAMI I SZUMEM PC")
    print("="*75)
    
    fan_samples = [
        ("tests/adversarial_lab/samples/fan_pc_yt.wav", "Real PC Fan 100% PWM"),
        ("tests/adversarial_lab/samples/datacenter_rack_yt.wav", "Server Room / Datacenter Turbines"),
        ("tests/adversarial_lab/samples/vacuum_cleaner_yt.wav", "Vacuum Cleaner Aerodynamic Noise"),
        ("tests/adversarial_lab/samples/hair_dryer_yt.wav", "Hair Dryer High Speed")
    ]
    
    fan_success = 0
    for p, desc in fan_samples:
        if os.path.exists(p):
            c, is_san = evaluate_wav(session, guard, p, desc)
            if c == "normal":
                fan_success += 1
                
    print(f"\nSkuteczność neutralizacji szumów aerodynamicznych do 'normal': {fan_success}/{len(fan_samples)}")
    assert fan_success >= 3, "Co najmniej 75% szumów wentylatorów powinno zostać zneutralizowanych do normal!"
    
    print("\n" + "-"*75)
    print("🔍 TEST REGRESJI: Weryfikacja autentycznych nagrań laboratoryjnych uszkodzeń HDD...")
    print("-" * 75)
    
    hdd_samples = [
        ("tests/adversarial_lab/samples/hdd_hitachi_deathstar_yt.wav", "Hitachi Deskstar Click of Death (yt)"),
        ("tests/adversarial_lab/samples/hdd_seagate_stuck_yt.wav", "Seagate Spindle Stuck / Beep (yt)")
    ]
    
    for p, desc in hdd_samples:
        if os.path.exists(p):
            c, is_san = evaluate_wav(session, guard, p, desc)
            assert c != "normal", f"BŁĄD: Autentyczne uszkodzenie HDD {desc} zostało błędnie oznaczone jako normal!"
            print(f"  ✅ Autentyczne uszkodzenie poprawnie sklasyfikowane jako awaria: {c.upper()}")
            
    # Sprawdzenie losowych próbek ze zbioru dataset_encoded/test/
    test_files = glob.glob("dataset_encoded/test/*/*.npy")
    np.random.seed(42)
    sample_files = np.random.choice(test_files, min(40, len(test_files)), replace=False)
    
    regressions = 0
    for f in sample_files:
        true_cls = os.path.basename(os.path.dirname(f))
        tensor = np.load(f)
        tensor_cf = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        
        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: tensor_cf})[0]
        exp = np.exp(out - np.max(out))
        probs = (exp / np.sum(exp))[0]
        
        clean_p, is_san, _ = guard.protect_and_sanitize_predictions(probs, tensor)
        raw_pred = CLASSES[np.argmax(probs)]
        clean_pred = CLASSES[np.argmax(clean_p)]
        
        if true_cls in ["beeping", "clicking", "scratching"]:
            if raw_pred == true_cls and clean_pred != true_cls:
                regressions += 1
                print(f"  ⚠️ REGRESJA na {name}: {raw_pred} -> {clean_pred}")
                
    print(f"\nWynik testu regresji na zbiorze testowym: {regressions} regresji na {len(sample_files)} próbek.")
    assert regressions == 0, f"Dopuszczalne jest 0 regresji na autentycznych uszkodzeniach HDD! Znaleziono: {regressions}"
    
    print("\n" + "="*75)
    print("✅ WSZYSTKIE TESTY WALIDACYJNE ITERACJI 2 ZAKOŃCZONE PEŁNYM SUKCESEM!")
    print("="*75)

if __name__ == "__main__":
    main()
