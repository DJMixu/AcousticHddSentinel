#!/usr/bin/env python3
"""
Test Walidacyjny Iteracji 3: DSP Adversarial Guard (Obrona przed przesterowaniem mikrofonu i THD).
Sprawdza:
1. Neutralizację fałszywych alarmów scratching wywołanych przez Hard Clipping i dystorsje THD.
2. 0% False Positives (brak błędnej neutralizacji) na autentycznych uszkodzeniach HDD.
3. Pełny brak regresji na 40 losowych próbkach ze zbioru testowego.
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
    print("🧪 TESTY WALIDACYJNE ITERACJI 3: OBRONA PRZED PRZESTEROWANIEM MIKROFONU I THD")
    print("="*75)
    
    clipping_samples = [
        ("tests/adversarial_lab/samples/sine_1000hz_clip_060.wav", "Sine 1000Hz Hard Clip 0.60"),
        ("tests/adversarial_lab/samples/sine_3000hz_clip_060.wav", "Sine 3000Hz Hard Clip 0.60"),
        ("tests/adversarial_lab/samples/ambient_ventilador_clip_060.wav", "Ambient Ventilador Clip 0.60"),
        ("tests/adversarial_lab/samples/pc_fan_turbulent_clip_060.wav", "PC Fan Turbulent Clip 0.60")
    ]
    
    clip_success = 0
    for p, desc in clipping_samples:
        if os.path.exists(p):
            c, is_san = evaluate_wav(session, guard, p, desc)
            if c == "normal":
                clip_success += 1
                
    print(f"\nSkuteczność neutralizacji przesterowań i THD do 'normal': {clip_success}/{len(clipping_samples)}")
    assert clip_success == len(clipping_samples), "Wszystkie przesterowane sygnały testowe powinny zostać zneutralizowane do normal!"
    
    print("\n" + "-"*75)
    print("🔍 TEST REGRESJI: Weryfikacja autentycznych uszkodzeń HDD poddanych przesterowaniu...")
    print("-" * 75)
    
    hdd_samples = [
        ("tests/adversarial_lab/samples/hitachi_deathstar_click_clip_060.wav", "Hitachi Deskstar Click (Clip 0.60)"),
        ("tests/adversarial_lab/samples/hitachi_deathstar_click_combined_smart_mic.wav", "Hitachi Deskstar Click (Smart Mic AGC/THD)"),
        ("tests/adversarial_lab/samples/seagate_stuck_beep_clip_060.wav", "Seagate Stuck Beep (Clip 0.60)"),
        ("tests/adversarial_lab/samples/hdd_hitachi_deathstar_yt.wav", "Hitachi Deskstar Clean"),
        ("tests/adversarial_lab/samples/hdd_seagate_stuck_yt.wav", "Seagate Spindle Stuck Clean")
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
                print(f"  ⚠️ REGRESJA na {os.path.basename(f)}: {raw_pred} -> {clean_pred}")
                
    print(f"\nWynik testu regresji na zbiorze testowym: {regressions} regresji na {len(sample_files)} próbek.")
    assert regressions == 0, f"Dopuszczalne jest 0 regresji na autentycznych uszkodzeniach HDD! Znaleziono: {regressions}"
    
    print("\n" + "="*75)
    print("✅ WSZYSTKIE TESTY WALIDACYJNE ITERACJI 3 ZAKOŃCZONE PEŁNYM SUKCESEM!")
    print("="*75)

if __name__ == "__main__":
    main()
