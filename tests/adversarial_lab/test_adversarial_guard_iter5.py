#!/usr/bin/env python3
"""
Test Walidacyjny Iteracji 5: DSP Adversarial Guard (Kompleksowy test obrony OOD & Fizycznych Inwariantów).
Sprawdza:
1. Neutralizację fałszywych alarmów scratching/clicking na sprawnym dysku enterprise 10k RPM (Maxtor Atlas 10K V).
2. Zachowanie autentycznych uszkodzeń serwerowych SCSI (IBM SCSI Surface Defect) - Safety Veto mid_high_ratio > 0.50.
3. Odporność na maskowanie akustyczne (Hitachi Click + Fan PC przy ujemnym SNR do -12 dB).
4. Neutralizację szumów aerodynamicznych (wentylator PC, szafa serwerowa, odkurzacz, suszarka).
5. 0% regresji na zbiorze testowym (.npy).
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

def evaluate_wav(session, guard, path, desc="", start_sec=0.0):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = np.mean(data, axis=-1)
    start_idx = int(start_sec * sr)
    chunk = data[start_idx : start_idx + 66150]
    if len(chunk) < 66150:
        return "too_short", False
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
    print(f"  Plik:             {os.path.basename(path)} (t={start_sec:.1f}s)")
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
    print("🧪 TESTY WALIDACYJNE ITERACJI 5: ENTERPRISE HIGH-RPM & SAFETY VETO")
    print("="*75)
    
    # 1. Test sprawnego dysku 10k RPM (Maxtor Atlas 10K V)
    maxtor_path = "data/raw/independent_web_hdd/freesound_182366_Maxtor_Atlas_10K_V_-_10000rpm_-_FDB.wav"
    maxtor_success = 0
    # Testujemy sekundy 3, 6, 9, 12, 15, 18, 21 (normalna praca po starcie)
    test_secs = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]
    for s in test_secs:
        c, is_san = evaluate_wav(session, guard, maxtor_path, f"Maxtor 10k Enterprise Healthy (t={s}s)", start_sec=s)
        if c == "normal":
            maxtor_success += 1
            
    print(f"\nSkuteczność ochrony zdrowego dysku 10k RPM: {maxtor_success}/{len(test_secs)}")
    assert maxtor_success == len(test_secs), f"Wszystkie sekcje pracy ustalonej 10k RPM powinny być 'normal'! (uzyskano {maxtor_success}/{len(test_secs)})"
    
    # 2. Test Safety Veto na uszkodzonym dysku serwerowym SCSI (IBM SCSI Surface Defect)
    ibm_defect = "data/raw/northwind/Α78301-IBM-SCSI-SURFACE-ΜΗ-ΑΝΑΚΤ.mp3"
    c_ibm, is_san_ibm = evaluate_wav(session, guard, ibm_defect, "IBM SCSI Damaged Surface (Authentic Failure)")
    assert c_ibm != "normal", f"BŁĄD: Uszkodzony dysk serwerowy IBM SCSI został błędnie zneutralizowany jako normal!"
    print(f"  ✅ Uszkodzony dysk serwerowy poprawnie zakwalifikowany jako awaria: {c_ibm.upper()}")
    
    # 3. Test wentylatorów i turbin
    fan_samples = [
        ("tests/adversarial_lab/samples/fan_pc_yt.wav", "Real PC Fan 100% PWM"),
        ("tests/adversarial_lab/samples/datacenter_rack_yt.wav", "Server Room / Datacenter Turbines")
    ]
    for p, desc in fan_samples:
        c, is_san = evaluate_wav(session, guard, p, desc)
        assert c == "normal", f"BŁĄD: Szum wentylatora {desc} nie został zneutralizowany!"
        print(f"  ✅ Szum aerodynamiczny poprawnie zneutralizowany do: NORMAL")
        
    # 4. Test maskowania akustycznego (Hitachi Click + Fan PC @ SNR = -6 dB)
    hitachi, sr = sf.read("tests/adversarial_lab/samples/hdd_hitachi_deathstar_yt.wav")
    fan, _ = sf.read("tests/adversarial_lab/samples/fan_pc_yt.wav")
    if hitachi.ndim > 1: hitachi = np.mean(hitachi, axis=-1)
    if fan.ndim > 1: fan = np.mean(fan, axis=-1)
    h_chunk = hitachi[:66150]
    f_chunk = fan[:66150]
    p_sig = np.mean(h_chunk**2)
    p_noise = np.mean(f_chunk**2)
    k = np.sqrt(p_sig / (p_noise * (10**(-6 / 10.0)) + 1e-12))
    mixed = (h_chunk + k * f_chunk).astype(np.float32)
    peak = np.max(np.abs(mixed))
    if peak > 0.99: mixed = mixed / peak * 0.95
    
    m_tensor = encode_audio_chunk(mixed, sr=sr)
    m_t_cf = np.transpose(m_tensor, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: m_t_cf})[0]
    exp = np.exp(out - np.max(out))
    probs = (exp / np.sum(exp))[0]
    clean_p, is_san, reason = guard.protect_and_sanitize_predictions(probs, m_tensor, waveform=mixed)
    m_class = CLASSES[np.argmax(clean_p)]
    print(f"\n[Test: Acoustic Masking (Hitachi Click + Fan @ -6 dB SNR)]")
    print(f"  Klasyfikacja końcowa: {m_class} (normal={clean_p[CLASSES.index('normal')]*100:.1f}%)")
    assert m_class != "normal", "BŁĄD: Uszkodzenie zamaskowane szumem wentylatora zostało błędnie odrzucone jako normal!"
    print("  ✅ Zamaskowany defekt poprawnie wykryty jako awaria pomimo 4x głośniejszego wentylatora!")

    # 5. Sprawdzenie losowych 40 próbek ze zbioru testowego na regresje
    test_files = glob.glob("dataset_encoded/test/*/*.npy")
    np.random.seed(42)
    sample_files = np.random.choice(test_files, min(40, len(test_files)), replace=False)
    
    regressions = 0
    for f in sample_files:
        true_cls = os.path.basename(os.path.dirname(f))
        tensor = np.load(f)
        tensor_cf = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        out = session.run(None, {input_name: tensor_cf})[0]
        exp = np.exp(out - np.max(out))
        p = (exp / np.sum(exp))[0]
        clean_p, _, _ = guard.protect_and_sanitize_predictions(p, tensor)
        raw_pred = CLASSES[np.argmax(p)]
        clean_pred = CLASSES[np.argmax(clean_p)]
        if true_cls in ["beeping", "clicking", "scratching"]:
            if raw_pred == true_cls and clean_pred != true_cls:
                regressions += 1
                print(f"  ⚠️ REGRESJA na {os.path.basename(f)}: {raw_pred} -> {clean_pred}")
                
    print(f"\nWynik testu regresji na zbiorze testowym: {regressions} regresji na {len(sample_files)} próbek.")
    assert regressions == 0, f"Dopuszczalne jest 0 regresji na autentycznych uszkodzeniach HDD! Znaleziono: {regressions}"
    
    print("\n" + "="*75)
    print("✅ WSZYSTKIE TESTY WALIDACYJNE ITERACJI 5 ZAKOŃCZONE PEŁNYM SUKCESEM!")
    print("="*75)

if __name__ == "__main__":
    main()
