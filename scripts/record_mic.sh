#!/usr/bin/env bash
# ==============================================================================
# Skrypt Rejestracji Audio z Fizycznego Mikrofonu Użytkownika (Acoustic HDD Sentinel)
# Wymusza właściwe źródło: alsa_input.pci-0000_7b_00.6.analog-stereo (ALC1220 / Płyta)
# ==============================================================================

set -e

DURATION="${1:-10}"
OUTPUT_FILE="${2:-tests/noise_experiment/recorded_mic_audio.wav}"
SAMPLE_RATE="${3:-22050}"
CHANNELS="1"

# Właściwe źródło sprzętowe w systemie użytkownika
MIC_SOURCE="alsa_input.pci-0000_7b_00.6.analog-stereo"

echo "======================================================================"
echo "   REJESTRACJA AUDIO Z FIZYCZNEGO MIKROFONU (ALC1220 ANALOG)          "
echo "======================================================================"
echo "-> Źródło dźwięku:   $MIC_SOURCE"
echo "-> Czas trwania:     $DURATION s"
echo "-> Plik wyjściowy:   $OUTPUT_FILE"
echo "-> Próbkowanie:      $SAMPLE_RATE Hz (Mono)"

mkdir -p "$(dirname "$OUTPUT_FILE")"

# Ustawienie domyślnego źródła w PipeWire / PulseAudio dla pewności
if command -v pactl &> /dev/null; then
    pactl set-default-source "$MIC_SOURCE" 2>/dev/null || true
fi

# Nagrywanie za pomocą ffmpeg
ffmpeg -y -hide_banner -loglevel error \
    -f pulse -i "$MIC_SOURCE" \
    -t "$DURATION" \
    -ar "$SAMPLE_RATE" \
    -ac "$CHANNELS" \
    "$OUTPUT_FILE"

echo "-> Nagranie zakończone pomyślnie: $OUTPUT_FILE"

# Szybka weryfikacja parametrów przez Python
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python -c "
import soundfile as sf
import numpy as np

y, sr = sf.read('$OUTPUT_FILE')
rms = float(np.sqrt(np.mean(y**2)))
peak = float(np.max(np.abs(y)))
cf = peak / max(rms, 1e-9)
clip_pct = float(np.mean(np.abs(y) >= 0.985) * 100)

print(f'-> Statystyki nagrania: RMS={rms:.4f} ({20*np.log10(max(rms, 1e-9)):.1f} dBFS) | Peak={peak:.4f} | Crest Factor={cf:.2f} | Clipping={clip_pct:.1f}%')
if rms < 0.0004:
    print('⚠️ OSTRZEŻENIE: Bardzo niski poziom RMS (< 0.0004). Sprawdź czy mikrofon nie jest wyciszony.')
else:
    print('✅ Sygnał akustyczny zarejestrowany pomyślnie!')
"
fi
echo "======================================================================"
