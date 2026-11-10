#!/usr/bin/env bash
# stop_services.sh — Zatrzymywanie usług LLM (zwalnianie zasobów i RAM/VRAM)
echo "=== Zatrzymywanie usług LLM ==="

# 1. Zatrzymanie modeli w Ollama
ollama stop --all 2>/dev/null || true

# 2. Zatrzymanie kontenera Open WebUI
echo "[...] Zatrzymywanie kontenera open-webui..."
podman stop open-webui >/dev/null 2>&1 || true

# 3. Zatrzymanie usługi Ollama (opcjonalne)
if systemctl is-active --quiet ollama.service; then
    echo "[...] Zatrzymywanie ollama.service..."
    sudo systemctl stop ollama.service 2>/dev/null || systemctl stop ollama.service 2>/dev/null || true
fi

echo "[OK] Wszystkie usługi LLM zostały zatrzymane."
