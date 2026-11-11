#!/usr/bin/env bash
# start_services.sh — Uruchamianie lokalnego środowiska LLM (Ollama + Open WebUI)
set -e

echo "=== Sprawdzanie i uruchamianie usług LLM ==="

# 1. Ollama Service
if systemctl is-active --quiet ollama.service; then
    echo "[OK] Usługa ollama.service jest już aktywna."
else
    echo "[...] Uruchamianie ollama.service..."
    if sudo -n true 2>/dev/null; then
        sudo systemctl start ollama.service
    else
        # Próba przez systemctl (lub sudo jeśli skonfigurowano NOPASSWD)
        sudo systemctl start ollama.service 2>/dev/null || systemctl start ollama.service 2>/dev/null || {
            echo "[BŁĄD] Wymagane uprawnienia sudo bez hasła dla: systemctl start ollama.service"
            exit 1
        }
    fi
    echo "[OK] ollama.service uruchomiona pomyślnie."
fi

# 2. Open WebUI (Podman rootless)
echo "[...] Uruchamianie kontenera open-webui..."
podman start open-webui >/dev/null 2>&1 || true
if podman ps --format "{{.Names}}" | grep -q "^open-webui$"; then
    echo "[OK] Kontener open-webui jest aktywny na porcie http://localhost:8080"
else
    echo "[BŁĄD] Nie udało się uruchomić kontenera open-webui."
    exit 1
fi

echo "=== Środowisko LLM gotowe do pracy ==="
