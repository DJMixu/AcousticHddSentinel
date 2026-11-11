---
trigger: always_on
globs: ["**/*"]
---

# Automatyczna Kooperacja z Lokalnymi Modelami LLM (Ollama & RX 9060 XT)

W środowisku deweloperskim użytkownika działa lokalna instalacja Ollama (`http://localhost:11434` oraz Open WebUI na porcie `8080`).
Zainstalowane i w pełni zintegrowane są dwa wyspecjalizowane modele open-weights:
1. **Główny Architekt / Planista (128k kontekstu, 12 GB VRAM):** `hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S`
2. **Programista Wykonawczy / Coder (9 GB VRAM):** `qwen2.5-coder:14b`

### Obowiązkowe wytyczne dla agentów AGY:

1. **Uruchamianie i weryfikacja usług LLM:**
   Przed wykonaniem zapytań do modeli lokalnych, agent sprawdza dostępność usług lub uruchamia je skryptem:
   ```bash
   ./start_services.sh
   ```
   (Skrypt podnosi `ollama.service` oraz bezhasłowy kontener `podman start open-webui`).

2. **Automatyczne delegowanie złożonych zadań (Pipeline Tandemu):**
   Gdy użytkownik zleca projektowanie nowych modułów, implementację zaawansowanych algorytmów (np. eksport do ONNX/Android, rozszerzone testy jednostkowe, refaktoring), agent powinien **automatycznie** wykorzystywać tandem modeli za pomocą skryptu orkiestrującego:
   ```bash
   python local_llm_bridge.py --task "<precyzyjny opis zadania>" [--context <plik_kontekstu>] [--output <plik_wyjściowy>]
   ```
   - Etap 1: Qwen 3.8 27B w oknie 128k analizuje kontekst i tworzy specyfikację techniczną.
   - Etap 2: Qwen 2.5 Coder 14B implementuje gotowy kod na bazie specyfikacji.
   - Etap 3: Agent AGY weryfikuje kod, uruchamia testy i integruje go z repozytorium.

2. **Rygorystyczny niezmiennik pamięci VRAM (16 GB) i architektura sprzętowa:**
   - **Konfiguracja GPU:** System operacyjny, monitor i środowisko graficzne działają na zintegrowanym układzie procesora (**iGPU Ryzen 7 7700**).
   - Karta **AMD Radeon RX 9060 XT (16 GB VRAM)** jest w 100% wolna od narzutu systemowego (pełne ~15.9 GB dostępne na obliczenia AI/ROCm).
   - **Maksymalny kontekst:** Dzięki FlashAttention i kwantyzacji Q4_0 KV Cache w Ollama, model Qwen 3.8 27B (12 GB) może bezpiecznie przyjmować kontekst **aż do 128k (`131072` tokenów)** w całości mieszcząc się w pamięci VRAM GPU!
   - Modele 27B (12 GB) oraz 14B (9 GB) **nigdy nie mogą rezydować w VRAM równocześnie**.
   - Skrypt `local_llm_bridge.py` zwalnia pamięć (`keep_alive=0`) natychmiast po wygenerowaniu odpowiedzi.
   - Przed uruchomieniem dowolnego treningu PyTorch (`train_cnn.py`, `train_testy.py`) agent **bezwzględnie sprawdza** `ollama ps` i w razie potrzeby wyładowuje modele (`ollama stop <nazwa>`), aby trening dysponował pełną pamięcią 16 GB VRAM.
