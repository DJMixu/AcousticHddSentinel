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
   ./scripts/start_services.sh
   ```
   (Skrypt podnosi `ollama.service` oraz bezhasłowy kontener `podman start open-webui`).

2. **Automatyczne delegowanie złożonych zadań (Pipeline Tandemu i Profile Presets):**
   Gdy użytkownik zleca projektowanie nowych modułów, implementację zaawansowanych algorytmów (np. eksport do ONNX/Android, rozszerzone testy jednostkowe, refaktoring), agent powinien **automatycznie** wykorzystywać tandem modeli za pomocą skryptu orkiestrującego:
   ```bash
   python scripts/local_llm_bridge.py --task "<precyzyjny opis zadania>" [--preset <nazwa_profilu>] [--context <plik_kontekstu>] [--output <plik_wyjściowy>]
   ```
   **Dostępne profile zadań (`--preset`):**
   - `--preset code-review`: Niezależny Code Review, bug hunting, wyścigi wątkowe (Qwen 2.5 Coder)
   - `--preset dsp-translate`: Translacja DSP Python/NumPy $\to$ 100% czysta Java 17 bez Kotlina (Qwen 2.5 Coder)
   - `--preset adversarial-gen`: Matematyczna synteza fal adwersarialnych (vibrato mowy, szum wentylatorów BPF, clipping MEMS) (Qwen 3.8 27B)
   - `--preset android-opt`: Niskopoziomowa optymalizacja Android Zero-Allocation i minimalnej latencji (Qwen 2.5 Coder)
   - `--preset smpc-arch`: Architektura wielomodalnych prototypów podcentrów i strat metric learning (Qwen 3.8 27B)
   - `--preset num-audit`: Audyt numeryczny zmiennoprzecinkowy IEEE 754 (NaN, log(0), błędy zaokrągleń) (Qwen 2.5 Coder)
   - `--preset smart-telemetry`: Analiza i korelacja SMART z sygnaturami akustycznymi w oknie 128k (Qwen 3.8 27B)

3. **Sprawdzanie gotowości środowiska przed delegacją:**
   ```bash
   python scripts/local_llm_bridge.py --check-ready   # Zwraca JSON ze stanem Ollama i VRAM
   python scripts/local_llm_bridge.py --free-vram      # Wyładowuje wszystkie modele z VRAM
   ```

4. **Rygorystyczny niezmiennik pamięci VRAM (16 GB) i architektura sprzętowa:**
   - **Konfiguracja GPU:** System operacyjny, monitor i środowisko graficzne działają na zintegrowanym układzie procesora (**iGPU Ryzen 7 7700**).
   - Karta **AMD Radeon RX 9060 XT (16 GB VRAM)** jest w 100% wolna od narzutu systemowego (pełne ~15.9 GB dostępne na obliczenia AI/ROCm).
   - **Maksymalny kontekst:** Dzięki FlashAttention i kwantyzacji Q4_0 KV Cache w Ollama, model Qwen 3.8 27B (12 GB) może bezpiecznie przyjmować kontekst **aż do 128k (`131072` tokenów)** w całości mieszcząc się w pamięci VRAM GPU!
   - Modele 27B (12 GB) oraz 14B (9 GB) **nigdy nie mogą rezydować w VRAM równocześnie**.
   - Skrypt `local_llm_bridge.py` zwalnia pamięć (`keep_alive=0`) natychmiast po wygenerowaniu odpowiedzi.
   - Przed uruchomieniem dowolnego treningu PyTorch (`train_cnn.py`, `train_ensemble.py`) agent **bezwzględnie sprawdza** `ollama ps` i w razie potrzeby wyładowuje modele (`python scripts/local_llm_bridge.py --free-vram`), aby trening dysponował pełną pamięcią 16 GB VRAM.

5. **Oficjalny Tryb Wieloagentowy: Natywny `swarm` (Zastępuje generyczny `/teamwork`):**
   Gdy użytkownik zleca pracę zespołową, prosi o swarm lub teamwork:
   - Domyślnym i zalecanym mechanizmem jest **narzędzie `swarm`** w połączeniu ze skillem `local-llm-teamwork`.
   - **Zakaz pisania kodu z głowy:** Koordynator i chmurowe workery AGY **mają bezwzględny zakaz pisania kodu produkcyjnego z głowy**. W briefach `swarm{action:"add", ...}` nakazują wykonanie implementacji przez `scripts/local_llm_bridge.py` na GPU.
   - **Oszczędność tokenów:** Użycie lokalnego swarma oszczędza ~90% tokenów chmurowych, eliminując błędy limitów 429 i zapewniając kompletny kod bez placeholderów.
   - **Sekwencyjność GPU:** Tylko jeden worker na raz deleguje do lokalnego GPU (kolejność: architekt $\to$ coder $\to$ verifier).
   - **Fallback:** Tylko w przypadku fizycznej awarii Ollama worker wykonuje zadanie samodzielnie i powiadamia użytkownika.

6. **Zasada Wczesnego Uruchamiania (Front-Loaded Execution — ZAKAZ OPÓŹNIANIA):**
   Lokalne modele generują z fizyczną prędkością GPU (Qwen 3.8: ~19 tok/s, Qwen 2.5 Coder: ~40 tok/s), przez co są kluczowym komponentem czasowym potoku.
   - **Bezwzględna reguła kolejkowania:** Zadania dla modeli Qwen (`architect`, `coder`, `smpc-arch`, `code-review`) **MUSZĄ być zlecane w pierwszej kolejności**, natychmiast na początku iteracji/etapu, a nie odkładane po długich dyskusjach.
