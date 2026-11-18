#!/usr/bin/env python3
"""
local_llm_bridge.py — Dynamiczny most i orkiestracja lokalnych modeli Ollama
Dedykowany dla karty AMD Radeon RX 9060 XT (16 GB VRAM) z akceleracją ROCm.

Łączy w tandem:
1. Architekt / Planista: hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S (lub qwen3.8-architect)
   - 12 GB VRAM, do 128k kontekstu (131072 tokenów)
   - Architektura Reasoning (RCO) z dedykowanym kanałem myślenia (thinking stream)
   - Zoptymalizowana niska temperatura (0.2) i top_p (0.95) eliminujące halucynacje
   - Nielimitowane generowanie (num_predict = -1) bez przedwczesnego ucinania
2. Programista / Wykonawca: qwen2.5-coder:14b (lub qwen2.5-coder-expert)
   - 9 GB VRAM, specjalizacja kod produkcyjny
   - Deterministyczna temperatura (0.1), top_p (0.90) dla maksymalnej poprawności składni i typów

Zapewnia płynne sekwencyjne zarządzanie pamięcią VRAM (keep_alive=0s),
aby nigdy nie przekroczyć 16 GB VRAM i utrzymać 100% obliczeń na GPU.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

OLLAMA_API_URL = "http://localhost:11434/api"

# Domyślne identyfikatory modeli zgodne z regułami globalnymi i AGENTS.md
MODEL_ARCHITECT = "hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S"
MODEL_CODER = "qwen2.5-coder:14b"

# Domyślne parametry generowania
DEFAULT_ARCHITECT_PARAMS = {
    "temperature": 0.2,       # Precyzyjne rozumowanie techniczne bez halucynacji
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.05,
    "num_predict": -1         # Nielimitowana długość odpowiedzi
}

DEFAULT_CODER_PARAMS = {
    "temperature": 0.1,       # Niemal deterministyczne generowanie kodu produkcyjnego
    "top_p": 0.90,
    "top_k": 40,
    "repeat_penalty": 1.05,
    "num_predict": -1         # Nielimitowana długość kodu
}

# Predefiniowane profile zadań wyspecjalizowanych dla modeli Qwen
PRESETS = {
    "code-review": {
        "role": "coder-only",
        "description": "Rygorystyczny Code Review: analiza bugów, wyścigów wielowątkowych, wycieków pamięci i niezmienników ADR-001/ADR-002",
        "temp": 0.1,
        "ctx": 32768,
        "system_prefix": (
            "Jesteś bezkompromisowym Lead Reviewerem kodu i Audytorem Bezpieczeństwa w Pythonie oraz czystej Javie 17.\n"
            "Twoim celem jest rygorystyczne wykrywanie: wyścigów wątkowych (concurrency), błędów off-by-one, niestabilności numerycznych, "
            "wycieków pamięci oraz naruszeń ADR-001 (Zero Data Leakage, DSP) oraz ADR-002 (100% Java 17, bezwzględny zakaz Kotlina)."
        )
    },
    "dsp-translate": {
        "role": "coder-only",
        "description": "Matematyczna translacja algorytmów DSP z Pythona (NumPy/Scipy) do 100% czystej Javy 17 (płaskie float[], zero Kotlina)",
        "temp": 0.05,
        "ctx": 32768,
        "system_prefix": (
            "Jesteś Głównym Inżynierem DSP i Architektem Javy 17.\n"
            "Twoim zadaniem jest wierna, bitowo tożsama translacja algorytmów cyfrowego przetwarzania sygnałów z Pythona "
            "(NumPy/Scipy/Librosa) do 100% czystej Javy 17 (płaskie tablice float[], brak zewnętrznych bibliotek, bezwzględny brak Kotlina, "
            "FFT Radix-2, filtry Mel Slaney, wagi Savitzky-Golay, filtry IIR Butterwortha)."
        )
    },
    "adversarial-gen": {
        "role": "arch-only",
        "description": "Matematyczna synteza trudnych sygnałów akustycznych i fal adwersarialnych (vibrato głosu, szum wentylatorów BPF, clipping MEMS)",
        "temp": 0.2,
        "ctx": 32768,
        "system_prefix": (
            "Jesteś Wybitnym Akustykiem i Inżynierem Sygnałowym.\n"
            "Twoim zadaniem jest synteza precyzyjnych równań matematycznych trudnych sygnałów testowych i adwersarialnych do weryfikacji "
            "odporności detektorów akustycznych (vibrato mowy, szumy wentylatorów chłodzenia PC, przesterowania mikrofonów MEMS, zjawiska rezonansowe stojana BLDC)."
        )
    },
    "android-opt": {
        "role": "coder-only",
        "description": "Niskopoziomowa optymalizacja kodu Androida pod kątem Zero-Allocation i minimalnej latencji (czysta Java 17)",
        "temp": 0.1,
        "ctx": 32768,
        "system_prefix": (
            "Jesteś Architektem Wydajności Androida i Low-Latency Java Systems.\n"
            "Optymalizujesz kod pod kątem eliminacji alokacji pamięci na stercie w pętli audio (Zero GC pressure), "
            "wykorzystania buforów kołowych in-place, synchronizacji wątkowej i minimalnej latencji przetwarzania w 100% czystej Javie 17."
        )
    },
    "smpc-arch": {
        "role": "arch-only",
        "description": "Projektowanie wielomodalnych prototypów podcentrów (SMPC) i funkcji straty metrycznej dla klas uszkodzeń",
        "temp": 0.2,
        "ctx": 65536,
        "system_prefix": (
            "Jesteś Głównym Badaczem AI/ML specjalizującym się w Deep Metric Learning i Open-Set Recognition.\n"
            "Projektujesz geometrie przestrzeni cech, mechanizmy wielomodalnych podcentrów (Sub-Center Multi-Prototype Cosine Margin), "
            "dynamiczne marginesy kątowe i bramki OOD oparte o metryki Mahalanobisa."
        )
    },
    "num-audit": {
        "role": "coder-only",
        "description": "Audyt bezpieczeństwa numerycznego zmiennoprzecinkowego (NaN, Inf, log(0), nasycenia filtrów IIR, błędy kwantyzacji)",
        "temp": 0.05,
        "ctx": 32768,
        "system_prefix": (
            "Jesteś Audytorem Matematycznym i Specjalistą od Numeryki Zmiennoprzecinkowej IEEE 754.\n"
            "Analizujesz kod pod kątem ryzyka dzielenia przez zero, utraty precyzji w float32, niestabilności filtrów IIR wyższych rzędów, "
            "anomalii Z-Score i przepełnień numerycznych."
        )
    },
    "smart-telemetry": {
        "role": "arch-only",
        "description": "Korelacja parametrów SMART i telemetrii zużycia dysków fizycznych z sygnaturami akustycznymi",
        "temp": 0.2,
        "ctx": 131072,
        "system_prefix": (
            "Jesteś Ekspertem Diagnostyki Sprzętowej HDD i Forensyki Nośników Magnetycznych.\n"
            "Analizujesz i korelujesz dane telemetrii SMART (Raw Read Error Rate, Spin-Up Time, Seek Error Rate) "
            "z rzeczywistymi nagraniami akustycznymi dysków laboratoryjnych i fizyką łożysk FDB."
        )
    }
}


def query_ollama(
    model_name: str,
    messages: list,
    stream: bool = True,
    keep_alive: str = "0s",
    num_ctx: int = 32768,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    repeat_penalty: float = 1.05,
    num_predict: int = -1,
    show_thinking: bool = True
) -> str:
    """
    Wysyła zapytanie do API czatu Ollama z automatycznym zwalnianiem VRAM,
    obsługą strumieniowania tokenów rozumowania (thinking) oraz treści (content).
    """
    url = f"{OLLAMA_API_URL}/chat"
    
    options = {
        "num_ctx": num_ctx,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty
    }
    if num_predict != 0:
        options["num_predict"] = num_predict

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": stream,
        "keep_alive": keep_alive,
        "options": options
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    response_text = ""
    thinking_text = ""
    in_thinking_phase = False
    in_content_phase = False

    try:
        # Brak sztywnego timeoutu (zgodnie z regułą dla modeli o oknie 128k)
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                msg = chunk.get("message", {})
                
                # Obsługa tokenów rozumowania (dla modeli Reasoning, np. Qwen 3.8 RCO)
                chunk_thinking = msg.get("thinking", "")
                if chunk_thinking:
                    if not in_thinking_phase:
                        in_thinking_phase = True
                        if stream and show_thinking:
                            sys.stdout.write("\n💭 [PROCES MYŚLOWY / REASONING]:\n")
                            sys.stdout.flush()
                    thinking_text += chunk_thinking
                    if stream and show_thinking:
                        sys.stdout.write(chunk_thinking)
                        sys.stdout.flush()

                # Obsługa docelowej treści odpowiedzi
                chunk_content = msg.get("content", "")
                if chunk_content:
                    if in_thinking_phase and not in_content_phase:
                        in_content_phase = True
                        if stream:
                            sys.stdout.write("\n\n📋 [ODPOWIEDŹ]:\n")
                            sys.stdout.flush()
                    elif not in_content_phase:
                        in_content_phase = True

                    response_text += chunk_content
                    if stream:
                        sys.stdout.write(chunk_content)
                        sys.stdout.flush()

                if chunk.get("done", False):
                    break

    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        print(f"\n[BŁĄD HTTP {e.code}] {err_msg}", file=sys.stderr)
        return ""
    except urllib.error.URLError as e:
        print(f"\n[BŁĄD] Nie można połączyć się z Ollama: {e}", file=sys.stderr)
        return ""

    if stream:
        print()
    return response_text


def unload_model(model_name: str):
    """Natychmiast zwalnia model z pamięci VRAM GPU."""
    url = f"{OLLAMA_API_URL}/generate"
    payload = {"model": model_name, "keep_alive": 0}
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            pass
    except Exception:
        pass


def auto_load_project_context(context_file: str = None) -> str:
    """Wczytuje podany plik kontekstu lub automatycznie ekstraktuje niezmienniki z AGENTS.md / ADR-001."""
    if context_file and os.path.exists(context_file):
        try:
            with open(context_file, "r", encoding="utf-8") as f:
                return f"\n\n--- KONTEKST ZADANIA ({context_file}) ---\n" + f.read()
        except Exception as e:
            print(f"[OSTRZEŻENIE] Nie udało się wczytać podanego pliku kontekstu: {e}")

    # Autodetekcja reguł projektu w repozytorium
    repo_root = Path(__file__).resolve().parent.parent
    agents_path = repo_root / "AGENTS.md"

    injected = []
    if agents_path.exists():
        try:
            content = agents_path.read_text(encoding="utf-8")
            if "## 2. Niezmienniki techniczne" in content:
                part = content.split("## 2. Niezmienniki techniczne")[1].split("---")[0]
                injected.append("### NIEZMIENNIKI TECHNICZNE PROJEKTU (AGENTS.md):\n" + part.strip())
        except Exception:
            pass

    if injected:
        return "\n\n--- AUTOMATYCZNIE WSTRZYKNIĘTY KONTEKST PROJEKTU (ADR-001 / AGENTS.md) ---\n" + "\n\n".join(injected)
    return ""


def run_pipeline(
    task_description: str,
    context_file: str = None,
    output_file: str = None,
    num_ctx: int = 32768,
    temp_arch: float = 0.2,
    temp_coder: float = 0.1,
    num_predict: int = -1,
    arch_only: bool = False,
    coder_only: bool = False,
    preset: str = None
):
    """Uruchamia sekwencyjną współpracę Architekta (27B) i Programisty (Coder 14B)."""
    preset_data = None
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"Nieznany preset: {preset}. Dostępne: {list(PRESETS.keys())}")
        preset_data = PRESETS[preset]
        print(f"\n[PRESET] Aktywowano profil: '{preset}' — {preset_data['description']}")
        
        # Jeśli użytkownik nie wymusił ręcznie flag, przejmij ustawienia profilu
        if not arch_only and not coder_only:
            if preset_data["role"] == "arch-only":
                arch_only = True
            elif preset_data["role"] == "coder-only":
                coder_only = True

        if num_ctx == 32768 and "ctx" in preset_data:
            num_ctx = preset_data["ctx"]
        if "temp" in preset_data:
            if arch_only:
                temp_arch = preset_data["temp"]
            elif coder_only:
                temp_coder = preset_data["temp"]

    context_content = auto_load_project_context(context_file)
    spec_output = ""

    # -------------------------------------------------------------------------
    # ETAP 1: ARCHITEKT (Qwen 3.8 27B)
    # -------------------------------------------------------------------------
    if not coder_only:
        print("=" * 75)
        print("🏛️  ETAP 1: Architekt (Qwen 3.8 27B) — Analiza, Architektura i Specyfikacja")
        print(f"Model: {MODEL_ARCHITECT} | Temp: {temp_arch} | Ctx: {num_ctx} | Predict: {num_predict}")
        print("=" * 75)

        arch_prefix = preset_data["system_prefix"] + "\n\n" if preset_data and arch_only else ""
        architect_system_prompt = (
            f"{arch_prefix}"
            "Jesteś Głównym Architektem Oprogramowania i Inżynierem Systemów AI/ML & DSP.\n"
            "Twoją rolą jest głębokie przeanalizowanie wymagań, weryfikacja niezmienników technicznych "
            "oraz przygotowanie rygorystycznej, precyzyjnej specyfikacji technicznej dla programisty.\n"
            "Zawsze projektujesz architekturę odporną na wycieki pamięci, błędy wymiarów tensorów i corner-cases.\n"
            "BARDZO WAŻNE: Twoje wstępne myślenie w tagu <think> musi być zwięzłe i celowe (maksymalnie 10-15 zdań planowania). "
            "Natychmiast po krótkiej analizie zamknij tag </think> i generuj pełną, wyczerpującą specyfikację techniczną w treści odpowiedzi."
        )

        architect_user_prompt = f"""Działasz jako Główny Architekt Systemu.
Przeanalizuj poniższe zadanie i przygotuj WYCZERPUJĄCĄ, FORMALNĄ SPECYFIKACJI TECHNICZNĄ dla programisty wykonawczego.

STRUKTURA SPECYFIKACJI ARCHITEKTONICZNEJ:
1. CEL I ZAKRES: Dokładne zdefiniowanie celu modułu/funkcji i jego integracji w systemie.
2. NIEZMIENNIKI I MODEL DANYCH:
   - Dokładne kształty tensorów wejściowych i wyjściowych (z oznaczeniem wymiarów: B, C, H, W).
   - Typy danych (float32, int64), zakresy wartości i wymagane normalizacje.
   - Zgodność z architekturą ADR-001 (DSP) oraz ADR-002 (Czysta Java 17).
3. ARCHITEKTURA KLAS I SYGNATURY METOD:
   - Nazwy klas, metod, argumentów wraz z kompletnymi typami (Python / Java 17).
4. LOGIKA ALGORYTMICZNA KROK PO KROKU:
   - Pseudokod lub schemat blokowy przetwarzania.
5. CORNER-CASES, BEZPIECZEŃSTWO I OBSŁUGA BŁĘDÓW:
   - Obsługa pustych danych, brakujących plików, izolacja zasobów GPU/VRAM.

ZADANIE DO ZREALIZOWANIA:
{task_description}
{context_content}
"""

        architect_messages = [
            {"role": "system", "content": architect_system_prompt},
            {"role": "user", "content": architect_user_prompt}
        ]

        spec_output = query_ollama(
            MODEL_ARCHITECT,
            architect_messages,
            stream=True,
            keep_alive="0s",
            num_ctx=num_ctx,
            temperature=temp_arch,
            top_p=DEFAULT_ARCHITECT_PARAMS["top_p"],
            top_k=DEFAULT_ARCHITECT_PARAMS["top_k"],
            repeat_penalty=DEFAULT_ARCHITECT_PARAMS["repeat_penalty"],
            num_predict=num_predict,
            show_thinking=True
        )

        # Natychmiastowe zwolnienie VRAM karty GPU przed etapem kodera
        unload_model(MODEL_ARCHITECT)
        time.sleep(1)

        if arch_only:
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(spec_output)
                print(f"\n[SUKCES] Wynik Architekta zapisano do: {output_file}")
            print("\n" + "=" * 75)
            print("✅ ZAKOŃCZONO: Zadanie Architekta ukończone, VRAM w 100% zwolniony.")
            print("=" * 75)
            return

    # Jeśli tryb coder_only, treść specyfikacji pobierana jest z zadania/kontekstu
    if coder_only:
        spec_output = task_description + (f"\n\n{context_content}" if context_content else "")

    # -------------------------------------------------------------------------
    # ETAP 2: PROGRAMISTA WYKONAWCZY (Qwen 2.5 Coder 14B)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("💻 ETAP 2: Programista (Qwen 2.5 Coder 14B) — Generowanie / Audyt Kodu")
    print(f"Model: {MODEL_CODER} | Temp: {temp_coder} | Ctx: {num_ctx} | Predict: {num_predict}")
    print("=" * 75)

    coder_prefix = preset_data["system_prefix"] + "\n\n" if preset_data and coder_only else ""
    coder_system_prompt = (
        f"{coder_prefix}"
        "Jesteś elitarnym inżynierem oprogramowania produkcyjnego (Staff Software Engineer) "
        "specjalizującym się w PyTorch, DSP, optymalizacji numerycznej i systemach wysokiej wydajności.\n"
        "Twoje zasady:\n"
        "1. Generujesz wyłącznie gotowy do bezpośredniego wdrożenia kod (w Pythonie lub w 100% czystej Javie 17 bez Kotlina).\n"
        "2. Zero placeholderów, skrótów, 'TODO', '# implement here' ani pustych implementacji.\n"
        "3. Kompletne typowanie (typing / Java strong typing), docstringi oraz ścisła obsługa wyjątków i błędów.\n"
        "4. Bezwzględna zgodność ze specyfikacją, kształtami tensorów i niezmiennikami ADR-001 / ADR-002."
    )

    coder_user_prompt = f"""Działasz jako Senior Staff Software Engineer / Lead Reviewer.
Poniżej znajduje się specyfikacja lub kod do zrealizowania / audytu:

{spec_output}

TWOJE ZADANIE:
Napisz kompletny, w 100% produkcyjny kod lub przeprowadź rygorystyczny audyt/review zgodnie z wymaganiami.
Wymagania bezwzględne:
- Wszystkie funkcje i klasy muszą być w pełni zaimplementowane lub szczegółowo zrecenzowane.
- Dodaj niezbędne importy, obsługę wyjątków (try/except / try-catch).
- Zadbaj o optymalną wydajność i poprawne zwalnianie zasobów.
- Zwróć kod / raport w czytelnym formacie Markdown.
"""

    coder_messages = [
        {"role": "system", "content": coder_system_prompt},
        {"role": "user", "content": coder_user_prompt}
    ]

    code_output = query_ollama(
        MODEL_CODER,
        coder_messages,
        stream=True,
        keep_alive="0s",
        num_ctx=num_ctx,
        temperature=temp_coder,
        top_p=DEFAULT_CODER_PARAMS["top_p"],
        top_k=DEFAULT_CODER_PARAMS["top_k"],
        repeat_penalty=DEFAULT_CODER_PARAMS["repeat_penalty"],
        num_predict=num_predict,
        show_thinking=False
    )

    # Zwolnienie VRAM karty GPU po zakończeniu kodowania
    unload_model(MODEL_CODER)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            if not coder_only:
                f.write(f"# Specyfikacja Architekta ({MODEL_ARCHITECT}):\n\"\"\"\n{spec_output}\n\"\"\"\n\n")
            f.write(f"# Wynik Wygenerowany przez ({MODEL_CODER}):\n{code_output}\n")
        print(f"\n[SUKCES] Wynik zapisano do: {output_file}")

    print("\n" + "=" * 75)
    print("✅ ZAKOŃCZONO: Pipeline LLM zakończony pomyślnie. VRAM w 100% wolny.")
    print("=" * 75)


def check_ollama_ready() -> dict:
    """
    Sprawdza gotowość środowiska Ollama i dostępność VRAM.
    Zwraca słownik z informacjami o stanie.
    """
    result = {
        "ollama_running": False,
        "vram_free": False,
        "loaded_models": [],
        "available_models": [],
        "ready": False,
        "message": ""
    }

    # Sprawdź czy Ollama jest dostępna
    try:
        url = f"{OLLAMA_API_URL}/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            result["ollama_running"] = True
            result["available_models"] = [m["name"] for m in data.get("models", [])]
    except Exception as e:
        result["message"] = f"Ollama niedostępna: {e}"
        return result

    # Sprawdź czy VRAM jest wolny (żaden model nie jest załadowany)
    try:
        url = f"{OLLAMA_API_URL}/ps"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            loaded = data.get("models", [])
            result["loaded_models"] = [m["name"] for m in loaded] if loaded else []
            result["vram_free"] = len(result["loaded_models"]) == 0
    except Exception as e:
        result["message"] = f"Nie można sprawdzić stanu VRAM: {e}"
        return result

    # Sprawdź dostępność wymaganych modeli
    has_architect = MODEL_ARCHITECT in result["available_models"]
    has_coder = MODEL_CODER in result["available_models"]

    if not has_architect and not has_coder:
        result["message"] = f"Brak wymaganych modeli. Zainstaluj: {MODEL_ARCHITECT} i {MODEL_CODER}"
    elif not has_architect:
        result["message"] = f"Brak modelu Architekta: {MODEL_ARCHITECT}"
    elif not has_coder:
        result["message"] = f"Brak modelu Programisty: {MODEL_CODER}"
    elif not result["vram_free"]:
        result["message"] = f"VRAM zajęty przez: {result['loaded_models']}. Zwolnij przed użyciem."
    else:
        result["ready"] = True
        result["message"] = "Środowisko gotowe do pracy z lokalnymi modelami LLM."

    return result


def ensure_vram_free():
    """Próbuje zwolnić VRAM wyładowując wszystkie załadowane modele."""
    try:
        url = f"{OLLAMA_API_URL}/ps"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for model in data.get("models", []):
                model_name = model.get("name", "")
                if model_name:
                    print(f"[VRAM] Wyładowuję model: {model_name}")
                    unload_model(model_name)
                    time.sleep(1)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Dynamiczny tandem lokalnych modeli Ollama: Qwen 3.8 27B (Architekt) + Qwen 2.5 Coder (Programista)"
    )
    parser.add_argument("--task", type=str, default=None, help="Opis zadania do zrealizowania")
    parser.add_argument("--context", type=str, default=None, help="Opcjonalny plik kontekstu (np. kontekst.md)")
    parser.add_argument("--output", type=str, default=None, help="Plik wyjściowy do zapisu rezultatu")
    parser.add_argument("--ctx", type=int, default=32768, help="Rozmiar okna kontekstu (domyślnie 32768, wspiera do 131072 / 128k)")
    parser.add_argument("--temp-arch", type=float, default=0.2, help="Temperatura dla Architekta (domyślnie 0.2)")
    parser.add_argument("--temp-coder", type=float, default=0.1, help="Temperatura dla Programisty (domyślnie 0.1)")
    parser.add_argument("--num-predict", type=int, default=-1, help="Maksymalna liczba generowanych tokenów (-1 = nielimitowane)")
    parser.add_argument("--arch-only", action="store_true", help="Uruchom tylko etap 1 (Architekt / Specyfikacja)")
    parser.add_argument("--coder-only", action="store_true", help="Uruchom tylko etap 2 (Programista / Implementacja)")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default=None,
                        help=f"Predefiniowany profil zadania: {', '.join(PRESETS.keys())}")
    parser.add_argument("--list-presets", action="store_true", help="Wyświetl listę dostępnych profili zadań i zakończ")
    parser.add_argument("--check-ready", action="store_true", help="Sprawdź gotowość Ollama i VRAM (zwraca JSON)")
    parser.add_argument("--free-vram", action="store_true", help="Wyładuj wszystkie modele z VRAM")
    parser.add_argument("--json-output", action="store_true", help="Wynik w formacie JSON (do integracji z AGY)")

    args = parser.parse_args()

    # Tryb wyświetlania dostępnych profili
    if args.list_presets:
        print("=" * 75)
        print("🎯 DOSTĘPNE PROFILE ZADAŃ DLA MODELI QWEN (scripts/local_llm_bridge.py):")
        print("=" * 75)
        for name, p in PRESETS.items():
            role_label = "Architekt (Qwen 3.8 27B)" if p["role"] == "arch-only" else "Programista (Qwen 2.5 Coder 14B)"
            print(f"  • {name:<18} [{role_label}]")
            print(f"    {p['description']} (ctx: {p.get('ctx', 32768)}, temp: {p.get('temp', 0.1)})")
        print("=" * 75)
        sys.exit(0)

    # Tryb sprawdzania gotowości środowiska
    if args.check_ready:
        status = check_ollama_ready()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        sys.exit(0 if status["ready"] else 1)

    # Tryb zwalniania VRAM
    if args.free_vram:
        ensure_vram_free()
        status = check_ollama_ready()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        sys.exit(0 if status["vram_free"] else 1)

    # Walidacja: --task jest wymagany dla pipeline'u
    if not args.task:
        parser.error("Argument --task jest wymagany (chyba że używasz --check-ready, --free-vram lub --list-presets)")

    run_pipeline(
        task_description=args.task,
        context_file=args.context,
        output_file=args.output,
        num_ctx=args.ctx,
        temp_arch=args.temp_arch,
        temp_coder=args.temp_coder,
        num_predict=args.num_predict,
        arch_only=args.arch_only,
        coder_only=args.coder_only,
        preset=args.preset
    )


if __name__ == "__main__":
    main()
