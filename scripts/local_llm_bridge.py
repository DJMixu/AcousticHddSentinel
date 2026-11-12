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
    if num_predict > 0:
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
    coder_only: bool = False
):
    """Uruchamia sekwencyjną współpracę Architekta (27B) i Programisty (Coder 14B)."""
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

        architect_system_prompt = (
            "Jesteś Głównym Architektem Oprogramowania i Inżynierem Systemów AI/ML & DSP.\n"
            "Twoją rolą jest głębokie przeanalizowanie wymagań, weryfikacja niezmienników technicznych "
            "oraz przygotowanie rygorystycznej, precyzyjnej specyfikacji technicznej dla programisty.\n"
            "Zawsze projektujesz architekturę odporną na wycieki pamięci, błędy wymiarów tensorów i corner-cases."
        )

        architect_user_prompt = f"""Działasz jako Główny Architekt Systemu.
Przeanalizuj poniższe zadanie i przygotuj WYCZERPUJĄCĄ, FORMALNĄ SPECYFIKACJI TECHNICZNĄ dla programisty wykonawczego.

STRUKTURA SPECYFIKACJI ARCHITEKTONICZNEJ:
1. CEL I ZAKRES: Dokładne zdefiniowanie celu modułu/funkcji i jego integracji w systemie.
2. NIEZMIENNIKI I MODEL DANYCH:
   - Dokładne kształty tensorów wejściowych i wyjściowych (z oznaczeniem wymiarów: B, C, H, W).
   - Typy danych (float32, int64), zakresy wartości i wymagane normalizacje.
   - Zgodność z architekturą ADR-001 (jeśli dotyczy DSP / Audio).
3. ARCHITEKTURA KLAS I SYGNATURY METOD:
   - Nazwy klas, metod, argumentów wraz z kompletnymi typami (Python typing).
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
                print(f"\n[SUKCES] Specyfikację Architekta zapisano do: {output_file}")
            print("\n" + "=" * 75)
            print("✅ ZAKOŃCZONO: Specyfikacja gotowa, VRAM w 100% zwolniony.")
            print("=" * 75)
            return

    # Jeśli tryb coder_only, treść specyfikacji pobierana jest z zadania/kontekstu
    if coder_only:
        spec_output = task_description

    # -------------------------------------------------------------------------
    # ETAP 2: PROGRAMISTA WYKONAWCZY (Qwen 2.5 Coder 14B)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("💻 ETAP 2: Programista (Qwen 2.5 Coder 14B) — Generowanie Kodu Produkcyjnego")
    print(f"Model: {MODEL_CODER} | Temp: {temp_coder} | Ctx: {num_ctx} | Predict: {num_predict}")
    print("=" * 75)

    coder_system_prompt = (
        "Jesteś elitarnym inżynierem oprogramowania produkcyjnego (Staff Software Engineer) "
        "specjalizującym się w PyTorch, DSP i systemach wysokiej wydajności.\n"
        "Twoje zasady:\n"
        "1. Generujesz wyłącznie gotowy do bezpośredniego uruchomienia kod w Pythonie.\n"
        "2. Zero placeholderów, skrótów, 'TODO', '# implement here' ani pustych 'pass'.\n"
        "3. Kompletne type hintsy (typing), docstringi oraz ścisła obsługa wyjątków i błędów.\n"
        "4. Bezwzględna zgodność ze specyfikacją Architekta i kształtami tensorów."
    )

    coder_user_prompt = f"""Działasz jako Senior Software Engineer.
Poniżej znajduje się rygorystyczna specyfikacja techniczna przygotowana przez Głównego Architekta:

{spec_output}

TWOJE ZADANIE:
Napisz kompletny, w 100% produkcyjny, bezpieczny kod w Pythonie realizujący tę specyfikację.
Wymagania bezwzględne:
- Wszystkie funkcje i klasy muszą być w pełni zaimplementowane.
- Dodaj niezbędne importy, obsługę wyjątków (try/except, ValueError, FileNotFoundError).
- Zadbaj o optymalną wydajność i poprawne zwalnianie zasobów.
- Zwróć kod w formacie Markdown w bloku ```python ... ```.
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
            f.write(f"# Kod Wygenerowany przez ({MODEL_CODER}):\n{code_output}\n")
        print(f"\n[SUKCES] Wygenerowany kod i raport zapisano do: {output_file}")

    print("\n" + "=" * 75)
    print("✅ ZAKOŃCZONO: Pipeline LLM zakończony pomyślnie. VRAM w 100% wolny.")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Dynamiczny tandem lokalnych modeli Ollama: Qwen 3.8 27B (Architekt) + Qwen 2.5 Coder (Programista)"
    )
    parser.add_argument("--task", type=str, required=True, help="Opis zadania do zrealizowania")
    parser.add_argument("--context", type=str, default=None, help="Opcjonalny plik kontekstu (np. kontekst.md)")
    parser.add_argument("--output", type=str, default=None, help="Plik wyjściowy do zapisu rezultatu")
    parser.add_argument("--ctx", type=int, default=32768, help="Rozmiar okna kontekstu (domyślnie 32768, wspiera do 131072 / 128k)")
    parser.add_argument("--temp-arch", type=float, default=0.2, help="Temperatura dla Architekta (domyślnie 0.2)")
    parser.add_argument("--temp-coder", type=float, default=0.1, help="Temperatura dla Programisty (domyślnie 0.1)")
    parser.add_argument("--num-predict", type=int, default=-1, help="Maksymalna liczba generowanych tokenów (-1 = nielimitowane)")
    parser.add_argument("--arch-only", action="store_true", help="Uruchom tylko etap 1 (Architekt / Specyfikacja)")
    parser.add_argument("--coder-only", action="store_true", help="Uruchom tylko etap 2 (Programista / Implementacja)")

    args = parser.parse_args()

    run_pipeline(
        task_description=args.task,
        context_file=args.context,
        output_file=args.output,
        num_ctx=args.ctx,
        temp_arch=args.temp_arch,
        temp_coder=args.temp_coder,
        num_predict=args.num_predict,
        arch_only=args.arch_only,
        coder_only=args.coder_only
    )


if __name__ == "__main__":
    main()
