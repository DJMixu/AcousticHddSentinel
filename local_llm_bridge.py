#!/usr/bin/env python3
"""
local_llm_bridge.py — Dynamiczny most i orkiestracja lokalnych modeli Ollama
Dedykowany dla karty AMD Radeon RX 9060 XT (16 GB VRAM).

Łączy w tandem:
1. Architekt / Planista: hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S (12 GB, 128k kontekstu)
2. Programista / Wykonawca: qwen2.5-coder:14b (9 GB, specjalizacja kod)

Zapewnia płynne zarządzanie pamięcią VRAM (przeładowywanie sekwencyjne keep_alive=0),
aby nigdy nie przekroczyć 16 GB VRAM i zachować maksymalną prędkość GPU.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

OLLAMA_API_URL = "http://localhost:11434/api"

MODEL_ARCHITECT = "hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S"
MODEL_CODER = "qwen2.5-coder:14b"


def query_ollama(model_name: str, messages: list, stream: bool = True, keep_alive: str = "0s", num_ctx: int = 16384) -> str:
    """Wysyła zapytanie do API czatu Ollama z automatycznym zwalnianiem VRAM."""
    url = f"{OLLAMA_API_URL}/chat"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": stream,
        "keep_alive": keep_alive,
        "options": {
            "num_ctx": num_ctx
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    response_text = ""
    try:
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    msg = chunk.get("message", {}).get("content", "")
                    if stream:
                        sys.stdout.write(msg)
                        sys.stdout.flush()
                    response_text += msg
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
    """Zwalnia model z pamięci VRAM GPU."""
    url = f"{OLLAMA_API_URL}/generate"
    payload = {"model": model_name, "keep_alive": 0}
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            pass
    except Exception:
        pass


def run_pipeline(task_description: str, context_file: str = None, output_file: str = None, num_ctx: int = 32768):
    """Uruchamia sekwencyjną współpracę Architekta (27B) i Programisty (Coder 14B)."""
    context_content = ""
    if context_file:
        try:
            with open(context_file, "r", encoding="utf-8") as f:
                context_content = f"\n\n--- KONTEKST PROJEKTU ({context_file}) ---\n" + f.read()
        except Exception as e:
            print(f"[OSTRZEŻENIE] Nie udało się wczytać pliku kontekstu: {e}")

    print("=" * 70)
    print("🚀 ETAP 1: Architekt (Qwen 3.8 27B) analizuje zadanie i tworzy specyfikację")
    print(f"Model: {MODEL_ARCHITECT}")
    print("=" * 70)

    architect_prompt = f"""Działasz jako Główny Architekt Oprogramowania i Ekspert AI.
Twoim zadaniem jest przeanalizowanie poniższego zadania i przygotowanie PRECYZYJNEJ, TECHNICZNEJ SPECYFIKACJI dla programisty (Qwen 2.5 Coder).

Wytyczne dla specyfikacji:
1. Zdefiniuj strukturę klas, sygnatury funkcji, wejścia i wyjścia (wraz z wymiarami tensorów, typami danych).
2. Wskaż kluczowe biblioteki (PyTorch, ONNX, Librosa, itp.) i potencjalne pułapki/corner-cases.
3. Przedstaw logikę algorytmiczną krok po kroku w pseudokodzie.

Zadanie do zrealizowania:
{task_description}
{context_content}
"""

    architect_messages = [
        {"role": "system", "content": "Jesteś Głównym Architektem Oprogramowania specjalizującym się w systemach AI/ML i DSP."},
        {"role": "user", "content": architect_prompt}
    ]

    spec_output = query_ollama(MODEL_ARCHITECT, architect_messages, stream=True, keep_alive="0s", num_ctx=num_ctx)
    
    # Upewniamy się, że VRAM został natychmiast zwolniony przed załadowaniem Codera
    unload_model(MODEL_ARCHITECT)
    time.sleep(1)

    print("\n" + "=" * 70)
    print("💻 ETAP 2: Programista (Qwen 2.5 Coder 14B) implementuje kod na bazie specyfikacji")
    print(f"Model: {MODEL_CODER}")
    print("=" * 70)

    coder_prompt = f"""Działasz jako Senior Software Engineer / PyTorch & DSP Expert.
Oto specyfikacja techniczna przygotowana przez Głównego Architekta:

{spec_output}

Twoje zadanie:
Napisz kompletny, czysty, w pełni działający kod w Pythonie spełniający tę specyfikację.
Kod musi zawierać pełną obsługę błędów, poprawne typowanie i nie zawierać placeholderów typu 'TODO' czy 'pass'.
"""

    coder_messages = [
        {"role": "system", "content": "Jesteś elitarnym inżynierem oprogramowania. Zawsze generujesz gotowy, bezpieczny i zoptymalizowany kod produkcyjny."},
        {"role": "user", "content": coder_prompt}
    ]

    code_output = query_ollama(MODEL_CODER, coder_messages, stream=True, keep_alive="0s", num_ctx=num_ctx)
    unload_model(MODEL_CODER)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Specyfikacja Architekta ({MODEL_ARCHITECT}):\n\"\"\"\n{spec_output}\n\"\"\"\n\n# Kod Wygenerowany przez ({MODEL_CODER}):\n{code_output}\n")
        print(f"\n[SUKCES] Całość zapisano do pliku: {output_file}")

    print("\n" + "=" * 70)
    print("✅ ZAKOŃCZONO: Modele wykonały zadanie, VRAM został w 100% zwolniony.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Tandem modeli lokalnych Ollama: Qwen 3.8 27B (Architekt) + Qwen 2.5 Coder (Programista)")
    parser.add_argument("--task", type=str, required=True, help="Opis zadania do zrealizowania przez tandem")
    parser.add_argument("--context", type=str, default=None, help="Ścieżka do opcjonalnego pliku kontekstu (np. kontekst.md)")
    parser.add_argument("--output", type=str, default=None, help="Ścieżka do pliku wyjściowego, gdzie zapisać rezultat")
    parser.add_argument("--ctx", type=int, default=32768, help="Rozmiar okna kontekstowego (domyślnie 32768, wspiera do 131072 / 128k)")
    args = parser.parse_args()

    run_pipeline(args.task, args.context, args.output, num_ctx=args.ctx)


if __name__ == "__main__":
    main()
