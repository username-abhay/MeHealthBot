# src/cli_app.py
"""
Main CLI for the health-log chatbot.
Flow:
- load dataset
- extract schema
- initialize Gemini with a strict system prompt (sent once)
- send schema
- interactive loop: send user question -> get Gemini response -> extract code -> validate -> execute -> print
"""

import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from nlp_adapter.gemini_client import GeminiAdapter
from utils import schema_manager
from executor.safe_executor import execute_query

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/Data_clean.csv")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def load_dataset(path=DATA_PATH) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        print(f"[!] Dataset not found at: {p.resolve()}")
        return None
    try:
        df = pd.read_csv(p, parse_dates=["Date"])
    except Exception as e:
        print(f"[!] Failed to read CSV: {e}")
        return None
    print(f"[+] Loaded dataset: {len(df)} rows × {len(df.columns)} columns")
    return df


# Utility: extract python code block from Gemini response
def extract_code_block(text: str) -> str | None:
    """
    Return Python code extracted from triple-backtick block or inline code.
    """
    # 1) triple-backtick block (```python ... ``` or ``` ... ```)
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()

    # 2) look for contiguous Python-like lines
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(result\s*=|df\[|df\.|import\s+|from\s+|for\s+|if\s+|with\s+)", stripped):
            lines.append(line)
        elif re.match(r"^[\w_]+\s*=", stripped):
            lines.append(line)
        elif stripped.startswith(("return", "print")):
            lines.append(line)
        elif re.search(r"[A-Za-z0-9].*\b(the|is|are|you|please)\b", stripped) and len(stripped.split()) > 6:
            continue
    if lines:
        return "\n".join(lines).strip()
    return None


def main():
    if not GEMINI_API_KEY:
        print("[!] GEMINI_API_KEY not found in .env. Please add it.")
        return

    df = load_dataset()
    if df is None:
        return

    # schema
    schema = schema_manager.extract_schema(df)
    schema_str = schema_manager.schema_to_string(schema)

    print("\n[+] Extracted Schema (ready for Gemini):\n")

    # Init Gemini adapter
    gemini = GeminiAdapter(api_key=GEMINI_API_KEY)

    # Strict system prompt (sent once)
    system_prompt = """SYSTEM INSTRUCTION:
You are a Python data analysis assistant with access to:
- A pandas DataFrame named `df` containing a health log dataset.
- Pre-trained ML functions available for analysis:
    1. predict_disease(symptom: str, fever: str, temp: float) -> predicts disease based on symptoms and temperature.
    2. forecast_temperature(days=7) -> forecasts average temperature for next N days.
    3. cluster_symptom_patterns(k=2) -> clusters symptom patterns into k groups.
IMPORTANT RULES:
- ALWAYS respond with a single executable Python code block, nothing else.
- The code MUST assign the final output to a variable named `result`.
- NEVER use multi-line if/else statements, loops, or imports.
- If you need conditions, use inline expressions (e.g., `x if cond else y`).
- If you need multiple outputs, return them as a dict or DataFrame assigned to `result`.
- Allowed libraries: pandas, numpy only.
- You may call the ML functions above directly in your code.
- If you cannot answer in code, reply exactly: ERROR:UNSUPPORTED
"""

    # Send system prompt first (establish role)
    gemini.send_message(system_prompt)

    # Send schema afterwards so the assistant knows columns
    schema_prompt = f"Dataset schema:\n{schema_str}\n"
    gemini.send_message(schema_prompt)

    print("\n[+] Schema and system instruction sent to Gemini. You can now ask questions.\n")

    while True:
        user_input = input("Ask your question (or 'exit' to quit):\n> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        # send user query (Gemini will have system + schema context)
        resp_text = gemini.send_message(user_input)
        print("\n[Gemini raw response]:\n", resp_text)

        # Try to extract code
        code = extract_code_block(resp_text)
        if code is None:
            retry_prompt = "Please respond ONLY with executable Python code that assigns the final output to variable named 'result'. If impossible, reply exactly: ERROR:UNSUPPORTED"
            resp_text2 = gemini.send_message(retry_prompt)
            print("\n[Gemini retry response]:\n", resp_text2)
            code = extract_code_block(resp_text2)

        if code is None:
            print("\n[Result]:\n⚠ Gemini did not provide executable code. Try rephrasing the question.")
            continue

        # Execute safely
        output = execute_query(code, df)
        print("\n[Result]:\n", output)


if __name__ == "__main__":
    main()
