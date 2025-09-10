# src/cli_app.py
"""
Health-log chatbot CLI + API wrapper.
- CLI loop commented for dev testing.
- Use `ask_question(user_input: str)` to send a question and get final conversational answer.
"""

import os
import re
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from src.nlp_adapter.gemini_client import GeminiAdapter
from src.utils import schema_manager
from src.executor.safe_executor import execute_query

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


def extract_code_block(text: str) -> str | None:
    """Extract Python code block from Gemini response."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    return None


# ---------- GLOBAL INITIALIZATION ----------
df = load_dataset()
if df is None:
    raise RuntimeError("[!] Could not load dataset. Exiting.")

# Extract schema
schema = schema_manager.extract_schema(df)
schema_str = schema_manager.schema_to_string(schema)

# Init Gemini adapter
gemini = GeminiAdapter(api_key=GEMINI_API_KEY)

# Strict system prompt
system_prompt = """SYSTEM INSTRUCTION:
You are a Python data analysis assistant with access to:
- A pandas DataFrame named `df` containing a health log dataset.
IMPORTANT RULES:
- ALWAYS respond with a single executable Python code block, nothing else.
- The code MUST assign the final output to a variable named `result`.
- NEVER import libraries (pandas and numpy already available).
- NEVER use loops or multi-line if/else; use vectorized pandas/numpy instead.
- Allowed libraries: pandas, numpy only.
"""
gemini.send_message(system_prompt)
gemini.send_message(f"Dataset schema:\n{schema_str}")
print("\n[+] Schema and system instruction sent to Gemini. Ready to answer questions.\n")


def ask_question(user_input: str) -> str:
    """
    Pass a user question to Gemini.
    Returns the final conversational answer as a string.
    """
    # ---------- PASS 1: Computation ----------
    comp_prompt = (
        f"User question: {user_input}\n"
        "Step 1: Return ONLY Python code that computes the raw answer from df. "
        "Do not add explanations. Assign the output to variable 'result'."
    )
    resp_text = gemini.send_message(comp_prompt)
    code = extract_code_block(resp_text)
    if code is None:
        return "⚠ No computable code returned."

    raw_output = execute_query(code, df)

    # ---------- PASS 2: Analysis ----------
    analysis_prompt = (
        f"The user asked: {user_input}\n"
        f"The computed result is: {raw_output}\n"
        "Step 2: Analyze this result, answer any follow-up parts of the question, "
        "and return ONLY Python code that assigns a final conversational string to 'result'."
    )
    resp_text2 = gemini.send_message(analysis_prompt)
    final_code = extract_code_block(resp_text2)
    if final_code is None:
        return "⚠ Gemini did not provide final analysis code."

    final_output = execute_query(final_code, df)
    return final_output


# ---------- CLI LOOP FOR DEV TESTING (COMMENTED) ----------
# if __name__ == "__main__":
#     while True:
#         user_input = input("Ask your question (or 'exit' to quit):\n> ").strip()
#         if user_input.lower() in {"exit", "quit"}:
#             break
#         if not user_input:
#             continue
#         answer = ask_question(user_input)
#         print("\n[Final Answer]:\n", answer)
