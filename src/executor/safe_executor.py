import re
import pandas as pd
from typing import Optional
from models import ml_models

# simple blacklist -- improve if needed
FORBIDDEN_TOKENS = [
    "import os", "import sys", "subprocess", "socket", "open(", "__import__",
    "eval(", "exec(", "os.", "sys.", "shutil", "requests", "fork", "threading"
]

def contains_forbidden(code: str) -> Optional[str]:
    lower = code.lower()
    for t in FORBIDDEN_TOKENS:
        if t in lower:
            return t
    return None


def execute_query(code: str, df: pd.DataFrame) -> str:
    """
    Execute sanitized code. Expect Gemini to set a variable named `result`.
    Returns readable string for CLI.
    Supports both Pandas and ML function calls.
    """
    # quick sanitize
    bad = contains_forbidden(code)
    if bad:
        return f"⚠ Rejected: disallowed token or operation found -> {bad}"

    # Prepare safe environment
    safe_globals = {"pd": pd}

    # Provide ML functions to Gemini safely
    safe_globals.update({
        "predict_disease": lambda symptom, fever, temp: ml_models.predict_disease(symptom, fever, temp),
        "forecast_temperature": lambda days=7: ml_models.forecast_temperature(df, days),
        "cluster_symptom_patterns": lambda k=2: ml_models.cluster_symptom_patterns(df, k),
    })

    # Use a copy of df so Gemini cannot mutate user's original DF accidentally
    local_env = {"df": df.copy(), "result": None}

    try:
        # Execute the code block
        exec(code, safe_globals, local_env)
    except Exception as e:
        return f"⚠ Could not process the query. Error during execution: {e}"

    # Retrieve result
    res = local_env.get("result", None)
    if res is None:
        return "⚠ Query executed but no 'result' variable was set by the code."

    # Format output
    if isinstance(res, pd.DataFrame):
        return res.to_string(index=False)
    if isinstance(res, pd.Series):
        return res.to_string()
    if isinstance(res, list):
        return str(res)
    if isinstance(res, dict):
        # format dict nicely
        return "\n".join(f"{k}: {v}" for k, v in res.items())
    return str(res)
