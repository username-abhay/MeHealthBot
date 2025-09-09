import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from nlp_adapter import gemini_client
from utils import schema_manager
from executor import safe_executor

# Load .env file (API keys, dataset path, etc.)
load_dotenv()

# Default dataset path from environment variable
DATA_PATH = os.getenv("DATA_PATH")


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """
    Loads the CSV dataset into a pandas DataFrame.
    Returns DataFrame if successful, None otherwise.
    """
    p = Path(path)
    if not p.exists():
        print(f"[!] Dataset not found at: {p.resolve()}")
        return None

    try:
        df = pd.read_csv(p, parse_dates=["Date"], infer_datetime_format=True)
    except Exception as e:
        print(f"[!] Failed to read CSV: {e}")
        return None

    print(f"[+] Loaded dataset: {len(df)} rows × {len(df.columns)} columns")
    # print("Columns:", list(df.columns))
    # print("\nFirst 5 rows:")
    # print(df.head(5).to_string(index=False))
    return df


def main():
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("[!] GEMINI_API_KEY not found in .env file. Please add it.")
        return

    df = load_dataset()
    if df is None:
        return

    # Extract schema
    schema = schema_manager.extract_schema(df)
    schema_str = schema_manager.schema_to_string(schema)

    print("\n[+] Extracted Schema (ready for Gemini):\n")
    # print(schema_str)

     # Initialize Gemini Adapter
    gemini = gemini_client.GeminiAdapter(api_key=GEMINI_API_KEY)

    # Send schema as first message
    system_prompt = """You are a Python data analysis assistant. You ONLY respond with fully executable Pandas/ML code blocks. No natural language, no explanations, no comments. Assign your final output to a variable named 'result'. The DataFrame is always named df."""

    schema_prompt = f"{system_prompt}\nHere is the dataset schema:\n{schema_str}\n"
    print("\n[+] Sending schema to Gemini...\n")
    # print(schema_prompt)
    gemini.send_message(schema_prompt)

    # Prototype loop for user queries
    while True:
        user_input = input("\nAsk your question (or 'exit' to quit):\n> ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response_text = gemini.send_message(user_input)
        print("\n[Gemini response code]:\n", response_text)

        # Execute safely
        output = safe_executor.execute_query(response_text, df)
        print("\n[Result]:\n", output)

    # TODO: Phase 4 onward - integrate Gemini, safe query execution, ML insights


if __name__ == "__main__":
    main()
