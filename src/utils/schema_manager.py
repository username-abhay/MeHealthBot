from typing import List, Dict
import pandas as pd


def extract_schema(df: pd.DataFrame) -> List[Dict]:
   
    schema = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique = int(df[col].nunique()) if len(df) > 0 else 0
        sample = df[col].dropna().unique()[:5].tolist() if len(df) > 0 else []
        schema.append({
            "column": col,
            "dtype": dtype,
            "unique_count": unique,
            "sample_values": sample
        })
    return schema


def schema_to_string(schema: List[Dict]) -> str:
  
    lines = []
    for col_meta in schema:
        col_name = col_meta["column"]
        dtype = col_meta["dtype"]
        unique_count = col_meta["unique_count"]
        sample = ", ".join([str(v) for v in col_meta["sample_values"]])
        line = f"- {col_name} (type: {dtype}, unique: {unique_count}, sample: [{sample}])"
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test: load a CSV and print schema
    import os
    import pandas as pd

    DATA_PATH = "D:\Coterie\chatbot\data\Data_clean.csv"
    if not os.path.exists(DATA_PATH):
        print(f"[!] CSV not found at {DATA_PATH}")
    else:
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        schema = extract_schema(df)
        print("Extracted Schema:\n")
        print(schema_to_string(schema))
