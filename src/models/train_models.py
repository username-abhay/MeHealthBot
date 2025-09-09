import pandas as pd
from pathlib import Path
from src.models import ml_models

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "Data_clean.csv"

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    print("[+] Training classification model...")
    ml_models.train_classification(df)
    print("[+] Disease model saved.")

    print("[+] Training regression model...")
    ml_models.train_regression(df)
    print("[+] Temperature model saved.")

    print("[✓] All models trained and saved in src/models/")

if __name__ == "__main__":
    main()
