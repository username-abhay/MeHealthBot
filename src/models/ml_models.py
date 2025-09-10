import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import joblib
import os

MODEL_DIR = os.path.dirname(__file__)

# --------------------------
# Classification: Predict Disease
# --------------------------
def train_classification(df: pd.DataFrame):
    """
    Train a RandomForest model to predict Disease from symptoms + vitals.
    """
    # Use Fever, Symptoms, Temperature for classification
    X = pd.get_dummies(df[["Fever", "Symptoms"]], drop_first=True)
    X["Temperature"] = df["Temperature"]

    y = df["Disease"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Save model + feature names
    path = os.path.join(MODEL_DIR, "disease_model.pkl")
    joblib.dump({"model": model, "features": X.columns.tolist()}, path)
    return model


def predict_disease(symptom: str, fever: str, temp: float):
    """
    Load the saved classification model and predict disease for given input.
    """
    path = os.path.join(MODEL_DIR, "disease_model.pkl")
    if not os.path.exists(path):
        return "[!] No trained disease model found. Train first."

    bundle = joblib.load(path)
    model, model_features = bundle["model"], bundle["features"]

    # Input row
    df_input = pd.DataFrame({"Fever": [fever], "Symptoms": [symptom], "Temperature": [temp]})

    X = pd.get_dummies(df_input, drop_first=True)

    # Align columns
    for col in model_features:
        if col not in X.columns:
            X[col] = 0
    X = X[model_features]

    prediction = model.predict(X)
    return prediction[0]


# --------------------------
# Regression: Forecast Temperature
# --------------------------
def train_regression(df: pd.DataFrame):
    """
    Train a LinearRegression model to predict Temperature from Date index.
    """
    df = df.copy()
    df["DayIndex"] = (df["Date"] - df["Date"].min()).dt.days

    X = df[["DayIndex"]]
    y = df["Temperature"]

    model = LinearRegression()
    model.fit(X, y)

    path = os.path.join(MODEL_DIR, "temp_model.pkl")
    joblib.dump(model, path)
    return model


def forecast_temperature(df: pd.DataFrame, days=7):
    """
    Forecast average temperature for next N days.
    """
    path = os.path.join(MODEL_DIR, "temp_model.pkl")
    if not os.path.exists(path):
        return "[!] No trained temperature model found. Train first."

    model = joblib.load(path)

    last_day = (df["Date"].max() - df["Date"].min()).days
    future_days = [[last_day + i] for i in range(1, days + 1)]

    preds = model.predict(future_days)
    return preds.tolist()


# --------------------------
# Clustering: Symptom Patterns
# --------------------------
def cluster_symptom_patterns(df: pd.DataFrame, k=3):
    """
    Cluster health log into k groups based on symptoms + vitals.
    """
    X = pd.get_dummies(df[["Fever", "Symptoms"]], drop_first=True)
    X["Temperature"] = df["Temperature"]

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    return labels.tolist()
