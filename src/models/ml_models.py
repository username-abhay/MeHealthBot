import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import joblib
import os

MODEL_DIR = os.path.dirname(__file__)

# Classification: Predict Disease
def train_classification(df: pd.DataFrame):
    """
    Train a RandomForest model to predict Disease from symptoms + vitals.
    """
    # Simplify: Convert categorical yes/no + symptoms into numeric
    X = pd.get_dummies(df[["Fever", "Symptoms"]], drop_first=True)
    X["Temperature"] = df["Temperature"]

    y = df["Disease"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model
    path = os.path.join(MODEL_DIR, "disease_model.pkl")
    joblib.dump(model, path)
    return model


def predict_disease(symptom: str, fever: str, temp: float):
    """
    Load the saved classification model and predict disease for given input.
    """
    path = os.path.join(MODEL_DIR, "disease_model.pkl")
    if not os.path.exists(path):
        return "[!] No trained disease model found. Train first."

    model = joblib.load(path)

    # Build input row
    data = {"Fever": [fever], "Symptoms": [symptom], "Temperature": [temp]}
    df_input = pd.DataFrame(data)

    X = pd.get_dummies(df_input, drop_first=True)

    # Align with model features safely
    model_features = model.feature_names_in_ 
    missing_cols = [col for col in model_features if col not in X.columns]
    if missing_cols:
        X = pd.concat([X, pd.DataFrame({col: [0] for col in missing_cols})], axis=1)

# Reorder columns exactly as model expects
    X = X[model_features].copy()  # copy() to de-fragment

    prediction = model.predict(X)
    return prediction[0]



# Regression: Forecast Temperature

def train_regression(df: pd.DataFrame):
    """
    Train a simple LinearRegression model to predict Temperature from Date.
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



# Clustering: Symptom Patterns

def cluster_symptom_patterns(df: pd.DataFrame, k=2):
    """
    Cluster health log into k groups based on symptoms + vitals.
    """
    X = pd.get_dummies(df[["Fever", "Symptom"]], drop_first=True)
    X["Temperature"] = df["Temperature"]

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    return labels.tolist()
