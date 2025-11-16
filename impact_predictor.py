import os, joblib, numpy as np

MODEL_PATH = os.path.join("models", "market_impact_model.pkl")

def load_impact_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

def predict_impact(model, text):
    try:
        features = np.array([[len(text), sum(c.isupper() for c in text)]])
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(features)[0].max())
        return float(model.predict(features)[0])
    except Exception:
        return 0.0
