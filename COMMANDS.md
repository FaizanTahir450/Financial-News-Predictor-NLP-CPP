# Commands to Run Financial News Predictor (No Transformer Version)

## Setup
```bash
python -m venv venv
# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

pip install -r requirements.txt
```

## Run the App
```bash
streamlit run app.py
```

## Retrain Model (optional)
Delete existing model files, then rerun app:
```bash
rm models/baseline_event_clf.pkl models/tfidf_vectorizer.pkl  # Linux/Mac
del models\baseline_event_clf.pkl models\tfidf_vectorizer.pkl  # Windows
streamlit run app.py
```
