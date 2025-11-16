import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report
from event_classifier import load_dataset # Reuse your data loader
from tqdm import tqdm

print("Loading dataset...")
df = load_dataset('data/financial_news.csv')

if df is None:
    print("Failed to load dataset. Exiting.")
    exit()

print("Loading FinBERT model... (This may take a moment)")

sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

print("Splitting data...")

# our event_classifier.py uses test_size=0.2 and random_state=42
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df["news"], # using the raw news text, not the preprocessed text
    df["label"], 
    test_size=0.2, 
    random_state=42, 
    stratify=df["label"]
)

print(f"Running predictions on {len(y_test)} test samples...")

# We use 'news' (raw text) because FinBERT does its own tokenization.
y_pred = []
for text in tqdm(X_test, desc="Predicting with FinBERT"):
    # The pipeline returns a list of dictionaries, e.g., [{'label': 'positive', 'score': 0.9...}]
    result = sentiment_pipeline(text)
    y_pred.append(result[0]['label']) 

print("\n--- FinBERT Model Results ---")
print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred, output_dict=True)
import json
with open("results/finbert_classification_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("\nComparison report saved to results/finbert_classification_report.json")