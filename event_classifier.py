import os, json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_text

MODEL_PATH = os.path.join("models", "baseline_event_clf.pkl")
VECT_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
RESULTS_DIR = os.path.join("results")

def load_dataset(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    
    # Try different encodings and CSV read parameters
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            # First attempt: assume file has a header with 'news' and 'label'
            df = pd.read_csv(path, encoding=encoding, on_bad_lines='skip', engine='python')
            if "news" in df.columns and "label" in df.columns:
                print(f"Successfully read CSV with encoding: {encoding} (header present)")
                df = df[["news", "label"]].dropna()
                df["text"] = df["news"].astype(str).apply(preprocess_text)
                return df

            # Second attempt: file may have no header and contain rows like: label, "news text"
            df2 = pd.read_csv(path, encoding=encoding, header=None, names=['label', 'news'], on_bad_lines='skip', engine='python')
            if 'label' in df2.columns and 'news' in df2.columns:
                print(f"Successfully read CSV with encoding: {encoding} (no header)")
                df2 = df2[["news", "label"]].dropna()
                df2["text"] = df2["news"].astype(str).apply(preprocess_text)
                return df2

        except Exception as e:
            print(f"Failed with encoding {encoding}: {str(e)}")
            continue

    print(f"Failed to read CSV file with any encoding ({', '.join(encodings)})")
    return None

def train_and_evaluate(df):
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_t = vect.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_t, y_train)
    X_test_t = vect.transform(X_test)
    y_pred = clf.predict(X_test_t)
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(os.path.join(RESULTS_DIR, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vect, VECT_PATH)
    return clf, vect

def load_or_train_baseline(data_path=None):
    if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
        clf = joblib.load(MODEL_PATH)
        vect = joblib.load(VECT_PATH)
        return clf, vect
    df = load_dataset(data_path)
    if df is None or df.empty:
        print("No dataset found for training.")
        return None, None
    print(f"Training model on dataset of size {df.shape[0]} ...")
    return train_and_evaluate(df)
