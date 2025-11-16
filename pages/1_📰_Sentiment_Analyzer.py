import os
import streamlit as st
import pandas as pd
from data_preprocessing import preprocess_text
from event_classifier import load_or_train_baseline
from impact_predictor import load_impact_model, predict_impact

from finbert_predictor import load_finbert_model, predict_finbert_sentiment

st.set_page_config(page_title="Financial News Predictor", page_icon="💹", layout="centered")

st.markdown("<h1 style='text-align:center;'>💹 Financial News Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Predict sentiment and market impact from financial headlines</p>", unsafe_allow_html=True)

def load_csv_with_fallbacks(path):
    
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, header=None, names=['label', 'news'])
            return df
        except Exception:
            continue
    return None

DATA_PATH = os.path.join("data", "financial_news.csv")

if os.path.exists(DATA_PATH):
    df = load_csv_with_fallbacks(DATA_PATH)
    if df is None:
        st.error("⚠️ Could not read dataset. Please check file encoding.")
        st.stop()
    with st.expander("📊 View Dataset Sample", expanded=False):
        st.dataframe(df.head(10))
else:
    st.error("⚠️ Dataset not found. Please place financial_news.csv under the 'data/' folder.")
    st.stop()

with st.spinner("Loading models... (FinBERT may download on first run)"):
    clf, vectorizer = load_or_train_baseline(data_path=DATA_PATH)
    
    finbert_pipeline = load_finbert_model()

if clf is None or vectorizer is None:
    st.error("❌ Baseline model could not be trained. Check dataset format and retry.")
    st.stop()
if finbert_pipeline is None:
    st.error("❌ Advanced FinBERT model could not be loaded. Check internet connection or 'transformers' installation.")
    

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<h3>📰 Single News Prediction</h3>", unsafe_allow_html=True)

st.markdown("**1. Select a Model**")
model_choice = st.selectbox(
    "Choose the prediction model:",
    ("Baseline (TF-IDF + Logistic Regression)", "Advanced (FinBERT)")
)

st.markdown("**2. Enter Text**")
text = st.text_area("Enter a financial news headline:", placeholder="e.g., Tesla shares surge after record deliveries")

col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("🔍 Predict Sentiment", use_container_width=True)
with col2:
    clear_btn = st.button("🧹 Clear", use_container_width=True)

if clear_btn:
    st.rerun()

if predict_btn:
    if not text.strip():
        st.error("⚠️ Please enter text before predicting.")
    else:
        # --- Option 1: Baseline Model ---
        if model_choice.startswith("Baseline"):
            st.markdown("<h4>🧠 Baseline Model Prediction</h4>", unsafe_allow_html=True)
            cleaned = preprocess_text(text)
            X = vectorizer.transform([cleaned])
            label = clf.predict(X)[0]
            st.success(f"**Predicted Sentiment:** {label.upper()}")
        
        else:
            if finbert_pipeline:
                st.markdown("<h4>✨ Advanced Model (FinBERT) Prediction</h4>", unsafe_allow_html=True)
                label, score = predict_finbert_sentiment(finbert_pipeline, text)
                if label:
                    st.success(f"**Predicted Sentiment:** {label}")
                    st.info(f"**Confidence Score:** {score:.3f}")
                else:
                    st.error("Error during FinBERT prediction.")
            else:
                st.error("Advanced model is not available. Please check app logs.")

        model = load_impact_model()
        if model:
            impact = predict_impact(model, text)
            st.info(f"📈 **Predicted Market Impact Score:** {impact:.3f}")
        else:
            st.warning("No market impact model available.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:gray;'>© 2025 Financial News Predictor | Streamlit App</p>",
    unsafe_allow_html=True
)