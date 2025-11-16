import streamlit as st
from transformers import pipeline

# Use Streamlit's cache to load the model only once
@st.cache_resource
def load_finbert_model():
    """
    Loads and caches the FinBERT sentiment analysis pipeline.
    """
    print("Loading FinBERT model for the first time...")
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        print("FinBERT model loaded successfully.")
        return sentiment_pipeline
    except Exception as e:
        print(f"Error loading FinBERT model: {e}")
        return None

def predict_finbert_sentiment(pipeline, text):
    """
    Predicts sentiment for a single text using the loaded FinBERT pipeline.
    """
    if not pipeline or not text:
        return None, None

    try:
        # The pipeline returns a list of dictionaries
        result = pipeline(text)
        
        # Extract the label and score
        label = result[0]['label'].upper()
        score = result[0]['score']
        return label, score
    except Exception as e:
        print(f"Error during FinBERT prediction: {e}")
        return None, None