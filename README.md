# Financial NLP Project Suite

## Overview

This project is a multi-page Streamlit application that combines two powerful Natural Language Processing (NLP) tools for financial analysis:

1.  **📰 Sentiment Analyzer:** A tool to predict the sentiment (positive, negative, neutral) of financial news headlines. It features a live comparison between a classical "Baseline" model (TF-IDF + Logistic Regression) and an "Advanced" model (FinBERT transformer).
2.  **🤖 AI Financial Research Assistant:** An advanced Retrieval-Augmented Generation (RAG) pipeline. This tool allows you to ask complex, natural-language questions about SEC 10-K and 10-Q filings for multiple companies and receive synthesized answers.

This application is designed to fulfill the requirements of the CT-485 NLP course, demonstrating both classical text classification (CLO 3) and an innovative, complex solution (CP 2, CP 3).

## Project Structure

```
financial-nlp-project/
├── .env                  # Stores API keys (GOOGLE_API_KEY, PINECONE_API_KEY)
├── app.py                # The main homepage and Streamlit router
├── requirements.txt      # All required Python packages
├── data/                 # Folder for datasets
│   └── financial_news.csv  # Required for the Sentiment Analyzer
├── models/               # Stores the auto-trained baseline model
│   └── baseline_event_clf.pkl
│   └── tfidf_vectorizer.pkl
├── pages/                # Contains the app's sub-pages
│   ├── 1_📰_Sentiment_Analyzer.py
│   └── 2_🤖_Financial_Assistant.py
├── data_preprocessing.py   # Text cleaning for the baseline model
├── event_classifier.py     # Training & logic for the baseline model
├── finbert_predictor.py    # Logic for the FinBERT sentiment model
└── results/              # Stores classification reports after training
```

## 🛠️ Setup Instructions

**1. Create Folders:**
Create the `data/`, `models/`, and `results/` folders in your project's root directory.

**2. Set Up API Keys:**
Create a file named `.env` in the root directory. This is **required** for the AI Financial Assistant to work. Add your keys:

```
GOOGLE_API_KEY="your-google-api-key"
PINECONE_API_KEY="your-pinecone-api-key"
```

**3. Install Dependencies:**
Create a virtual environment and install all required packages from the `requirements.txt` file.

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

*(See the new `requirements.txt` in the previous step)*

**4. Download Sentiment Dataset:**

  * Go to: `https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news`
  * Download the dataset.
  * Rename the file to `financial_news.csv`.
  * Place it in the `data/` folder.

**5. (Prerequisite) Pinecone Index:**
The AI Financial Assistant assumes you have a pre-populated Pinecone index named `financial-semantic-chunker-v3` containing embeddings of SEC filings. The code in this project *queries* the index but does not *create* it.

## 🚀 Run the Application

With your virtual environment activated, run the main `app.py` file:

```bash
streamlit run app.py
```

Your browser will open to the project's homepage. You can navigate between the Sentiment Analyzer and the AI Financial Assistant using the sidebar.

-----

## 📰 Tool 1: Sentiment Analyzer

  * **Function:** Predicts the sentiment of a single financial news headline.
  * **Models:**
      * **Baseline:** A TF-IDF Vectorizer + Logistic Regression classifier.
      * **Advanced:** The `ProsusAI/finbert` transformer model.
  * **Auto-Training Behavior:**
      * On the first run, the app will automatically train the **Baseline** model using the `financial_news.csv` file.
      * It saves the trained model and vectorizer to the `models/` folder.
      * It saves evaluation metrics (classification report, confusion matrix) to the `results/` folder.
      * On all future runs, it loads the saved models instantly.

## 🤖 Tool 2: AI Financial Research Assistant

  * **Function:** Answers natural language questions about SEC 10-K and 10-Q filings.
  * **Workflow (RAG):**
    1.  **Identify Entities:** Uses Gemini to extract company names, tickers, years, and quarters from your question.
    2.  **Identify Sections:** Uses Gemini to determine the relevant filing sections (e.g., "Item 1A. Risk Factors").
    3.  **Retrieve:** Performs a semantic search in Pinecone, using metadata filters for the extracted entities.
    4.  **Generate:** Uses Gemini to synthesize a final answer based *only* on the retrieved document chunks.
  * **Dependencies:** This tool will **not** work without the valid `GOOGLE_API_KEY` and `PINECONE_API_KEY` in your `.env` file and a correctly populated Pinecone index.