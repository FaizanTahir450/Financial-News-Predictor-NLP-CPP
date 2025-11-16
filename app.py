import streamlit as st

st.set_page_config(
    page_title="Financial NLP Suite",
    page_icon=" briefcase ",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title(" Financial NLP Project Suite")
st.sidebar.success("Select a tool above.")

st.markdown("""
Welcome to your Computer Science Final Year Project!

This application demonstrates two different Natural Language Processing (NLP) solutions for financial analysis,
fulfilling the requirements for CT-485.

### 1.  Sentiment Analyzer
This tool uses two models (a TF-IDF baseline and an advanced FinBERT transformer) to
predict the sentiment of a single financial news headline.

**Go to the ' Sentiment Analyzer' page to try it.**

### 2.  Financial Assistant (RAG)
This is an advanced AI assistant that uses a Retrieval-Augmented Generation (RAG)
pipeline to answer complex questions about SEC 10-K and 10-Q filings.

**Go to the ' Financial Assistant' page to try it.**
""")