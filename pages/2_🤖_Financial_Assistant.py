import streamlit as st
import os
import re
import yfinance as yf
import google.generativeai as genai
from dotenv import load_dotenv
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph
from langchain_huggingface import HuggingFaceEmbeddings as HuggingFaceBgeEmbeddings

load_dotenv() 

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY must be set in your .env file.")
    st.stop()
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY must be set in your .env file.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Could not configure google.generativeai. Check API key. Error: {e}")
    st.stop()

PINECONE_INDEX_NAME = "financial-semantic-chunker-v3"
EMBEDDING_MODEL_NAME_BGE = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_NAME_SBERT = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash" 

RELEVANT_SECTIONS_10K = [
    "Item 1. Business", "Item 1A. Risk Factors", "Item 1B. Unresolved Staff Comments",
    "Item 1C. Cybersecurity", "Item 2. Properties", "Item 3. Legal Proceedings",
    "Item 4. Mine Safety Disclosures", "Item 5. Market for Registrant’s Common Equity",
    "Item 6. [Reserved]", "Item 7. Management’s Discussion and Analysis",
    "Item 7A. Quantitative and Qualitative Disclosures", "Item 8. Financial Statements and Supplementary Data",
    "Item 9. Changes in and Disagreements with Accountants", "Item 9A. Controls and Procedures",
    "Item 9B. Other Information", "Item 9C. Disclosure Regarding Foreign Jurisdictions",
    "Item 10. Directors, Executive Officers, and Corporate Governance", "Item 11. Executive Compensation",
    "Item 12. Security Ownership of Certain Beneficial Owners", "Item 13. Certain Relationships and Related Transactions",
    "Item 14. Principal Accountant Fees and Services", "Item 15. Exhibit and Financial Statement Schedules",
    "Item 16. Form 10-K Summary"
]
RELEVANT_SECTIONS_10Q = [
    "Item 1. Financial Statements", "Item 2. Management’s Discussion and Analysis",
    "Item 3. Quantitative and Qualitative Disclosures", "Item 4. Controls and Procedures",
    "Item 1. Legal Proceedings", "Item 1A. Risk Factors",
    "Item 2. Unregistered Sales of Equity Securities", "Item 3. Defaults Upon Senior Securities",
    "Item 4. Mine Safety Disclosures", "Item 5. Other Information"
]
ALL_RELEVANT_SECTIONS = sorted(list(set(RELEVANT_SECTIONS_10K + RELEVANT_SECTIONS_10Q)))


class SentenceTransformersEmbeddings:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME_SBERT, device=None):
        self.model = SentenceTransformer(model_name, device=device)
    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0].tolist()


@st.cache_data
def get_gemini_extraction_model():
    try:
        return genai.GenerativeModel(LLM_MODEL)
    except Exception as e:
        st.error(f"Could not initialize {LLM_MODEL} for name extraction. {e}")
        return None

gemini_extraction_model = get_gemini_extraction_model()

def identify_form_type(question: str) -> list[str]:
    q = question.lower()
    form_types = []
    if any(word in q for word in ["annual", "10-k", "yearly", "fiscal year", "full year report"]):
        form_types.append("10-K")
    if any(word in q for word in ["quarter", "10-q", "quarterly", "q1", "q2", "q3", "q4"]):
        form_types.append("10-Q")
    form_types = sorted(list(set(form_types)))
    if not form_types and gemini_extraction_model:
        try:
            response = gemini_extraction_model.generate_content(f"""
            Determine which SEC form(s) the user is referring to: "{question}"
            Respond with one of: ["10-K"], ["10-Q"], or ["10-K", "10-Q"]
            """)
            match = re.findall(r"10-[KQ]", response.text)
            form_types = sorted(list(set(match)))
        except Exception as e:
            print(f"Form type classification failed: {e}")
    if not form_types:
        form_types = ["10-K", "10-Q"] # Default to both
    print(f"📄 Detected Form Types: {form_types}")
    return form_types

def extract_company_names(prompt: str):
    if not gemini_extraction_model: return []
    try:
        response = gemini_extraction_model.generate_content(f"""
        Extract all company names from the text. Correct misspellings.
        Text: "{prompt}"
        Return as a Python list of strings. Example: ["Apple", "Microsoft"]
        """)
        text = response.text.strip()
        match = re.search(r"\[([^\]]*)\]", text)
        if match:
            list_content = match.group(1)
            companies = [c.strip().strip("'\"") for c in list_content.split(",") if c.strip()]
            print(f"✅ Extracted company names: {companies}")
            return companies
        return []
    except Exception as e:
        print(f"⚠️ Error during company name extraction: {e}")
        return []

def extract_years(prompt: str):
    regex_years = re.findall(r"\b(20\d{2})\b", prompt)
    years = sorted(list(set(int(y) for y in regex_years)))
    # (Skipping Gemini fallback for brevity in Streamlit app)
    print(f"📅 Extracted Years: {years}")
    return years

def extract_quarters(prompt: str) -> list[int]:
    if not gemini_extraction_model: return []
    try:
        response = gemini_extraction_model.generate_content(f"""
        Identify all specific quarters (1, 2, 3, or 4) mentioned in: "{prompt}"
        Return a Python list of integers. Example: [1, 3]
        """)
        text = response.text.strip()
        match = re.search(r"\[([^\]]*)\]", text)
        if match:
            list_content = match.group(1)
            quarters = [int(q) for q in re.findall(r"\b([1-4])\b", list_content)]
            quarters = sorted(list(set(quarters)))
            print(f"🧭 Extracted Quarters (LLM): {quarters}")
            return quarters
        return []
    except Exception as e:
        print(f"⚠️ Error extracting quarters via Gemini: {e}")
        return []

def get_tickers_from_yfinance(company_names):
    tickers = {}
    headers = {'User-Agent': 'Mozilla/5.0'}
    def fetch_ticker(name):
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={name}"
            response = requests.get(url, headers=headers, timeout=5)
            search_result = response.json()
            if "quotes" in search_result and search_result["quotes"]:
                return name, search_result["quotes"][0]["symbol"]
            return name, None
        except Exception as e:
            return name, None
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_name = {executor.submit(fetch_ticker, name): name for name in company_names}
        for future in as_completed(future_to_name):
            name, ticker = future.result()
            tickers[name] = ticker
    return tickers


def identify_entities(state):
    print("--- Identifying entities ---")
    question = state["question"]
    company_names = extract_company_names(question)
    ticker_map = get_tickers_from_yfinance(company_names)
    target_tickers = list(set([t for t in ticker_map.values() if t]))
    print(f"✅ Extracted Tickers: {target_tickers}")
    return {
        "target_tickers": target_tickers,
        "target_years": extract_years(question),
        "target_forms": identify_form_type(question),
        "target_quarters": extract_quarters(question),
        "question": question
    }

def identify_sections(state):
    print("\n--- Routing to sections ---")
    question = state["question"]
    target_forms = state.get("target_forms", [])
    sections_for_prompt = []
    if "10-K" in target_forms: sections_for_prompt.extend(RELEVANT_SECTIONS_10K)
    if "10-Q" in target_forms: sections_for_prompt.extend(RELEVANT_SECTIONS_10Q)
    if not sections_for_prompt: sections_for_prompt = ALL_RELEVANT_SECTIONS
    sections_for_prompt = sorted(list(set(sections_for_prompt)))
    all_sections_str = "\n".join(sections_for_prompt)
    form_type_str = " and ".join(target_forms) if target_forms else "10-K or 10-Q"
    
    routing_prompt_template = f"""
    You are an expert financial analyst. Determine ALL relevant sections of a {form_type_str} filing to answer:
    "{question}"
    
    Choose ONLY from this list:
    {all_sections_str}

    Respond with a JSON object: {{"sections": ["List of sections"]}}
    If no specific sections seem relevant, return {{"sections": []}}.
    """
    
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, response_mime_type="application/json")
    parser = JsonOutputParser()
    router_chain = llm | parser # Simpler chain for this prompt
    
    try:
        result = router_chain.invoke(routing_prompt_template)
        target_sections = result.get("sections", [])
        target_sections = [s for s in target_sections if s in sections_for_prompt]
        print(f"✅ Routing to sections: {target_sections}")
        state["target_sections"] = target_sections
    except Exception as e:
        print(f"⚠️ Error in section routing: {e}. Defaulting to no section filter.")
        state["target_sections"] = []
    return state

def retrieve_documents(state):
    print("\n--- Retrieving documents ---")
    embeddings = get_sbert_embeddings()
    index = get_pinecone_index()
    
    question = state["question"]
    query_vector = embeddings.embed_query(question)
    
    filters = []
    if state.get("target_tickers"): filters.append({"ticker": {"$in": state["target_tickers"]}})
    if state.get("target_sections"): filters.append({"section": {"$in": state["target_sections"]}})
    if state.get("target_years"): filters.append({"year": {"$in": state["target_years"]}})
    if state.get("target_forms"): filters.append({"form_type": {"$in": state["target_forms"]}})
    if state.get("target_quarters"): filters.append({"quarter": {"$in": state["target_quarters"]}})
    
    final_filter = {"$and": filters} if len(filters) > 1 else (filters[0] if filters else {})
    print(f"🔍 Applying filter: {final_filter}")

    dynamic_top_k = 20 + len(state.get("target_tickers", [])) * 5
    
    retrieved_docs = index.query(
        vector=query_vector,
        top_k=dynamic_top_k,
        filter=final_filter,
        include_metadata=True
    )
    
    context = ""
    if not retrieved_docs.get('matches'):
        print("No documents found.")
    for i, match in enumerate(retrieved_docs.get('matches', [])):
        metadata = match.get('metadata', {})
        fulltext = metadata.get('_full_text', 'No full text available.')
        context += f"Source {i+1} (Ticker: {metadata.get('ticker')}, Form: {metadata.get('form_type')}, Section: {metadata.get('section')}, Year: {metadata.get('year')}):\n{fulltext}\n\n"
    
    state["documents"] = context
    return state

def generate_answer(state):
    print("\n--- Generating Answer ---")
    question = state["question"]
    documents = state["documents"]
    if not documents:
        state["generation"] = "I'm sorry, but I couldn't find any documents matching your query."
        return state

    prompt_template = """
    You are an expert financial analyst. Answer the user's question based *only* on the provided context.
    Cite the sources (Company, Form, Year) for your answer.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    ANSWER:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    state["generation"] = generation
    return state


@st.cache_resource
def get_sbert_embeddings():
    return SentenceTransformersEmbeddings()

@st.cache_resource
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

@st.cache_resource
def get_rag_workflow():
    """
    Compiles and caches the LangGraph workflow.
    """
    workflow = StateGraph(dict)
    
    workflow.add_node("identify_entities", identify_entities)
    workflow.add_node("identify_sections", identify_sections)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)

    workflow.set_entry_point("identify_entities")
    workflow.add_edge("identify_entities", "identify_sections")
    workflow.add_edge("identify_sections", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()
    print("--- RAG Workflow Compiled ---")
    return app

st.set_page_config(page_title="AI Financial Assistant", page_icon="🤖")
st.title("🤖 AI Financial Research Assistant")
st.markdown("Query 10-K and 10-Q filings for multiple companies.")

try:
    app = get_rag_workflow()
except Exception as e:
    st.error(f"Failed to initialize RAG workflow. Check connections. Error: {e}")
    st.stop()

question = st.text_input(
    "Ask a question:",
    placeholder="What are the main risks for Apple and Google in 2024?"
)

if st.button("Get Answer"):
    if question:
        inputs = {"question": question}
        
        with st.status("Thinking... (Running RAG pipeline)") as status:
            final_generation = {}
            
            for output in app.stream(inputs):
                for key, value in output.items():
                    status.update(label=f"Running node: {key}")
                    if key == "generate":
                        final_generation = value
            
            status.update(label="Answer Generated!", state="complete")

        st.markdown("### Answer")
        st.write(final_generation.get("generation", "No answer could be generated."))
    
    else:
        st.error("Please enter a question.")