import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import chromadb

# Import core agent functionality
from rag_agent import (
    get_chroma_client,
    get_collection,
    run_agent,
    ingest_pdf
)

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Multi-Agent RAG Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium look and styling
st.markdown("""
<style>
    /* Main background and styling */
    .reportview-container {
        background: #f7f9fc;
    }
    
    /* Title styling */
    .title-container {
        padding: 1.5rem 0rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .title-container h1 {
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
        color: white !important;
    }
    .title-container p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Routing Badge styling */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .badge-calculator {
        background-color: #f3e5f5;
        color: #7b1fa2;
        border: 1px solid #e1bee7;
    }
    .badge-retrieval {
        background-color: #e3f2fd;
        color: #1565c0;
        border: 1px solid #bbdefb;
    }
    .badge-web {
        background-color: #fff3e0;
        color: #e65100;
        border: 1px solid #ffe0b2;
    }
    .badge-general {
        background-color: #efebe9;
        color: #4e342e;
        border: 1px solid #d7ccc8;
    }
    
    /* Source context styling */
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #90caf9;
        padding: 0.8rem;
        margin-top: 0.5rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #333;
    }
    
    /* Card panel styling */
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eef2f6;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3c72;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {"history": []}

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.image("https://img.icons8.com/clouds/200/brain.png", width=120)
st.sidebar.title("Configuration & DB")

# API Keys Configuration
st.sidebar.subheader("🔌 API Integrations")
groq_api_key = os.getenv("GROQ_API_KEY", "")

# Allow override or input if missing
if not groq_api_key:
    groq_api_key = st.sidebar.text_input(
        "Enter Groq API Key:",
        type="password",
        help="Obtain key from console.groq.com"
    )
else:
    st.sidebar.success("✅ Groq API Key Configured (.env)")

# Initialize DB connection
chroma_client = get_chroma_client()
collection = get_collection(chroma_client)

# Document Ingestion Section
st.sidebar.subheader("📄 Document Ingestion")

# Auto-ingest default PDF if database is empty
DEFAULT_PDF = "economics_research_reference.pdf"
if collection.count() == 0 and os.path.exists(DEFAULT_PDF):
    with st.sidebar.status("🔄 Ingesting default reference PDF...", expanded=True) as status:
        try:
            num_chunks = ingest_pdf(DEFAULT_PDF, collection)
            status.update(label=f"✅ Ingested {DEFAULT_PDF} ({num_chunks} chunks)", state="complete")
        except Exception as e:
            status.update(label=f"❌ Ingestion failed: {str(e)}", state="error")

# Upload new document
uploaded_file = st.sidebar.file_uploader(
    "Upload additional PDF",
    type=["pdf"],
    help="Add more reference material to the vector database."
)

if uploaded_file is not None:
    # Check API key before processing
    if not groq_api_key:
        st.sidebar.error("Please configure your Groq API Key first!")
    else:
        # Ingest PDF
        with st.sidebar.status("⏳ Processing and embedding document...", expanded=True) as status:
            try:
                # Save uploaded file to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Ingest document
                num_chunks = ingest_pdf(tmp_path, collection)
                os.unlink(tmp_path)  # clean up temp file
                
                status.update(label=f"✅ Indexed {num_chunks} new chunks!", state="complete")
                st.balloons()
            except Exception as e:
                status.update(label=f"❌ Error: {str(e)}", state="error")

# Vector Database Status
st.sidebar.subheader("📊 Vector DB Status")
doc_count = collection.count()

st.sidebar.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{doc_count}</div>
    <div class="metric-label">Embedded Document Chunks</div>
</div>
""", unsafe_allow_html=True)

# DB Reset option
if st.sidebar.button("🗑️ Clear Vector Database", use_container_width=True):
    try:
        chroma_client.delete_collection(name=collection.name)
        # Re-fetch clean collection
        collection = get_collection(chroma_client)
        st.sidebar.success("Database cleared successfully!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error resetting database: {str(e)}")

# Architecture Details
st.sidebar.info(
    "**How Routing Works:**\n"
    "- If query is simple math (e.g. `23 * 78`), it goes to the **Calculator Agent**.\n"
    "- Otherwise, it goes to **Local PDF Retrieval**.\n"
    "- If no matching documents are found (similarity threshold > 1.0), it falls back to **Web Search**."
)

# ----------------- MAIN PLAYGROUND INTERFACE -----------------

# Page title header
st.markdown("""
<div class="title-container">
    <h1>🧠 Multi-Agent RAG Playground</h1>
    <p>A simple, clean interface to interact with a custom-routed retrieve-or-search assistant</p>
</div>
""", unsafe_allow_html=True)

# Verify API key
if not groq_api_key:
    st.info("⚠️ Please enter or configure your Groq API Key in the sidebar to get started.")
    st.stop()

# Initialize Chat Model
chat_model = ChatGroq(model='llama-3.1-8b-instant', api_key=groq_api_key)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show routing badge if it exists for assistant messages
        if msg["role"] == "assistant" and "route" in msg:
            route = msg["route"]
            badge_class = "badge-general"
            if route == "Calculator":
                badge_class = "badge-calculator"
                icon = "🧮"
            elif route == "Local PDF Retrieval":
                badge_class = "badge-retrieval"
                icon = "📄"
            elif route == "Web Search Fallback":
                badge_class = "badge-web"
                icon = "🌐"
            else:
                icon = "🤖"
                
            st.markdown(f'<span class="badge {badge_class}">{icon} {route}</span>', unsafe_allow_html=True)
            
        st.markdown(msg["content"])
        
        # Show expander with source logs/context if available
        if msg["role"] == "assistant" and "details" in msg:
            details = msg["details"]
            if "retrieved_chunks" in details and details["retrieved_chunks"]:
                with st.expander("📄 View Retrieved Document Context"):
                    for idx, chunk in enumerate(details["retrieved_chunks"]):
                        st.markdown(f"**Chunk {idx+1}:**")
                        st.markdown(f'<div class="source-box">{chunk}</div>', unsafe_allow_html=True)
            elif "web_search_results" in details and details["web_search_results"]:
                with st.expander("🌐 View Web Search Context"):
                    st.markdown(f'<div class="source-box">{details["web_search_results"]}</div>', unsafe_allow_html=True)

# User query input
if user_query := st.chat_input("Ask a question, do calculations (e.g. 54 * 23), or request info..."):
    
    # 1. Display User Message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 2. Run multi-agent pipeline
    with st.chat_message("assistant"):
        with st.spinner("Agent router processing query..."):
            try:
                answer, metadata, updated_state = run_agent(
                    query=user_query,
                    collection=collection,
                    chat_model=chat_model,
                    state=st.session_state.agent_state
                )
                
                # Update local session agent state
                st.session_state.agent_state = updated_state
                
                # Render Routing Badge
                route = metadata["route"]
                badge_class = "badge-general"
                icon = "🤖"
                if route == "Calculator":
                    badge_class = "badge-calculator"
                    icon = "🧮"
                elif route == "Local PDF Retrieval":
                    badge_class = "badge-retrieval"
                    icon = "📄"
                elif route == "Web Search Fallback":
                    badge_class = "badge-web"
                    icon = "🌐"
                
                st.markdown(f'<span class="badge {badge_class}">{icon} {route}</span>', unsafe_allow_html=True)
                
                # Render Answer
                st.markdown(answer)
                
                # Render Source logs
                details = metadata.get("details", {})
                if "retrieved_chunks" in details and details["retrieved_chunks"]:
                    with st.expander("📄 View Retrieved Document Context"):
                        for idx, chunk in enumerate(details["retrieved_chunks"]):
                            st.markdown(f"**Chunk {idx+1}:**")
                            st.markdown(f'<div class="source-box">{chunk}</div>', unsafe_allow_html=True)
                elif "web_search_results" in details and details["web_search_results"]:
                    with st.expander("🌐 View Web Search Context"):
                        st.markdown(f'<div class="source-box">{details["web_search_results"]}</div>', unsafe_allow_html=True)
                
                # Save assistant message to memory
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "route": route,
                    "details": details
                })
                
            except Exception as e:
                error_msg = f"An error occurred in the agent execution flow: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
