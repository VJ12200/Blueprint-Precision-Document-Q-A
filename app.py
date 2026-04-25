import streamlit as st
import json
import numpy as np
import faiss
from search_answer import embed_query, llm_answer
from evaluation import evaluate_grounding
import os


st.set_page_config(page_title="Blueprint;", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {
        background-color: #0e1117;  /* Dark background */
    }
    
    /* Optional: Change stApp for full page background */
    .stApp {
        background-color: #0e1117;
    }
                   
    .chat-message {
        padding: 1rem;
        margin: 1.5rem 0.5rem;
        border-radius: 0.5rem;
    }
    .user-message {
        background-color: #23262b;  /* Purple */
        color: white;
        text-align: right;
        padding: 1rem;
        margin: 1rem 5rem;
        border-radius: 0.55rem
    }
    .assistant-message {
        background-color: #313335;  /* Orange */
        color: white;
    }
    .chunk-info {
        background-color: #3b3b3c;
        padding: 2.5px;
        border-radius: 2.5px;
        margin: 2.5px 0;
        border-left: 2px solid #3b3b3c;
    }
    .score-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px 0;
    }
    .score-high {background-color: #4caf50; color: white;}
    .score-medium {background-color: #ff9800; color: white;}
    .score-low {background-color: #f44336; color: white;}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "results" not in st.session_state:
    st.session_state.results = {}


@st.cache_resource
def load_rag_data():
    with open("data/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    index = faiss.read_index("data/faiss.index")
    return chunks, index

chunks, index = load_rag_data()

def search_and_answer(question, top_k=3):
    
    
    q_emb = np.array([embed_query(question)], dtype="float32")
    _, ids = index.search(q_emb, top_k)
    ids = ids[0]
    top_chunks = [chunks[i] for i in ids]
    
    
    context = "\n\n".join(
        f"Source:{c['doc']} chunk_id:{c['chunk_id']}\n{c['text']}"
        for c in top_chunks
    )
    answer = llm_answer(context, question)
    
    
    chunks_text = [c["text"] for c in top_chunks]
    grounding_score, unsupported = evaluate_grounding(answer, chunks_text)
    
    return {
        "answer": answer,
        "chunks": top_chunks,
        "ids": ids,
        "grounding_score": grounding_score,
        "unsupported_claims": unsupported
    }


st.title("Ask Blueprint")
st.markdown("Hello! I am Blueprint. Ask me anything. I'll do my best to help you out! 😊")


with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Retrieved Chunks (top-k)", min_value=1, max_value=10, value=3)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.results = {}
        st.rerun()

# Chat container
chat_container = st.container()

# Input container
col1, col2 = st.columns([15, 7])
with col1:
    user_input = st.chat_input("Ask Blueprint")
with col2:
    pass  

# Process user input
if user_input:
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("🔍 Searching and generating answer..."):
        result = search_and_answer(user_input, top_k=top_k)
    
    st.session_state.results[user_input] = result
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "metadata": {
            "chunks": result["chunks"],
            "ids": result["ids"],
            "grounding_score": result["grounding_score"],
            "unsupported_claims": result["unsupported_claims"]
        }
    })
    
    st.rerun()

with chat_container:
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><b></b> {message["content"]}</div>', unsafe_allow_html=True)
        else:
           
            col1, col2 = st.columns([20, 1])
            
            with col1:
                st.markdown(f'<div class="chat-message assistant-message"><b>🏠 </b>\n\n{message["content"]}</div>', unsafe_allow_html=True)
                
                if "metadata" in message:
                    metadata = message["metadata"]
                    
                    score = metadata["grounding_score"]
                    if score >= 0.7:
                        score_class = "score-high"
                        score_text = "✓ High Grounding"
                    elif score >= 0.4:
                        score_class = "score-medium"
                        score_text = "⚠ Medium Grounding"
                    else:
                        score_class = "score-low"
                        score_text = "✗ Low Grounding"
                    
                    st.markdown(f'<span class="score-badge {score_class}" title="Grounding Score: {score:.1%}. Unsupported: {len(metadata["unsupported_claims"])}">{score_text} ({score:.0%})</span>', unsafe_allow_html=True)
                    
                    # Retrieved chunks in expandable section
                    with st.expander("📃 Retrieved Documents"):
                        for idx, chunk in enumerate(metadata["chunks"]):
                            st.markdown(f'<div class="chunk-info">', unsafe_allow_html=True)
                            st.markdown(f"**Source:** {chunk['doc']} | **Chunk ID:** {chunk['chunk_id']}")
                            st.markdown(f"**Content:** {chunk['text'][:200]}..." if len(chunk['text']) > 200 else f"**Content:** {chunk['text']}")
                            st.markdown(f'</div>', unsafe_allow_html=True)
