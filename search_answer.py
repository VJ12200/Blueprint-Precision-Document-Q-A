import time
import sys
import os, json
import random
import numpy as np, faiss, requests
from dotenv import load_dotenv
import os

load_dotenv() 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not os.getenv("OPENROUTER_API_KEY"):
    raise RuntimeError("OPENROUTER_API_KEY is missing; set env var or load .env file")

EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
# LLM_MODEL = "minimax/minimax-m2.5:free"  # OpenRouter chat model

def embed_query(query):
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type":"application/json"}
    data = {"model": EMBED_MODEL, "input": [query]}
    r = requests.post(url, headers=headers, json=data, timeout=90); r.raise_for_status()
    return r.json()["data"][0]["embedding"]

def llm_answer(context, question):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type":"application/json"}
    system = "Use only the provided context (top-k chunks). If not answerable, respond 'not in docs'."
    user = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    body = {
      "model": "openrouter/free",
      "messages": [
        {"role":"system","content":system},
        {"role":"user","content":user}
      ]
    }
  
    max_retries = 3
    for attempt in range(max_retries):
      try:
        r = requests.post(url, headers=headers, json=body, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
      except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
          if attempt < max_retries - 1:
            wait_time = (2 ** attempt) + random.uniform(1, 3)
            print(f"Rate limited. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
          else:
            raise Exception("Rate limit exceeded. Please try again later.")
        else:
          raise

def main(question, top_k=3):
    with open("data/chunks.json","r",encoding="utf-8") as f:
        chunks = json.load(f)
    index = faiss.read_index("data/faiss.index")

    q_emb = np.array([embed_query(question)], dtype="float32")
    _, ids = index.search(q_emb, top_k)
    ids = ids[0]
    top_chunks = [chunks[i] for i in ids]

    context = "\n\n".join(
      f"Source:{c['doc']} chunk_id:{c['chunk_id']}\n{c['text']}"
      for c in top_chunks
    )
    answer = llm_answer(context, question)

    result = {"answer": answer, "top_k_chunks": [
      {"idx": int(i), "score": None, "doc": chunks[i]["doc"], "text": chunks[i]["text"]}
      for i in ids
    ]}
    
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("Result saved to result.json")

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is in doc1?"
    main(q, top_k=3)