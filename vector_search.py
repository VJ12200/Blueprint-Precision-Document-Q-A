import numpy as np, faiss, os

def load_data():
    import json
    with open("data/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load("data/embeddings.npy")
    return chunks, embeddings

def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d) 
    index.add(embeddings)
    faiss.write_index(index, "data/faiss.index")
    print("faiss index built, n_items=", index.ntotal)
    return index

if __name__ == "__main__":
    _, embs = load_data()
    build_faiss_index(embs)