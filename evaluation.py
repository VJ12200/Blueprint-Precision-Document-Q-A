import json, sys, re

def evaluate_grounding(answer, chunks_text):
    sentences = re.split(r'[.!?]+', answer)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    unsupported = []
    for sent in sentences:
        words = set(sent.lower().split())
        combined = " ".join(chunks_text).lower()
        coverage = len(words & set(combined.split())) / len(words) if words else 1.0
        if coverage < 0.4:
            unsupported.append(sent)
    
    grounding_ratio = 1.0 - (len(unsupported) / len(sentences)) if sentences else 1.0
    return grounding_ratio, unsupported

if __name__ == "__main__":
    with open("result.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    answer = data["answer"] 
    chunks = [c["text"] for c in data["top_k_chunks"]]

    grounding, unsupported = evaluate_grounding(answer, chunks)

    evaluation = {
        "grounding_score": grounding,
        "has_hallucinations": len(unsupported) > 0,
        "unsupported_claims": unsupported
    }

    data["evaluation"] = evaluation
    print(json.dumps(data, ensure_ascii=False, indent=2))