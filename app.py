"""Local PDF RAG API: FAISS retrieval -> Cross-Encoder reranking -> vLLM.

Start your existing vLLM-Metal server on port 8000 first.
Then run this API with single-thread settings on port 8001.
"""

import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent
PDF_FILE = ROOT / "data" / "enterprise_knowledge_base_demo.pdf"
VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "mlx-community/Qwen2.5-3B-Instruct-4bit"

app = FastAPI(title="Enterprise Knowledge Assistant", version="0.1.0")


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def load_chunks():
    if not PDF_FILE.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_FILE}")

    reader = PdfReader(str(PDF_FILE))
    result = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        start = 0
        while start < len(text):
            end = min(start + 700, len(text))
            section = text[start:end].strip()
            if len(section) >= 80:
                result.append({"page": page_number, "text": section})
            if end == len(text):
                break
            start = end - 120
    if not result:
        raise ValueError("This PDF contains no extractable text.")
    print(f"Loaded {len(reader.pages)} PDF pages and {len(result)} text chunks.", flush=True)
    return result


print("Loading PDF and embedding model (first start may take a minute)...", flush=True)
chunks = load_chunks()
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
vectors = embedder.encode(
    [item["text"] for item in chunks], normalize_embeddings=True
).astype("float32")
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

print("Loading reranking model...", flush=True)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")
print("RAG retrieval system is ready.", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": len(chunks)}


@app.post("/ask")
def ask(payload: QuestionRequest):
    question = payload.question.strip()
    if len(question) < 3:
        raise HTTPException(status_code=422, detail="Please enter a longer question.")

    query_vector = embedder.encode(
        [question], normalize_embeddings=True
    ).astype("float32")
    count = min(8, len(chunks))
    _, positions = index.search(query_vector, count)
    candidates = [chunks[int(i)] for i in positions[0]]

    pairs = [[question, item["text"]] for item in candidates]
    rerank_scores = reranker.predict(pairs)
    ranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )[:3]

    context = "\n\n".join(
        f"[Page {item['page']}] {item['text']}" for item, _ in ranked
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions using ONLY the supplied document excerpts. "
                "If the answer is not in the excerpts, say you cannot find it. "
                "Answer briefly in English and cite the relevant page as [Page N]."
            ),
        },
        {
            "role": "user",
            "content": f"Document excerpts:\n{context}\n\nQuestion: {question}",
        },
    ]

    try:
        response = requests.post(
            VLLM_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 180,
            },
            timeout=180,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model request failed; check that vLLM is running on port 8000. {exc}",
        ) from exc

    sources = [
        {
            "page": item["page"],
            "rerank_score": round(float(score), 3),
            "excerpt": item["text"],
        }
        for item, score in ranked
    ]
    return {"question": question, "answer": answer, "sources": sources}
