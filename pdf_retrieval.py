import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder

# 1. Read PDF and split text into chunks
pdf_path = Path("data/enterprise_knowledge_base_demo.pdf")
reader = PdfReader(str(pdf_path))

chunks = []

for page_number, page in enumerate(reader.pages, 1):
    text = " ".join((page.extract_text() or "").split())

    start = 0
    while start < len(text):
        end = min(start + 700, len(text))
        chunk = text[start:end]

        if len(chunk.strip()) >= 80:
            chunks.append({
                "page": page_number,
                "text": chunk
            })

        if end == len(text):
            break

        start = end - 120

print(f"PDF pages: {len(reader.pages)}")
print(f"Text chunks: {len(chunks)}")

if not chunks:
    raise ValueError("No readable text found in the PDF.")

# 2. Create embeddings
print("\nLoading embedding model...")
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

texts = [item["text"] for item in chunks]

vectors = embedder.encode(
    texts,
    normalize_embeddings=True
).astype("float32")

# 3. Build FAISS index
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

# 4. Retrieve relevant chunks
question = "What should we check to maintain data quality?"

query_vector = embedder.encode(
    [question],
    normalize_embeddings=True
).astype("float32")

k = min(8, len(chunks))
scores, indices = index.search(query_vector, k)

candidates = [chunks[int(i)] for i in indices[0]]

print("\nQUESTION:", question)
print("\nBEFORE RERANK:")

for rank, item in enumerate(candidates, 1):
    print(f"{rank}. Page {item['page']}: {item['text'][:100]}...")

# 5. Rerank the retrieved chunks
print("\nLoading reranking model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    device="cpu"
)

pairs = [[question, item["text"]] for item in candidates]
rerank_scores = reranker.predict(pairs)

ranked = sorted(
    zip(candidates, rerank_scores),
    key=lambda pair: float(pair[1]),
    reverse=True
)

print("\nAFTER RERANK:")

for rank, (item, score) in enumerate(ranked[:5], 1):
    print(f"\nRank {rank} | Page {item['page']} | Score {float(score):.3f}")
    print(item["text"][:250])

print("\nPDF retrieval completed successfully!")
