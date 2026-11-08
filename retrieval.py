from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# 1. Read documents
text = Path("data/company_knowledge.txt").read_text(encoding="utf-8")
documents = []

for block in text.split("\n\n"):
    block = block.strip()
    if block.startswith("Document "):
        documents.append(block.replace("\n", " "))

print(f"Loaded {len(documents)} documents.")

# 2. Create embeddings
print("Loading embedding model...")
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

vectors = embedding_model.encode(
    documents, normalize_embeddings=True
).astype("float32")

# 3. Build FAISS index
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

# 4. Search documents
question = "What should we check to ensure data quality?"
query_vector = embedding_model.encode(
    [question], normalize_embeddings=True
).astype("float32")

scores, indices = index.search(query_vector, len(documents))
retrieved = [documents[i] for i in indices[0]]

print("\nQuestion:", question)
print("\nBefore Rerank:")
for i, doc in enumerate(retrieved, 1):
    print(f"{i}. {doc.split(':')[0]}")

# 5. Rerank retrieved documents
print("\nLoading reranking model...")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    device="cpu"
)

pairs = [[question, doc] for doc in retrieved]
rerank_scores = reranker.predict(pairs)

ranked = sorted(
    zip(retrieved, rerank_scores),
    key=lambda item: float(item[1]),
    reverse=True
)

print("\nAfter Rerank:")
for i, (doc, score) in enumerate(ranked, 1):
    print(f"{i}. {doc.split(':')[0]} | Score: {float(score):.3f}")

print("\nRetrieval completed successfully!")
