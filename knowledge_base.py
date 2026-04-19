"""
knowledge_base.py

Upgraded RAG implementation using vector embeddings and semantic search.

How it works:
1. Loads the knowledge base text file
2. Splits it into chunks (paragraphs)
3. Generates vector embeddings for each chunk using sentence-transformers
4. Saves embeddings to disk so they don't need to be recalculated every restart
5. When a user asks a question, embeds the query and finds the most
   semantically similar chunks using cosine similarity
6. Returns only the relevant chunks, not the entire knowledge base

Why this is better than injecting the full knowledge base:
- Scales to any size document (100 pages, not just 5)
- Reduces token usage by approximately 90% per message
- More accurate answers, less irrelevant context confusing the model
- This is how production RAG systems actually work

Tech stack:
- sentence-transformers: generates embeddings locally (no API cost)
- PyTorch: powers the underlying model (torch is a dependency)
- cosine similarity: measures how similar two vectors are (0-1 scale)
- numpy: handles vector math efficiently
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

EMBEDDINGS_CACHE = "embeddings_cache.json"
CHUNK_SIZE = 150
TOP_K = 4
MODEL_NAME = "all-MiniLM-L6-v2"

print(f"[Vector Search] Loading embedding model: {MODEL_NAME}")
embedder = SentenceTransformer(MODEL_NAME)
print(f"[Vector Search] Model loaded successfully")

_chunks = []
_embeddings = None


def chunk_text(text, chunk_size=CHUNK_SIZE):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        word_count = len(para.split())
        if current_word_count + word_count > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-1:]
            current_word_count = len(current_chunk[0].split()) if current_chunk else 0
        current_chunk.append(para)
        current_word_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"[Vector Search] Knowledge base split into {len(chunks)} chunks")
    return chunks


def generate_and_cache_embeddings(chunks):
    if os.path.isfile(EMBEDDINGS_CACHE):
        with open(EMBEDDINGS_CACHE, "r") as f:
            cache = json.load(f)
        if cache.get("chunks") == chunks:
            print(f"[Vector Search] Loading embeddings from cache")
            embeddings = np.array(cache["embeddings"], dtype=np.float32)
            return embeddings

    print(f"[Vector Search] Generating embeddings for {len(chunks)} chunks...")
    embeddings = embedder.encode(chunks, convert_to_numpy=True).astype(np.float32)

    cache = {
        "chunks": chunks,
        "embeddings": embeddings.tolist()
    }
    with open(EMBEDDINGS_CACHE, "w") as f:
        json.dump(cache, f)

    print(f"[Vector Search] Embeddings generated and cached. Shape: {embeddings.shape}")
    return embeddings


def load_knowledge_base(filepath):
    global _chunks, _embeddings

    if not os.path.isfile(filepath):
        print(f"[Vector Search] Warning: {filepath} not found, using default")
        default = "AI Automation Agency. Services: AI chatbots, voice bots, CRM automation. Book a free 30-minute discovery call."
        _chunks = [default]
        _embeddings = embedder.encode([default], convert_to_numpy=True).astype(np.float32)
        return default

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"[Vector Search] Loaded {len(content)} characters from {filepath}")
    _chunks = chunk_text(content)
    _embeddings = generate_and_cache_embeddings(_chunks)
    return content


def semantic_search(query, top_k=TOP_K):
    global _chunks, _embeddings

    if not _chunks or _embeddings is None:
        return "Knowledge base not loaded."

    query_embedding = embedder.encode(query, convert_to_numpy=True).astype(np.float32)
    similarities = util.cos_sim(query_embedding, _embeddings)[0].numpy()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_indices = sorted(top_indices)
    relevant_chunks = [_chunks[i] for i in top_indices]
    result = "\n\n".join(relevant_chunks)
    scores = [f"{similarities[i]:.2f}" for i in top_indices]
    print(f"[Vector Search] Retrieved {len(relevant_chunks)} chunks. Similarity scores: {scores}")
    return result
