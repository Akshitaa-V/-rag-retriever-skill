"""
Local RAG (Retrieval-Augmented Generation) Retriever
=====================================================

A fully local, offline implementation of the *retrieval* half of a RAG
pipeline. No API keys, no internet access, and no paid services are used.

Pipeline stages demonstrated:
    1. INDEXING    -> load .txt notes, split into chunks, vectorize (TF-IDF)
    2. RETRIEVAL   -> embed the query, rank chunks by cosine similarity
    3. CONTEXT     -> assemble the top-k chunks into a clearly structured
                      context block (this is the "context engineering" step:
                      deciding what goes in, in what order, and how it's
                      formatted, before it would be handed to a model)

Note: This script stops at context assembly. It does NOT call a generative
model, since that would require a paid API key. The printed "ASSEMBLED
CONTEXT" block is exactly what would be inserted into a prompt sent to an
LLM in a full RAG system.

Usage:
    python rag_retriever.py "What is context engineering?"
    python rag_retriever.py "How does an MCP server expose tools?" --top_k 2
"""

import argparse
import glob
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_documents(notes_dir: str) -> list[dict]:
    """Load every .txt file in notes_dir into memory."""
    docs = []
    for path in sorted(glob.glob(os.path.join(notes_dir, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    return docs


def chunk_text(text: str, max_words: int = 60) -> list[str]:
    """
    Split text into paragraph-based chunks, then further split any
    paragraph longer than max_words into smaller word-count chunks.
    This is a simple but realistic chunking strategy.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            chunks.append(para)
        else:
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i : i + max_words]))
    return chunks


def build_index(notes_dir: str):
    """Indexing stage: load docs, chunk them, and fit a TF-IDF vectorizer."""
    documents = load_documents(notes_dir)
    chunk_records = []
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            chunk_records.append({"source": doc["source"], "chunk": chunk})

    corpus = [r["chunk"] for r in chunk_records]
    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_vectors = vectorizer.fit_transform(corpus)

    return chunk_records, vectorizer, chunk_vectors


def retrieve(query: str, chunk_records, vectorizer, chunk_vectors, top_k: int = 3):
    """Retrieval stage: embed the query and rank chunks by cosine similarity."""
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, chunk_vectors).flatten()

    ranked_indices = scores.argsort()[::-1][:top_k]
    results = []
    for idx in ranked_indices:
        results.append(
            {
                "score": float(scores[idx]),
                "source": chunk_records[idx]["source"],
                "chunk": chunk_records[idx]["chunk"],
            }
        )
    return results


def assemble_context(results: list[dict]) -> str:
    """
    Context engineering stage: format retrieved chunks into a structured
    block, the way they would be inserted into an LLM prompt. Each chunk
    is labeled with its source and relevance score so the model (or a
    human reviewer) can judge how much to trust each piece of context.
    """
    lines = ["--- RETRIEVED CONTEXT (highest relevance first) ---"]
    for i, r in enumerate(results, start=1):
        lines.append(f"\n[Chunk {i} | source: {r['source']} | relevance: {r['score']:.3f}]")
        lines.append(r["chunk"])
    lines.append("\n--- END CONTEXT ---")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Local RAG retriever (no API key needed).")
    parser.add_argument("query", type=str, help="The question to retrieve context for.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--notes_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "notes"),
        help="Folder containing .txt notes to index.",
    )
    args = parser.parse_args()

    print(f"\nQuery: {args.query}\n")
    print("Stage 1/3 — Indexing notes...")
    chunk_records, vectorizer, chunk_vectors = build_index(args.notes_dir)
    print(f"  Indexed {len(chunk_records)} chunks from notes in '{args.notes_dir}'.")

    print("\nStage 2/3 — Retrieving relevant chunks...")
    results = retrieve(args.query, chunk_records, vectorizer, chunk_vectors, top_k=args.top_k)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['source']}")

    print("\nStage 3/3 — Assembling context for the model prompt...\n")
    context_block = assemble_context(results)
    print(context_block)

    print(
        "\n(In a full RAG system, the block above would now be inserted into "
        "a prompt and sent to a generative model. This script "
        "stops here since calling a live model requires a paid API key.)"
    )


if __name__ == "__main__":
    main()
