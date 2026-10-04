# rag-retriever-skill

A small local RAG (Retrieval-Augmented Generation) retriever, packaged as a reusable agent skill.

I built this to get hands-on with the retrieval side of a RAG pipeline and with the skill format that AI assistants use to load reusable capabilities. It runs fully offline: no API keys, no paid services, nothing to sign up for.

## What it does

Point it at a folder of `.txt` notes and ask it a question. It will:

1. Split the notes into chunks
2. Turn the chunks and the question into TF-IDF vectors
3. Rank the chunks by cosine similarity to the question
4. Print the top matches as a labelled context block

That context block is what you would hand to a language model as the "augmented" part of RAG. I stopped there on purpose: generating an answer would mean calling a paid LLM API, which I wanted to avoid for this project.

## Packaged as a skill

Instead of shipping a bare script, the retriever comes with a `SKILL.md` under `skills/rag-retriever/` that describes when it should be used and how to call it. Any assistant that supports the skill format can load the folder and use the retriever to ground its answers in local documents.

## About MCP

There is no MCP server in this repository; that is a separate piece of work (see [mcp-notes-server](https://github.com/Akshitaa-V/mcp-notes-server)). The retrieval logic (`retrieve()` in `rag_retriever.py`) is written so it can be wrapped as an MCP tool, and the SKILL.md notes this as a next step.

## Project structure

```
rag-retriever-skill/
├── rag_retriever.py
├── notes/
│   ├── genai_concepts.txt
│   └── skills_hooks_and_mcp.txt
└── skills/
    └── rag-retriever/
        └── SKILL.md
```

## Running it

```bash
pip install scikit-learn
python rag_retriever.py "How does an MCP server expose tools?" --top_k 2
```

Sample output:

```
Stage 1/3 — Indexing notes...
  Indexed 15 chunks from notes in 'notes'.

Stage 2/3 — Retrieving relevant chunks...
  [0.620] skills_hooks_and_mcp.txt
  [0.337] skills_hooks_and_mcp.txt

Stage 3/3 — Assembling context for the model prompt...

--- RETRIEVED CONTEXT (highest relevance first) ---

[Chunk 1 | source: skills_hooks_and_mcp.txt | relevance: 0.620]
The Model Context Protocol, or MCP, is an open protocol that standardizes how applications provide context...
```

## What I would add next

- Wrap `retrieve()` as an MCP server tool
- Swap TF-IDF for sentence embeddings so retrieval is semantic, not just keyword-based
- Add a generation step on top of the retrieved context
