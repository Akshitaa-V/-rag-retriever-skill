---
name: rag-retriever
description: Use this skill when the user wants to retrieve relevant context from a local folder of .txt notes before answering a question - e.g. "search my notes for X", "what do my notes say about Y", or any task that should be grounded in a local knowledge base instead of general knowledge.
---

# RAG Retriever Skill

This skill handles the retrieval half of a RAG pipeline over a local folder of `.txt` notes. No external API calls.

## When to use it

- The user's question should be answered from their own notes/documents rather than general knowledge.
- The user asks to "search my notes," "look up X in my files," or similar.

## How to use it

1. Take the user's question and the path to the notes folder (defaults to `notes/` next to this skill).
2. Run:
   ```
   python rag_retriever.py "<question>" --notes_dir <path_to_notes> --top_k 3
   ```
3. The script prints a context block with the top-k chunks, each labeled with source file and relevance score.
4. Use that context to ground the answer, and mention which file each fact came from.

## What it doesn't do

- No call to an external/paid LLM — it only does retrieval and context assembly. Generation is left to whatever model is running this skill.
- No live database or remote service connection.

## On MCP

This skill is local-only by design. Wrapping `retrieve()` as an MCP server tool (e.g. `search_notes(query)`) would let any MCP-compatible client use it - that's a reasonable next step, not something this skill currently does.
