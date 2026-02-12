# Basic RAG Chat

[![CI](https://github.com/marlonbarreto-git/basic-rag-chat/actions/workflows/ci.yml/badge.svg)](https://github.com/marlonbarreto-git/basic-rag-chat/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Chat with documents using Retrieval-Augmented Generation with source citations.

## Overview

Basic RAG Chat implements a complete RAG pipeline: chunk documents with configurable text splitting, store embeddings in ChromaDB, and answer questions using retrieved context with OpenAI. The system provides source citations for every answer, letting users verify information against the original documents.

## Architecture

```
Documents (text)
  |
  v
DocumentProcessor (RecursiveCharacterTextSplitter)
  |
  v
Chunks (content + source + index)
  |
  v
VectorStore (ChromaDB, cosine similarity)
  |
  v
RAGChain.query(question)
  |
  +---> VectorStore.search(question, k=3)
  |        |
  |        v
  |     Top-k chunks with similarity scores
  |
  +---> OpenAI Chat Completion (context + question)
  |
  v
RAGResponse (answer + source citations + token usage)
```

## Features

- Document chunking with configurable size and overlap
- Vector storage and retrieval via ChromaDB with cosine similarity
- Configurable top-k retrieval results
- Context-grounded answers with source citations
- Token usage tracking per query
- Automatic system prompt construction with retrieved context

## Tech Stack

- Python 3.11+
- LangChain Text Splitters
- ChromaDB (vector store)
- OpenAI SDK
- FastAPI + Uvicorn
- Pydantic

## Quick Start

```bash
git clone https://github.com/marlonbarreto-git/basic-rag-chat.git
cd basic-rag-chat
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export OPENAI_API_KEY=your-key
pytest
```

## Project Structure

```
src/basic_rag_chat/
  __init__.py
  document_processor.py  # Text chunking with RecursiveCharacterTextSplitter
  vector_store.py        # ChromaDB wrapper for embedding storage and search
  rag_chain.py           # Retrieval + generation pipeline with citations
tests/
  test_document_processor.py
  test_vector_store.py
  test_rag_chain.py
```

## Testing

```bash
pytest -v --cov=src/basic_rag_chat
```

16 tests covering document chunking, vector store operations, and RAG chain query flow.

## License

MIT