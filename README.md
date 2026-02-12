# basic-rag-chat

Chat with PDF documents using Retrieval-Augmented Generation. Upload PDFs, chunk them, embed in ChromaDB, and query with source citations.

## Features

- **PDF ingestion**: Load and chunk PDF documents with configurable strategies
- **Vector search**: ChromaDB with cosine similarity for semantic retrieval
- **Source citations**: Every answer includes the exact chunks used
- **RAG chain**: Retrieval + LLM generation with context-aware prompts
- **Configurable**: Chunk size, overlap, top-k results, LLM model

## Architecture

```
basic_rag_chat/
├── document_processor.py  # Text chunking with RecursiveCharacterTextSplitter
├── vector_store.py        # ChromaDB wrapper for embeddings and search
└── rag_chain.py           # Retrieval + OpenAI generation with citations
```

## How It Works

```
PDF → Chunk (RecursiveCharacterTextSplitter)
    → Embed (ChromaDB default embeddings)
    → Store (ChromaDB)

Query → Search (cosine similarity, top-k)
      → Build context from retrieved chunks
      → LLM generates answer with citations
      → Return answer + source references
```

## Quick Start

```bash
uv sync
export OPENAI_API_KEY="sk-..."

# Python usage
from basic_rag_chat.document_processor import DocumentProcessor
from basic_rag_chat.vector_store import VectorStore
from basic_rag_chat.rag_chain import RAGChain

# 1. Process document
processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
chunks = processor.chunk_text(pdf_text, source="document.pdf")

# 2. Store embeddings
store = VectorStore(collection_name="my_docs")
store.add_chunks(chunks)

# 3. Query
chain = RAGChain(vector_store=store, openai_api_key="sk-...")
response = await chain.query("What is this document about?")
print(response.answer)
print(response.sources)  # Source citations
```

## Development

```bash
uv sync --all-extras
uv run pytest tests/ -v
```

## Roadmap

- **v2**: Conversation memory, multi-PDF support, RAGAS evaluation
- **v3**: Hybrid search (semantic + keyword), reranking, streaming

## License

MIT
