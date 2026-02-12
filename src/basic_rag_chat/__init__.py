"""Basic RAG Chat - Chat with PDF documents using Retrieval-Augmented Generation."""

__all__ = [
    "Chunk",
    "DocumentProcessor",
    "RAGChain",
    "RAGResponse",
    "SearchResult",
    "VectorStore",
]

__version__ = "0.1.0"

from .document_processor import Chunk, DocumentProcessor
from .rag_chain import RAGChain, RAGResponse
from .vector_store import SearchResult, VectorStore
