"""Vector store wrapper around ChromaDB for document embeddings."""

from dataclasses import dataclass

import chromadb

from basic_rag_chat.document_processor import Chunk

DEFAULT_TOP_K: int = 3


@dataclass
class SearchResult:
    """A single search hit returned from the vector store."""

    content: str
    source: str
    chunk_index: int
    score: float  # cosine similarity (1.0 = identical, 0.0 = orthogonal)


class VectorStore:
    """ChromaDB-backed store for adding and querying document embeddings."""

    def __init__(self, collection_name: str = "documents") -> None:
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Upsert document chunks into the collection.

        Args:
            chunks: Chunks to store; no-op if the list is empty.
        """
        if not chunks:
            return

        self._collection.add(
            documents=[c.content for c in chunks],
            metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
            ids=[f"{c.source}_{c.chunk_index}" for c in chunks],
        )

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        """Return the top-k most similar chunks to the query.

        Args:
            query: Natural-language search query.
            k: Maximum number of results to return.

        Returns:
            List of SearchResult ordered by descending similarity.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
        )

        search_results = []
        for i in range(len(results["documents"][0])):
            search_results.append(
                SearchResult(
                    content=results["documents"][0][i],
                    source=results["metadatas"][0][i]["source"],
                    chunk_index=results["metadatas"][0][i]["chunk_index"],
                    score=1 - results["distances"][0][i],
                )
            )
        return search_results

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self._collection.count()
