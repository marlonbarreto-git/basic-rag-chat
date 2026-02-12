"""Vector store wrapper around ChromaDB for document embeddings."""

from dataclasses import dataclass

import chromadb

from basic_rag_chat.document_processor import Chunk


@dataclass
class SearchResult:
    content: str
    source: str
    chunk_index: int
    score: float


class VectorStore:
    def __init__(self, collection_name: str = "documents"):
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        self._collection.add(
            documents=[c.content for c in chunks],
            metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
            ids=[f"{c.source}_{c.chunk_index}" for c in chunks],
        )

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
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
                    score=1 - results["distances"][0][i],  # Convert distance to similarity
                )
            )
        return search_results

    def count(self) -> int:
        return self._collection.count()
