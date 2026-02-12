"""Tests for the vector store (ChromaDB wrapper)."""



from basic_rag_chat.document_processor import Chunk
from basic_rag_chat.vector_store import VectorStore, SearchResult


class TestVectorStore:
    def test_create_store(self):
        store = VectorStore(collection_name="test")
        assert store is not None

    def test_add_chunks(self):
        store = VectorStore(collection_name="test_add")
        chunks = [
            Chunk(content="Python is a programming language.", source="test.pdf", chunk_index=0),
            Chunk(content="RAG stands for Retrieval-Augmented Generation.", source="test.pdf", chunk_index=1),
        ]
        store.add_chunks(chunks)
        assert store.count() == 2

    def test_search_returns_results(self):
        store = VectorStore(collection_name="test_search")
        chunks = [
            Chunk(content="Python is great for data science.", source="doc.pdf", chunk_index=0),
            Chunk(content="JavaScript is used for web development.", source="doc.pdf", chunk_index=1),
            Chunk(content="Go is excellent for backend systems.", source="doc.pdf", chunk_index=2),
        ]
        store.add_chunks(chunks)

        results = store.search("data science language", k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].content is not None
        assert results[0].source is not None
        assert results[0].score is not None

    def test_search_empty_store(self):
        store = VectorStore(collection_name="test_empty_search")
        results = store.search("anything", k=3)
        assert results == []

    def test_search_result_has_metadata(self):
        store = VectorStore(collection_name="test_metadata")
        chunks = [
            Chunk(content="Test content about AI.", source="ai.pdf", chunk_index=5),
        ]
        store.add_chunks(chunks)

        results = store.search("AI", k=1)
        assert results[0].source == "ai.pdf"
        assert results[0].chunk_index == 5

    def test_count_empty(self):
        store = VectorStore(collection_name="test_count_empty")
        assert store.count() == 0
