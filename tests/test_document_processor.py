"""Tests for document processing (loading, chunking)."""

import pytest

from basic_rag_chat.document_processor import DocumentProcessor, Chunk


class TestDocumentProcessor:
    def setup_method(self):
        self.processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)

    def test_chunk_text(self):
        text = "This is a test document. " * 50  # ~1250 chars
        chunks = self.processor.chunk_text(text, source="test.pdf")
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_has_metadata(self):
        text = "Hello world. This is a test document with some content."
        chunks = self.processor.chunk_text(text, source="doc.pdf")
        assert chunks[0].source == "doc.pdf"
        assert chunks[0].chunk_index == 0

    def test_chunk_content_not_empty(self):
        text = "Some meaningful content here that should be preserved in the chunk."
        chunks = self.processor.chunk_text(text, source="test.pdf")
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0

    def test_empty_text_returns_empty(self):
        chunks = self.processor.chunk_text("", source="empty.pdf")
        assert chunks == []

    def test_short_text_single_chunk(self):
        text = "Short text."
        chunks = self.processor.chunk_text(text, source="short.pdf")
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_custom_chunk_size(self):
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        text = "word " * 100  # 500 chars
        chunks = processor.chunk_text(text, source="test.pdf")
        assert len(chunks) > 5

    def test_chunk_dataclass(self):
        chunk = Chunk(content="test", source="file.pdf", chunk_index=0)
        assert chunk.content == "test"
        assert chunk.source == "file.pdf"
        assert chunk.chunk_index == 0
