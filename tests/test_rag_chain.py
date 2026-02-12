"""Tests for the RAG chain (retrieval + generation)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from basic_rag_chat.rag_chain import RAGChain, RAGResponse
from basic_rag_chat.vector_store import SearchResult


class TestRAGChain:
    def _make_search_results(self) -> list[SearchResult]:
        return [
            SearchResult(
                content="Python was created by Guido van Rossum in 1991.",
                source="python.pdf",
                chunk_index=0,
                score=0.95,
            ),
            SearchResult(
                content="Python is widely used in AI and data science.",
                source="python.pdf",
                chunk_index=1,
                score=0.85,
            ),
        ]

    @pytest.mark.asyncio
    async def test_query_returns_response(self):
        mock_store = MagicMock()
        mock_store.search.return_value = self._make_search_results()

        with patch("basic_rag_chat.rag_chain.AsyncOpenAI") as mock_openai_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Python was created by Guido in 1991."
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 20
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            chain = RAGChain(vector_store=mock_store, openai_api_key="test-key")
            result = await chain.query("Who created Python?")

        assert isinstance(result, RAGResponse)
        assert "Guido" in result.answer
        assert len(result.sources) == 2
        assert result.sources[0].source == "python.pdf"

    @pytest.mark.asyncio
    async def test_query_with_no_context(self):
        mock_store = MagicMock()
        mock_store.search.return_value = []

        with patch("basic_rag_chat.rag_chain.AsyncOpenAI") as mock_openai_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "I don't have enough context."
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 10
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            chain = RAGChain(vector_store=mock_store, openai_api_key="test-key")
            result = await chain.query("Something unknown")

        assert result.sources == []

    @pytest.mark.asyncio
    async def test_query_passes_context_to_llm(self):
        mock_store = MagicMock()
        mock_store.search.return_value = self._make_search_results()

        with patch("basic_rag_chat.rag_chain.AsyncOpenAI") as mock_openai_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Answer"
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 10
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            chain = RAGChain(vector_store=mock_store, openai_api_key="test-key")
            await chain.query("Who created Python?")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        # System message should contain the context
        system_msg = messages[0]["content"]
        assert "Guido van Rossum" in system_msg
