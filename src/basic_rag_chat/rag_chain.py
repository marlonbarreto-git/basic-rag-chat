"""RAG chain: retrieval + generation with source citations."""

from dataclasses import dataclass, field

from openai import AsyncOpenAI

from basic_rag_chat.vector_store import DEFAULT_TOP_K, SearchResult, VectorStore

DEFAULT_MODEL: str = "gpt-4o-mini"
DEFAULT_TEMPERATURE: float = 0.3
DEFAULT_MAX_TOKENS: int = 1024


@dataclass
class RAGResponse:
    """Container for a RAG-generated answer and its metadata."""

    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    input_tokens: int = 0  # prompt tokens consumed by the LLM call
    output_tokens: int = 0  # completion tokens produced by the LLM call


class RAGChain:
    """Orchestrates retrieval from a VectorStore and generation via OpenAI."""

    def __init__(
        self,
        vector_store: VectorStore,
        openai_api_key: str,
        model: str = DEFAULT_MODEL,
        k: int = DEFAULT_TOP_K,
    ) -> None:
        self._store = vector_store
        self._client = AsyncOpenAI(api_key=openai_api_key)
        self._model = model
        self._k = k

    async def query(self, question: str) -> RAGResponse:
        """Retrieve relevant chunks and generate an answer.

        Args:
            question: The user's natural-language question.

        Returns:
            RAGResponse with the generated answer, sources, and token usage.
        """
        results = self._store.search(question, k=self._k)

        context = self._build_context(results)
        system_prompt = self._build_system_prompt(context)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        return RAGResponse(
            answer=response.choices[0].message.content,
            sources=results,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    def _build_context(self, results: list[SearchResult]) -> str:
        if not results:
            return "No relevant context found."

        sections = []
        for i, r in enumerate(results, 1):
            sections.append(f"[Source {i}: {r.source}]\n{r.content}")
        return "\n\n".join(sections)

    def _build_system_prompt(self, context: str) -> str:
        return (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Use ONLY the information from the context below to answer. "
            "If the context doesn't contain enough information, say so. "
            "Always cite which source(s) you used.\n\n"
            f"Context:\n{context}"
        )
