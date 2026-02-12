"""RAG chain: retrieval + generation with source citations."""

from dataclasses import dataclass, field

from openai import AsyncOpenAI

from basic_rag_chat.vector_store import SearchResult, VectorStore


@dataclass
class RAGResponse:
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class RAGChain:
    def __init__(
        self,
        vector_store: VectorStore,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
        k: int = 3,
    ):
        self._store = vector_store
        self._client = AsyncOpenAI(api_key=openai_api_key)
        self._model = model
        self._k = k

    async def query(self, question: str) -> RAGResponse:
        results = self._store.search(question, k=self._k)

        context = self._build_context(results)
        system_prompt = self._build_system_prompt(context)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=1024,
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
