"""Document processing: loading and chunking text from PDFs."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    content: str
    source: str
    chunk_index: int


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_text(self, text: str, source: str) -> list[Chunk]:
        if not text.strip():
            return []

        splits = self._splitter.split_text(text)
        return [
            Chunk(content=split, source=source, chunk_index=i)
            for i, split in enumerate(splits)
        ]
