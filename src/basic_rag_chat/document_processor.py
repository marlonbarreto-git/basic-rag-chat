"""Document processing: loading and chunking text from PDFs."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 100


@dataclass
class Chunk:
    """A single text chunk extracted from a source document."""

    content: str
    source: str  # identifier of the originating document (e.g. filename)
    chunk_index: int


class DocumentProcessor:
    """Splits raw text into overlapping chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_text(self, text: str, source: str) -> list[Chunk]:
        """Split text into chunks, tagging each with its source.

        Args:
            text: The raw text to split.
            source: Identifier for the originating document.

        Returns:
            Ordered list of Chunk objects (empty if text is blank).
        """
        if not text.strip():
            return []

        splits = self._splitter.split_text(text)
        return [
            Chunk(content=split, source=source, chunk_index=i)
            for i, split in enumerate(splits)
        ]
