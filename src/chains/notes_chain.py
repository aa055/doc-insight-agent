from typing import Generator, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.core.llm import get_llm
from src.utils.prompts import NOTES_PROMPT


class NotesChain:
    """Chain for generating structured notes from documents."""

    def __init__(self):
        """Initialize the notes chain."""
        self.llm = get_llm(temperature=0.3)

    def generate_notes(
        self,
        doc_name: str,
        documents: List[Document],
    ) -> Generator[str, None, None]:
        """Generate structured notes for a document.

        Args:
            doc_name: Name of the document to create notes from
            documents: List of all document chunks

        Yields:
            Notes chunks as they are generated
        """
        # Filter documents by source
        doc_content = [
            doc for doc in documents if doc.metadata.get("doc_name") == doc_name
        ]

        if not doc_content:
            yield f"No content found for document: {doc_name}"
            return

        # Sort by page number if available
        doc_content.sort(key=lambda x: x.metadata.get("page", 0))

        # Combine content (limit to avoid token limits)
        combined_content = "\n\n".join(
            [doc.page_content for doc in doc_content[:25]]  # First 25 chunks
        )

        # Stream notes generation
        chain = NOTES_PROMPT | self.llm | StrOutputParser()

        for chunk in chain.stream({"doc_name": doc_name, "content": combined_content}):
            yield chunk
