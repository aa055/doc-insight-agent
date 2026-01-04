from typing import Dict, Generator, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.core.llm import get_llm
from src.utils.prompts import ALL_DOCS_SUMMARY_PROMPT, SUMMARIZE_PROMPT


class SummarizationChain:
    """Chain for summarizing documents."""

    def __init__(self):
        """Initialize the summarization chain."""
        self.llm = get_llm(temperature=0.5)

    def summarize_document(
        self,
        doc_name: str,
        documents: List[Document],
    ) -> Generator[str, None, None]:
        """Summarize a specific document.

        Args:
            doc_name: Name of the document to summarize
            documents: List of all document chunks

        Yields:
            Summary chunks as they are generated
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
            [doc.page_content for doc in doc_content[:30]]  # First 30 chunks
        )

        # Stream summarization
        chain = SUMMARIZE_PROMPT | self.llm | StrOutputParser()

        for chunk in chain.stream({"doc_name": doc_name, "content": combined_content}):
            yield chunk

    def summarize_all_documents(
        self,
        documents: List[Document],
    ) -> Generator[str, None, None]:
        """Summarize all uploaded documents together.

        Args:
            documents: List of all document chunks

        Yields:
            Summary chunks as they are generated
        """
        if not documents:
            yield "No documents available to summarize."
            return

        # Group by document
        doc_groups: Dict[str, List[Document]] = {}
        for doc in documents:
            doc_name = doc.metadata.get("doc_name", "Unknown")
            if doc_name not in doc_groups:
                doc_groups[doc_name] = []
            doc_groups[doc_name].append(doc)

        # Format content with document labels
        formatted_parts = []
        for doc_name, chunks in doc_groups.items():
            # Sort by page
            chunks.sort(key=lambda x: x.metadata.get("page", 0))
            # Take first few chunks from each document
            sample_content = "\n".join([c.page_content for c in chunks[:10]])
            formatted_parts.append(f"### {doc_name}\n{sample_content}")

        combined = "\n\n---\n\n".join(formatted_parts)

        # Stream summarization
        chain = ALL_DOCS_SUMMARY_PROMPT | self.llm | StrOutputParser()

        for chunk in chain.stream({"content": combined}):
            yield chunk
