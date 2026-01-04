import hashlib
from pathlib import Path
from typing import Dict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


class DocumentProcessor:
    """Handles PDF document loading and text splitting."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        """Initialize the document processor.

        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.processed_documents: Dict[str, List[Document]] = {}

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a single PDF and return documents with enhanced metadata.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of Document objects with metadata
        """
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Generate unique document ID and extract name
        doc_id = self._generate_doc_id(file_path)
        doc_name = Path(file_path).stem

        # Enhance metadata for each document
        for doc in documents:
            doc.metadata["doc_id"] = doc_id
            doc.metadata["doc_name"] = doc_name
            doc.metadata["file_path"] = file_path

        return documents

    def load_multiple_pdfs(self, file_paths: List[str]) -> List[Document]:
        """Load multiple PDFs and combine their documents.

        Args:
            file_paths: List of paths to PDF files

        Returns:
            Combined list of Document objects from all PDFs
        """
        all_documents = []
        for path in file_paths:
            docs = self.load_pdf(path)
            all_documents.extend(docs)
            self.processed_documents[path] = docs
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks while preserving metadata.

        Args:
            documents: List of Document objects to split

        Returns:
            List of chunked Document objects with preserved metadata
        """
        return self.text_splitter.split_documents(documents)

    def process_pdfs(self, file_paths: List[str]) -> List[Document]:
        """Load and split multiple PDFs in one operation.

        Args:
            file_paths: List of paths to PDF files

        Returns:
            List of chunked Document objects ready for embedding
        """
        documents = self.load_multiple_pdfs(file_paths)
        return self.split_documents(documents)

    def get_document_names(self) -> List[str]:
        """Get list of processed document names.

        Returns:
            List of document names (without extension)
        """
        return [Path(path).stem for path in self.processed_documents.keys()]

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate a unique ID for document tracking.

        Args:
            file_path: Path to the document

        Returns:
            8-character hash ID
        """
        return hashlib.md5(file_path.encode()).hexdigest()[:8]

    def clear(self):
        """Clear all processed documents."""
        self.processed_documents.clear()
