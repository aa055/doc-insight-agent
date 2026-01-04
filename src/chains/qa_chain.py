from typing import Dict, Generator, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.config import settings
from src.core.llm import get_llm
from src.core.vector_store import VectorStoreManager
from src.utils.prompts import QA_PROMPT


class QAChain:
    """Question-answering chain with retrieval and conversation memory."""

    def __init__(self, vector_store_manager: VectorStoreManager):
        """Initialize the QA chain.

        Args:
            vector_store_manager: Vector store manager instance
        """
        self.vector_store = vector_store_manager
        self.llm = get_llm(temperature=0.3)
        self.chat_history: List[Tuple[str, str]] = []

    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents with source attribution.

        Args:
            docs: List of retrieved documents

        Returns:
            Formatted string with source information
        """
        if not docs:
            return ""

        formatted = []
        for doc in docs:
            source = doc.metadata.get("doc_name", "Unknown")
            page = doc.metadata.get("page", "N/A")
            page_num = page + 1 if isinstance(page, int) else page
            formatted.append(f"[Source: {source}, Page: {page_num}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def _get_chat_history_messages(self) -> List:
        """Get formatted chat history for the prompt.

        Returns:
            List of message objects for the prompt
        """
        messages = []
        # Keep last 10 exchanges to avoid token limits
        for role, content in self.chat_history[-10:]:
            if role == "human":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        return messages

    def query(
        self,
        question: str,
        filter_source: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Process a question and stream the response.

        Args:
            question: User's question
            filter_source: Optional document name to filter retrieval

        Yields:
            Response chunks as they are generated
        """
        # Build filter if specific source requested
        filter_dict = {"doc_name": filter_source} if filter_source else None

        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(
            question,
            k=settings.RETRIEVAL_K,
            filter_dict=filter_dict,
        )

        # Check if we have relevant content
        if not docs:
            response = "I could not find relevant information in the uploaded documents for your question."
            yield response
            self.chat_history.append(("human", question))
            self.chat_history.append(("assistant", response))
            return

        # Format context
        context = self._format_docs(docs)

        # Build chain with LCEL
        chain = QA_PROMPT | self.llm | StrOutputParser()

        # Stream response
        full_response = ""
        for chunk in chain.stream(
            {
                "context": context,
                "chat_history": self._get_chat_history_messages(),
                "question": question,
            }
        ):
            full_response += chunk
            yield chunk

        # Update chat history
        self.chat_history.append(("human", question))
        self.chat_history.append(("assistant", full_response))

    def get_source_documents(
        self,
        question: str,
        filter_source: Optional[str] = None,
    ) -> List[Document]:
        """Get source documents for a question.

        Args:
            question: User's question
            filter_source: Optional document name to filter

        Returns:
            List of relevant documents
        """
        filter_dict = {"doc_name": filter_source} if filter_source else None
        return self.vector_store.similarity_search(
            question,
            k=settings.RETRIEVAL_K,
            filter_dict=filter_dict,
        )

    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []
