from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Q&A Chain Prompt - answers questions based on retrieved context
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that answers questions based ONLY on the provided document context.

IMPORTANT RULES:
1. Only answer based on the provided context below
2. If the answer is not found in the context, clearly state: "I could not find this information in the uploaded documents."
3. When answering, cite which document(s) the information came from using the document name in brackets
4. Be concise and accurate
5. Use bullet points for lists when appropriate

Context from documents:
{context}""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Summarization Prompt - for summarizing a single document
SUMMARIZE_PROMPT = ChatPromptTemplate.from_template(
    """Summarize the following document content. Provide a comprehensive but concise summary that captures:
- Main topics and themes
- Key arguments or points
- Important details and findings
- Conclusions or recommendations (if any)

Document: {doc_name}

Content:
{content}

Summary:"""
)

# Structured Notes Prompt - for generating organized notes
NOTES_PROMPT = ChatPromptTemplate.from_template(
    """Create well-structured notes from the following document content.

Organize the notes with:
- Clear main topics and subtopics using headers
- Key points as bullet points under each topic
- Important definitions, terms, or concepts highlighted
- Notable quotes, statistics, or data points
- Action items or takeaways (if applicable)

Document: {doc_name}

Content:
{content}

Structured Notes:"""
)

# All Documents Summary Prompt - for summarizing multiple documents together
ALL_DOCS_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """Provide a comprehensive summary of all the following documents.

Your summary should:
- Identify the main theme or purpose of each document
- Highlight common themes across documents
- Note any contrasting or complementary information
- Provide unique insights from each document
- End with key takeaways from all documents combined

Documents Content:
{content}

Comprehensive Summary:"""
)
