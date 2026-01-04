from langchain_ollama import ChatOllama

from src.config import settings


def get_llm(temperature: float = 0.3, streaming: bool = True) -> ChatOllama:
    """Initialize and return Ollama chat model.

    Args:
        temperature: Controls randomness in responses (0.0 to 1.0)
        streaming: Whether to enable streaming responses

    Returns:
        Configured ChatOllama instance
    """
    return ChatOllama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
        streaming=streaming,
    )
