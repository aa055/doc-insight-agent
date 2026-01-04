from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application configuration settings."""

    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2"
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # Document processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval settings
    RETRIEVAL_K: int = 4

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    VECTOR_STORE_DIR: Path = BASE_DIR / "data" / "vector_store"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
