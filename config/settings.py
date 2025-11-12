from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings for environment variable management and validation."""

    # API Keys
    google_api_key: Optional[str] = None
    huggingfacehub_api_token: Optional[str] = None

    # Application Settings
    app_env: str = "development"  # Options: development, staging, production
    log_level: str = "INFO"
    max_file_size_mb: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Paths
    vector_store_path: str = "./data/vector_store"
    upload_directory: str = "./data/uploads"

    # Model Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L12-v2"
    chat_model: str = "gemini-flash-latest"
    temperature: float = 0.1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @field_validator("vector_store_path", "upload_directory")
    def ensure_directory_exists(cls, v: str) -> str:
        """Ensure that the specified directory exists; create it if it doesn't."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @property
    def max_file_size_bytes(self) -> int:
        """Convert maximum file size from megabytes to bytes."""
        return self.max_file_size_mb * 1024 * 1024


settings = Settings()
