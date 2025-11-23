import logging
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Professional Embedding Service for generating and managing embeddings."""

    def __init__(self):
        logger.info("EmbeddingService initialized.")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32  # this is optimized for performance
            }
        )

        logger.info("Embedding model loaded: %s on device: %s",
                    settings.embedding_model,
                    settings.embedding_device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple docs with progress tracking."""
        if not texts:
            logger.warning("No texts provided for embedding.")
            return []
        logger.info("Embedding %d documents.", len(texts))

        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info("Successfully embedded %d documents.", len(texts))
            return embeddings
        except Exception as e:
            logger.error("Error embedding documents: %s", str(e))
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if not text:
            logger.warning("No text provided for embedding.")
            return []
        logger.info("Embedding query.")

        try:
            embedding = self.embeddings.embed_query(text)
            logger.info("Successfully embedded query.")
            return embedding
        except Exception as e:
            logger.error("Error embedding query: %s", str(e))
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        sample_embedding = self.embed_query("sample text")
        dimension = len(sample_embedding)
        logger.info("Embedding dimension: %d", dimension)
        return dimension


