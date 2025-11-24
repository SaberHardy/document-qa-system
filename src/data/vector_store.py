import logging
from typing import Optional, List
from xml.dom.minidom import Document

from langchain_community.vectorstores import Chroma

from config.settings import settings
from src.core.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Professional vector store manager with persistence and monitoring"""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store: Optional[Chroma] = None

        logging.info("VectorStoreManager initialized with path: %s", self.vector_store)

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create or load a persistent vector store."""
        logger.info("Creating/loading vector store at: %s", self.vector_store)

        if not documents:
            logger.warning("No documents provided to create vector store.")
            raise ValueError("Document list is empty.")

        try:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_service.embeddings,
                persist_directory=settings.vector_store_path
            )
            # This one is to force persistence to disk
            self.vector_store.persist()
            logger.info("Vector store created/loaded successfully with %d documents.", len(documents))

            doc_count = self.vector_store._collection.count()
            logger.info("Vector store contains %d documents.", doc_count)

            return self.vector_store
        except Exception as e:
            logger.error("Error creating/loading vector store: %s", str(e))
            raise

    def load_vector_store(self) -> Chroma:
        """Load an existing vector store from disk."""
        logger.info("Loading vector store from path: %s", settings.vector_store_path)

        try:
            self.vector_store = Chroma(
                persist_directory=settings.vector_store_path,
                embedding_function=self.embedding_service.embeddings
            )
            logger.info("Vector store loaded successfully.")

            doc_count = self.vector_store._collection.count()
            logger.info("Vector store contains %d documents.", doc_count)
            return self.vector_store
        except Exception as e:
            logger.error("Error loading vector store: %s", str(e))
            raise

    def get_retriever(self, search_type: str = "similarity", k: int = 4):

        """Get a retriever from the vector store."""
        if not self.vector_store:
            self.load_vector_store()

        retriever = self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})

        logger.info("Creating retriever with top_k=%d", k)
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        if not self.vector_store:
            self.load_vector_store()

        doc_count = self.vector_store._collection.count()
        logger.info("Vector store contains %d documents.", doc_count)
        return doc_count

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform a similarity search in the vector store."""
        if not self.vector_store:
            self.load_vector_store()

        logger.info("Performing similarity search for query: '%s' with top_k=%d", query, k)
        results = self.vector_store.similarity_search(query, k=k)
        logger.info("Similarity search returned %d results.", len(results))
        return results