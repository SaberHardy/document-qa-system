import logging
from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': Docx2txtLoader,
            '.doc': UnstructuredMarkdownLoader
        }

    def load_document(self, file_path: str) -> List[Document]:
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Document not found: {file_path}")

        file_extension = file_path.suffix.lower()
        if file_extension not in self.supported_extensions:
            logger.error(f"Unsupported file type: {file_extension}")
            raise ValueError(f"Unsupported file type: {file_extension}, "
                             f"supported types are: {list(self.supported_extensions.keys())}")

        try:
            loader_class = self.supported_extensions[file_extension]
            loader = loader_class(str(file_path))
            documents = loader.load()

            for doc in documents:
                doc.metadata.update({
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_extension
                })
            logger.info(f"Successfully loaded document: {file_path.name}, total pages/lines: {len(documents)}")
            return documents
        except Exception as e:
            logger.error(f"Error loading document {file_path.name}: {e}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits documents into smaller chunks."""
        if not documents:
            logger.warning("No documents to split.")
            return []

        logger.info(f"Chunking documents into size {self.chunk_size} with overlap {self.chunk_overlap}")
        chunks = self.text_splitter.split_documents(documents)

        # Calculate statistics
        total_characters = sum(len(chunk.page_content) for chunk in chunks)
        average_chunk_size = total_characters / len(chunks) if chunks else 0

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents, "
                    f"(avg. chunk size: {average_chunk_size:.2f} characters)")
        return chunks

    def process_directory(self, directory_path: str) -> List[Document]:
        """Processes all supported documents in a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        all_documents = []
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    documents = self.load_document(str(file_path))
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Failed to process file {file_path.name}: {e}")
                    continue

        logger.info(f"Total documents loaded from directory: {len(all_documents)}")
        return all_documents
