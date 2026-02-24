"""
RAG Database for the Pt/Pd Printing Assistant.

Uses ChromaDB for vector storage and SentenceTransformers for embeddings.
"""

import logging
from pathlib import Path

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    raise ImportError("RAG dependencies are required. Install with: pip install ptpd-calibration[rag]")

from ptpd_calibration.config import get_settings

logger = logging.getLogger(__name__)


class RAGDatabase:
    """
    Manages the Retrieval-Augmented Generation database.
    """

    def __init__(self, persist_directory: Path | None = None, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG database.

        Args:
            persist_directory: Directory to store the database. Defaults to a path in the data_dir.
            embedding_model: The sentence-transformer model to use for embeddings.
        """
        settings = get_settings()
        if persist_directory is None:
            self._persist_directory = settings.data_dir / "rag_db"
        else:
            self._persist_directory = persist_directory

        self._persist_directory.mkdir(parents=True, exist_ok=True)

        # Use a persistent client to store data on disk
        self._client = chromadb.PersistentClient(path=str(self._persist_directory))

        # Create an embedding function
        self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # Get or create the collection
        self._collection = self._client.get_or_create_collection(
            name="ptpd_documents",
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine distance for similarity
        )
        logger.info(f"RAG database initialized at: {self._persist_directory}")

    def add_documents(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """
        Add documents to the RAG database.

        Args:
            documents: A list of document texts.
            metadatas: Optional list of metadata dictionaries corresponding to the documents.
        """
        if not documents:
            return

        # Generate IDs for the documents
        # This is a simple approach; for production, you might want more robust ID generation.
        start_id = self._collection.count()
        ids = [f"doc_{start_id + i}" for i in range(len(documents))]

        logger.info(f"Adding {len(documents)} documents to the RAG collection.")
        self._collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query: str, n_results: int = 3) -> list[str]:
        """
        Search for relevant documents in the database.

        Args:
            query: The user's query string.
            n_results: The number of relevant documents to retrieve.

        Returns:
            A list of the most relevant document texts.
        """
        if not query:
            return []

        logger.info(f"Searching RAG DB for query: '{query}'")
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results
        )

        return results.get("documents", [[]])[0]

    def get_document_count(self) -> int:
        """Returns the number of documents in the database."""
        return self._collection.count()


# Singleton instance
_rag_db_instance: RAGDatabase | None = None

def get_rag_db() -> RAGDatabase:
    """Get the singleton RAGDatabase instance."""
    global _rag_db_instance
    if _rag_db_instance is None:
        _rag_db_instance = RAGDatabase()
    return _rag_db_instance
