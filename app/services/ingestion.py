"""
Document Ingestion Service Module

Business logic for document ingestion and indexing.
Handles:
- Document loading from files/directories
- Chunking and embedding generation
- Vector store indexing
- Index management and updates
"""

from typing import Optional, Dict, Any, List
import os
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.pipeline import get_pipeline
from app.rag.loader import DocumentLoader
from app.rag.chunker import get_chunker
from app.rag.embeddings import get_embedding_client
from app.rag.vector_store import get_vector_store

logger = get_logger(__name__)


class IngestionService:
    """
    Service for document ingestion and vector store management.
    """
    
    def __init__(self):
        """Initialize ingestion service with RAG components"""
        self.pipeline = get_pipeline()
        self.loader = DocumentLoader()
        self.chunker = get_chunker()
        self.embedding_client = get_embedding_client()
        self.vector_store = get_vector_store()
        
        logger.info("Initialized IngestionService")
    
    def ingest_file(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Ingest a single document file.
        
        Args:
            file_path: Path to document file
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
        
        Returns:
            Ingestion result with statistics
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            # Validate file type
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in settings.SUPPORTED_FILE_TYPES:
                error_msg = f"Unsupported file type: {file_ext}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            # Validate file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > settings.MAX_FILE_SIZE_MB:
                error_msg = f"File too large: {file_size_mb:.1f}MB (max {settings.MAX_FILE_SIZE_MB}MB)"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            logger.info(f"Ingesting file: {file_path}")
            
            # Use pipeline for ingestion
            result = self.pipeline.ingest_document(file_path)
            
            return {
                "status": result.get("status", "success"),
                "message": f"Successfully ingested {result.get('filename', 'document')}",
                **result
            }
            
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def ingest_directory(
        self,
        directory_path: str,
        file_types: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Ingest all supported documents from directory.
        
        Args:
            directory_path: Path to directory
            file_types: Specific file types to ingest
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
        
        Returns:
            Batch ingestion results
        """
        try:
            # Validate directory
            if not os.path.isdir(directory_path):
                error_msg = f"Directory not found: {directory_path}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            logger.info(f"Ingesting directory: {directory_path}")
            
            # Use pipeline for batch ingestion
            result = self.pipeline.ingest_directory(directory_path)
            
            return {
                "status": result.get("status", "success"),
                "message": f"Ingestion complete: {result.get('documents_processed', 0)} documents",
                **result
            }
            
        except Exception as e:
            logger.error(f"Error ingesting directory {directory_path}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def rebuild_index(
        self,
        directory_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rebuild vector index from scratch.
        
        Args:
            directory_path: Path to documents (uses default if not provided)
        
        Returns:
            Rebuild operation result
        """
        try:
            path = directory_path or settings.DATA_RAW_PATH
            
            logger.info(f"Rebuilding index from: {path}")
            
            # Use pipeline rebuild
            result = self.pipeline.rebuild_index(path)
            
            return {
                "status": result.get("status", "success"),
                "message": "Index rebuild complete",
                **result
            }
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def clear_index(self) -> Dict[str, Any]:
        """
        Clear all documents from vector store.
        
        Returns:
            Operation result
        """
        try:
            logger.warning("Clearing vector store index")
            
            self.vector_store.clear()
            
            logger.info("Index cleared successfully")
            return {
                "status": "success",
                "message": "Vector store cleared"
            }
            
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Index statistics
        """
        try:
            stats = self.pipeline.get_stats()
            
            return {
                "status": "success",
                **stats
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def delete_documents(self, doc_ids: List[str]) -> Dict[str, Any]:
        """
        Delete specific documents from index.
        
        Args:
            doc_ids: List of document IDs to delete
        
        Returns:
            Operation result
        """
        try:
            if not doc_ids:
                return {
                    "status": "error",
                    "message": "No document IDs provided"
                }
            
            logger.info(f"Deleting {len(doc_ids)} documents from index")
            
            self.vector_store.delete(doc_ids)
            
            return {
                "status": "success",
                "message": f"Deleted {len(doc_ids)} documents",
                "deleted_count": len(doc_ids)
            }
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return {
                "status": "error",
                "message": str(e)
            }


# Global service instance
_ingestion_service = None


def get_ingestion_service() -> IngestionService:
    """Get or create ingestion service instance"""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service


# Convenience functions
def ingest_file(file_path: str, **kwargs) -> Dict[str, Any]:
    """Ingest a single file"""
    return get_ingestion_service().ingest_file(file_path, **kwargs)


def ingest_directory(directory_path: str, **kwargs) -> Dict[str, Any]:
    """Ingest all files from directory"""
    return get_ingestion_service().ingest_directory(directory_path, **kwargs)


def rebuild_index(directory_path: Optional[str] = None) -> Dict[str, Any]:
    """Rebuild vector index"""
    return get_ingestion_service().rebuild_index(directory_path)
