"""
Retriever Module

This module handles semantic search and retrieval of relevant documents
from the vector store.

Retrieval Process:
1. Convert user query to embedding
2. Search vector store for similar embeddings
3. Filter by similarity threshold
4. Rank and return top-k results
5. Add context metadata

This is the "R" in "RAG" - retrieves relevant context before generation.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.embeddings import get_embedding_client
from app.rag.vector_store import get_vector_store

logger = get_logger(__name__)


class Retriever:
    """
    Retrieves relevant document chunks from vector store based on query similarity.
    """
    
    def __init__(
        self,
        vector_store=None,
        embedding_client=None,
        top_k: int = settings.RETRIEVAL_TOP_K,
        similarity_threshold: float = settings.SIMILARITY_THRESHOLD
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: Vector store instance (uses default if None)
            embedding_client: Embedding client (uses default if None)
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score (0.0-1.0)
        """
        self.vector_store = vector_store or get_vector_store()
        self.embedding_client = embedding_client or get_embedding_client()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        logger.info(
            f"Initialized Retriever: top_k={top_k}, "
            f"threshold={similarity_threshold}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            top_k: Override default top_k (number of results)
            threshold: Override default similarity threshold
        
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            if not query or not query.strip():
                logger.warning("Empty query provided")
                return []
            
            # Use provided values or defaults
            k = top_k or self.top_k
            thresh = threshold if threshold is not None else self.similarity_threshold
            
            logger.debug(f"Retrieving documents for query: {query[:100]}...")
            
            # Convert query to embedding
            query_embedding = self.embedding_client.embed_text(query)
            
            # Search vector store
            results = self.vector_store.search(
                embedding=query_embedding,
                top_k=k * 2,  # Get more to filter by threshold
                threshold=None  # Filter manually for control
            )
            
            # Filter by threshold
            filtered_results = [
                r for r in results 
                if r["similarity"] >= thresh
            ]
            
            # Keep top_k after filtering
            filtered_results = filtered_results[:k]
            
            logger.info(
                f"Retrieved {len(filtered_results)} relevant chunks "
                f"(similarity >= {thresh})"
            )
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with similarity scores (for ranking).
        
        Args:
            query: User query string
            top_k: Number of results
        
        Returns:
            List of results with similarity scores
        """
        results = self.retrieve(query, top_k=top_k, threshold=0.0)
        
        # Sort by similarity (descending)
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        
        return results
    
    def retrieve_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve specific documents by ID.
        
        Args:
            ids: List of document IDs
        
        Returns:
            List of documents
        """
        try:
            # Chroma get() method
            results = self.vector_store.collection.get(ids=ids)
            
            retrieved = []
            for doc, metadata in zip(results["documents"], results["metadatas"]):
                retrieved.append({
                    "text": doc,
                    "metadata": metadata,
                    "similarity": None
                })
            
            logger.debug(f"Retrieved {len(retrieved)} documents by ID")
            return retrieved
            
        except Exception as e:
            logger.error(f"Error retrieving by IDs: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        try:
            total_docs = self.vector_store.get_count()
            
            return {
                "total_documents": total_docs,
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
                "vector_store_type": "chroma",
                "embedding_model": self.embedding_client.model
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


# Global retriever instance
_retriever = None


def get_retriever(
    top_k: int = settings.RETRIEVAL_TOP_K,
    similarity_threshold: float = settings.SIMILARITY_THRESHOLD
) -> Retriever:
    """Get or create retriever instance"""
    global _retriever
    if _retriever is None:
        _retriever = Retriever(
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
    return _retriever


def retrieve(
    query: str,
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Convenience function to retrieve documents"""
    return get_retriever().retrieve(query, top_k=top_k)
