"""
Vector Store Module

This module manages the Chroma vector database for storing and retrieving
document embeddings.

Vector Database:
- Stores embeddings (768-dim vectors) of all document chunks
- Enables fast semantic similarity search
- Persists data to disk for reuse across sessions
- Supports CRUD operations (Create, Read, Update, Delete)

Process:
1. Create collection in Chroma
2. Add embeddings + metadata for each chunk
3. Query by similarity to find relevant chunks
4. Return top-k most similar chunks
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    logger.error("chromadb not installed. Install with: pip install chromadb")
    raise


class VectorStore:
    """
    Chroma vector store wrapper for managing embeddings and similarity search.
    """
    
    def __init__(
        self,
        persist_directory: str = settings.VECTOR_STORE_PATH,
        collection_name: str = settings.CHROMA_COLLECTION_NAME
    ):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory to persist vector database
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if not exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory
            )
            logger.info(f"Initialized Chroma client at {persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {e}")
            raise
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        logger.info(f"Using collection: {collection_name}")
    
    def add_texts(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add text embeddings to vector store.
        
        Args:
            texts: List of text documents
            embeddings: List of embedding vectors (numpy arrays)
            ids: Optional custom IDs for documents
            metadatas: Optional metadata for each document
        
        Returns:
            List of document IDs
        """
        try:
            if not texts or not embeddings:
                logger.warning("Empty texts or embeddings provided")
                return []
            
            if len(texts) != len(embeddings):
                raise ValueError("texts and embeddings must have same length")
            
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(texts))]
            
            # Convert embeddings to lists (Chroma expects lists, not numpy arrays)
            embedding_lists = [emb.tolist() for emb in embeddings]
            
            # Sanitize metadatas - Chroma only accepts str, int, float, list
            sanitized_metadatas = []
            if metadatas:
                for meta in metadatas:
                    sanitized_meta = {}
                    for key, value in meta.items():
                        # Only keep valid types: str, int, float, bool -> convert to str/int/float
                        if isinstance(value, bool):
                            sanitized_meta[key] = int(value)  # Convert bool to 0/1
                        elif isinstance(value, (str, int, float)):
                            sanitized_meta[key] = value
                        elif isinstance(value, list):
                            # Only allow lists of str/int/float
                            sanitized_meta[key] = [str(v) for v in value]
                        else:
                            # Convert other types to string, skip if conversion fails
                            try:
                                sanitized_meta[key] = str(value)
                            except:
                                logger.warning(f"Skipping metadata field '{key}' with unsupported type {type(value)}")
                    sanitized_metadatas.append(sanitized_meta)
            
            # Batch add to handle large datasets (Chroma has max batch size of ~5461)
            batch_size = 5000
            all_ids = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = embedding_lists[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                batch_metadatas = sanitized_metadatas[i:i+batch_size] if sanitized_metadatas else []
                
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas or []
                )
                
                all_ids.extend(batch_ids)
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch_texts)} documents to vector store")
            
            logger.info(f"Completed adding {len(texts)} documents total to vector store")
            return all_ids
            
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {e}")
            raise
    
    def search(
        self,
        embedding: np.ndarray,
        top_k: int = settings.RETRIEVAL_TOP_K,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search vector store for similar embeddings.
        
        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity score (0.0-1.0)
        
        Returns:
            List of results with text, metadata, and similarity score
        """
        try:
            if embedding is None or len(embedding) == 0:
                logger.warning("Empty embedding provided for search")
                return []
            
            # Convert numpy array to list
            query_embedding = embedding.tolist()
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            
            if results and results["documents"] and len(results["documents"]) > 0:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0] if results["metadatas"] else []
                distances = results["distances"][0] if results["distances"] else []
                
                for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                    # Chroma returns distance (0 for identical, 2 for opposite)
                    # Convert to similarity (1 - distance/2)
                    similarity = 1 - (distance / 2)
                    
                    # Apply threshold if specified
                    if threshold and similarity < threshold:
                        continue
                    
                    result = {
                        "text": doc,
                        "metadata": metadata,
                        "similarity": float(similarity),
                        "distance": float(distance)
                    }
                    search_results.append(result)
            
            logger.debug(f"Found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents from vector store.
        
        Args:
            ids: List of document IDs to delete
        
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from vector store")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def update(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Update existing documents in vector store.
        
        Args:
            ids: List of document IDs to update
            texts: New text content
            embeddings: New embedding vectors
            metadatas: Updated metadata
        
        Returns:
            True if successful
        """
        try:
            embedding_lists = [emb.tolist() for emb in embeddings]
            
            self.collection.upsert(
                ids=ids,
                embeddings=embedding_lists,
                documents=texts,
                metadatas=metadatas or []
            )
            
            logger.info(f"Updated {len(ids)} documents in vector store")
            return True
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            raise
    
    def get_count(self) -> int:
        """Get total number of documents in collection"""
        try:
            count = self.collection.count()
            return count
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0
    
    def clear(self) -> bool:
        """Clear all documents from collection"""
        try:
            # Get all IDs
            all_items = self.collection.get()
            if all_items and all_items["ids"]:
                self.collection.delete(ids=all_items["ids"])
            
            logger.info("Cleared vector store")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise
    
    def delete_collection(self) -> bool:
        """Delete entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise


# Global vector store instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get or create vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
