"""
Embeddings Module

This module handles generation of vector embeddings using Ollama.
Embeddings convert text into numerical vectors that capture semantic meaning,
enabling similarity-based retrieval.

Process:
1. Take input text (query or document chunk)
2. Send to Ollama embedding model (nomic-embed-text)
3. Get back vector representation (768-dimensional by default)
4. Store in vector database for similarity search
"""

import numpy as np
from typing import List, Union, Optional
import requests
from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingClient:
    """
    Embedding client for generating vector representations using Ollama.
    """
    
    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.EMBEDDING_MODEL,
        embedding_dim: int = settings.EMBEDDING_DIMENSION,
        timeout: int = 120
    ):
        """
        Initialize embedding client.
        
        Args:
            base_url: Ollama server URL
            model: Embedding model name (e.g., 'nomic-embed-text')
            embedding_dim: Dimension of output embeddings
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embedding_dim = embedding_dim
        self.timeout = timeout
        
        logger.info(f"Initialized Embedding Client with model: {self.model}")
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify connection to Ollama and check if embedding model is available"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info(f"Connected to Ollama at {self.base_url}")
            
            # Check if embedding model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            if self.model not in model_names:
                logger.warning(
                    f"Embedding model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}"
                )
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
        
        Returns:
            Numpy array of embeddings (shape: embedding_dim,)
        
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_dim)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": text.strip()},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            embeddings = response.json().get("embeddings", [])
            if embeddings:
                embedding = np.array(embeddings[0], dtype=np.float32)
                logger.debug(f"Generated embedding for text (length: {len(text)})")
                return embedding
            else:
                logger.warning("No embeddings returned from Ollama")
                return np.zeros(self.embedding_dim)
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of numpy arrays (one per text)
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Error embedding text {i}: {e}, using zero vector")
                embeddings.append(np.zeros(self.embedding_dim))
        
        logger.info(f"Generated embeddings for {len(embeddings)} texts")
        return embeddings
    
    def embed_batch(self, texts: List[str], batch_size: int = 128) -> List[np.ndarray]:
        """
        Generate embeddings in batches (more efficient).
        
        Args:
            texts: List of input texts
            batch_size: Number of texts per batch
        
        Returns:
            List of numpy arrays
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1} ({len(batch)} texts)")
            
            batch_embeddings = self.embed_texts(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)


# Global embedding client instance
_embedding_client = None


def get_embedding_client() -> EmbeddingClient:
    """Get or create embedding client"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


def embed_text(text: str) -> np.ndarray:
    """Convenience function to embed single text"""
    return get_embedding_client().embed_text(text)


def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """Convenience function to embed multiple texts"""
    return get_embedding_client().embed_texts(texts)
