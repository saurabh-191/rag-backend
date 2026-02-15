"""
Text Chunking Module

This module handles splitting large documents into smaller, overlapping chunks
to ensure that:
1. Each chunk fits within model's context window
2. Context is preserved across chunks (via overlap)
3. Relevant information isn't split awkwardly

Chunking strategy:
- Recursive splitting: Splits on sentences/paragraphs first, then characters
- Overlap: Maintains context continuity between chunks
- Size: Configurable chunk size based on model requirements
"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class TextChunker:
    """
    Text chunking utility for splitting documents into manageable chunks.
    Uses LangChain's RecursiveCharacterTextSplitter for intelligent splitting.
    """
    
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        separators: List[str] = None
    ):
        """
        Initialize text chunker with parameters.
        
        Args:
            chunk_size: Maximum size of each chunk (characters)
            chunk_overlap: Overlap between chunks (characters)
            separators: Custom separators for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators: paragraph, sentence, word, character
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
        
        logger.info(
            f"Initialized TextChunker: chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Input text to chunk
            metadata: Additional metadata (source, page, etc.)
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        try:
            # Split text into chunks
            chunks = self.splitter.split_text(text)
            
            logger.debug(f"Split text into {len(chunks)} chunks")
            
            # Add metadata to each chunk
            chunked_data = []
            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    "text": chunk,
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    **(metadata or {})
                }
                chunked_data.append(chunk_dict)
            
            return chunked_data
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise
    
    def chunk_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents at once.
        
        Args:
            documents: List of dicts with 'text' and optional 'metadata'
        
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            # Add document index to metadata
            metadata["doc_index"] = doc_idx
            
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks


# Default chunker instance
_chunker = None


def get_chunker(
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
) -> TextChunker:
    """Get or create default chunker instance"""
    global _chunker
    if _chunker is None:
        _chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return _chunker
