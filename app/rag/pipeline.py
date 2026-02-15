"""
RAG Pipeline Module

This module orchestrates the complete RAG (Retrieval Augmented Generation) pipeline.

Complete RAG Flow:
1. INGESTION: Load documents → Chunk → Generate embeddings → Store in vector DB
2. RETRIEVAL: Convert query to embedding → Search vector DB → Get top-k chunks
3. GENERATION: Build prompt with context → Send to LLM → Generate response
4. STREAMING: Send response chunks to client in real-time

This module coordinates all RAG components into a single unified pipeline.
"""

from typing import List, Dict, Any, Optional, Generator
import time

from app.core.config import settings
from app.core.logging import get_logger

from app.rag.loader import DocumentLoader
from app.rag.chunker import get_chunker
from app.rag.embeddings import get_embedding_client
from app.rag.vector_store import get_vector_store
from app.rag.retriever import get_retriever
from app.rag.prompt import get_prompt_builder
from app.llm.client import get_llm_client

logger = get_logger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline for document ingestion and question answering.
    """
    
    def __init__(self):
        """Initialize RAG pipeline components"""
        self.loader = DocumentLoader()
        self.chunker = get_chunker()
        self.embedding_client = get_embedding_client()
        self.vector_store = get_vector_store()
        self.retriever = get_retriever()
        self.prompt_builder = get_prompt_builder()
        self.llm_client = get_llm_client()
        
        logger.info("Initialized RAG Pipeline")
    
    # ============ INGESTION PHASE ============
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single document into the RAG system.
        
        Process:
        1. Load document from file
        2. Split into chunks
        3. Generate embeddings
        4. Store in vector database
        
        Args:
            file_path: Path to document file
        
        Returns:
            Ingestion statistics
        """
        try:
            start_time = time.time()
            logger.info(f"Starting ingestion: {file_path}")
            
            # Step 1: Load document
            document = self.loader.load_document(file_path)
            logger.debug(f"Loaded document: {document['filename']}")
            
            # Step 2: Chunk document
            chunks = self.chunker.chunk_text(
                text=document["text"],
                metadata={
                    "source": document["source"],
                    "file_type": document["file_type"],
                    "filename": document["filename"]
                }
            )
            logger.debug(f"Created {len(chunks)} chunks")
            
            if not chunks:
                logger.warning(f"No chunks created for {file_path}")
                return {
                    "status": "warning",
                    "message": "Document produced no chunks",
                    "chunks_created": 0
                }
            
            # Step 3: Generate embeddings
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_client.embed_batch(texts)
            logger.debug(f"Generated {len(embeddings)} embeddings")
            
            # Step 4: Store in vector database
            doc_ids = self.vector_store.add_texts(
                texts=texts,
                embeddings=embeddings,
                metadatas=[chunk for chunk in chunks]
            )
            
            elapsed_time = time.time() - start_time
            
            logger.info(
                f"Successfully ingested: {document['filename']} "
                f"({len(chunks)} chunks in {elapsed_time:.2f}s)"
            )
            
            return {
                "status": "success",
                "filename": document["filename"],
                "chunks_created": len(chunks),
                "doc_ids": doc_ids,
                "processing_time_s": elapsed_time,
                "document_size_bytes": document.get("file_size_bytes", 0)
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            raise
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Ingest all documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
        
        Returns:
            Ingestion summary statistics
        """
        try:
            start_time = time.time()
            logger.info(f"Starting batch ingestion: {directory_path}")
            
            # Load all documents
            documents = self.loader.load_documents_from_directory(directory_path)
            
            if not documents:
                logger.warning(f"No documents found in {directory_path}")
                return {
                    "status": "warning",
                    "message": "No documents found",
                    "documents_processed": 0
                }
            
            # Chunk all documents
            all_chunks = self.chunker.chunk_documents(documents)
            logger.debug(f"Created {len(all_chunks)} total chunks")
            
            if not all_chunks:
                logger.warning("No chunks created from documents")
                return {
                    "status": "warning",
                    "message": "Documents produced no chunks",
                    "documents_processed": len(documents),
                    "total_chunks": 0
                }
            
            # Generate embeddings for all chunks
            texts = [chunk["text"] for chunk in all_chunks]
            embeddings = self.embedding_client.embed_batch(texts)
            logger.debug(f"Generated {len(embeddings)} embeddings")
            
            # Store all in vector database
            doc_ids = self.vector_store.add_texts(
                texts=texts,
                embeddings=embeddings,
                metadatas=all_chunks
            )
            
            elapsed_time = time.time() - start_time
            
            logger.info(
                f"Batch ingestion complete: "
                f"{len(documents)} documents, {len(all_chunks)} chunks "
                f"in {elapsed_time:.2f}s"
            )
            
            return {
                "status": "success",
                "documents_processed": len(documents),
                "total_chunks": len(all_chunks),
                "doc_ids": doc_ids,
                "processing_time_s": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Error ingesting directory: {e}")
            raise
    
    # ============ RETRIEVAL & GENERATION PHASE ============
    
    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a query using RAG (non-streaming).
        
        Process:
        1. Retrieve relevant context
        2. Build prompt with context
        3. Generate response from LLM
        
        Args:
            query_text: User question
            top_k: Number of context chunks to retrieve
            include_context: Whether to include context in response
        
        Returns:
            Query result with answer and context
        """
        try:
            start_time = time.time()
            logger.info(f"Processing query: {query_text[:100]}...")
            
            # Step 1: Retrieve context
            context_chunks = self.retriever.retrieve(
                query=query_text,
                top_k=top_k
            )
            logger.debug(f"Retrieved {len(context_chunks)} context chunks")
            
            # Step 2: Build prompt
            if context_chunks and include_context:
                prompt = self.prompt_builder.build_rag_prompt(
                    query=query_text,
                    context_chunks=context_chunks
                )
            else:
                prompt = self.prompt_builder.build_simple_prompt(query_text)
            
            # Step 3: Generate response
            response_text = self.llm_client.generate(prompt)
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"Query processed in {elapsed_time:.2f}s")
            
            return {
                "status": "success",
                "query": query_text,
                "answer": response_text,
                "context": context_chunks if include_context else None,
                "context_count": len(context_chunks),
                "processing_time_s": elapsed_time,
                "model": self.llm_client.model
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def query_stream(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        include_context: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Answer a query with streaming response.
        
        Args:
            query_text: User question
            top_k: Number of context chunks to retrieve
            include_context: Whether to include context in response
        
        Yields:
            Response chunks as they are generated
        """
        try:
            logger.info(f"Starting streaming query: {query_text[:100]}...")
            
            # Retrieve context
            context_chunks = self.retriever.retrieve(
                query=query_text,
                top_k=top_k
            )
            
            # Build prompt
            if context_chunks and include_context:
                prompt = self.prompt_builder.build_rag_prompt(
                    query=query_text,
                    context_chunks=context_chunks
                )
            else:
                prompt = self.prompt_builder.build_simple_prompt(query_text)
            
            # Yield initial response with context
            yield {
                "type": "start",
                "query": query_text,
                "context": context_chunks if include_context else None,
                "context_count": len(context_chunks)
            }
            
            # Stream response chunks
            for chunk in self.llm_client.stream(prompt):
                yield {
                    "type": "chunk",
                    "data": chunk
                }
            
            # Yield final response
            yield {
                "type": "end",
                "model": self.llm_client.model,
                "status": "success"
            }
            
            logger.info("Streaming query completed")
            
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield {
                "type": "error",
                "message": str(e)
            }
    
    # ============ UTILITY METHODS ============
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        try:
            return {
                "total_documents": self.vector_store.get_count(),
                "vector_store": settings.VECTOR_STORE_TYPE,
                "llm_model": self.llm_client.model,
                "embedding_model": self.embedding_client.model,
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "top_k": settings.RETRIEVAL_TOP_K
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def rebuild_index(self, directory_path: str) -> Dict[str, Any]:
        """
        Rebuild vector index from documents in directory.
        
        Args:
            directory_path: Path to documents
        
        Returns:
            Rebuild statistics
        """
        try:
            logger.info(f"Rebuilding index from: {directory_path}")
            
            # Clear existing index
            self.vector_store.clear()
            
            # Reingest documents
            result = self.ingest_directory(directory_path)
            
            logger.info("Index rebuild complete")
            return result
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            raise


# Global pipeline instance
_pipeline = None


def get_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# Convenience functions
def ingest_document(file_path: str) -> Dict[str, Any]:
    """Ingest a single document"""
    return get_pipeline().ingest_document(file_path)


def ingest_directory(directory_path: str) -> Dict[str, Any]:
    """Ingest all documents from directory"""
    return get_pipeline().ingest_directory(directory_path)


def query_rag(query_text: str, top_k: int = 5) -> Dict[str, Any]:
    """Execute RAG query"""
    return get_pipeline().query(query_text, top_k=top_k)


def stream_rag_query(query_text: str, top_k: int = 5) -> Generator:
    """Stream RAG query"""
    return get_pipeline().query_stream(query_text, top_k=top_k)
