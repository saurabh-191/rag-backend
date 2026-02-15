"""
Chat Service Module

Business logic layer for chat operations.
Handles:
- Query validation and safety checks
- Context retrieval
- Prompt preparation
- Response generation
"""

from typing import Optional, Dict, Any, Generator
import time

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.pipeline import get_pipeline
from app.rag.retriever import get_retriever
from app.rag.prompt import get_prompt_builder
from app.utils.guards import is_safe_query, detect_prompt_injection
from app.llm.client import get_llm_client

logger = get_logger(__name__)


class ChatService:
    """
    Service for handling chat operations and RAG queries.
    """
    
    def __init__(self):
        """Initialize chat service with RAG components"""
        self.pipeline = get_pipeline()
        self.retriever = get_retriever()
        self.prompt_builder = get_prompt_builder()
        self.llm_client = get_llm_client()
        
        logger.info("Initialized ChatService")
    
    def validate_query(self, query: str) -> tuple[bool, Optional[str]]:
        """
        Validate user query for safety and injection attacks.
        
        Args:
            query: User query string
        
        Returns:
            Tuple of (is_safe, error_message)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query) > 10000:
            return False, "Query too long (max 10000 characters)"
        
        # Safety check
        if not is_safe_query(query):
            return False, "Query contains potentially harmful content"
        
        # Prompt injection detection
        if settings.ENABLE_PROMPT_INJECTION_GUARD:
            is_safe, message = detect_prompt_injection(query)
            if not is_safe:
                logger.warning(f"Potential prompt injection detected: {message}")
                return False, f"Suspicious query pattern detected: {message}"
        
        return True, None
    
    def prepare_chat_prompt(
        self,
        query: str,
        include_context: bool = True,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Prepare prompt for chat with context and validation.
        
        Args:
            query: User query
            include_context: Include retrieved context
            top_k: Number of context chunks
        
        Returns:
            Dict with prompt, context, and metadata
        
        Raises:
            ValueError: If query is invalid
        """
        try:
            # Validate query
            is_safe, error_msg = self.validate_query(query)
            if not is_safe:
                logger.warning(f"Invalid query: {error_msg}")
                raise ValueError(error_msg)
            
            logger.debug(f"Preparing chat prompt for: {query[:100]}...")
            
            # Retrieve context
            context_chunks = []
            if include_context:
                context_chunks = self.retriever.retrieve(
                    query=query,
                    top_k=top_k
                )
            
            # Build prompt
            if context_chunks and include_context:
                prompt = self.prompt_builder.build_rag_prompt(
                    query=query,
                    context_chunks=context_chunks
                )
            else:
                prompt = self.prompt_builder.build_simple_prompt(query)
            
            return {
                "prompt": prompt,
                "context_chunks": context_chunks,
                "context_count": len(context_chunks),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error preparing chat prompt: {e}")
            raise
    
    def generate_response(
        self,
        query: str,
        include_context: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate response for user query (non-streaming).
        
        Args:
            query: User query
            include_context: Include RAG context
            temperature: LLM temperature override
            max_tokens: Maximum tokens override
        
        Returns:
            Response with answer and metadata
        """
        try:
            start_time = time.time()
            
            # Prepare prompt
            prompt_data = self.prepare_chat_prompt(query, include_context)
            
            # Generate response
            response = self.llm_client.generate(
                prompt=prompt_data["prompt"],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"Generated response in {elapsed_time:.2f}s")
            
            return {
                "status": "success",
                "query": query,
                "response": response,
                "context": prompt_data["context_chunks"],
                "context_count": prompt_data["context_count"],
                "processing_time_ms": elapsed_time * 1000,
                "model": self.llm_client.model
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "status": "error",
                "query": query,
                "error": str(e)
            }
    
    def stream_response(
        self,
        query: str,
        include_context: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream response for user query.
        
        Args:
            query: User query
            include_context: Include RAG context
            temperature: LLM temperature override
            max_tokens: Maximum tokens override
        
        Yields:
            Response chunks with metadata
        """
        try:
            # Prepare prompt
            prompt_data = self.prepare_chat_prompt(query, include_context)
            
            # Yield initial metadata
            yield {
                "type": "start",
                "status": "streaming",
                "query": query,
                "context": prompt_data["context_chunks"],
                "context_count": prompt_data["context_count"]
            }
            
            # Stream response
            for chunk in self.llm_client.stream(
                prompt=prompt_data["prompt"],
                temperature=temperature,
                max_tokens=max_tokens
            ):
                yield {
                    "type": "chunk",
                    "data": chunk
                }
            
            # Yield completion
            yield {
                "type": "end",
                "status": "complete",
                "model": self.llm_client.model
            }
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield {
                "type": "error",
                "status": "error",
                "error": str(e)
            }
    
    def get_conversation_context(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get context for a query without generating response.
        Useful for frontend to show relevant documents.
        
        Args:
            query: User query
            top_k: Number of context chunks
        
        Returns:
            Context chunks with metadata
        """
        try:
            context_chunks = self.retriever.retrieve(
                query=query,
                top_k=top_k
            )
            
            return {
                "status": "success",
                "query": query,
                "context": context_chunks,
                "context_count": len(context_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


# Global service instance
_chat_service = None


def get_chat_service() -> ChatService:
    """Get or create chat service instance"""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service


# Convenience functions
def validate_query(query: str) -> tuple[bool, Optional[str]]:
    """Validate query"""
    return get_chat_service().validate_query(query)


def generate_response(query: str, **kwargs) -> Dict[str, Any]:
    """Generate response for query"""
    return get_chat_service().generate_response(query, **kwargs)


def stream_response(query: str, **kwargs) -> Generator:
    """Stream response for query"""
    return get_chat_service().stream_response(query, **kwargs)
