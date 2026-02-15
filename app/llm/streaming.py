from typing import Generator, Optional

from app.llm.client import get_llm_client
from app.core.logging import get_logger

logger = get_logger(__name__)


def stream_completion(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 40,
    max_tokens: int = 2048,
) -> Generator[str, None, None]:
    """
    Stream completion from LLM with given prompt
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt for context
        temperature: Sampling temperature (0.0 to 2.0)
        top_p: Nucleus sampling parameter
        top_k: Top-K sampling parameter
        max_tokens: Maximum tokens to generate
    
    Yields:
        Text chunks as they are generated
    
    Raises:
        Exception: If streaming fails
    """
    try:
        logger.debug(f"Starting stream completion for prompt: {prompt[:50]}...")
        
        client = get_llm_client()
        
        # Stream from the LLM client
        for chunk in client.stream(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        ):
            yield chunk
        
        logger.debug("Stream completion finished successfully")
        
    except Exception as e:
        logger.error(f"Error in stream completion: {e}")
        raise


async def stream_completion_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    **kwargs
) -> Generator[str, None, None]:
    """
    Async version of stream completion (for FastAPI streaming)
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt for context
        **kwargs: Additional parameters for streaming
    
    Yields:
        Text chunks as they are generated
    """
    try:
        logger.debug(f"Starting async stream completion")
        
        for chunk in stream_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        ):
            yield chunk
            
    except Exception as e:
        logger.error(f"Error in async stream completion: {e}")
        raise
