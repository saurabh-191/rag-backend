import requests
import json
from typing import Generator, Optional, Dict, Any, List
from abc import ABC, abstractmethod

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream response from LLM"""
        pass


class OllamaClient(LLMClient):
    """Ollama LLM Client for local Llama 3.2 model"""
    
    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.LLM_MODEL_NAME,
        temperature: float = settings.LLM_TEMPERATURE,
        top_p: float = settings.LLM_TOP_P,
        top_k: int = settings.LLM_TOP_K,
        context_window: int = settings.LLM_CONTEXT_WINDOW,
        timeout: int = 120
    ):
        """
        Initialize Ollama LLM client
        
        Args:
            base_url: Ollama server base URL
            model: Model name (e.g., 'llama3.2')
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter
            top_k: Top-K sampling parameter
            context_window: Model context window size
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.context_window = context_window
        self.timeout = timeout
        
        logger.info(f"Initialized Ollama client with model: {self.model}")
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify connection to Ollama server"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info(f"Successfully connected to Ollama at {self.base_url}")
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            if self.model not in model_names:
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available models: {model_names}. "
                    f"Pull it with: ollama pull {self.model}"
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            raise
    
    def _build_payload(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build request payload for Ollama"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
            "num_predict": kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
        }
        return payload
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response from Llama 3.2 via Ollama
        
        Args:
            prompt: Input prompt
            system_prompt: System prompt for context
            **kwargs: Additional parameters (temperature, top_p, top_k, max_tokens)
        
        Returns:
            Generated text response
        
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        try:
            payload = self._build_payload(prompt, system_prompt, **kwargs)
            
            logger.debug(f"Generating response with model: {self.model}")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            text = result.get("response", "").strip()
            
            logger.debug(f"Generated response ({len(text)} tokens)")
            return text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating response: {e}")
            raise
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing Ollama response: {e}")
            raise
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream response from Llama 3.2 via Ollama
        
        Args:
            prompt: Input prompt
            system_prompt: System prompt for context
            **kwargs: Additional parameters (temperature, top_p, top_k, max_tokens)
        
        Yields:
            Text chunks as they are generated
        
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        try:
            payload = self._build_payload(prompt, system_prompt, **kwargs)
            payload["stream"] = True
            
            logger.debug(f"Streaming response with model: {self.model}")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing streaming response line: {e}")
                        continue
            
            logger.debug("Streaming response completed")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error streaming response: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            response = requests.get(
                f"{self.base_url}/api/show",
                params={"name": self.model},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting model info: {e}")
            raise


class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    _clients = {
        "ollama": OllamaClient,
    }
    
    @classmethod
    def create_client(
        cls,
        client_type: str = settings.LLM_TYPE,
        **kwargs
    ) -> LLMClient:
        """
        Create LLM client instance
        
        Args:
            client_type: Type of client ('ollama')
            **kwargs: Additional arguments for client initialization
        
        Returns:
            LLMClient instance
        
        Raises:
            ValueError: If client type is not supported
        """
        if client_type not in cls._clients:
            raise ValueError(
                f"Unsupported LLM client type: {client_type}. "
                f"Supported types: {list(cls._clients.keys())}"
            )
        
        client_class = cls._clients[client_type]
        logger.info(f"Creating {client_type} LLM client")
        return client_class(**kwargs)


# Default client instance
llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the LLM client"""
    global llm_client
    if llm_client is None:
        llm_client = LLMClientFactory.create_client()
    return llm_client


def set_llm_client(client: LLMClient):
    """Set custom LLM client"""
    global llm_client
    llm_client = client
