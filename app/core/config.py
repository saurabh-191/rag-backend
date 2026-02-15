import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration"""

    # ============ APP SETTINGS ============
    APP_NAME: str = "RAG Chatbot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # ============ SERVER SETTINGS ============
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    RELOAD: bool = os.getenv("RELOAD", "True").lower() == "true"
    
    # ============ CORS SETTINGS ============
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    
    # ============ LLM SETTINGS (Llama 3.2) ============
    # Using Ollama for local Llama 3.2
    LLM_TYPE: str = "ollama"  # ollama, local, or api
    LLM_MODEL_NAME: str = "llama3.2"  # Model name for Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # LLM Model Parameters
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.7))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", 0.95))
    LLM_TOP_K: int = int(os.getenv("LLM_TOP_K", 40))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 2048))
    LLM_CONTEXT_WINDOW: int = int(os.getenv("LLM_CONTEXT_WINDOW", 8192))
    
    # ============ EMBEDDING SETTINGS ============
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", 768))
    
    # ============ VECTOR STORE SETTINGS ============
    VECTOR_STORE_TYPE: str = "chroma"  # chroma or faiss
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/chroma")
    CHROMA_COLLECTION_NAME: str = "rag-documents"
    
    # ============ RAG SETTINGS ============
    # Document Loading
    DATA_RAW_PATH: str = os.getenv("DATA_RAW_PATH", "./data/raw")
    DATA_PROCESSED_PATH: str = os.getenv("DATA_PROCESSED_PATH", "./data/processed")
    
    # Chunking Strategy
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Retrieval
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", 5))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.3))
    
    # Supported Document Types
    SUPPORTED_FILE_TYPES: list = [".pdf", ".txt", ".docx", ".md"]
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 50))
    
    # ============ LOGGING SETTINGS ============
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "logs/app.log")
    
    # ============ SECURITY SETTINGS ============
    API_KEY: Optional[str] = os.getenv("API_KEY", None)
    ENABLE_PROMPT_INJECTION_GUARD: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Instantiate settings
settings = Settings()
