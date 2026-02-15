from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


class ChatMessageRole(str, Enum):
    """Chat message role enumeration"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Single chat message"""
    role: ChatMessageRole
    content: str = Field(..., min_length=1, max_length=50000)


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    stream: bool = Field(default=True, description="Enable streaming response")
    temperature: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=2.0, 
        description="LLM temperature (0.0 to 2.0)"
    )
    top_p: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Nucleus sampling parameter"
    )
    top_k: Optional[int] = Field(
        default=None, 
        ge=1, 
        description="Top-K sampling parameter"
    )
    max_tokens: Optional[int] = Field(
        default=None, 
        ge=1, 
        description="Maximum tokens to generate"
    )
    include_context: bool = Field(
        default=True, 
        description="Include retrieved context in response"
    )
    context_count: int = Field(
        default=5, 
        ge=1, 
        le=20, 
        description="Number of context chunks to retrieve"
    )
    
    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        """Validate message is not just whitespace"""
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace only")
        return v.strip()


class HealthCheckRequest(BaseModel):
    """Health check request model"""
    check_llm: bool = Field(default=True, description="Check LLM connection")
    check_vector_store: bool = Field(default=True, description="Check vector store")


class DocumentUploadRequest(BaseModel):
    """Document upload request model"""
    filename: str = Field(..., description="Name of the document")
    file_type: str = Field(..., description="File type (pdf, txt, docx, md)")
    
    @field_validator("file_type")
    @classmethod
    def validate_file_type(cls, v):
        """Validate file type is supported"""
        supported = ["pdf", "txt", "docx", "md"]
        if v.lower() not in supported:
            raise ValueError(f"Unsupported file type. Supported: {supported}")
        return v.lower()


class IngestionRequest(BaseModel):
    """Document ingestion request model"""
    file_path: str = Field(..., description="Path to the document")
    chunk_size: Optional[int] = Field(
        default=None, 
        ge=100, 
        description="Chunk size for splitting"
    )
    chunk_overlap: Optional[int] = Field(
        default=None, 
        ge=0, 
        description="Overlap between chunks"
    )


class ContextRetrievalRequest(BaseModel):
    """Context retrieval request model"""
    query: str = Field(..., min_length=1, max_length=10000, description="Search query")
    top_k: int = Field(
        default=5, 
        ge=1, 
        le=20, 
        description="Number of results to retrieve"
    )
    similarity_threshold: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Minimum similarity score"
    )


class RebuildIndexRequest(BaseModel):
    """Rebuild vector index request model"""
    clear_existing: bool = Field(
        default=False, 
        description="Clear existing index before rebuild"
    )
    data_path: Optional[str] = Field(
        default=None, 
        description="Custom path to data directory"
    )
