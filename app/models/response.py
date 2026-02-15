from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ResponseStatus(str, Enum):
    """Response status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PENDING = "pending"


class ContextChunk(BaseModel):
    """Single context chunk"""
    text: str = Field(..., description="Chunk text content")
    source: str = Field(..., description="Source document")
    page: Optional[int] = Field(None, description="Page number (if applicable)")
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ChatResponse(BaseModel):
    """Chat response model"""
    status: ResponseStatus = Field(..., description="Response status")
    message: str = Field(..., description="AI response message")
    context: Optional[List[ContextChunk]] = Field(None, description="Retrieved context chunks")
    tokens_generated: Optional[int] = Field(None, ge=0, description="Number of tokens generated")
    processing_time_ms: Optional[float] = Field(None, ge=0, description="Processing time in milliseconds")
    model_used: Optional[str] = Field(None, description="Model name used for generation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "This is the AI response",
                "context": [
                    {
                        "text": "Context snippet...",
                        "source": "document.pdf",
                        "similarity_score": 0.95
                    }
                ],
                "tokens_generated": 150,
                "processing_time_ms": 2500.5,
                "model_used": "llama3.2"
            }
        }


class StreamingChatResponse(BaseModel):
    """Streaming chat response model"""
    status: ResponseStatus = Field(..., description="Response status")
    chunk: str = Field(..., description="Response chunk")
    is_final: bool = Field(default=False, description="Is this the final chunk")
    context: Optional[List[ContextChunk]] = Field(None, description="Context (only in first chunk)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "chunk": "This is a chunk ",
                "is_final": False
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: ResponseStatus = Field(..., description="Overall status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    llm_status: Optional[str] = Field(None, description="LLM connection status")
    vector_store_status: Optional[str] = Field(None, description="Vector store status")
    model_name: Optional[str] = Field(None, description="Loaded model name")
    uptime_seconds: Optional[float] = Field(None, ge=0, description="Service uptime")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


class DocumentUploadResponse(BaseModel):
    """Document upload response model"""
    status: ResponseStatus = Field(..., description="Upload status")
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Uploaded filename")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    chunks_created: int = Field(..., ge=0, description="Number of chunks created")
    embedding_time_ms: float = Field(..., ge=0, description="Time to generate embeddings")
    message: Optional[str] = Field(None, description="Additional message")


class IngestionResponse(BaseModel):
    """Document ingestion response model"""
    status: ResponseStatus = Field(..., description="Ingestion status")
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Document filename")
    total_chunks: int = Field(..., ge=0, description="Total chunks created")
    chunks_indexed: int = Field(..., ge=0, description="Chunks successfully indexed")
    processing_time_ms: float = Field(..., ge=0, description="Total processing time")
    message: str = Field(..., description="Status message")


class ContextRetrievalResponse(BaseModel):
    """Context retrieval response model"""
    status: ResponseStatus = Field(..., description="Retrieval status")
    query: str = Field(..., description="Original query")
    results: List[ContextChunk] = Field(..., description="Retrieved context chunks")
    total_results: int = Field(..., ge=0, description="Total results found")
    processing_time_ms: float = Field(..., ge=0, description="Retrieval time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "query": "What is RAG?",
                "results": [
                    {
                        "text": "RAG stands for Retrieval Augmented Generation...",
                        "source": "doc1.pdf",
                        "similarity_score": 0.98
                    }
                ],
                "total_results": 1,
                "processing_time_ms": 150.5
            }
        }


class RebuildIndexResponse(BaseModel):
    """Rebuild index response model"""
    status: ResponseStatus = Field(..., description="Rebuild status")
    documents_processed: int = Field(..., ge=0, description="Number of documents processed")
    total_chunks: int = Field(..., ge=0, description="Total chunks indexed")
    processing_time_ms: float = Field(..., ge=0, description="Total processing time")
    message: str = Field(..., description="Status message")


class ErrorResponse(BaseModel):
    """Error response model"""
    status: ResponseStatus = Field(default=ResponseStatus.ERROR, description="Error status")
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "error_code": "LLM_CONNECTION_ERROR",
                "message": "Failed to connect to LLM service",
                "timestamp": "2024-01-10T12:30:00Z"
            }
        }


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=10, ge=1, le=100, description="Items per page")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    status: ResponseStatus = Field(..., description="Response status")
    data: List[Dict[str, Any]] = Field(..., description="Response data")
    total_count: int = Field(..., ge=0, description="Total number of items")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
