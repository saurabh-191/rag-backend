from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio

from app.core.config import settings
from app.core.logging import log_startup_info, get_logger
from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.services.chat_service import get_chat_service
from app.services.ingestion import get_ingestion_service

logger = get_logger(__name__)

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_service = get_chat_service()
ingestion_service = get_ingestion_service()


@app.on_event("startup")
async def startup_event():
    log_startup_info()
    logger.info("Application startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown")


@app.get("/health")
async def health():
    """Health check endpoint: checks basic service components."""
    try:
        stats = ingestion_service.get_index_stats()
        # Minimal LLM check (best-effort)
        llm_ok = True
        try:
            client = chat_service.llm_client
            # call lightweight method if available
            _ = getattr(client, "get_model_info", lambda: {})()
        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            llm_ok = False

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok" if llm_ok else "degraded",
                "llm_ok": llm_ok,
                "index": stats,
            },
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    try:
        result = chat_service.generate_response(
            query=request.message,
            include_context=request.include_context,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events (SSE)."""
    async def event_generator():
        try:
            for item in chat_service.stream_response(
                query=request.message,
                include_context=request.include_context,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                # Convert dict to SSE data event
                data = json.dumps(item, default=str)
                yield f"data: {data}\n\n"
                # small sleep to yield control
                await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            err = json.dumps({"type": "error", "error": str(e)})
            yield f"data: {err}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/ingest/file")
async def ingest_file(payload: dict):
    """Ingest a single file by path (expects JSON {"file_path": "..."})."""
    file_path = payload.get("file_path")
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required")
    result = ingestion_service.ingest_file(file_path)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


@app.post("/ingest/directory")
async def ingest_directory(payload: dict):
    """Ingest all documents from a directory."""
    directory = payload.get("directory")
    if not directory:
        raise HTTPException(status_code=400, detail="directory is required")
    result = ingestion_service.ingest_directory(directory)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
    )
