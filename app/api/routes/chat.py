from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from rag.retriever import retrieve
from rag.prompt import build_prompt
from llm.streaming import stream_completion

router = APIRouter()

@router.get("/chat")
def chat(query: str):
    context_chunks = retrieve(query)
    prompt = build_prompt(context_chunks, query)

    return StreamingResponse(
        stream_completion(prompt),
        media_type="text/event-stream"
    )
