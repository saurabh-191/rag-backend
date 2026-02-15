from types import SimpleNamespace
from app.services.chat_service import ChatService


def test_generate_response_returns_structure():
    svc = ChatService()

    # Mock retriever to return a small context
    svc.retriever = SimpleNamespace(retrieve=lambda query, top_k=None: [{"text": "ctx", "metadata": {"source": "doc1"}, "similarity": 0.9}])

    # Mock prompt builder
    svc.prompt_builder = SimpleNamespace(build_rag_prompt=lambda query, context_chunks: "PROMPT:")+SimpleNamespace()

    # Mock LLM client
    class FakeLLM:
        model = "fake-model"
        def generate(self, prompt, **kwargs):
            return "This is an answer"
    svc.llm_client = FakeLLM()

    result = svc.generate_response("Hello world")
    assert result["status"] == "success"
    assert "response" in result
    assert result["response"] == "This is an answer"
    assert result["model"] == svc.llm_client.model


def test_stream_response_yields_chunks():
    svc = ChatService()

    svc.retriever = SimpleNamespace(retrieve=lambda query, top_k=None: [])

    # prompt builder
    svc.prompt_builder = SimpleNamespace(build_simple_prompt=lambda q: "p")

    class FakeLLMStream:
        model = "fake-model"
        def stream(self, prompt, **kwargs):
            yield "chunk1"
            yield "chunk2"
    svc.llm_client = FakeLLMStream()

    stream = svc.stream_response("Hi")
    items = list(stream)

    # Expect start, chunks..., end
    assert items[0]["type"] == "start"
    assert any(i for i in items if i.get("type") == "chunk")
    assert items[-1]["type"] == "end"
