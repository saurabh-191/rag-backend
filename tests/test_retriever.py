import numpy as np
from app.rag.retriever import Retriever


def test_retriever_filters_and_limits():
    # Fake embedding client
    class FakeEmbeddingClient:
        def embed_text(self, text: str) -> np.ndarray:
            return np.array([1.0])

    # Fake vector store returns a set of results (already similarity computed)
    class FakeVectorStore:
        def search(self, embedding, top_k=10, threshold=None):
            return [
                {"text": "doc A chunk", "metadata": {"source": "A"}, "similarity": 0.9, "distance": 0.1},
                {"text": "doc B chunk", "metadata": {"source": "B"}, "similarity": 0.4, "distance": 0.6},
                {"text": "doc C chunk", "metadata": {"source": "C"}, "similarity": 0.75, "distance": 0.25},
            ]

    fake_vs = FakeVectorStore()
    fake_emb = FakeEmbeddingClient()

    retriever = Retriever(vector_store=fake_vs, embedding_client=fake_emb, top_k=2, similarity_threshold=0.5)

    results = retriever.retrieve("some query")

    # Should filter out similarity < 0.5 and limit to top_k=2
    assert len(results) <= 2
    for r in results:
        assert r["similarity"] >= 0.5


def test_retrieve_with_scores_sorts_descending():
    class FakeEmbeddingClient:
        def embed_text(self, text: str) -> np.ndarray:
            return np.array([1.0])

    class FakeVectorStore:
        def search(self, embedding, top_k=10, threshold=None):
            return [
                {"text": "low", "metadata": {}, "similarity": 0.2, "distance": 1.6},
                {"text": "high", "metadata": {}, "similarity": 0.95, "distance": 0.1},
                {"text": "mid", "metadata": {}, "similarity": 0.6, "distance": 0.8},
            ]

    retriever = Retriever(vector_store=FakeVectorStore(), embedding_client=FakeEmbeddingClient(), top_k=3, similarity_threshold=0.0)
    results = retriever.retrieve_with_scores("q", top_k=3)

    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)
