from app.retrieval.retrieval_bm25 import BM25Retriever


def test_bm25_returns_context_and_references():
    retriever = BM25Retriever()
    context, references = retriever.retrieve("How do I authenticate to the GitHub REST API?", top_k=2)

    assert isinstance(context, list)
    assert isinstance(references, list)
    assert len(context) > 0
    assert len(references) > 0