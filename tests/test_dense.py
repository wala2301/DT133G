from app.retrieval.retrieval_dense import DenseRetriever


def test_dense_returns_context_and_references():
    retriever = DenseRetriever()
    context, references = retriever.retrieve("How do I authenticate to the GitHub REST API?", top_k=2)

    assert isinstance(context, list)
    assert isinstance(references, list)
    assert len(context) > 0
    assert len(references) > 0