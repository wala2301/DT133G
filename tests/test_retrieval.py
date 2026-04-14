from app.retrieval.retrieval_tfidf import TFIDFRetriever


def test_retrieve_context_returns_list():
    retriever = TFIDFRetriever()
    context, references = retriever.retrieve("Can I take antibiotics?")

    assert isinstance(context, list)
    assert len(context) > 0
    assert isinstance(context[0], str)


def test_retrieve_context_top_k():
    retriever = TFIDFRetriever()
    context, references = retriever.retrieve("Can I take antibiotics?", top_k=2)

    assert len(context) == 2