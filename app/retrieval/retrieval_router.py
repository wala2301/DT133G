from app.retrieval.retrieval_tfidf import TFIDFRetriever
from config import RETRIEVAL_METHOD
from app.retrieval.retrieval_bm25 import BM25Retriever
from app.retrieval.retrieval_dense import DenseRetriever

# Retrieval prompt selects the appropriate recovery method from config.py
class RetrievalRouter:

    def __init__(self):
        if RETRIEVAL_METHOD == "tfidf":
            self.retriever = TFIDFRetriever()
        elif RETRIEVAL_METHOD == "bm25":
            self.retriever = BM25Retriever()
        elif RETRIEVAL_METHOD == "dense":
            self.retriever = DenseRetriever()
        else:
            raise ValueError(f"Unknown retrieval method:  {RETRIEVAL_METHOD}")


    def retrieve_context_bundle(self, question: str, top_k: int):
        return self.retriever.retrieve(question, top_k)