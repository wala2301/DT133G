from app.retrieval.retrieval_tfidf import TFIDFRetriever
from config import RETRIEVAL_METHOD


class RetrievalRouter:

    def __init__(self):
        if RETRIEVAL_METHOD == "tfidf":
            self.retriever = TFIDFRetriever()
        else:
            raise ValueError("Unknown retrieval method")


    def retrieve_context_bundle(self, question: str, top_k: int):
        return self.retriever.retrieve(question, top_k)