from rank_bm25 import BM25Okapi
from app.retrieval.retrieval_base import BaseRetriever
from app.retrieval.retrieval_utils import load_documents, validate_question
from config import RETRIEVAL_TOP_K


# A BM25-based retrieval system.
class BM25Retriever(BaseRetriever):

    def __init__(self):
        self.documents = load_documents()
        self.doc_texts = [doc["text"] for doc in self.documents]
        self.tokenized_docs = [text.lower().split() for text in self.doc_texts]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, question: str, top_k: int | None = None):
        valid, result = validate_question(question)

        if not valid:
            return [], []

        question = result
        tokenized_query = question.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        effective_top_k = top_k or RETRIEVAL_TOP_K
        safe_top_k = max(1, min(effective_top_k, len(self.documents)))

        ranked_indices = scores.argsort()[-safe_top_k:][::-1]

        context = [self.documents[i]["text"] for i in ranked_indices]
        references = [
            {
                "doc_id": self.documents[i]["doc_id"],
                "source_file": self.documents[i]["source_file"],
                "preview": self.documents[i]["text"][:120]
            }
            for i in ranked_indices
        ]

        return context, references