from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.retrieval.retrieval_base import BaseRetriever
from app.retrieval.retrieval_utils import load_documents, validate_question
from config import RETRIEVAL_TOP_K, DENSE_MODEL_NAME

# A semantic retrieval that relies on embeddings using SentenceTransformer.
class DenseRetriever(BaseRetriever):

    def __init__(self):
        self.documents = load_documents()
        self.doc_texts = [doc["text"] for doc in self.documents]

        self.model = SentenceTransformer(DENSE_MODEL_NAME)
        self.doc_embeddings = self.model.encode(self.doc_texts)

    def retrieve(self, question: str, top_k: int | None = None):
        valid, result = validate_question(question)

        if not valid:
            return [], []

        question = result
        question_embedding = self.model.encode([question])
        similarities = cosine_similarity(question_embedding, self.doc_embeddings)[0]

        effective_top_k = top_k or RETRIEVAL_TOP_K
        safe_top_k = max(1, min(effective_top_k, len(self.documents)))

        ranked_indices = similarities.argsort()[-safe_top_k:][::-1]

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