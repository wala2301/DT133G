import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.retrieval.retrieval_base import BaseRetriever
from config import RETRIEVAL_TOP_K, Data_DOC_PATH

class TFIDFRetriever(BaseRetriever):
    def __init__(self):
        self.documents = self.load_documents()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)

    # Load the document
    def load_documents(self, folder_path = Data_DOC_PATH):
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"{folder} not found!")

        documents = []

        for file in folder.glob("*.txt"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # Strip structured metadata prefix if present, index only the text content
                            if "| Text:" in line:
                                line = line.split("| Text:", 1)[1].strip()
                            documents.append(line)
            except Exception as e:
                print(f"Warning: Could not read {file}: {e}")

        if not documents:
            raise ValueError("No valid documents were loaded.")

        return documents

    def retrieve(self, question: str, top_k: int | None = None):

            valid, result = self.validate_question(question)

            if not valid:
                return [], []

            question = result
            # Transform question to vector
            question_vector = self.vectorizer.transform([question])
            # Compute similarity
            similarities = cosine_similarity(question_vector, self.doc_vectors)
            # threshold to avoid irrelevant results
            if similarities.max() < 0.05:
                return [], []

            effective_top_k = top_k or RETRIEVAL_TOP_K

            safe_top_k = max(1, min(effective_top_k, len(self.documents)))

            ranked = similarities.argsort()[0][-safe_top_k:][::-1] # Get best matches

            context = [self.documents[i] for i in ranked]

            references = [
                {"doc_id": int(i), "preview": self.documents[i][:120]}
                for i in ranked
            ]

            return context, references


# Input validation
    def validate_question(self, question: str):
        if question is None:
            return False, "Question cannot be None."
        question = question.strip()
        if question == "":
            return False, "Question cannot be empty."

        if len(question) < 2:
            return False, "Question is too short."

        if not any(c.isalpha() for c in question):
            return False, "Question must contain letters."

        return True, question
