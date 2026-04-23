import json
from statistics import mean

from config import RETRIEVAL_METHOD, RETRIEVAL_TOP_K
from app.retrieval.retrieval_router import RetrievalRouter
from app.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

# Load JSON file
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Evaluate only the retrieval phase without the generation phase
def evaluate_retrieval():
    # Load question and ground truth
    questions = load_json("data/questions.json")
    ground_truth = load_json("data/ground_truth.json")
    # Convert ground truth to a dictionary based on question_id
    gt_map = {item["question_id"]: item for item in ground_truth}
    retrieval_system = RetrievalRouter()  # Create retrieval router
    results = []  # Save the results for each question

    # Addressing every question
    for q in questions:
        context, references = retrieval_system.retrieve_context_bundle(q["question"], RETRIEVAL_TOP_K)
        # Documents recovered by the regime
        retrieved_ids = [ref["doc_id"] for ref in references]
        # Correct documents
        relevant_ids = gt_map[q["id"]]["relevant_docs"]

        # Calculating measurements
        result = {
            "question_id": q["id"],
            "precision@1": precision_at_k(retrieved_ids, relevant_ids, 1),
            "precision@3": precision_at_k(retrieved_ids, relevant_ids, 3),
            "precision@5": precision_at_k(retrieved_ids, relevant_ids, 5),
            "recall@1": recall_at_k(retrieved_ids, relevant_ids, 1),
            "recall@3": recall_at_k(retrieved_ids, relevant_ids, 3),
            "recall@5": recall_at_k(retrieved_ids, relevant_ids, 5),
            "ndcg@1": ndcg_at_k(retrieved_ids, relevant_ids, 1),
            "ndcg@3": ndcg_at_k(retrieved_ids, relevant_ids, 3),
            "ndcg@5": ndcg_at_k(retrieved_ids, relevant_ids, 5)
        }
        results.append(result)

    # Calculating Final Averages
    summary = {
        "retrieval_method": RETRIEVAL_METHOD,
        "avg_precision@1": mean(r["precision@1"] for r in results),
        "avg_precision@3": mean(r["precision@3"] for r in results),
        "avg_precision@5": mean(r["precision@5"] for r in results),
        "avg_recall@1": mean(r["recall@1"] for r in results),
        "avg_recall@3": mean(r["recall@3"] for r in results),
        "avg_recall@5": mean(r["recall@5"] for r in results),
        "avg_ndcg@1": mean(r["ndcg@1"] for r in results),
        "avg_ndcg@3": mean(r["ndcg@3"] for r in results),
        "avg_ndcg@5": mean(r["ndcg@5"] for r in results)
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    evaluate_retrieval()