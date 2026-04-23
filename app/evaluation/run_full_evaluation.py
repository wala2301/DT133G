import json
import csv
from pathlib import Path
from statistics import mean
from time import perf_counter

from config import RETRIEVAL_METHOD, RETRIEVAL_TOP_K
from app.retrieval.retrieval_router import RetrievalRouter
from app.llm.llm import generate_answer
from app.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    compute_rouge_l,
    compute_bertscore_batch
)
from app.evaluation.statistics_analysis import (
    check_normality,
    compare_three_groups,
    correlation_analysis
)


# Load JSON
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Converting non-save JSON types such as NumPy scalars to regular Python types.
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, "item"):  # NumPy scalars
        return obj.item()
    else:
        return obj


# Save data in JSON form
def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    safe_data = make_json_serializable(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_data, f, indent=2, ensure_ascii=False)

# Save the results in CSV
def save_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Perform a complete evaluation of a single retrieval method
def evaluate_full():
    # Load the questions and ground truth
    questions = load_json("data/questions.json")
    ground_truth = load_json("data/ground_truth.json")
    # Convert ground truth to a dictionary based on question_id
    gt_map = {item["question_id"]: item for item in ground_truth}

    retrieval_system = RetrievalRouter()  # Create a rollback prompt according to config.py
    detailed_results = []  # Store the detailed results for each question here

    # Lists for saving values for later use in normality and correlation
    precision3_scores = []
    recall3_scores = []
    ndcg3_scores = []
    rouge_scores = []
    bert_scores = []

    # Go through each question
    for q in questions:
        question_id = q["id"]
        question_text = q["question"]
        # Measures retrieval time only
        retrieval_start = perf_counter()
        context, references = retrieval_system.retrieve_context_bundle(
            question_text,
            RETRIEVAL_TOP_K
        )
        retrieval_latency = perf_counter() - retrieval_start
        # Extracting IDs of retrieved documents
        retrieved_ids = [ref["doc_id"] for ref in references]
        # Extracting correct documents from ground truth
        relevant_ids = gt_map[question_id]["relevant_docs"]
        # Extracting the reference answer
        reference_answer = gt_map[question_id]["reference_answer"]

        # Calculating retrieval metrics
        p1 = precision_at_k(retrieved_ids, relevant_ids, 1)
        p3 = precision_at_k(retrieved_ids, relevant_ids, 3)
        p5 = precision_at_k(retrieved_ids, relevant_ids, 5)

        r1 = recall_at_k(retrieved_ids, relevant_ids, 1)
        r3 = recall_at_k(retrieved_ids, relevant_ids, 3)
        r5 = recall_at_k(retrieved_ids, relevant_ids, 5)

        n1 = ndcg_at_k(retrieved_ids, relevant_ids, 1)
        n3 = ndcg_at_k(retrieved_ids, relevant_ids, 3)
        n5 = ndcg_at_k(retrieved_ids, relevant_ids, 5)

        # Measure generation time only
        generation_start = perf_counter()
        generated_answer = generate_answer(question_text, context)  # Generate answer
        generation_latency = perf_counter() - generation_start

        # Total time = Retrieval time + Generation time
        total_latency = retrieval_latency + generation_latency

        # Calculate ROUGE-L
        rouge_l = compute_rouge_l(reference_answer, generated_answer)

        # Save the detailed result for this question
        detailed_results.append(
            {
                "question_id": question_id,
                "question": question_text,
                "retrieval_method": RETRIEVAL_METHOD,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": relevant_ids,
                "precision@1": p1,
                "precision@3": p3,
                "precision@5": p5,
                "recall@1": r1,
                "recall@3": r3,
                "recall@5": r5,
                "ndcg@1": n1,
                "ndcg@3": n3,
                "ndcg@5": n5,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "rouge_l": rouge_l,
                "retrieval_latency": retrieval_latency,
                "generation_latency": generation_latency,
                "total_latency": total_latency,

            }
        )

        # Storing analytical values
        precision3_scores.append(p3)
        recall3_scores.append(r3)
        ndcg3_scores.append(n3)
        rouge_scores.append(rouge_l)
    # Calculate BERTScore after completing all questions
    reference_answers = [r["reference_answer"] for r in detailed_results]
    generated_answers = [r["generated_answer"] for r in detailed_results]

    bertscore_values = compute_bertscore_batch(reference_answers, generated_answers)

    # Add BERTScore for each question
    for i, score in enumerate(bertscore_values):
        detailed_results[i]["bertscore_f1"] = score

    bert_scores = bertscore_values

    # Calculating Final Averages
    summary = {
        "retrieval_method": RETRIEVAL_METHOD,
        "top_k": RETRIEVAL_TOP_K,
        "avg_precision@1": mean([r["precision@1"] for r in detailed_results]),
        "avg_precision@3": mean([r["precision@3"] for r in detailed_results]),
        "avg_precision@5": mean([r["precision@5"] for r in detailed_results]),
        "avg_recall@1": mean([r["recall@1"] for r in detailed_results]),
        "avg_recall@3": mean([r["recall@3"] for r in detailed_results]),
        "avg_recall@5": mean([r["recall@5"] for r in detailed_results]),
        "avg_ndcg@1": mean([r["ndcg@1"] for r in detailed_results]),
        "avg_ndcg@3": mean([r["ndcg@3"] for r in detailed_results]),
        "avg_ndcg@5": mean([r["ndcg@5"] for r in detailed_results]),
        "avg_rouge_l": mean([r["rouge_l"] for r in detailed_results]),
        "avg_bertscore_f1": mean([r["bertscore_f1"] for r in detailed_results]),
        "avg_retrieval_latency": mean([r["retrieval_latency"] for r in detailed_results]),
        "avg_generation_latency": mean([r["generation_latency"] for r in detailed_results]),
        "avg_total_latency": mean([r["total_latency"] for r in detailed_results]),
    }

    # Normality checks
    normality = {
        "precision@3": check_normality(precision3_scores),
        "recall@3": check_normality(recall3_scores),
        "ndcg@3": check_normality(ndcg3_scores),
        "rouge_l": check_normality(rouge_scores),
        "bertscore_f1": check_normality(bert_scores),
    }

    # Correlation analysis between recall quality and response quality
    correlations = {
        "precision3_vs_rouge": correlation_analysis(precision3_scores, rouge_scores, method="pearson"),
        "precision3_vs_bertscore": correlation_analysis(precision3_scores, bert_scores, method="pearson"),
        "recall3_vs_rouge": correlation_analysis(recall3_scores, rouge_scores, method="pearson"),
        "ndcg3_vs_rouge": correlation_analysis(ndcg3_scores, rouge_scores, method="pearson"),
    }

    # Compiling all outputs
    final_output = {
        "summary": summary,
        "normality": normality,
        "correlations": correlations,
        "detailed_results": detailed_results
    }

    # Save results in JSON form
    save_json(f"results/{RETRIEVAL_METHOD}_full_evaluation.json", final_output)

    # Save CSV summary row
    save_csv(
        f"results/{RETRIEVAL_METHOD}_summary.csv",
        [summary],
        fieldnames=list(summary.keys())
    )

    # Save CSV detailed rows
    detailed_fieldnames = [
        "question_id",
        "question",
        "retrieval_method",
        "retrieved_ids",
        "relevant_ids",
        "precision@1",
        "precision@3",
        "precision@5",
        "recall@1",
        "recall@3",
        "recall@5",
        "ndcg@1",
        "ndcg@3",
        "ndcg@5",
        "reference_answer",
        "generated_answer",
        "rouge_l",
        "bertscore_f1"
    ]

    csv_ready_rows = []
    for row in detailed_results:
        csv_row = row.copy()
        csv_row["retrieved_ids"] = ", ".join(map(str, csv_row["retrieved_ids"]))
        csv_row["relevant_ids"] = ", ".join(map(str, csv_row["relevant_ids"]))
        csv_ready_rows.append(csv_row)

    save_csv(
        f"results/{RETRIEVAL_METHOD}_detailed_results.csv",
        csv_ready_rows,
        fieldnames=detailed_fieldnames
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSaved files:")
    print(f"- results/{RETRIEVAL_METHOD}_full_evaluation.json")
    print(f"- results/{RETRIEVAL_METHOD}_summary.csv")
    print(f"- results/{RETRIEVAL_METHOD}_detailed_results.csv")



if __name__ == "__main__":
    evaluate_full()