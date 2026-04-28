import json
import csv
from pathlib import Path
from statistics import mean
from time import perf_counter

from app.retrieval.retrieval_utils import load_documents
from app.llm.llm import generate_answer
from app.evaluation.metrics import (
    compute_rouge_l,
    compute_bertscore_batch
)


# Load JSON
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Converting non-serializable types such as NumPy scalars to regular Python types.
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, "item"):
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


def evaluate_baseline():
    """
    Baseline evaluation: no retrieval step.
    All documents are concatenated and passed directly as context to the LLM.
    This measures the upper-bound answer quality when the LLM receives the full corpus,
    and serves as a comparison point for the retrieval-based methods.
    """
    questions = load_json("data/questions.json")
    ground_truth = load_json("data/ground_truth.json")
    gt_map = {item["question_id"]: item for item in ground_truth}

    # Load all documents and concatenate their text as the fixed context
    all_documents = load_documents()
    full_context = [doc["text"] for doc in all_documents]

    detailed_results = []
    rouge_scores = []

    for q in questions:
        question_id = q["id"]
        question_text = q["question"]
        reference_answer = gt_map[question_id]["reference_answer"]

        # No retrieval — measure only generation latency
        generation_start = perf_counter()
        generated_answer = generate_answer(question_text, full_context)
        generation_latency = perf_counter() - generation_start

        rouge_l = compute_rouge_l(reference_answer, generated_answer)

        detailed_results.append(
            {
                "question_id": question_id,
                "question": question_text,
                "retrieval_method": "baseline_full_context",
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "rouge_l": rouge_l,
                "retrieval_latency": 0.0,
                "generation_latency": generation_latency,
                "total_latency": generation_latency,
            }
        )

        rouge_scores.append(rouge_l)

    # Compute BERTScore in batch after all questions
    reference_answers = [r["reference_answer"] for r in detailed_results]
    generated_answers = [r["generated_answer"] for r in detailed_results]
    bertscore_values = compute_bertscore_batch(reference_answers, generated_answers)

    for i, score in enumerate(bertscore_values):
        detailed_results[i]["bertscore_f1"] = score

    summary = {
        "retrieval_method": "baseline_full_context",
        "num_documents_in_context": len(all_documents),
        "avg_rouge_l": mean(rouge_scores),
        "avg_bertscore_f1": mean(bertscore_values),
        "avg_generation_latency": mean([r["generation_latency"] for r in detailed_results]),
        "avg_total_latency": mean([r["total_latency"] for r in detailed_results]),
    }

    final_output = {
        "summary": summary,
        "detailed_results": detailed_results
    }

    save_json("results/baseline_full_evaluation.json", final_output)
    save_csv(
        "results/baseline_summary.csv",
        [summary],
        fieldnames=list(summary.keys())
    )

    print("Baseline evaluation complete.")
    print(f"  Avg ROUGE-L:      {summary['avg_rouge_l']:.4f}")
    print(f"  Avg BERTScore F1: {summary['avg_bertscore_f1']:.4f}")
    print(f"  Avg latency:      {summary['avg_total_latency']:.4f}s")
    return final_output


if __name__ == "__main__":
    evaluate_baseline()
