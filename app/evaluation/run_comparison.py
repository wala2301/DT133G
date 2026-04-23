import json
import csv
from pathlib import Path

from app.evaluation.statistics_analysis import compare_three_groups, cohens_d


# Load JSON file
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Save the data as JSON
def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Save the data as CSV
def save_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Extracting a list of values for a specific metric from detailed_results
def extract_metric(data, metric_name):
    return [item[metric_name] for item in data["detailed_results"]]


# Statistically compare the three methods TF-IDF, BM25 and Dense across all metrics.
def run_comparison():
    # Load the results for the three methods
    tfidf = load_json("results/tfidf_full_evaluation.json")
    bm25 = load_json("results/bm25_full_evaluation.json")
    dense = load_json("results/dense_full_evaluation.json")

    # List of metrics we want to compare
    metrics_to_compare = [
        "precision@1",
        "precision@3",
        "precision@5",
        "recall@1",
        "recall@3",
        "recall@5",
        "ndcg@1",
        "ndcg@3",
        "ndcg@5",
        "rouge_l",
        "bertscore_f1"
    ]

    # List to save results to be written in CSV
    comparison_results = []
    json_output = {}  # Dictionary for saving detailed results in JSON

    # Compare each metric separately
    for metric in metrics_to_compare:
        # Extracting the values of this scale from each method
        tfidf_scores = extract_metric(tfidf, metric)
        bm25_scores = extract_metric(bm25, metric)
        dense_scores = extract_metric(dense, metric)

        # Comparing the three methods statistically
        overall_test = compare_three_groups(tfidf_scores, bm25_scores, dense_scores)

        # Cohen's calculation between each pair of roads
        pairwise = {
            "tfidf_vs_bm25": cohens_d(tfidf_scores, bm25_scores),
            "tfidf_vs_dense": cohens_d(tfidf_scores, dense_scores),
            "bm25_vs_dense": cohens_d(bm25_scores, dense_scores)
        }

        metric_result = {
            "metric": metric,
            "test_used": overall_test["test"],
            "statistic": overall_test["statistic"],
            "p_value": overall_test["p_value"],
            "effect_size_tfidf_vs_bm25": pairwise["tfidf_vs_bm25"],
            "effect_size_tfidf_vs_dense": pairwise["tfidf_vs_dense"],
            "effect_size_bm25_vs_dense": pairwise["bm25_vs_dense"]
        }

        comparison_results.append(metric_result)
        # Detailed version of the JSON file
        json_output[metric] = {
            "overall_test": overall_test,
            "effect_sizes": pairwise
        }

        # Save the final results
    save_json("results/comparison_results.json", json_output)

    save_csv(
        "results/comparison_results.csv",
        comparison_results,
        fieldnames=[
            "metric",
            "test_used",
            "statistic",
            "p_value",
            "effect_size_tfidf_vs_bm25",
            "effect_size_tfidf_vs_dense",
            "effect_size_bm25_vs_dense"
        ]
    )

    print("Saved files:")
    print("- results/comparison_results.json")
    print("- results/comparison_results.csv")


if __name__ == "__main__":
    run_comparison()