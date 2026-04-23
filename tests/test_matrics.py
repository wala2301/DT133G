from app.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k


def test_precision_at_k():
    retrieved = ["1", "2", "3"]
    relevant = ["1", "3"]

    assert precision_at_k(retrieved, relevant, 1) == 1.0
    assert precision_at_k(retrieved, relevant, 3) == 2 / 3


def test_recall_at_k():
    retrieved = ["1", "2", "3"]
    relevant = ["1", "3"]

    assert recall_at_k(retrieved, relevant, 1) == 0.5
    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_ndcg_at_k_returns_value():
    retrieved = ["1", "2", "3"]
    relevant = ["1", "3"]

    value = ndcg_at_k(retrieved, relevant, 3)
    assert 0.0 <= value <= 1.0