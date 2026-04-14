import time

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_nfr1_average_response_time_under_3_seconds(monkeypatch):
    def fake_retrieve_context_bundle(question: str, top_k: int | None = None):
        return ["Context row 1", "Context row 2"], [{"doc_id": 0, "preview": "Context row 1"}]

    monkeypatch.setattr("app.api.routes.retrieve_context_bundle", fake_retrieve_context_bundle)
    monkeypatch.setattr("app.api.routes.generate_answer", lambda question, context: "fast answer")
    monkeypatch.setattr("app.api.routes.log_conversation", lambda *args, **kwargs: None)

    iterations = 30
    total_seconds = 0.0

    for _ in range(iterations):
        start = time.perf_counter()
        response = client.post("/ask", json={"question": "Can I take antibiotics?", "top_k": 3})
        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        total_seconds += elapsed

    average_seconds = total_seconds / iterations
    assert average_seconds < 3.0
