from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_ask_endpoint_success():
    response = client.post(
        "/ask",
        json={"question": "Can I take antibiotics?"}
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


def test_ask_endpoint_empty_question():
    response = client.post(
        "/ask",
        json={"question": ""}
    )

    assert response.status_code == 422


def test_ask_endpoint_returns_generated_response(monkeypatch):
    class FakeRetrievalSystem:
        def retrieve_context_bundle(self, question: str, top_k: int | None = None):
            assert top_k == 3
            return ["Context row 1", "Context row 2"], [{"doc_id": 0, "preview": "Context row 1"}]

    def fake_generate_answer(question: str, context: list[str]):
        assert question == "Can I take antibiotics?"
        assert context == ["Context row 1", "Context row 2"]
        return "Generated response from model"

    monkeypatch.setattr("app.api.routes.retrieval_system", FakeRetrievalSystem())
    monkeypatch.setattr("app.api.routes.generate_answer", fake_generate_answer)
    monkeypatch.setattr("app.api.routes.log_conversation", lambda *args, **kwargs: None)

    response = client.post(
        "/ask",
        json={"question": "Can I take antibiotics?"}
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "Generated response from model"}


def test_ask_endpoint_accepts_top_k(monkeypatch):
    class FakeRetrievalSystem:
        def retrieve_context_bundle(self, question: str, top_k: int | None = None):
            assert top_k == 2
            return ["Context row 1", "Context row 2"], [{"doc_id": 0, "preview": "Context row 1"}, {"doc_id": 1, "preview": "Context row 2"}]

    monkeypatch.setattr("app.api.routes.retrieval_system", FakeRetrievalSystem())
    monkeypatch.setattr("app.api.routes.generate_answer", lambda question, context: "Generated response from model")
    monkeypatch.setattr("app.api.routes.log_conversation", lambda *args, **kwargs: None)

    response = client.post(
        "/ask",
        json={"question": "Can I take antibiotics?", "top_k": 2}
    )

    assert response.status_code == 200