import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
from app.logging.logging import anonymize_text, log_conversation, delete_old_records, LOG_FILE

# Temporary file for recording conversations
@pytest.fixture
def temp_log_file(monkeypatch):
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".jsonl") as tf:
        path = Path(tf.name)
        tf.write("")
        tf.flush()

    # Temporarily change LOG_FILE
    from app.logging import logging as log_module
    monkeypatch.setattr(log_module, "LOG_FILE", path)

    yield path
    path.unlink(missing_ok=True)


# Anonymization Test
def test_anonymize_text():
    text = "Contact me at test@example.com or +46701234567. ID: 123456"
    anonymized = anonymize_text(text)

    assert "[EMAIL]" in anonymized
    assert "[PHONE]" in anonymized
    assert "[ID]" in anonymized
    assert "test@example.com" not in anonymized
    assert "+46701234567" not in anonymized
    assert "123456" not in anonymized


# log_conversation test
def test_log_conversation(temp_log_file):
    question = "Contact me at test@example.com"
    answer = "Call me +46701234567"

    log_conversation(
        question,
        answer,
        retrieval_method="tfidf",
        retrieved_references=[{"doc_id": 1, "preview": "Medical row"}],
        top_k=2,
    )

    with open(temp_log_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    assert len(data) == 1
    record = data[0]

    assert "[EMAIL]" in record["question"]
    assert "[PHONE]" in record["answer"]
    assert "question_hash" in record
    assert "answer_hash" in record
    assert record["top_k"] == 2
    assert record["retrieved_references"] == [{"doc_id": 1, "preview": "Medical row"}]


# Retention policy test
def test_delete_old_records(temp_log_file):
    # Add an old record 31 days
    old_record = {
        "timestamp": (datetime.now(timezone.utc) - timedelta(days=31)).isoformat(),
        "question": "old_q",
        "answer": "old_a"
    }
  # Add a new record
    new_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": "new_q",
        "answer": "new_a"
    }

    with open(temp_log_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(old_record) + "\n")
        f.write(json.dumps(new_record) + "\n")

    # Implementing the deletion policy
    delete_old_records(days=30)

    with open(temp_log_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Only new record should remain
    assert len(data) == 1
    assert data[0]["question"] == "new_q"