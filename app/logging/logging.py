from pathlib import Path
import json
from datetime import datetime, timedelta, timezone
import hashlib
import re
from config import LOG_FILE_PATH, LOG_RETENTION_DAYS, LOG_CLEANUP_EVERY_N_WRITES




LOG_FILE = LOG_FILE_PATH
_WRITE_COUNT = 0

# NFR_3 Hide Sensitive Data
def anonymize_text(text: str) -> str:
    text = re.sub(r"\S+@\S+", "[EMAIL]", text)
    text = re.sub(r"\+?\d[\d\s\-]{7,}", "[PHONE]", text)
    text = re.sub(r"\b\d{6,}\b", "[ID]", text)
    return text


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Load existing logs
def _load_records(file_path: Path) -> list[dict]:
    if not file_path.exists():
        return []

    raw_text = file_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    if raw_text.startswith("["):
        try:
            data = json.loads(raw_text)
            if isinstance(data, list):
                return [record for record in data if isinstance(record, dict)]
        except json.JSONDecodeError:
            return []

    records = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records

# Write logs
def _write_records(file_path: Path, records: list[dict]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Delete old logs
def delete_old_records(days: int = LOG_RETENTION_DAYS):
    records = _load_records(LOG_FILE)
    if not records:
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered_data = []
    for record in records:
        timestamp = datetime.fromisoformat(record["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)

        if timestamp > cutoff:
            filtered_data.append(record)
    _write_records(LOG_FILE, filtered_data)

# NFR_7 Log interaction
def log_conversation(question: str, answer: str, retrieval_method: str, retrieved_references: list[dict] | None = None, top_k: int | None = None,  latency: float | None = None
):
    global _WRITE_COUNT
    _WRITE_COUNT += 1

    if _WRITE_COUNT % max(1, LOG_CLEANUP_EVERY_N_WRITES) == 0:
        delete_old_records(days=LOG_RETENTION_DAYS)

    anonymized_question = anonymize_text(question)
    anonymized_answer = anonymize_text(answer)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # hashed data (privacy safe)
        "question_hash": hash_text(question),
        "answer_hash": hash_text(answer),
        # anonymized text
        "question": anonymized_question,
        "answer": anonymized_answer,
        # retrieval metadata
        "retrieval_method": retrieval_method,
        "top_k": top_k,
        # evaluation metadata
        "latency": latency,
        "num_references": len(retrieved_references or []),
        "retrieved_references": retrieved_references or []
    }

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")