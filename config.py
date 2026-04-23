import os
from pathlib import Path
from dotenv import load_dotenv

# Project general settings file
load_dotenv()

# Ensure system component interchangeability (tfidf / bm25 / dense)
RETRIEVAL_METHOD = os.getenv("RETRIEVAL_METHOD", "dense")


# LLM configuration
LLM_API_URL: str = os.getenv("LLM_API_URL")
LLM_API_KEY: str = os.getenv("LLM_API_KEY")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", 2.5))

# Retrieval configuration
Data_DOC_PATH: Path = Path(os.getenv("Data_DOC_PATH", "data/documents")) # Document folder path
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 3)) # Number of most relevant documents
RETRIEVAL_MAX_TOP_K = int(os.getenv("RETRIEVAL_MAX_TOP_K", 10))

# Logging configuration
LOG_FILE_PATH: Path = Path(os.getenv("LOG_FILE_PATH", "logs/ai_backend.json"))
LOG_RETENTION_DAYS: int = int(os.getenv("LOG_RETENTION_DAYS", 30))
LOG_CLEANUP_EVERY_N_WRITES: int = int(os.getenv("LOG_CLEANUP_EVERY_N_WRITES", 20))

# Key to encrypt questions and answers
FERNET_KEY = os.getenv("FERNET_KEY")

DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "all-MiniLM-L6-v2")