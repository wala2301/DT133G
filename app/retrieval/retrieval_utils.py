from pathlib import Path
from config import RETRIEVAL_TOP_K, Data_DOC_PATH


# Load text documents from the documents folder, each text file is read as a single text string.
def load_documents(folder_path=Data_DOC_PATH):
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"{folder} not found!")

    documents = []

    for file in sorted(folder.glob("*.txt")):
        try:
            text = file.read_text(encoding="utf-8").strip()
            if text:
                documents.append(
                    {
                        "doc_id": file.stem,
                        "source_file": file.name,
                        "text": text
                    }
                )
        except Exception as e:
            print(f"Warning: Could not read {file}: {e}")

    if not documents:
        raise ValueError("No valid documents were loaded.")

    return documents


# InPut validation, verify the question before passing it to the retrieval system
def validate_question(question: str):

    if question is None:
        return False, "Question cannot be None."

    question = question.strip()

    if question == "":
        return False, "Question cannot be empty."

    if len(question) < 2:
        return False, "Question is too short."

    if not any(c.isalpha() for c in question):
        return False, "Question must contain letters."

    return True, question