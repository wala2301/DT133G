# Repository Review — Group 2

**Course:** Applied Datateknik, Mid Sweden University  
**Group:** 2  
**Project name:** Backend Architecture for a Domain-Adapted Conversational AI System  
**Repository:** https://github.com/wala2301/DT002G  
**Review date:** 2026-03-04  
**Reviewer:** GitHub Copilot (automated evidence-based review)  

---

## 1. Repository Summary

### Project purpose
This project implements a Retrieval-Augmented Generation (RAG) backend system designed for a domain-adapted conversational AI assistant. Users send natural-language queries via a REST API; the system retrieves the most relevant passages from a local medical knowledge base using TF-IDF cosine similarity, constructs a structured prompt, and forwards it to a configurable external LLM (e.g., GPT-4o-mini) to generate a response. All interactions are logged in anonymized form with a 30-day retention policy. The primary audience is developers or researchers building domain-specific AI query interfaces.

### Technology stack
Python 3 with FastAPI (REST API), Pydantic (input validation), scikit-learn TF-IDF + cosine similarity (retrieval), httpx (LLM HTTP client), python-dotenv (configuration), Uvicorn (ASGI server), and pytest + FastAPI TestClient (testing). No containerisation or CI configuration is present.

### Current development state
Relatively mature for a course project. Five of six functional requirements and all seven non-functional requirements have identifiable implementations. The commit graph shows feature-by-feature delivery through four merged pull requests. Tests cover all major subsystems. The only apparent gap is FR-3, which is implemented in code but not described in the README goal list.

---

## 2. Evidence-Based Checklist of Good Practices

Scale: **Yes / Partly / No / Unclear**  
For every item, include evidence from the repository.

### 2.1 Structure and organization

**The repository has a clear and logical folder structure.**  
Assessment: Yes  
Evidence: Top-level `main.py`, `config.py`, `requirements.txt`, `README.md`; sub-packages `app/api/`, `app/llm/`, `app/logging/`, `app/retrieval/`; separate `data/`, `logs/`, `tests/`.  
Comment: Separation of concerns is clearly reflected in the directory layout.

**Source code, tests, configuration, and documentation are separated appropriately.**  
Assessment: Yes  
Evidence: Application logic in `app/`; all configuration in `config.py` (using `os.getenv`); tests in `tests/`; documentation in `README.md`; data in `data/`.  
Comment: Clean separation. `logs/ai_backend.json` is committed (see clutter item below), but this is the only stray artefact.

**File and folder names are meaningful and consistent.**  
Assessment: Yes  
Evidence: `routes.py`, `schemas.py`, `retrieval.py`, `logging.py`, `llm.py`, `config.py` all describe their content exactly.  
Comment: The directory `app/logging/` shadows Python's built-in `logging` standard library module, which could cause import confusion if standard logging is ever needed.

**The repository avoids unnecessary generated files or clutter.**  
Assessment: Partly  
Evidence: `.gitignore` excludes `.env`, `__pycache__/`, `venv/`, `*.pyc`; however `logs/ai_backend.json` is committed and contains a live test record with an older schema (no hashes, no references).  
Comment: The log file should either be excluded via `.gitignore` or the entry removed before committing.

---

### 2.2 Code quality

**The code appears correct and runnable.**  
Assessment: Yes  
Evidence: `main.py` creates a `FastAPI` instance and includes the router; `uvicorn main:app --reload` instruction in README; `config.py` uses `load_dotenv()` with safe defaults so the system can run without a `.env` file (falling back to returning the first retrieved document as the answer).  
Comment: The `requirements.txt` does not pin versions, which could cause reproducibility issues over time.

**The code follows consistent style and coding conventions.**  
Assessment: Yes  
Evidence: Consistent use of type annotations (`str`, `int | None`, `list[str]`), Pydantic models, docstring-style inline comments, and snake_case throughout. No obvious PEP-8 violations observed.  
Comment: Minor: a tab character is used for indentation inside `llm.py` while the rest of the codebase uses spaces (Python style violation).

**The code is readable and reasonably modular.**  
Assessment: Yes  
Evidence: Each module exposes a narrow public interface (`retrieve_context_bundle`, `generate_answer`, `log_conversation`). `routes.py` orchestrates these without duplicating logic. `config.py` centralises all tuneable parameters.  
Comment: The private helper `_rank_indices` in `retrieval.py` correctly encapsulates repeated ranking logic.

**There are no major obvious code smells (duplication, overly large files, unclear naming).**  
Assessment: Yes  
Evidence: No duplicated retrieval or logging logic detected. Largest file is `logging.py` (~90 lines). All functions have clear single responsibilities.  
Comment: `Data_FILE_PATH` in `config.py` uses mixed case (PascalCase start), inconsistent with the surrounding `UPPER_SNAKE_CASE` constants.

---

### 2.3 Documentation

**The repository contains a clear README.**  
Assessment: Yes  
Evidence: `README.md` (121 lines) contains Environment & Tools, Purpose, Concrete Goals aligned to FRs, Procedures section describing each implemented FR, a Discussion section, and run/test commands.  
Comment: FR-3 appears as a blank line (`- FR3:`) in the goals list, suggesting it was added to the code later without updating the README.

**The README explains how to install, run, and use the system.**  
Assessment: Partly  
Evidence: Run command (`uvicorn main:app --reload`) and test command (`python -m pytest -v`) are provided. Tool list includes all dependencies.  
Comment: No step-by-step install instructions (e.g., `pip install -r requirements.txt`), no `.env.example` file showing required environment variables (`LLM_API_URL`, `LLM_API_KEY`, etc.), and no example API request.

**The documentation would help a new developer understand and contribute to the project.**  
Assessment: Partly  
Evidence: The Procedures section of the README explains the rationale and implementation of each FR with code snippets. Architecture is described textually.  
Comment: A new developer could understand the system from the README but would struggle to run it without knowing which environment variables are required. An `.env.example` and a short curl/Postman example would close this gap.

**Important design decisions or setup details are documented.**  
Assessment: Partly  
Evidence: README explains choice of TF-IDF, modular architecture, separation of concerns, and the anonymization/retention design. `config.py` itself is self-documenting via parameter names.  
Comment: The decision to fall back to the top retrieved document when the LLM is unavailable (`if not api_url or not api_key: return context[0]`) is not explained in the README.

---

### 2.4 Testing

**The repository contains tests.**  
Assessment: Yes  
Evidence: Four test files: `test_api.py`, `test_retrieval.py`, `test_logging.py`, `test_performance.py`, totalling 14+ test functions.  
Comment: Good coverage across all subsystems.

**Tests are relevant to the main functionality.**  
Assessment: Yes  
Evidence: `test_api.py` tests the POST `/ask` endpoint (success, validation, monkeypatched LLM, top_k override); `test_retrieval.py` tests list return and top_k parameter; `test_logging.py` tests anonymization, log writing with hashes, and retention deletion; `test_performance.py` tests 30-iteration average latency < 3 s.  
Comment: All tests map directly to FRs or NFRs.

**Tests can be executed with clear instructions.**  
Assessment: Yes  
Evidence: README states: "Using `python -m pytest -v` all tests were run and their success was confirmed."  
Comment: No CI pipeline is configured; tests are manual only.

**Tests appear to pass, or there is evidence that they have been run successfully.**  
Assessment: Partly  
Evidence: README claims all tests pass. `test_logging.py`'s `test_delete_old_records` calls `delete_old_records()` without overriding the global `LOG_FILE`, so it operates on the real log file in `logs/`; the committed `logs/ai_backend.json` uses an older schema (missing `timestamp` in UTC ISO format required by the parser), which could cause that test to fail.  
Comment: No CI run output or test report is committed. The log fixture isolation in `test_logging.py` partially works (it patches `LOG_FILE` for `log_conversation` tests) but the retention test does not redirect the file.

---

### 2.5 Collaboration and development practices

**Commit history suggests incremental development.**  
Assessment: Yes  
Evidence: 12 commits: initial commit → project structure (PR #1) → FR1 & FR2 → FR4 + LLM (PR #2) → FR5 + FR6 → SRS alignment (PR #3) → NFR-1 performance (PR #4).  
Comment: Each PR corresponds to one or two requirements, showing disciplined incremental delivery.

**Commit messages are meaningful.**  
Assessment: Yes  
Evidence: Examples: "Implement FR4 LLM integration and strengthen FR1 input validation", "FR5_implement anonymized conversation logging to protect sensitive data", "Align FR5 and FR6 with SRS", "Implement NFR-1 performance optimizations".  
Comment: All messages clearly state what changed and why.

**There is evidence of collaboration between both students.**  
Assessment: Yes  
Evidence: Two distinct git authors: "Wala Alshaqa" and "Lina Meander". Feature branches were merged via pull requests (#1–#4), and the merge commits are authored by a different contributor from the feature commits.  
Comment: The split of work between the two authors is not reflected in individual commit authorship for most feature commits, but the PR-based workflow confirms collaborative review.

---

## 3. What Is Being Done Well

1. **Clean modular architecture.** Each concern (API, retrieval, LLM, logging) lives in its own package with a narrow public interface, making the codebase easy to navigate and extend.
2. **Privacy-by-design logging.** Regex anonymization of emails, phone numbers, and IDs plus SHA-256 hashing and a 30-day retention policy are all implemented and tested.
3. **Incremental, traceable development via PRs.** Four pull requests with descriptive commit messages map directly to requirement IDs, giving a clear audit trail of what was added when.
4. **Comprehensive test suite.** All four subsystems have dedicated tests; monkeypatching isolates external dependencies so tests do not require a live LLM key.
5. **Graceful degradation.** When the LLM is unreachable or unconfigured, the system falls back to returning the top retrieved document rather than crashing, and the configurable timeout (default 2.5 s) bounds worst-case latency.

---

## 4. What Needs Improvement

1. **Missing `.env.example` and setup instructions.** The README does not list which environment variables are required (`LLM_API_URL`, `LLM_API_KEY`, `LLM_MODEL`, etc.), making the system difficult to run for the first time without reading `config.py`.
2. **`logs/ai_backend.json` should not be committed.** The file contains a live test record in an older schema and should be added to `.gitignore`. Its presence could cause the retention-policy test to operate on real data.
3. **FR-3 is undocumented in the README.** The `build_prompt()` function in `llm.py` implements prompt construction with context, but the README goal list has a blank `- FR3:` entry, leaving the requirement unnarrated.
4. **`requirements.txt` does not pin versions.** All six dependencies are listed without version constraints, risking silent breakage if upstream packages introduce incompatible changes.
5. **`app/logging/` shadows the Python built-in `logging` module.** If any future code needs to import the standard library `logging`, the module name collision will produce a confusing `ImportError`. Renaming to `app/conversation_log/` or similar would avoid this risk.

---

## 5. Evaluation Against Requirements

### 5.1 Functional Requirements

| Requirement | Description | Status | Evidence | Comment |
|---|---|---|---|---|
| FR-1 | Receive user queries through a REST API endpoint and validate input format. | Implemented | `POST /ask` in `routes.py`; `QuestionRequest` in `schemas.py` uses Pydantic `StringConstraints(min_length=1)`; returns HTTP 422 on empty question (tested in `test_ask_endpoint_empty_question`). | Fully implemented and tested. |
| FR-2 | Retrieve domain-relevant information from a structured knowledge base using similarity search. | Implemented | `retrieval.py` loads `data/document.txt`, builds a TF-IDF matrix once, computes cosine similarity per request, and returns top-k ranked documents. Tested in `test_retrieval.py`. | TF-IDF is a simple but adequate similarity search for a course project. |
| FR-3 | Construct a structured prompt containing user input and retrieved context before sending it to the external language model. | Implemented | `build_prompt()` in `llm.py` formats retrieved context as a bulleted list and appends the user question with a domain-assistant system message. | Not mentioned in the README goal list (`- FR3:` is blank); implementation exists in code. |
| FR-4 | Return a generated response in JSON format via the API. | Implemented | `QuestionResponse(answer=answer)` is returned by the route and serialised to JSON by FastAPI. Response model is declared on the route decorator. Tested in `test_api.py`. | JSON output is enforced by Pydantic. |
| FR-5 | Record anonymized conversation history including timestamps and retrieved document references. | Implemented | `log_conversation()` in `logging.py` anonymizes text (email/phone/ID regex), stores SHA-256 hashes, records `retrieved_references` and UTC timestamp. 30-day retention implemented and tested. | Committed `logs/ai_backend.json` contains an older-format record without hashes; should be excluded from the repo. |
| FR-6 | Allow configuration of retrieval parameters such as the number of retrieved documents. | Implemented | `config.py` exposes `RETRIEVAL_TOP_K` and `RETRIEVAL_MAX_TOP_K` via environment variables. `top_k` can also be passed per-request in the JSON body (validated with `Field(ge=1, le=10)`). Tested in `test_ask_endpoint_accepts_top_k`. | Both global and per-request configuration are supported. |

### 5.2 Non-Functional Requirements

| Requirement | Description | Status | Evidence | Comment |
|---|---|---|---|---|
| NFR-1 | Response time below 3 seconds. | Implemented | LLM timeout set to 2.5 s via `LLM_TIMEOUT_SECONDS`; TF-IDF matrix built once at startup (not per request); `test_performance.py` measures 30-iteration average and asserts < 3 s. | Test mocks the LLM so it measures routing/retrieval overhead only, not real LLM latency. |
| NFR-2 | Be usable without training. | Partly | The system exposes a single `POST /ask` endpoint with a simple JSON body (`{"question": "..."}`) — minimal API surface. No GUI or interactive documentation beyond FastAPI's auto-generated `/docs` (Swagger UI). | Swagger UI is available at `/docs` by default in FastAPI, which helps discoverability, but this is not mentioned in the README. |
| NFR-3 | Encrypt sensitive data or avoid storing sensitive data. | Implemented | `.env` is excluded from git (`.gitignore`). Logs store anonymized text and SHA-256 hashes, not raw PII. Regex patterns remove emails, phone numbers, and long IDs. | Raw question/answer are never persisted; only anonymized versions and hashes are written. |
| NFR-4 | Support replacement of system components. | Implemented | LLM integration is isolated in `app/llm/llm.py`; retrieval in `app/retrieval/retrieval.py`; all endpoints configurable via environment variables in `config.py`. Swapping the LLM provider requires only changing `LLM_API_URL` and `LLM_MODEL`. | Loose coupling is achieved through the narrow interfaces called from `routes.py`. |
| NFR-5 | Follow a modular codebase structure with independent modules. | Implemented | Four independent sub-packages: `app/api/`, `app/llm/`, `app/logging/`, `app/retrieval/`. Each has its own `__init__.py` and a single-responsibility `.py` file. | Modules have no circular imports. |
| NFR-6 | Handle invalid or malformed API requests without crashing and return appropriate error messages. | Implemented | Pydantic `StringConstraints(min_length=1)` returns HTTP 422 Unprocessable Entity on empty or missing question (tested). `generate_answer` catches `httpx.HTTPError`, `KeyError`, `IndexError`, `TypeError`, `ValueError` and returns a fallback string. | Error handling in the LLM layer is broad; specific error types could be logged for better observability. |
| NFR-7 | Include sufficient metadata in logs to support later thesis evaluation. | Implemented | Each log record contains: `timestamp` (UTC ISO-8601), `question_hash`, `answer_hash`, `question` (anonymized), `answer` (anonymized), `retrieved_references` (list of `{doc_id, preview}`), `top_k`. | The committed `logs/ai_backend.json` contains a legacy record missing hashes and references; the current code schema is richer. |

---

## 6. Overall Assessment

### Summary judgment
This is a solid, well-structured course project that fulfils its stated requirements. All six FRs are implemented in code (FR-3 is undocumented in the README but present in `llm.py`). All seven NFRs have identifiable implementations. The modular architecture, privacy-aware logging, and use of pull requests for incremental delivery are highlights. The main weaknesses are operational: missing environment-variable documentation, a committed log file that should be excluded, and unpinned dependencies. These are relatively minor and straightforward to fix.

### Confidence in this review
High — the full source code, tests, commit history, and README were examined directly.

### Limitations of this review
Tests were not executed; pass/fail status is inferred from code inspection only. The real-LLM integration path (FR-4, NFR-1 under real latency) could not be verified without a valid API key. No external architecture or SRS document was available beyond what is in the README.

---

## 7. Suggested Improvements

1. **Add a `.env.example` file and expand the README setup section.** Create a file listing all required environment variables (`LLM_API_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_TIMEOUT_SECONDS`, `Data_FILE_PATH`, `RETRIEVAL_TOP_K`, `RETRIEVAL_MAX_TOP_K`, `LOG_FILE_PATH`, `LOG_RETENTION_DAYS`, `LOG_CLEANUP_EVERY_N_WRITES`) with placeholder values and short descriptions. Add a numbered install walkthrough (`git clone` → `pip install -r requirements.txt` → copy `.env.example` to `.env` → `uvicorn main:app --reload`) and at least one example `curl` request. This would make the project immediately runnable by any new developer.

2. **Exclude `logs/ai_backend.json` from version control.** Add `logs/` (or `logs/*.json`) to `.gitignore`. The committed file contains a live test record in an older schema that is missing the `question_hash`, `answer_hash`, and `retrieved_references` fields expected by the current code. Its presence can interfere with the retention-policy test (`test_delete_old_records`) because that test does not redirect `LOG_FILE` to a temporary path — it operates on the real log.

3. **Fix the test isolation bug in `test_delete_old_records`.** The pytest fixture `temp_log_file` correctly monkeypatches `LOG_FILE` for `test_log_conversation`, but `test_delete_old_records` calls `delete_old_records()` directly without going through the fixture, so it reads and writes the real `logs/ai_backend.json`. Pass the temporary path explicitly to `delete_old_records` (add a `file_path` parameter to the function, consistent with `_load_records`/`_write_records`) or extend the fixture to also cover the deletion test.

4. **Pin dependency versions in `requirements.txt`.** Replace the six unpinned entries with exact or minimum-bounded versions (e.g., `fastapi>=0.110,<1`, `scikit-learn>=1.4,<2`). This prevents the project from silently breaking when upstream packages release incompatible updates and ensures that `python -m pytest -v` produces reproducible results on any machine.

5. **Document FR-3 in the README and rename `app/logging/` to avoid shadowing the standard library.** Add a short paragraph under Concrete Goals describing `build_prompt()` — what it injects (system role, context list, user question) and why (grounding the LLM in retrieved context). Separately, rename the package from `app/logging/` to `app/conversation_log/` (and update all three import sites) to prevent a silent `ImportError` if any future code or dependency tries to import the Python standard library `logging` module.  
