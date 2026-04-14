from time import perf_counter
from fastapi import APIRouter
from app.api.schemas import QuestionRequest, QuestionResponse
from app.retrieval.retrieval_router import RetrievalRouter
from app.llm.llm import generate_answer
from app.logging.logging import log_conversation
from config import RETRIEVAL_TOP_K, RETRIEVAL_MAX_TOP_K, RETRIEVAL_METHOD

router = APIRouter()

# create retrieval system once
retrieval_system = RetrievalRouter()
@router.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    requested_top_k = request.top_k if request.top_k is not None else RETRIEVAL_TOP_K
    safe_top_k = max(1, min(requested_top_k, RETRIEVAL_MAX_TOP_K))

    start = perf_counter()
    context, references = retrieval_system.retrieve_context_bundle(request.question, safe_top_k)
    answer = generate_answer(request.question, context)

    latency = perf_counter() - start
    # Log anonymized conversation
    log_conversation(request.question, answer, retrieval_method=RETRIEVAL_METHOD, retrieved_references=references, top_k=safe_top_k, latency=latency)

    return QuestionResponse(answer=answer)