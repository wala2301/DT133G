from pydantic import BaseModel, Field, StringConstraints
from typing import Annotated

# Defining the format of input & output data using Pydantic
class QuestionRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    top_k: Annotated[int | None, Field(ge=1, le=10)] = None

class QuestionResponse(BaseModel):
    answer: str
