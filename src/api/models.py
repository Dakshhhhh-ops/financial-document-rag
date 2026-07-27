from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

class Source(BaseModel):
    document: str
    pages: list[int]


class QuestionResponse(BaseModel):
    answer: str
    sources: list[Source]