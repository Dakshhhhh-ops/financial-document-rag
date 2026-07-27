from fastapi import FastAPI
from src.api.models import QuestionRequest, QuestionResponse
from src.chain.rag_chain import ask_question

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Financial document RAG API is running"
    }


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    result = ask_question(request.question)
    return result