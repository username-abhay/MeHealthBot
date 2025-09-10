# api/routes.py
from fastapi import APIRouter
from src.cli_app import ask_question
from .schemas import QuestionRequest, AnswerResponse

router = APIRouter()

@router.post("/ask", response_model=AnswerResponse)
def ask_user_question(request: QuestionRequest):
    """
    Endpoint to pass user question to the chatbot and get the answer.
    """
    answer = ask_question(request.question)
    return AnswerResponse(answer=answer)
