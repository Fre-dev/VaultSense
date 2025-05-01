from typing import Literal, Optional
from pydantic import BaseModel


class ChatInput(BaseModel):
    question: str

class ChatAnswerAddons(BaseModel):
    type: str
    data: dict


class ChatAnswer(BaseModel):
    summary: str
    query: Optional[str] = None
    addons: Optional[dict] = None


class ChatOutput(BaseModel):
    chat_id: str
    response_id: int
    answer: ChatAnswer
    audio: Optional[str] = None
