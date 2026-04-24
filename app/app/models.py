"""
Shared Pydantic models — imported by both main.py and llm.py
to avoid circular imports.
"""

from pydantic import BaseModel
from typing import Optional


class Citation(BaseModel):
    chunk_id: str
    page: Optional[int] = None
    text: str
    score: float


class QuestionRequest(BaseModel):
    question: str
    document_id: str
    top_k: Optional[int] = 5


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    document_id: str
    question: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: str
