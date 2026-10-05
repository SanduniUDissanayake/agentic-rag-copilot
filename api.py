from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import core


@asynccontextmanager
async def lifespan(app: FastAPI):
    core.get_index()  # load the index once at startup
    yield


app = FastAPI(
    title="Annual Report Copilot API",
    description="Agentic RAG over BHP, CBA, Telstra and Woolworths 2026 annual reports.",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str
    provider: Literal["Groq", "Gemini"] = "Groq"


class ToolCall(BaseModel):
    name: str
    args: str


class Source(BaseModel):
    file: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    provider: str
    fallback_used: bool
    tool_calls: list[ToolCall]
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")
    try:
        return await core.answer(req.question, req.provider)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))