import os

import torch
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq as GroqLLM

from tools import calculate
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

COMPANY_FILES = {
    "BHP": "260818_bhpannualreport2026.pdf",
    "CBA": "CBA-2026-Annual-Report.pdf",
    "Telstra": "telstra-financial-results-and-annual-report-for-the-year-ended-30-jun-2026.pdf",
    "Woolworths": "woolworths-2026-annual-report.pdf",
}

load_dotenv()
_index = None


def get_index():
    global _index
    if _index is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device=device)
        ctx = StorageContext.from_defaults(persist_dir="storage")
        _index = load_index_from_storage(ctx, embed_model=embed_model)
    return _index


def get_llm(provider: str):
    if provider == "Groq":
        return GroqLLM(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))
    return GoogleGenAI(model="gemini-3.6-flash", api_key=os.getenv("GEMINI_API_KEY"))


def build_agent(provider: str):
    llm = get_llm(provider)
    index = get_index()
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
    doc_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="document_search",
        description="Searches all four annual reports (BHP, CBA, Telstra, Woolworths) at once.",
    )

    def company_search(company: str, question: str) -> str:
        """Search ONE company's annual report only. company must be exactly one of:
        BHP, CBA, Telstra, Woolworths. Use this for comparisons, once per company."""
        key = next((k for k in COMPANY_FILES if k.lower() == company.strip().lower()), None)
        if key is None:
            return "Unknown company. Choose from: " + ", ".join(COMPANY_FILES)
        filters = MetadataFilters(
            filters=[MetadataFilter(key="file_name", value=COMPANY_FILES[key])]
        )
        engine = index.as_query_engine(llm=llm, similarity_top_k=3, filters=filters)
        return str(engine.query(question))

    company_tool = FunctionTool.from_defaults(fn=company_search, name="company_search")
    calc_tool = FunctionTool.from_defaults(fn=calculate, name="calculator")
    return FunctionAgent(
        tools=[doc_tool, company_tool, calc_tool],
        llm=llm,
        system_prompt=(
            "You are a financial analyst assistant. Use document_search for general "
            "questions about the reports. For questions that compare or involve several "
            "companies, call company_search once per company. Use calculator for any "
            "arithmetic on retrieved numbers. Never do math in your head. Say which "
            "company each figure is from."
        ),
    )


async def run_agent(agent, question: str):
    calls = []
    handler = agent.run(question)
    async for event in handler.stream_events():
        if hasattr(event, "tool_name"):
            entry = {"name": event.tool_name, "args": str(getattr(event, "tool_kwargs", ""))}
            if entry not in calls:
                calls.append(entry)
    response = await handler
    return str(response), calls


def get_sources(question: str, k: int = 3):
    nodes = get_index().as_retriever(similarity_top_k=k).retrieve(question)
    return [
        {
            "file": n.metadata.get("file_name", "unknown"),
            "score": float(n.score or 0.0),
            "text": n.text[:400] + "...",
        }
        for n in nodes
    ]


async def answer(question: str, preferred: str = "Groq") -> dict:
    order = ["Groq", "Gemini"] if preferred == "Groq" else ["Gemini", "Groq"]
    errors = []
    for provider in order:
        try:
            text, calls = await run_agent(build_agent(provider), question)
            return {
                "answer": text,
                "provider": provider,
                "fallback_used": provider != order[0],
                "tool_calls": calls,
                "sources": get_sources(question),
            }
        except Exception as e:
            errors.append(f"{provider}: {str(e)[:200]}")
    raise RuntimeError("All providers failed. " + " | ".join(errors))