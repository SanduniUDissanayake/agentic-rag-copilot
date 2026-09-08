import os
import ast
import operator
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as GroqLLM
import asyncio

load_dotenv()

# --- Tool 1: calculator, so numeric comparisons aren't hallucinated ---
def calculate(expression: str) -> str:
    """Safely evaluates a basic arithmetic expression, e.g. '4.6 - 1.4' or '39 * 3'."""
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
    node = ast.parse(expression, mode="eval").body
    def ev(n):
        if isinstance(n, ast.BinOp):
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Constant):
            return n.value
        raise ValueError("Unsupported expression")
    return str(ev(node))

calculator_tool = FunctionTool.from_defaults(fn=calculate, name="calculator")

# --- Tool 2: the document retriever, wrapped as a callable tool ---
print("Loading embedding model...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cuda")

print("Loading document index...")
storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context, embed_model=embed_model)

llm = GroqLLM(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

retriever_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="document_search",
    description="Searches the annual reports (BHP, CBA, Telstra, Woolworths) for relevant financial and business information."
)

# --- Assemble the agent ---
agent = FunctionAgent(
    tools=[retriever_tool, calculator_tool],
    llm=llm,
    system_prompt="You are a financial analyst assistant. Use document_search to find facts from the annual reports. Use calculator for any arithmetic on numbers you've retrieved — never do math in your head."
)

async def main():
    question = "According to BHP's report, if copper capex was $4.6 billion this year and drops to $1.4 billion next year, what is the dollar decrease?"
    print(f"\nQuestion: {question}\n")

    handler = agent.run(question)
    async for event in handler.stream_events():
        if hasattr(event, "tool_name"):
            print(f"[TOOL CALLED] {event.tool_name} — input: {getattr(event, 'tool_kwargs', '')}")
    response = await handler
    print("\nAnswer:")
    print(response)

asyncio.run(main())