import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as GroqLLM

load_dotenv()

print("Loading embedding model...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cuda")

print("Loading your saved document index...")
storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context, embed_model=embed_model)

print("Connecting to Groq...")
llm = GroqLLM(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))

query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

question = "What were the key financial highlights mentioned across these reports?"
print(f"\nQuestion: {question}\n")
response = query_engine.query(question)
print("Answer:")
print(response)