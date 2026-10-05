import os
import asyncio
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as GroqLLM
from llama_index.llms.google_genai import GoogleGenAI
from eval.testset import eval_questions
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.run_config import RunConfig
from datasets import Dataset

load_dotenv()

print("Loading index and models...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cuda")
storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context, embed_model=embed_model)
llm = GroqLLM(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

results = []
print("\nRunning eval questions through the RAG pipeline...\n")
for item in eval_questions:
    response = query_engine.query(item["question"])
    retrieved_texts = [node.text for node in response.source_nodes]
    results.append({
        "question": item["question"],
        "answer": str(response),
        "contexts": retrieved_texts,
        "ground_truth": item["ground_truth"],
    })
    print(f"Q: {item['question']}")
    print(f"A: {response}\n")

# Save raw results so we can inspect them even before running RAGAS scoring
import json
with open("eval/raw_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved raw results to eval/raw_results.json")
print("\nScoring with RAGAS...")

ragas_llm = LlamaIndexLLMWrapper(llm)
ragas_embeddings = LlamaIndexEmbeddingsWrapper(embed_model)

dataset = Dataset.from_list([
    {
        "question": r["question"],
        "answer": r["answer"],
        "contexts": r["contexts"],
        "ground_truth": r["ground_truth"],
    }
    for r in results
])

scores = ragas_evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=RunConfig(max_workers=1, timeout=180),
)

print("\nRAGAS Scores:")
print(scores)

# Save scores to a markdown report for the README
with open("eval/results.md", "w") as f:
    f.write("# Evaluation Results\n\n")
    f.write(f"Evaluated on {len(eval_questions)} questions across BHP, CBA, Telstra, and Woolworths annual reports.\n\n")
    f.write("## RAGAS Scores\n\n")
    f.write(str(scores))
    f.write("\n")

print("\nSaved scores to eval/results.md")