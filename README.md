# Annual Report Copilot

[![CI/CD](https://github.com/SanduniUDissanayake/agentic-rag-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/SanduniUDissanayake/agentic-rag-copilot/actions)

An AI agent that answers questions about real company annual reports. It searches the documents for facts and uses a calculator tool for arithmetic, so numbers are computed rather than guessed.

The knowledge base is four 2026 annual reports (BHP, Commonwealth Bank, Telstra and Woolworths), about 775 pages across mining, banking, telecommunications and retail.

## How it works

```
Streamlit UI  ->  FastAPI service  ->  Agent (LlamaIndex)
                                          |-- document_search tool -> vector index of the reports
                                          |-- calculator tool      -> safe arithmetic evaluator
                                          '-- LLM: Groq, with automatic fallback to Gemini
```

1. **Ingestion:** PDFs are read with PyMuPDF, split into chunks and embedded locally with BAAI/bge-small-en-v1.5 (GPU when available). The index is stored on disk.
2. **Agent:** for each question the agent decides whether to search the reports, run a calculation, or both.
3. **Provider fallback:** if the preferred LLM provider fails or is rate limited, the request is retried on the other one.
4. **API:** a FastAPI service exposes `POST /query` (answer, provider used, tool calls, source passages) and `GET /health`, with typed request and response schemas and auto-generated docs at `/docs`.
5. **Frontend:** a Streamlit chat app calls the API and shows the agent trace and retrieved passages next to each answer.

## Stack

Python, LlamaIndex, FastAPI, Streamlit, PyMuPDF, Hugging Face sentence embeddings, Groq (openai/gpt-oss-120b), Google Gemini, RAGAS, pytest, ruff, GitHub Actions.

## Evaluation

Scored with RAGAS on a hand-written set of 4 questions with reference answers.

| Metric | Result | Runs |
|---|---|---|
| Faithfulness | 1.00 | 3 completed runs |
| Answer relevancy | 0.92 | 1 completed run |
| Context precision | 0.75 to 1.00 | 2 completed runs |

These numbers come from a small sample and should be read as a smoke test, not a benchmark. RAGAS uses an LLM as judge, which makes many extra calls. Running it on free tiers exhausted the daily quotas of both Groq and Gemini, so several runs returned partial results. The table only reports metrics that completed. A larger test set and a paid tier or result caching are the next step.

## Engineering notes

- **PDF extraction failure:** the default reader returned raw PDF bytes instead of text, so the first answers were vague. I traced it with an extraction check script and switched to PyMuPDF, which fixed retrieval.
- **Tool use is verified:** the agent trace shows the document search followed by the calculator for percentage questions.
- **Safe calculator:** arithmetic is parsed with Python's `ast` module rather than `eval`, and unit tests check that code injection is rejected.
- **Secrets:** API keys, source PDFs and the built index are excluded from the repository.

## Known limitations

- Broad cross-company questions can still favour one report in the general search. A per-company search tool (metadata-filtered) lets the agent query each report separately for comparisons. Reranking would improve this further.
- Retrieval can return a nearby but wrong passage. In testing, a "registered office" question for BHP returned the New Zealand share registry address instead of the head office, while rewording the question to match the report's own wording returned the correct one. The UI shows tool calls and source passages so this is visible. Reranking and answer-grounding checks are the next step.
- The "source passages" panel in the UI runs a separate retrieval for the same question, so it shows relevant context but not necessarily the exact passages the agent used.
- The evaluation set is small (4 questions), so the RAGAS results are a smoke test, not a benchmark.
- No authentication or rate limiting on the API yet.
- Dockerfile, image publishing and cloud deployment are not done yet.
## Run locally

```
pip install -r requirements.txt
python ingest.py
uvicorn api:app --port 8000
streamlit run app.py
```

Create a `.env` file with `GROQ_API_KEY` and `GEMINI_API_KEY`, and place the report PDFs in a `data/` folder.

## CI

Every push runs ruff and the pytest suite through GitHub Actions.

## Roadmap

1. Docker image build and publish to GitHub Container Registry
2. Deployment to a public URL, plus a self-hosted AWS EC2 instance
3. Larger evaluation set, cross-company retrieval and reranking