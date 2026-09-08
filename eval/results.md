# Evaluation Results

Evaluated on 4 questions across BHP, CBA, Telstra, and Woolworths annual reports, scored with RAGAS across multiple runs on Sept 9, 2026.

## RAGAS Scores

| Metric | Score | Confirmed across |
|---|---|---|
| Faithfulness | 1.0000 | 3 of 3 completed runs |
| Answer Relevancy | 0.9161 | 1 of 1 completed run |
| Context Precision | 0.75 – 1.00 | 2 completed runs (0.75, 1.0000) |

## Note on evaluation methodology

RAGAS scoring is LLM-judge-based, requiring many additional API calls beyond the original 4 questions (each metric decomposes answers into individual statements/sub-questions for verification). Across iterative testing in one session, this exhausted Groq's daily token quota (200,000 TPD) and Gemini's free-tier daily request quota (20 requests/day), causing partial failures (`nan`) in several runs when calls timed out waiting on rate-limit backoff. Faithfulness was the most consistently reproducible metric (1.0 across all 3 runs that reached it); Context Precision showed real run-to-run variance (0.75–1.0), plausibly due to non-deterministic LLM judging and retry-induced partial context loss. In production, this would be addressed with result caching, checkpointed batch evaluation, or a paid inference tier with higher throughput.