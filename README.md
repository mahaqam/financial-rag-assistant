# Financial QA Retrieval Assistant

A financial-document retrieval project built on the FinQA development split. The system ranks report text and table rows as evidence for financial questions, creating a grounded retrieval layer that can sit in front of a question-answering or LLM system.

## Dataset
The verified run used **883 FinQA development questions** with **1,513 annotated gold evidence items** across financial-report text and tables.

## Approach
- Convert report paragraphs and table rows into retrievable evidence units.
- Build word-level TF-IDF (1–2 grams) and character-level TF-IDF (3–5 grams).
- Combine similarity scores using a **70/30 word/character hybrid**.
- Rank evidence for each question.
- Evaluate against FinQA gold evidence using hit rate@k, mean gold-evidence recall@k, and mean reciprocal rank (MRR).

## Verified retrieval results
| Metric | Result |
|---|---:|
| MRR | 0.7789 |
| Hit rate @ 1 | 0.6738 |
| Hit rate @ 3 | 0.8618 |
| Hit rate @ 5 | 0.9173 |
| Mean gold-evidence recall @ 5 | 0.8203 |
| Hit rate @ 10 | 0.9592 |

At least one gold evidence item appeared in the top five retrieved units for **91.7%** of questions in this development split.

## Why this matters
Financial AI systems need grounding before generation. This project measures the retrieval layer separately instead of claiming end-to-end answer accuracy that was not evaluated. It provides a transparent foundation for a later RAG/LLM system and makes retrieval failures auditable.

## Run
```bash
python -m venv .venv
# activate the environment, then:
pip install -r requirements.txt
python src/evaluate_retrieval.py --data data/dev.json
python src/retrieve_evidence.py --data data/dev.json --doc-index 0 --question "What is the average payment volume per transaction for American Express?"
```

## Repository structure
- `src/evaluate_retrieval.py` — candidate construction, hybrid retrieval, benchmark metrics
- `src/retrieve_evidence.py` — lightweight evidence-retrieval CLI
- `results/retrieval_metrics.json` — verified aggregate metrics
- `results/question_level_retrieval.csv` — per-question retrieval outcomes

## Limitations and next steps
This repository currently evaluates **retrieval**, not full LLM answer generation. The uploaded file was the FinQA `dev.json` split only, so results should be described as development-set retrieval results. Next steps would add dense embeddings, reranking, answer generation, numerical reasoning, hallucination checks, and a deployment/API layer.
