# Hallucinations and Evaluation

## Say this (60–90 sec)
Hallucination is when an LLM generates confident but false or unsupported content — fake citations, wrong facts, invented APIs. Causes: training objective rewards plausible text not truth; parametric memory is fuzzy; no grounding in retrieval. Mitigations: RAG for factual QA, prompt to say "I don't know," lower temperature for fact tasks, fine-tune on honest refusals, tool use for verifiable lookups. Evaluating LLMs is hard — single metric isn't enough. Perplexity measures language modeling on held-out text. Task benchmarks: MMLU, HumanEval for code. Human eval for chat quality. RAG eval splits into retrieval metrics — recall at k — and generation metrics — faithfulness, answer relevance. LLM-as-judge is common but biased. Production needs offline eval plus online monitoring — user feedback, regression suites on every model change. Always test on your domain, not just public leaderboards.

## Why it matters
Top production concern for GenAI. Interviewers want root causes and a practical eval stack, not just "hallucinations are bad."

## How it works
- **Hallucination types**: factual error, fabricated reference, inconsistent reasoning, wrong tool args.
- **Detection**: consistency checks, entailment models, retrieval overlap, human review.
- **Metrics**: exact match, F1 on QA, BLEU/ROUGE (limited), faithfulness to context (RAG), win-rate vs baseline.
- **Red teaming**: adversarial prompts for safety and failure modes.
- **Regression sets**: fixed prompt suite run on every deploy.

## Tradeoffs
- Use when: designing any customer-facing LLM — plan eval before launch.
- Avoid when: trusting a single automated score — combine human, task-specific, and safety evals.

## If they dig deeper
- Calibration — model confidence vs accuracy; well-calibrated models know when unsure.
- Groundedness metrics — Ragas, TruLens frameworks for RAG pipelines.
- Factuality vs creativity tradeoff — low temperature and RAG help facts; hurt creative writing.
