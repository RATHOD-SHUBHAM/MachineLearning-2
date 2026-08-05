# Perplexity (Language Models)

## Say this (60–90 sec)
Perplexity measures how surprised a language model is by held-out text. It’s built from average negative log-likelihood of the next tokens: perplexity = exp(average cross-entropy per token). Intuition: perplexity roughly means “weighted average branching factor” — a perplexity of 20 is like choosing among 20 equally likely next tokens. Lower is better on the same tokenizer and dataset. It’s a standard offline metric for LMs, but it doesn’t measure truthfulness, helpfulness, or task success — a model can have great perplexity and still hallucinate. For GenAI interviews I say: use perplexity for language-modeling fit; use task metrics, human eval, or RAG faithfulness for product quality.

## Why it matters
Core LM pretraining metric. Interviewers check you know it ≠ “good chatbot.”

## How it works
- Tokenize held-out corpus.
- Compute mean cross-entropy loss per token.
- Perplexity = e^{mean CE} (natural base) or 2^{mean CE in bits}.
- Must compare models with the **same vocabulary/tokenizer** and similar data.

## Tradeoffs
- Use when: comparing LMs on next-token prediction; monitoring pretrain/finetune fit.
- Avoid when: evaluating chat quality, factuality, or tool use alone — pair with task benchmarks.

## If they dig deeper
- Bits-per-character / bits-per-byte — tokenizer-agnostic cousins.
- Domain shift: low perplexity on news ≠ good code model.
- Also mentioned in [`05_GenAI/08_hallucinations_eval.md`](../05_GenAI/08_hallucinations_eval.md).
