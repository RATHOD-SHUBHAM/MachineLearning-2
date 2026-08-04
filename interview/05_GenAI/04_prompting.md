# Prompting

## Say this (60–90 sec)
Prompting is how you steer an LLM without changing weights. Zero-shot: describe the task in natural language and ask for output. Few-shot: include examples in the prompt — input-output pairs — so the model picks up the pattern in context. Chain-of-thought: ask the model to think step by step before the final answer — improves reasoning on math and logic. System prompts set role and constraints — "you are a helpful assistant, answer concisely." Structure matters: clear instructions, delimiters for sections, specify output format like JSON. Prompt order and example selection affect results — similar examples help. Limitations: context window caps how much you can include; model may ignore instructions; sensitive to phrasing. Prompt engineering is fast to iterate but brittle across model versions — test when you upgrade checkpoints.

## Why it matters
Primary interface for LLM products before fine-tuning. Interviewers want concrete techniques beyond "write a good prompt."

## How it works
- **Zero-shot**: task instruction only — relies on pretrain knowledge.
- **Few-shot (in-context learning)**: k examples in prompt; no weight update.
- **Chain-of-thought (CoT)**: "Let's think step by step" or exemplars with reasoning traces.
- **Role/system prompt**: persistent behavior constraints across turns.
- **Structured output**: ask for JSON/XML; some models support constrained decoding.

## Tradeoffs
- Use when: rapid prototyping, low data, model already capable of task, need flexibility.
- Avoid when: strict reliability, proprietary behavior, or low latency at huge prompt size — fine-tune, RAG, or smaller specialized model.

## If they dig deeper
- Self-consistency — sample multiple CoT paths, majority vote on answer.
- ReAct — interleave reasoning and tool calls in prompt.
- Prompt injection — untrusted user content overriding system instructions — security concern.
