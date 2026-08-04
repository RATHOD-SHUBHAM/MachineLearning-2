# Tokens and Embeddings

## Say this (60–90 sec)
LLMs don't read raw text — they read tokens. Tokenization splits text into subword units — whole words, prefixes, suffixes — using algorithms like BPE or SentencePiece. "Unhappiness" might become "un", "happiness". Vocabulary size is typically 30k–100k tokens. Each token ID maps to a learned embedding vector — a dense representation in d dimensions, say 4096. Embeddings capture semantic similarity: related tokens end up close in vector space. Positional information is added separately — learned position embeddings or RoPE — because attention alone doesn't know order. The embedding layer is the first lookup: token IDs go in, vectors come out, then transformer blocks process the sequence. Tokenization choices affect everything — spelling, languages, code — and mismatches between training and inference tokenizers cause subtle bugs.

## Why it matters
Foundation of how LLMs represent language. Interviewers test whether you know tokens ≠ words and why embedding dimension and vocab size matter for memory.

## How it works
- **Tokenization**: text → list of token IDs via BPE/Unigram/SentencePiece.
- **Embedding lookup**: `E[token_id]` → vector ∈ R^d — weight matrix (vocab_size × d).
- **Subword benefit**: handles rare words, morphology, open vocabulary without huge word-level vocab.
- **Position encoding**: added to token embeddings so model knows sequence order.
- **Special tokens**: `[CLS]`, `[SEP]`, `<|endoftext|>`, padding — task-specific markers.

## Tradeoffs
- Use when: any NLP/LLM pipeline — always know your tokenizer matches the model checkpoint.
- Avoid when: assuming character-level is always better — longer sequences, harder learning; subword is the practical default.

## If they dig deeper
- BPE training — merge most frequent pairs iteratively.
- Embedding tying — share input and output embedding weights in LM head — saves parameters.
- Multilingual tokenizers — vocab shared across languages; token efficiency varies by language.
