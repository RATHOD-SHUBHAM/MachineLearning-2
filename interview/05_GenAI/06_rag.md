# RAG (Retrieval-Augmented Generation)

## Say this (60–90 sec)
RAG grounds LLM answers in external documents instead of relying only on parametric memory. Pipeline: user query → retrieve relevant chunks from a knowledge base via embedding similarity search → inject retrieved text into the prompt as context → LLM generates answer citing that context. Chunk documents, embed with an embedding model, store in vector database. At query time, embed the question, find top-k similar chunks, concatenate into prompt. Benefits: fresher facts, fewer hallucinations on company-specific data, easier updates — swap docs without retraining. Tradeoffs: retrieval quality is the ceiling — bad chunks mean bad answers; latency adds retrieval step; context window limits how much you can inject. Hybrid search — keyword plus semantic — often helps. Evaluate both retrieval recall and final answer quality.

## Why it matters
Standard enterprise GenAI pattern. Separates knowledge storage from generation — critical for production systems with private data.

## How it works
- **Indexing**: split docs → chunks → embed → store vectors + metadata in vector DB.
- **Retrieval**: query embedding → nearest neighbors (cosine/IP) → top-k chunks.
- **Generation**: prompt = instruction + retrieved context + user question → LLM response.
- **Reranking**: cross-encoder rescores candidates for better precision.
- **Citation**: chunk metadata enables source attribution.

## Tradeoffs
- Use when: private/proprietary knowledge, frequently updated facts, need traceable sources, reduce hallucination on domain QA.
- Avoid when: task is pure creative writing or model already has stable public knowledge — RAG adds complexity without gain.

## If they dig deeper
- Chunk size/overlap — too small loses context, too large dilutes relevance.
- Query transformation — HyDE, multi-query expansion for better recall.
- GraphRAG / agentic RAG — structured retrieval and tool loops beyond flat vector search.
