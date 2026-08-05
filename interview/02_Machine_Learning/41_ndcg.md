# NDCG — Normalized Discounted Cumulative Gain

## Say this (60–90 sec)
NDCG is a ranking metric — used in search, recommendations, and information retrieval. The idea: relevant items should appear near the top of the list, and highly relevant items matter more than weakly relevant ones. You compute DCG: sum relevance scores with a **log discount** so position 1 counts more than position 10. Then normalize by the ideal DCG (best possible ordering) to get NDCG between 0 and 1. People usually report NDCG@k — only the top k results. If I rank products for a query, NDCG@10 asks: how good is my top-10 ordering versus the perfect top-10?

## Why it matters
Standard ranking interview metric. Shows you think beyond classification accuracy for ordered results.

## How it works
- Gain at rank i: often (2^{rel_i} − 1) or just rel_i.
- Discount: / log2(i + 1) — lower ranks contribute less.
- DCG@k = Σ_{i=1..k} gain_i / log2(i + 1)
- NDCG@k = DCG@k / IDCG@k

## Tradeoffs
- Use when: order matters (search, recsys, learning-to-rank).
- Avoid when: you only need a binary decision, not a ranked list — precision/recall may suffice.

## If they dig deeper
- Binary relevance vs graded relevance (0–4 stars) — graded fits NDCG naturally.
- vs MRR / Hit Rate — NDCG uses graded gains and position discount; Hit Rate is coarser.
