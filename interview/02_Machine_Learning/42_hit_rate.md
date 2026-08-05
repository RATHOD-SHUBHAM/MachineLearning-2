# Hit Rate (Recall@k / Hit@k)

## Say this (60–90 sec)
Hit Rate — often Hit@k or Recall@k in recommenders — asks: did at least one relevant item appear in my top-k predictions? For each user or query, you score a hit if the true item (or any relevant item) is in the top k; then average over users. It’s simple and intuitive: “Out of 100 sessions, we surfaced the clicked item in the top 10 for 40 of them — Hit@10 = 0.4.” It ignores where in the top k the hit landed and ignores graded relevance — that’s why teams often pair it with NDCG or MRR.

## Why it matters
Common recsys / retrieval eval. Easy to explain; good sanity check next to NDCG.

## How it works
- For each user/query: Hit = 1 if relevant item ∈ top-k else 0.
- Hit Rate = mean Hit over users.
- Recall@k: fraction of all relevant items recovered in top-k (when multiple relevants exist).
- Related: MRR — reciprocal rank of first relevant item, averaged.

## Tradeoffs
- Use when: success = “show something useful in top k”; quick online/offline proxy.
- Avoid when: position inside the list matters a lot, or relevance is graded — prefer NDCG.

## If they dig deeper
- Hit@k ignores rank position within k — NDCG/MRR fix that.
- Choice of k is product-driven (UI shows 5 vs 20).
