# F1 Score

## Say this (60–90 sec)
**F1** is the harmonic mean of precision and recall: F1 = 2 × (precision × recall) / (precision + recall). It balances “when we say spam, are we right?” with “did we catch the spam?” The harmonic mean punishes extreme imbalance — if precision is 99% but recall is 10%, F1 is about 18%, not a misleading average near 55%. I use F1 for **imbalanced binary problems** like spam when both false alarms and misses matter and I want one number. F1 ignores TN — it’s only about the positive class behavior (TP, FP, FN). It assumes FP and FN costs are roughly equal; if missing spam is ten times worse than blocking ham, F1 may not match business needs — then weighted F-beta or cost-sensitive metrics fit better.

## Why it matters
F1 is the standard single metric for imbalanced classification interviews. Explaining why harmonic mean beats arithmetic mean shows depth.

## How it works
- **Precision** = TP/(TP+FP); **Recall** = TP/(TP+FN).
- **F1** = 2PR/(P+R). Equivalent: F1 = TP / (TP + ½(FP + FN)).
- **Spam example**: TP=40, FP=10, FN=50 → P=40/50=0.8, R=40/90≈0.44 → F1 ≈ 0.57.
- **Range**: 0 to 1; 1 only if precision and recall both 1.
- **Macro-F1** (multiclass): compute F1 per class, average — treats classes equally.

## Tradeoffs
- Use when: imbalanced spam/fraud; need one metric; FP and FN similarly bad.
- Avoid when: TN matters heavily or costs are asymmetric — use F2 (weight recall) or F0.5 (weight precision).
- Avoid optimizing F1 alone in production without inspecting confusion matrix at chosen threshold.

## If they dig deeper
- F-beta: Fβ = (1+β²)PR / (β²P + R) — β>1 weights recall; β<1 weights precision.
- Why accuracy can be high while F1 is low — many TN inflate accuracy, few TP hurt F1.
- PR-AUC vs F1 at one threshold — PR-AUC summarizes across thresholds.
