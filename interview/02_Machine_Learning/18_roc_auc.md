# ROC and AUC

## Say this (60–90 sec)
The **ROC curve** plots classifier performance across all thresholds. **X-axis**: false positive rate, FPR = FP/(FP+TN) = 1 − specificity — how often ham is wrongly flagged as spam. **Y-axis**: true positive rate, TPR = recall = TP/(TP+FN) — how much actual spam we catch. Each threshold gives one (FPR, TPR) point; connecting them forms the ROC curve. **AUC** — area under that curve — summarizes ranking quality from 0 to 1. AUC 0.5 is random guessing; 1.0 is perfect separation. AUC answers: if I pick a random spam and random ham email, how often is the spam score higher? It’s **threshold-independent** — useful for comparing models before choosing an operating point. For **heavy imbalance** like spam, PR-AUC can be more informative than ROC-AUC because ROC can look optimistic when TN dominates.

## Why it matters
ROC-AUC is a standard model comparison metric. Interviewers expect you to interpret axes, relate TPR to recall, and know ROC limits on imbalanced data.

## How it works
- **TPR (recall)**: TP/(TP+FN) — spam caught.
- **FPR**: FP/(FP+TN) — ham wrongly flagged; not the same as 1 − precision.
- **ROC**: sweep threshold from high to low; plot (FPR, TPR).
- **AUC**: probability model ranks random spam above random ham; geometrically area under ROC.
- **Spam intuition**: good filter pushes spam scores high, ham low → curve bows toward top-left → high AUC.
- **Diagonal line**: random classifier, AUC = 0.5.

## Tradeoffs
- Use ROC-AUC when: comparing models; balanced-ish classes; ranking quality matters; you’ll tune threshold later.
- Prefer PR-AUC when: rare spam (<5%); you care about precision at high recall; ROC looks inflated.
- Avoid when: you need a single deployed metric at one threshold — report precision/recall at chosen t, not AUC alone.
- Avoid interpreting AUC as “accuracy” — it’s ranking performance, not error rate.

## If they dig deeper
- Partial AUC — focus on low FPR region (operating only at strict thresholds).
- Class weights during training vs threshold tuning — both shift FP/FN tradeoff differently.
- Multiclass ROC — one-vs-rest AUC per class, macro average.
