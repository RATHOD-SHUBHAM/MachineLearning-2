# PR-AUC (Average Precision)

## Say this (60–90 sec)
PR-AUC is the area under the **Precision–Recall curve**. For each threshold you plot recall on the x-axis and precision on the y-axis, then take the area — often computed as **average precision**. For spam detection with heavy imbalance, PR-AUC is usually more honest than ROC-AUC: ROC can look great just because true negatives are easy when most email is ham. PR focuses on the positive class — when I call something spam, how often am I right, and how much spam do I catch? A model with high ROC-AUC but low PR-AUC is a red flag on rare events. I report both, but I trust PR-AUC more for fraud, rare disease, and spam-like problems.

## Why it matters
Standard follow-up after ROC-AUC: “Which AUC when data is imbalanced?” Answer: PR-AUC.

## How it works
- Build PR curve over score thresholds: Precision = TP/(TP+FP), Recall = TP/(TP+FN).
- **PR-AUC / Average Precision**: summarizes the curve into one number in [0, 1].
- Baseline for a random classifier ≈ positive class prevalence (not 0.5).
- Sklearn: `average_precision_score(y_true, y_score)`.

## Tradeoffs
- Use when: positives are rare; FP cost matters; comparing ranking of positive scores.
- Avoid when: classes are balanced and you care equally about both — ROC-AUC is fine and more familiar.

## If they dig deeper
- F1 is one operating point; PR-AUC summarizes all thresholds.
- vs ROC-AUC: same model ranking scores, different axes — PR ignores easy TNs.
- Always state the positive class prevalence when quoting PR-AUC.
