# Specificity

## Say this (60–90 sec)
**Specificity** — also called **true negative rate** — answers: among all actual not-spam email, how much did we correctly leave in the inbox? Formula: TN / (TN + FP). High specificity means we rarely flag legitimate mail as spam — few false alarms on the ham side. It’s the mirror of recall on the negative class: recall focuses on catching spam; specificity focuses on correctly clearing ham. If specificity is low, users see good email in the spam folder (high FP). In medical tests, specificity is “correctly identifying healthy patients.” For spam, both recall (catch spam) and specificity (don’t block ham) matter; which you weight depends on product policy. Specificity plus sensitivity (recall) fully describe binary classifier performance at a fixed threshold.

## Why it matters
Shows you understand both sides of the confusion matrix — not only the positive class. Pairs with recall for balanced accuracy and ROC curves.

## How it works
- **Formula**: Specificity = TN / (TN + FP). Denominator = all actual not-spam (ham).
- **Spam example**: 980 ham emails; TN=950, FP=30 → specificity = 950/980 ≈ **96.9%** — 30 ham wrongly marked spam.
- **Not affected by FN** — missed spam doesn’t enter the formula.
- **False positive rate (FPR)**: FP / (FP + TN) = 1 − specificity — used on ROC x-axis.
- **Perfect specificity (1.0)**: FP=0 — never block ham; may require conservative spam threshold (lower recall).

## Tradeoffs
- Use when: false positives on the negative class matter — wrong spam flags, false fraud blocks on legit users.
- Report alongside recall when: you need a full picture at one threshold.
- Avoid when: class is extremely imbalanced and you only care about catching rare spam — still mention specificity if users complain about blocked mail.

## If they dig deeper
- Sensitivity + specificity = recall + TNR — symmetric pair for binary tests.
- Balanced accuracy = (recall + specificity) / 2.
- ROC plots TPR (recall) vs FPR (1 − specificity) across thresholds — not a single number like specificity at one threshold.
