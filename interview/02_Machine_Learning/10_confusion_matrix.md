# Confusion Matrix

## Say this (60–90 sec)
For classification, a confusion matrix is a table of predicted vs actual classes — it shows not just whether we were right, but *how* we were wrong. In **spam detection**, positive class = spam. Rows are actual, columns predicted (convention varies — state yours). The four cells: **True Positive** — predicted spam, actually spam (correct catch). **True Negative** — predicted not spam, actually not spam (correct inbox). **False Positive** — predicted spam, actually not spam (false alarm — good email quarantined). **False Negative** — predicted not spam, actually spam (missed spam — junk in inbox). Accuracy alone hides imbalance: 98% not-spam data lets a dummy “never spam” model look great. The confusion matrix is the foundation for precision, recall, specificity, and F1 — always start here for imbalanced problems.

## Why it matters
Interviewers want you to read a confusion matrix cold and translate cells into business impact — false alarms vs missed spam.

## How it works
```
                    Predicted
                 Not spam   Spam
Actual Not spam    TN        FP
       Spam        FN        TP
```
- **Positive class**: spam (by our convention).
- **TP**: model said spam, truth is spam — good.
- **TN**: model said not spam, truth is not spam — good.
- **FP**: model said spam, truth is not spam — user loses legitimate email.
- **FN**: model said not spam, truth is spam — user sees junk/phishing.
- All metrics (precision, recall, etc.) derive from these four counts.

## Tradeoffs
- Use when: any binary classifier evaluation; especially imbalanced data.
- Avoid when: relying on accuracy from the matrix diagonal alone without checking class balance.
- Always clarify: which class is positive and row/column layout — conventions differ across libraries.

## If they dig deeper
- Multiclass confusion matrix — one-vs-rest breakdown per class.
- Normalized confusion matrix — percentages per row show error patterns across classes.
- Cost-sensitive learning — weight FP vs FN differently if business costs differ (missed fraud vs false block).
