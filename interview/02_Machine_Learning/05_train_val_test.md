# Train, Validation, and Test Splits

## Say this (60–90 sec)
We split data so we can honestly measure generalization. **Training set** — the model sees these examples and updates its parameters to minimize loss. **Validation set** — held out during training; we use it to tune hyperparameters, pick models, and decide when to stop — early stopping, learning rate, tree depth. **Test set** — touched only once at the end for an unbiased estimate of real-world performance. If you tune on the test set, you leak information and your reported metrics are optimistically biased. Typical splits might be 70/15/15 or 80/10/10 depending on data size. With little data, cross-validation on the train+val portion is common, keeping test strictly final. The rule I follow: train teaches, validation guides choices, test tells the truth once.

## Why it matters
Data leakage and optimistic metrics are red flags in interviews and in production. This split is the foundation of trustworthy evaluation.

## How it works
- **Train**: fit model parameters (weights).
- **Validation**: compare models/hyperparameters; no direct gradient updates on val loss (though early stopping watches it).
- **Test**: final report — simulate deployment on unseen data.
- **Split methods**: random split (i.i.d. assumption), stratified (preserve class ratios), time-based (for temporal data — never shuffle future into past).
- **Data leakage**: preprocessing fit on full dataset before split, duplicate/near-duplicate rows across splits, target leakage in features.

## Tradeoffs
- Use when: always, for any supervised project with enough data to split.
- Avoid when: reporting test performance after many rounds of test-set peeking — use validation instead.
- Small data: k-fold CV on train; keep a separate test or use nested CV if you must tune aggressively.

## If they dig deeper
- Nested cross-validation — outer loop for test-like estimate, inner for hyperparameter tuning.
- Time series — walk-forward validation, not random shuffle.
- Why validation loss can rise while train loss falls — overfitting signal for early stopping.
