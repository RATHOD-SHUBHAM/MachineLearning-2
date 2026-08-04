# Cross-Validation

## Say this (60–90 sec)
Cross-validation gives a more stable estimate of model performance when data is limited. The common **k-fold** approach: split data into k equal folds, train on k−1 folds, validate on the held-out fold, rotate until each fold served as validation once, then average the k scores. **Stratified k-fold** keeps class proportions in each fold — important for imbalanced classification like spam detection. CV uses the training/validation portion only; the final test set stays locked away. We use CV to compare models, tune hyperparameters, and detect overfitting without wasting data on a single random split that might be lucky or unlucky. **Leave-one-out** is k = n — low bias, high variance, expensive. For time series, use **time-series split** — always train on past, validate on future, never shuffle time.

## Why it matters
Single train/val splits are noisy. CV is standard for hyperparameter tuning and shows you know how to evaluate responsibly on small datasets.

## How it works
- **k-fold CV**: partition into k subsets; for each i, train on all except fold i, score on fold i; mean ± std of k scores.
- **Stratified**: preserve label distribution per fold — e.g. ~2% spam in each fold if base rate is 2%.
- **Nested CV**: outer loop estimates generalization; inner loop tunes hyperparameters — avoids tuning bias on one val set.
- **Shuffle split / repeated k-fold**: multiple random splits for more stable estimates.
- **Pipeline + CV**: include scaling, encoding inside CV folds — fit preprocessors only on training folds.

## Tradeoffs
- Use when: small to medium data, hyperparameter search, model selection.
- Avoid when: data is temporal — use forward chaining instead of random k-fold.
- Avoid leaking test set into any CV fold used for final reporting alongside tuning on the same holdout repeatedly.

## If they dig deeper
- Group k-fold — keep same user/session in one fold (avoid leakage across related rows).
- CV variance — report mean and std across folds, not just mean.
- Why test set is still needed — CV optimizes choices; test gives unbiased final number once.
