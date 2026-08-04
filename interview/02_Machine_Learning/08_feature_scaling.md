# Feature Scaling

## Say this (60–90 sec)
Feature scaling puts inputs on comparable numeric ranges so distance-based and gradient-based algorithms behave well. **Standardization** subtracts the mean and divides by standard deviation — each feature ends up roughly mean 0, std 1. Good when features are roughly Gaussian or when outliers are moderate; used by linear models, logistic regression, SVM, neural nets. **Normalization** — often min-max scaling to [0, 1] — rescales by min and max. Good when you need bounded inputs or for image pixels already in a range. Tree-based models don’t require scaling — splits are order-invariant. Scaling matters for k-NN, k-means, PCA, and anything using L2 distance or gradient descent on mixed-scale features. Critical rule: fit scaler on **training data only**, then transform train/val/test — otherwise you leak statistics from validation/test into training.

## Why it matters
Unscaled features can dominate distance metrics and slow or destabilize optimization. Leakage from misfit scaling is a common interview trap.

## How it works
- **Standardization (z-score)**: z = (x − μ) / σ. μ, σ computed on train set.
- **Min-max normalization**: x′ = (x − min) / (max − min). Sensitive to outliers compressing the range.
- **Robust scaling**: uses median and IQR — less outlier-sensitive.
- **When it helps**: gradient descent converges faster; SVM/k-NN/PCA treat dimensions fairly.
- **When it doesn’t matter**: decision trees, random forests — split on threshold, scale-invariant.

For deeper intuition, see `animated_linear_regression/my_learnings/01_standardization.md` and `02_standardization_vs_normalization.md`.

## Tradeoffs
- Use standardization when: features differ widely in units/magnitude; linear models, SVM, neural nets, PCA.
- Use normalization when: you need [0,1] bounds; neural nets with sigmoid; image pipelines.
- Avoid scaling trees — unnecessary compute; can confuse if interviewer asks why you scaled for RandomForest.
- Avoid fitting scaler on full dataset before split — data leakage.

## If they dig deeper
- Outliers inflate σ in standardization — robust scaler or clipping.
- Sparse high-dimensional text — scaling often skipped; TF-IDF already normalized per feature.
- Batch normalization in deep nets — different concept; normalizes activations per mini-batch during training.
