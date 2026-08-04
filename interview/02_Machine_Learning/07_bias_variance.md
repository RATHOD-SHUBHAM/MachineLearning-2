# Bias and Variance

## Say this (60–90 sec)
Bias and variance explain why models err. **Bias** is error from wrong assumptions — the model consistently misses the true pattern because it’s too simple. High bias → underfitting. **Variance** is error from sensitivity to the particular training set — small data changes cause big prediction swings. High variance → overfitting. Total error decomposes roughly into bias², variance, and irreducible noise. The **bias–variance tradeoff**: simpler models have higher bias, lower variance; complex models lower bias but higher variance. Our job is to find the sweet spot — or use techniques that reduce variance without crushing bias: more data, ensembling, regularization. In interviews I connect this to learning curves and validation gap: high train and val error means bias problem; low train, high val means variance problem.

## Why it matters
It’s the theoretical backbone for underfitting/overfitting and model selection. Shows you think beyond “try a bigger network.”

## How it works
- **Bias**: E[ŷ] − y — systematic error; model too rigid (linear fit to quadratic data).
- **Variance**: how much ŷ changes if we retrain on a different sample — model too flexible.
- **Irreducible error**: noise in labels and unmeasured factors — no model removes this.
- **Tradeoff curve**: as complexity increases, bias drops, variance rises; test error often U-shaped.
- **Ensembling** (bagging, random forest): averages models trained on different subsamples — cuts variance.

## Tradeoffs
- Use when: explaining why a model fails and what lever to pull — complexity, data, regularization.
- Avoid when: blaming “high variance” without checking data size, leakage, or label noise.
- High bias fixes: more features, nonlinear terms, deeper model.
- High variance fixes: regularization, simpler model, more data, cross-validation for stable estimates.

## If they dig deeper
- Double descent in over-parameterized nets — test error can fall again after interpolation point.
- Bias–variance vs train/val gap — practical diagnostics map cleanly but aren’t identical math.
- k-NN: small k → high variance; large k → high bias.
