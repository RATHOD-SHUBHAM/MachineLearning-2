# RMSE — Root Mean Squared Error

## Say this (60–90 sec)
RMSE is simply the square root of MSE. You still punish large errors via squaring, but then take the root so the metric is back in the original units — dollars, not dollars squared. If RMSE is 15k on house prices, I’m saying typical squared-error magnitude is like a 15k error. It’s usually what people report instead of raw MSE. Compared to MAE, RMSE is still more sensitive to outliers because of the square inside. If MAE and RMSE are very far apart, that often means a few huge errors are driving the score.

## Why it matters
The regression metric you’ll actually say in a meeting. Same intuition as MSE, human-readable scale.

## How it works
- RMSE = √MSE = √( (1/n) Σ (yᵢ − ŷᵢ)² )
- Same units as y.
- Always ≥ MAE (by math); gap hints at outlier influence.

## Tradeoffs
- Use when: reporting regression performance; comparing models where large errors matter.
- Avoid when: you need maximum robustness to outliers (prefer MAE); or relative/percentage error (MAPE).

## If they dig deeper
- “Is lower RMSE always better?” — on the same data and target scale, yes for comparison; don’t compare RMSE across differently scaled targets without normalizing.
- Normalized RMSE (divide by range or mean of y) — for comparing across datasets.
