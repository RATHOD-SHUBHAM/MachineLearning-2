# MAE — Mean Absolute Error

## Say this (60–90 sec)
MAE is a regression metric: the average absolute gap between prediction and truth. For house prices, if I predict 300k and the true price is 320k, the absolute error is 20k; MAE averages that over all houses. Formula: mean of |y − ŷ|. It’s in the same units as the target — dollars, degrees, whatever — so it’s easy to explain to non-ML people. Unlike MSE, it doesn’t square errors, so a few huge mistakes don’t dominate as hard. It’s also more robust to outliers. Downside: the absolute value isn’t smooth at zero, which matters more for optimization than for reporting a score.

## Why it matters
Go-to “how wrong am I on average?” number for regression interviews and dashboards.

## How it works
- Error per point: eᵢ = yᵢ − ŷᵢ
- MAE = (1/n) Σ |eᵢ|
- Same scale as y — interpret directly.

## Tradeoffs
- Use when: you want an interpretable average error; outliers shouldn’t crush the metric; stakeholder-facing reports.
- Avoid when: you especially want to punish large errors (prefer MSE/RMSE); or percentage error matters more (MAPE).

## If they dig deeper
- MAE vs MSE for training: MSE has nicer gradients for many optimizers; MAE (L1 loss) is more robust.
- Median Absolute Error — even more robust to outliers than MAE.
