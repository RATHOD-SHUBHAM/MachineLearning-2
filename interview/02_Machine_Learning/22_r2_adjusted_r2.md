# R² and Adjusted R²

## Say this (60–90 sec)
R² — the coefficient of determination — answers: how much of the variance in y does my model explain? It’s 1 minus (MSE of my model / variance of y), or more precisely 1 − SS_res/SS_tot. R² = 1 is perfect; R² = 0 means I’m no better than predicting the mean; negative means I’m worse than the mean. It’s scale-free, so easier to compare than raw RMSE across problems — but a “good” R² depends on the domain. Adjusted R² penalizes adding useless features: it only rises if a new feature improves fit enough to offset complexity. That’s why we use adjusted R² when comparing linear models with different numbers of predictors.

## Why it matters
Classic interview question for linear regression. Shows you know fit quality beyond “loss went down.”

## How it works
- SS_res = Σ (yᵢ − ŷᵢ)²  
- SS_tot = Σ (yᵢ − ȳ)²  
- R² = 1 − SS_res / SS_tot  
- Adjusted R² = 1 − (1−R²)(n−1)/(n−p−1) — p = number of features  
- Repo note: [`calculate_adjusted_R2_.ipynb`](../../calculate_adjusted_R2_.ipynb)

## Tradeoffs
- Use when: explaining variance captured; comparing nested linear models (prefer adjusted R²).
- Avoid when: non-linear models where R² can mislead; comparing across totally different target definitions; class imbalance problems (wrong family — use classification metrics).

## If they dig deeper
- R² always increases (or stays) when you add features in-sample — hence adjusted R² / holdout RMSE.
- High R² ≠ good causal model; doesn’t prove you should deploy.
- For non-linear models, report RMSE/MAE on a test set — don’t lean only on R².
