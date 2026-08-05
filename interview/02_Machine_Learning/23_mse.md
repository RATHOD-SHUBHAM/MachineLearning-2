# MSE — Mean Squared Error

## Say this (60–90 sec)
MSE is the average of squared prediction errors. For each house, take (true price − predicted price)², then average. Squaring does two things: all errors become positive, and large mistakes get punished much more than small ones — a 20k miss hurts four times a 10k miss in the square. Units are squared (dollars²), so MSE is great for math and training loss, but awkward to explain to a product manager. In linear regression we often minimize MSE — it pairs cleanly with gradient descent. If the interviewer asks “what loss for regression?”, MSE is the default answer unless they care about outliers.

## Why it matters
Default regression loss in textbooks and many training loops. Connects metrics to optimization.

## How it works
- eᵢ = yᵢ − ŷᵢ
- MSE = (1/n) Σ eᵢ²
- Related: RSS = Σ eᵢ² (sum, not mean); often used in closed-form derivations.

## Tradeoffs
- Use when: training with gradient-based methods; you want to heavily penalize big misses; comparing models on the same scale.
- Avoid when: reporting to humans (use RMSE/MAE); heavy outliers will inflate MSE and may mislead.

## If they dig deeper
- Why square? Differentiable, unique minimizer under Gaussian noise assumptions (MLE).
- MSE as train loss vs MSE as test metric — same formula, different role.
- Deep dive in this repo: [`animated_linear_regression/my_learnings/04_loss_and_gradients.md`](../../animated_linear_regression/my_learnings/04_loss_and_gradients.md).
