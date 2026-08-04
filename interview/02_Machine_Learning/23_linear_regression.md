# Linear Regression

## Say this (60–90 sec)
Linear regression predicts a continuous target from features by fitting a weighted sum plus a bias — for example, house price from square footage, bedrooms, and zip code. The model is y-hat equals w transpose x plus b. Training finds weights that minimize prediction error, usually mean squared error. Geometrically, you are fitting a line or hyperplane through the data cloud. It is simple, fast, and highly interpretable — each weight tells you how much that feature moves the prediction, holding others fixed. I use it as a baseline for any regression problem and as a building block for understanding loss, gradients, and regularization. It assumes roughly linear relationships and additive feature effects; when that holds, it often performs surprisingly well.

## Why it matters
It is the simplest supervised model and the gateway to gradient descent, regularization, and neural nets. Interviewers expect you to explain the prediction equation, loss, and when linearity breaks down.

## How it works
- **Model**: ŷ = wᵀx + b (one weight per feature plus intercept).
- **Loss**: MSE = average of (y − ŷ)² — penalizes large errors quadratically.
- **Training**: minimize MSE via closed-form normal equation or iterative gradient descent.
- **Interpretation**: weight wⱼ = expected change in y per unit change in feature j (if features are scaled comparably).
- **Example**: price ≈ 50k + 200 × sqft + 15k × bedrooms.

## Tradeoffs
- Use when: target is continuous, relationships are roughly linear, you need speed and interpretability, or as a strong baseline.
- Avoid when: interactions and nonlinearities dominate, outliers are heavy (MSE is sensitive), or you need calibrated probabilities.

## If they dig deeper
- Metrics: MAE / MSE / RMSE / R² — see notes `19`–`22` in this folder.
- Normal equation vs gradient descent — closed form is O(n³) in features; GD scales better for large n or online learning.
- Multicollinearity inflates weight variance — regularization or feature selection helps.
- Residual plots reveal nonlinearity — if curved, try polynomial features, trees, or a different model.
- Deeper dive: [animated_linear_regression/my_learnings/05_linear_regression_and_training.md](../../animated_linear_regression/my_learnings/05_linear_regression_and_training.md) and [Algorithm_from_Scratch/LinearRegression/](../../Algorithm_from_Scratch/LinearRegression/).
