# MAPE — Mean Absolute Percentage Error

## Say this (60–90 sec)
MAPE is the average absolute percent error: mean of |y − ŷ| / |y| × 100%. For revenue forecasting, MAPE of 8% means you’re off by about 8% of the true value on average — easy for business stakeholders. It is scale-free, so you can compare across products with different price ranges. The big catch: it blows up or is undefined when true y is zero or near zero, and it puts heavier weight on errors when y is small. So for house prices it’s often fine; for counts that can be zero, use MAE/RMSE or a variant like sMAPE instead.

## Why it matters
Common business KPI for forecasting. Interviewers test whether you know the zero-denominator trap.

## How it works
- MAPE = (100%/n) Σ |yᵢ − ŷᵢ| / |yᵢ|
- Same units: percent.
- Symmetric MAPE (sMAPE) uses |y| + |ŷ| in the denominator to soften zeros — variants exist.

## Tradeoffs
- Use when: targets are strictly positive and stakeholders think in percentages.
- Avoid when: y can be 0; heavy skew with tiny y values; optimizing MAPE directly can bias predictions low.

## If they dig deeper
- Why low bias: under-predicting can reduce MAPE on some distributions — don’t optimize MAPE blindly.
- Prefer MAE/RMSE for model selection; report MAPE for communication when appropriate.
