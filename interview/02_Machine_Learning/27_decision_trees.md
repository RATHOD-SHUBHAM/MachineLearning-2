# Decision Trees

## Say this (60–90 sec)
Decision trees learn a flowchart of if-then rules to classify or regress. At each node, the algorithm picks a feature and split threshold that best separates the labels — for spam, maybe "contains 'free'" or "sender not in contacts." It recurses until leaves are pure enough or a stopping rule fires. Predictions are the majority class or average target in the leaf. Trees handle nonlinear boundaries and feature interactions naturally — no need to engineer "sqft × bedrooms" manually. They are easy to explain to non-technical stakeholders. The downside is they overfit aggressively if you let them grow deep, so you control depth, min samples per leaf, or prune using validation performance.

## Why it matters
Trees are the foundation for random forests and gradient boosting — the workhorses of tabular ML. Interviewers want split criteria, overfitting controls, and why a single tree is unstable.

## How it works
- **Classification splits**: maximize information gain (entropy reduction) or Gini impurity decrease.
- **Regression splits**: minimize variance within child nodes — pick split that most reduces MSE.
- **Growth**: greedy, top-down; each split uses only one feature at a time.
- **Prediction**: traverse from root to leaf; classify by majority vote or regress by leaf mean.
- **Stopping**: max depth, min samples per split/leaf, max leaf nodes, or post-prune with cost-complexity pruning.

## Tradeoffs
- Use when: tabular data, nonlinear patterns, mixed feature types, need interpretability, or as base learners for ensembles.
- Avoid when: data is high-dimensional and sparse (text embeddings), you need smooth extrapolation (regression on unseen ranges), or a single tree must be stable — use an ensemble instead.

## If they dig deeper
- Greedy splits are locally optimal, not globally — not guaranteed best tree.
- Sensitive to small data changes — different bootstrap sample can yield a very different tree.
- Missing values: surrogate splits, imputation, or algorithms that handle NaN natively (e.g. XGBoost).
- Feature importance from total impurity decrease — useful but biased toward high-cardinality features.
