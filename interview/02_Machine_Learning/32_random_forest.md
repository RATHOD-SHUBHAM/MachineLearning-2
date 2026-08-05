# Random Forest

## Say this (60–90 sec)
Random forest is an ensemble of decision trees trained on bootstrapped data subsets with random feature subsets at each split. Each tree overfits differently; averaging their votes or predictions reduces variance without much bias increase. For spam classification, one tree might fixate on subject-line keywords while another picks up sender patterns — the forest combines both. Out-of-bag error gives a free validation estimate because each tree never sees roughly one-third of the data. It handles nonlinear interactions, needs minimal feature scaling, and gives feature importance scores. It is often the strongest out-of-the-box model for structured tabular data before you invest in heavy tuning on gradient boosting.

## Why it matters
Ensemble methods appear constantly in production ML on tabular data. Interviewers test bagging vs boosting, why averaging helps, and how randomness reduces tree correlation.

## How it works
- **Bagging**: train each tree on a bootstrap sample (sampling with replacement from training set).
- **Feature randomness**: at each split, consider only a random subset of features (e.g. √p for classification).
- **Classification**: majority vote across trees; **regression**: average predictions (house price from 500 trees).
- **OOB score**: predict each sample using trees that did not include it in training — internal validation.
- **Hyperparameters**: n_estimators, max_depth, min_samples_leaf, max_features.

## Tradeoffs
- Use when: tabular data, moderate-to-large datasets, mixed feature types, need robust default with limited tuning.
- Avoid when: inference latency is critical and hundreds of deep trees are too slow, data is very high-dimensional sparse text (linear or neural models often win), or you need a single interpretable tree.

## If they dig deeper
- Why random feature subsets? — decorrelates trees; averaging correlated trees helps less.
- More trees → lower variance, diminishing returns; does not overfit in the classic bias–variance sense (unlike boosting).
- vs Gradient Boosting — RF is parallel and robust; GBM often wins on accuracy with careful tuning but overfits easier.
- Feature importance via mean decrease in impurity — check with permutation importance for reliability.
