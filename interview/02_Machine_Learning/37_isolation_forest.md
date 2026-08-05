# Isolation Forest

## Say this (60–90 sec)
Isolation Forest is an unsupervised anomaly detector built on the idea that anomalies are few and different — so they’re easier to isolate with random splits. You grow many isolation trees: at each node pick a random feature and a random split value between min and max. Anomalies tend to land in short paths — they get separated early. Normal points need more splits to isolate. The anomaly score is based on average path length across trees: short path → more anomalous. It scales well to high dimensions, doesn’t assume spherical clusters, and works when you mostly have normal data with rare outliers. For sensor or transaction streams, it’s often my first baseline before denser or deeper methods.

## Why it matters
One of the most practical unsupervised anomaly algorithms in industry and interviews. Fast, few assumptions, strong baseline.

## How it works
- Build an ensemble of **isolation trees** on random subsamples.
- Each split: random feature + random threshold in that feature’s range.
- **Path length** h(x): edges from root to the leaf that isolates x.
- **Score**: shorter average path → higher anomaly score (normalized with expected path length for sample size).
- Threshold the score (or use contamination estimate) to label outliers.
- Implementation in this repo: [`Algorithm_from_Scratch/UnsupervisedLearning/IsolationForest/`](../../Algorithm_from_Scratch/UnsupervisedLearning/IsolationForest/).

## Tradeoffs
- Use when: tabular data, high dimensions, need a fast unsupervised baseline, anomalies are rare and different.
- Avoid when: anomalies are dense clusters of their own (many similar fraud patterns) — isolation can struggle; or when you need density-based structure (try LOF/DBSCAN).

## If they dig deeper
- Contamination hyperparameter: expected fraction of outliers — affects threshold, not the trees themselves as much.
- vs LOF: Isolation Forest is global/isolation-based; LOF is local density-based.
- Feature scaling: less sensitive than distance methods, but irrelevant features still add noise in random splits.
- Extended Isolation Forest: reduces bias from axis-aligned splits.
