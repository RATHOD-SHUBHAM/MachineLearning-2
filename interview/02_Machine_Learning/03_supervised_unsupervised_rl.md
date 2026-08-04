# Supervised, Unsupervised, and Reinforcement Learning

## Say this (60–90 sec)
The three main learning paradigms differ in what signal you get from data. **Supervised learning** uses labeled examples — input paired with the correct output — like emails tagged spam or not. You learn a mapping from features to labels: classification or regression. **Unsupervised learning** has no labels; you find structure in the data — clusters, low-dimensional representations, anomalies. Examples: customer segmentation with k-means, dimensionality reduction with PCA. **Reinforcement learning** is about an agent taking actions in an environment to maximize cumulative reward — no fixed label per input, but feedback over time through rewards and penalties. Chess, robotics, ad bidding. In practice most production ML I’ve seen is supervised; unsupervised supports exploration and preprocessing; RL shows up in games, recommendation, and control, but needs careful simulation or safe deployment.

## Why it matters
Shows you can place algorithms in the right bucket and pick the right problem formulation — a common opening question before “which model would you use?”

## How it works
- **Supervised**: (X, y) pairs → learn f(X) ≈ y. Tasks: classify, regress, rank.
- **Unsupervised**: X only → discover patterns. Tasks: cluster, embed, detect outliers, density estimate.
- **Semi-supervised**: small labeled set + large unlabeled set — uses structure in unlabeled data.
- **RL**: state, action, reward, policy π(a|s). Learn by trial and error; credit assignment over sequences.
- **Self-supervised**: generate labels from the data itself (predict next word) — bridges unsupervised data with supervised-style training.

## Tradeoffs
- Use supervised when: reliable labels exist and the target is clear.
- Use unsupervised when: exploring data, no labels, or labels are too expensive — clustering, visualization, pretraining.
- Use RL when: sequential decisions, delayed reward, and a simulatable or safe environment exist.
- Avoid RL when: a simpler supervised or bandit formulation works — RL is sample-inefficient and hard to debug.

## If they dig deeper
- Imbalanced labels in supervised — ties to precision/recall, not just accuracy.
- k-means assumes spherical clusters; DBSCAN for arbitrary shapes.
- RL vs contextual bandits — bandits when no long-term state matters.
