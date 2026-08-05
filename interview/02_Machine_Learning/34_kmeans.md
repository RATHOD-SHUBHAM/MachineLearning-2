# K-Means

## Say this (60–90 sec)
K-means is an unsupervised clustering algorithm that partitions data into K groups. You pick K centroids — either randomly or via k-means++ — then alternate two steps: assign each point to its nearest centroid, then move each centroid to the mean of its assigned points. Repeat until assignments stabilize. For customer segmentation, features might be purchase frequency and average spend; k-means groups similar customers so marketing can target each cluster differently. It assumes clusters are roughly spherical and similar in size. You choose K via domain knowledge, the elbow method on within-cluster sum of squares, or silhouette score. It is simple, fast, and scales well — a standard first pass for exploratory clustering.

## Why it matters
K-means is the canonical clustering algorithm. Interviewers test the algorithm loop, how to pick K, limitations of Euclidean distance, and initialization sensitivity.

## How it works
- **Input**: feature matrix X, number of clusters K, distance metric (usually Euclidean).
- **Assign step**: each point → nearest centroid (minimum distance).
- **Update step**: centroid_k = mean of all points assigned to cluster k.
- **Objective**: minimize within-cluster sum of squares (WCSS / inertia) — not convex, so result depends on init.
- **k-means++**: spread initial centroids apart — reduces bad local minima.

## Tradeoffs
- Use when: exploratory segmentation, preprocessing for downstream models, moderate K, roughly globular clusters of similar size.
- Avoid when: clusters are non-spherical, varying density, or overlapping — try DBSCAN, GMM, or hierarchical clustering.

## If they dig deeper
- K-means assumes equal-variance spherical clusters — GMM relaxes this with soft assignments and covariance.
- Scale features first — otherwise high-magnitude features dominate distance.
- Elbow plot: WCSS drops sharply until K is "right," then flattens — subjective in practice.
- Empty clusters can occur — reinitialize or use k-means++ and multiple random restarts (n_init).
