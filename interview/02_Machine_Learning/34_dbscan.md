# DBSCAN

## Say this (60–90 sec)
DBSCAN is a density-based clustering algorithm — Density-Based Spatial Clustering of Applications with Noise. Instead of picking K, you pick a neighborhood radius **eps** and a minimum neighbor count **min_samples**. A **core point** has at least min_samples points within eps. Clusters grow by connecting core points that are within eps of each other; **border points** are near a core but aren’t dense enough themselves; points that are neither are labeled **noise** — and those noise points are often treated as anomalies. It finds arbitrarily shaped clusters and doesn’t force every point into a group. Classic use: geospatial grouping, or flagging sparse outliers around dense regions of normal traffic.

## Why it matters
Interview favorite for unsupervised learning: contrasts with k-means (no K, non-spherical clusters, explicit noise/anomaly label).

## How it works
1. For each point, find neighbors within distance **eps**.
2. If neighbor count ≥ **min_samples** → core point.
3. Form clusters by linking core points in each other’s neighborhoods (density-reachability).
4. Assign border points to a nearby cluster; label the rest as **noise** (−1 in sklearn).
5. Distance metric and feature scaling matter a lot — usually standardize first.

## Tradeoffs
- Use when: clusters have irregular shapes, varying that density is still “dense vs sparse,” you want automatic outlier labeling.
- Avoid when: clusters have very different densities (one eps won’t fit all), high dimensionality (distance concentration), or huge datasets without approximate neighbors — eps/min_samples tuning is painful.

## If they dig deeper
- How to pick eps: k-distance plot (sorted distance to k-th neighbor) — look for the elbow.
- HDBSCAN: hierarchical extension — better with varying density.
- Anomaly view: noise points ≈ outliers; not a calibrated anomaly score like Isolation Forest.
- vs k-means: DBSCAN doesn’t need K; k-means always assigns every point to a cluster.
