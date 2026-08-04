# Local Outlier Factor (LOF)

## Say this (60–90 sec)
Local Outlier Factor scores how isolated a point is **relative to its neighbors’ density**. A point in a sparse region next to a dense cluster gets a high LOF even if it’s not far in absolute distance. You pick k neighbors, estimate local density via reachability distance, then compare a point’s density to its neighbors’ densities. LOF ≈ 1 means similar density to neighbors (inlier); LOF ≫ 1 means much sparser (outlier). This shines when “normal” has multiple clusters with different densities — a global distance threshold would fail. For anomaly detection on tabular or low-dimensional spatial data, LOF is the classic local-density baseline next to Isolation Forest.

## Why it matters
Shows you understand **local** vs **global** anomalies — a common interview distinction.

## How it works
- Choose **k** (number of neighbors).
- Compute k-distance and reachability distances to neighbors.
- Local reachability density (LRD): inverse of average reachability distance.
- **LOF(x)** ≈ average(LRD of neighbors) / LRD(x).
- Threshold LOF scores to flag anomalies (or rank top-N).

## Tradeoffs
- Use when: normal data has regions of different density; local context matters.
- Avoid when: very high dimensions (neighbors become less meaningful), huge datasets (LOF is heavier than Isolation Forest), or you need a single global “far from everything” score.

## If they dig deeper
- vs Isolation Forest: LOF = local density; IF = easy-to-isolate with random trees — try both as baselines.
- vs DBSCAN: DBSCAN hard-labels noise; LOF gives a continuous score.
- Choice of k: too small → noisy scores; too large → becomes more global.
