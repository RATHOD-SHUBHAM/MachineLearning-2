"""
Isolation Forest anomaly detection using scikit-learn (production-style pipeline).

Dependencies: numpy, scikit-learn, joblib; pandas optional for CSV CLI.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PIPELINE_VERSION = 2

MaxSamples = Union[int, float, str]


@dataclass
class IsolationForestConfig:
    n_estimators: int = 100
    max_samples: MaxSamples = 256
    contamination: float = 0.1
    max_features: Union[int, float] = 1.0
    bootstrap: bool = False
    random_state: Optional[int] = 42

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if isinstance(self.max_samples, str):
            if self.max_samples != "auto":
                raise ValueError('max_samples string must be "auto"')
        elif isinstance(self.max_samples, float):
            if not 0.0 < self.max_samples <= 1.0:
                raise ValueError("max_samples float must be in (0, 1]")
        elif isinstance(self.max_samples, int):
            if self.max_samples < 2:
                raise ValueError("max_samples int must be >= 2")
        else:
            raise TypeError("max_samples must be int, float, or 'auto'")
        if not 0.0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")

    def to_sklearn_kwargs(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
        }


@dataclass
class AnomalyPipelineArtifacts:
    """Serializable bundle for serving."""

    version: int
    sklearn_pipeline: Pipeline
    config: IsolationForestConfig
    feature_columns: Optional[list[str]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class IsolationForestAnomalyPipeline:
    """
    StandardScaler + IsolationForest in a sklearn Pipeline; persist with joblib + manifest JSON.
    """

    def __init__(self, config: Optional[IsolationForestConfig] = None) -> None:
        self.config = config or IsolationForestConfig()
        self.pipeline_: Optional[Pipeline] = None
        self.feature_columns_: Optional[list[str]] = None

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("iforest", IsolationForest(**self.config.to_sklearn_kwargs())),
            ]
        )

    def fit(self, X: np.ndarray, feature_columns: Optional[Iterable[str]] = None) -> IsolationForestAnomalyPipeline:
        X = np.asarray(X, dtype=np.float64)
        if feature_columns is not None:
            self.feature_columns_ = list(feature_columns)
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X)
        return self

    def _ensure_fitted(self) -> None:
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() before transform/predict.")

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float64)
        return self.pipeline_.named_steps["scaler"].transform(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        return self.pipeline_.score_samples(np.asarray(X, dtype=np.float64))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        return self.pipeline_.decision_function(np.asarray(X, dtype=np.float64))

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        return self.pipeline_.predict(np.asarray(X, dtype=np.float64))

    def save(self, directory: str | Path, metadata: Optional[dict[str, Any]] = None) -> None:
        self._ensure_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        iforest: IsolationForest = self.pipeline_.named_steps["iforest"]
        bundle = AnomalyPipelineArtifacts(
            version=PIPELINE_VERSION,
            sklearn_pipeline=self.pipeline_,
            config=self.config,
            feature_columns=self.feature_columns_,
            metadata=metadata or {},
        )
        joblib.dump(bundle, directory / "pipeline.joblib")
        n_fit = getattr(iforest, "n_samples_", None)
        sidecar = {
            "pipeline_version": PIPELINE_VERSION,
            "sklearn": {"IsolationForest": iforest.get_params(deep=False)},
            "n_features": int(iforest.n_features_in_),
            "n_samples_fit": int(n_fit) if n_fit is not None else None,
            "contamination": self.config.contamination,
            "n_estimators": self.config.n_estimators,
            "max_samples": self.config.max_samples,
            "offset_": float(iforest.offset_),
            "feature_columns": self.feature_columns_,
            "metadata": bundle.metadata,
        }
        (directory / "manifest.json").write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> IsolationForestAnomalyPipeline:
        directory = Path(directory)
        bundle: AnomalyPipelineArtifacts = joblib.load(directory / "pipeline.joblib")
        if bundle.version != PIPELINE_VERSION:
            warnings.warn(
                f"Loaded pipeline version {bundle.version}; code expects {PIPELINE_VERSION}.",
                UserWarning,
                stacklevel=2,
            )
        pipe = cls(config=bundle.config)
        pipe.pipeline_ = bundle.sklearn_pipeline
        pipe.feature_columns_ = bundle.feature_columns
        return pipe


def _load_credit_card_frame(path: Path, max_rows: Optional[int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("Install pandas to use --data creditcard.csv: pip install pandas") from e
    df = pd.read_csv(path, nrows=max_rows)
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column for evaluation")
    y = df["Class"].astype(int).to_numpy()
    feature_cols = [c for c in df.columns if c not in ("Class", "Time")]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    return X, y, feature_cols


def _eval_binary(y_true: np.ndarray, y_pred_outlier: np.ndarray) -> dict[str, float]:
    """y_pred_outlier: True if predicted anomaly (-1). y_true: 1 fraud, 0 normal."""
    actual = y_true.astype(bool)
    pred = y_pred_outlier
    tp = int(np.sum(actual & pred))
    fp = int(np.sum(~actual & pred))
    fn = int(np.sum(actual & ~pred))
    tn = int(np.sum(~actual & ~pred))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": prec, "recall": rec, "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolation Forest anomaly pipeline (scikit-learn)")
    parser.add_argument("--train", action="store_true", help="Fit pipeline and save artifact")
    parser.add_argument("--data", type=str, default="", help="Path to CSV (e.g. creditcard.csv)")
    parser.add_argument("--out", type=str, default="./artifacts/isolation_forest", help="Artifact directory")
    parser.add_argument("--max-rows", type=int, default=50000, help="Row cap for CSV training (speed)")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.train:
        parser.error("Specify --train to fit, or import IsolationForestAnomalyPipeline in code.")

    if not args.data:
        parser.error("--data is required with --train")

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Data file not found: {data_path}")

    X, y, feature_cols = _load_credit_card_frame(data_path, args.max_rows or None)
    if np.any(~np.isfinite(X)):
        warnings.warn("Non-finite values in X; replacing with 0 for demo.", UserWarning, stacklevel=2)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    config = IsolationForestConfig(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        contamination=args.contamination,
        random_state=args.seed,
    )
    pipe = IsolationForestAnomalyPipeline(config).fit(X, feature_columns=feature_cols)
    pred = pipe.predict(X)
    y_out = pred == -1
    metrics = _eval_binary(y, y_out)

    pipe.save(
        Path(args.out),
        metadata={"metrics_on_train_subset": metrics, "data_path": str(data_path.resolve())},
    )
    print(json.dumps({"saved_to": str(Path(args.out).resolve()), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
