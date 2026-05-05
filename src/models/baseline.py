from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.cleaning import clean_training_frame
from src.models.dataset import FEATURE_COLUMNS
from src.models.train_lgbm import score_predictions, split_date_range
from src.utils.config import project_path


def train_baseline_models(
    dataset: pd.DataFrame,
    target: str,
    report_dir: str | Path,
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    if dataset.empty:
        raise ValueError("Training dataset is empty. Run build-training-dataset first.")
    for split in ["train", "valid", "test"]:
        if split not in set(dataset["dataset_split"]):
            raise ValueError(f"Training dataset has no {split} split")

    clean_dataset = clean_training_frame(dataset, target=target).dropna(subset=[target]).copy()
    train = clean_dataset[clean_dataset["dataset_split"] == "train"].copy()
    valid = clean_dataset[clean_dataset["dataset_split"] == "valid"].copy()
    test = clean_dataset[clean_dataset["dataset_split"] == "test"].copy()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"baseline_{target}_{timestamp}_metrics.json"
    predictions_path = output_dir / f"baseline_{target}_{timestamp}_predictions.csv"

    predictions = []
    metrics: dict[str, Any] = {
        "target": target,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "rows": {
            "train": int(len(train)),
            "valid": int(len(valid)),
            "test": int(len(test)),
        },
        "date_ranges": {
            "train": split_date_range(train),
            "valid": split_date_range(valid),
            "test": split_date_range(test),
        },
        "models": {},
    }

    model_outputs = {
        "train_mean": predict_train_mean(train, [train, valid, test], target),
        "zero": predict_zero([train, valid, test]),
        "momentum_5d": predict_feature([train, valid, test], "return_5d"),
        "linear_regression": predict_sklearn_model(LinearRegression(), train, [train, valid, test], target),
        "ridge": predict_sklearn_model(Ridge(alpha=ridge_alpha), train, [train, valid, test], target),
    }

    split_frames = {"train": train, "valid": valid, "test": test}
    for model_name, split_predictions in model_outputs.items():
        metrics["models"][model_name] = {}
        for split_name, y_pred in split_predictions.items():
            split_frame = split_frames[split_name]
            metrics["models"][model_name][split_name] = score_predictions(split_frame[target], y_pred)
            predictions.append(build_prediction_frame(split_frame, target, y_pred, model_name))

    pd.concat(predictions, ignore_index=True).to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def predict_train_mean(
    train: pd.DataFrame,
    splits: list[pd.DataFrame],
    target: str,
) -> dict[str, np.ndarray]:
    mean_value = float(train[target].mean())
    return {split_name: np.full(len(split), mean_value) for split_name, split in zip(["train", "valid", "test"], splits)}


def predict_zero(splits: list[pd.DataFrame]) -> dict[str, np.ndarray]:
    return {split_name: np.zeros(len(split)) for split_name, split in zip(["train", "valid", "test"], splits)}


def predict_feature(splits: list[pd.DataFrame], feature: str) -> dict[str, np.ndarray]:
    return {
        split_name: split[feature].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
        for split_name, split in zip(["train", "valid", "test"], splits)
    }


def predict_sklearn_model(
    estimator: object,
    train: pd.DataFrame,
    splits: list[pd.DataFrame],
    target: str,
) -> dict[str, np.ndarray]:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("estimator", estimator),
        ]
    )
    model.fit(train[FEATURE_COLUMNS], train[target])
    return {
        split_name: model.predict(split[FEATURE_COLUMNS])
        for split_name, split in zip(["train", "valid", "test"], splits)
    }


def build_prediction_frame(
    split: pd.DataFrame,
    target: str,
    prediction: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    result = split[["symbol", "ts_code", "trade_date", "adjust", "dataset_split", target]].copy()
    result["model"] = model_name
    result["prediction"] = prediction
    result["actual_direction"] = np.sign(result[target])
    result["predicted_direction"] = np.sign(result["prediction"])
    result["direction_correct"] = result["actual_direction"] == result["predicted_direction"]
    return result
