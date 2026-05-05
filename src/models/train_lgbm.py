from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.cleaning import clean_training_frame
from src.models.dataset import FEATURE_COLUMNS
from src.utils.config import project_path


def train_lgbm_regressor(
    dataset: pd.DataFrame,
    target: str,
    model_config: dict[str, Any],
    model_dir: str | Path,
    report_dir: str | Path,
) -> dict[str, Any]:
    if dataset.empty:
        raise ValueError("Training dataset is empty. Build model_training_dataset first.")
    for split in ["train", "valid", "test"]:
        if split not in set(dataset["dataset_split"]):
            raise ValueError(f"Training dataset has no {split} split")

    clean_dataset = clean_training_frame(dataset, target=target).dropna(subset=[target]).copy()
    train = clean_dataset[clean_dataset["dataset_split"] == "train"].copy()
    valid = clean_dataset[clean_dataset["dataset_split"] == "valid"].copy()
    test = clean_dataset[clean_dataset["dataset_split"] == "test"].copy()

    params = dict(model_config)
    params.setdefault("objective", "regression")
    model = lgb.LGBMRegressor(**params)
    model.fit(
        train[FEATURE_COLUMNS],
        train[target],
        eval_set=[(valid[FEATURE_COLUMNS], valid[target])],
        eval_metric="rmse",
        callbacks=[lgb.log_evaluation(period=0)],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = project_path(model_dir) / f"lgbm_{target}_{timestamp}.txt"
    metrics_path = project_path(report_dir) / f"lgbm_{target}_{timestamp}_metrics.json"
    predictions_path = project_path(report_dir) / f"lgbm_{target}_{timestamp}_predictions.csv"
    importance_path = project_path(report_dir) / f"lgbm_{target}_{timestamp}_feature_importance.csv"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        "target": target,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "importance_path": str(importance_path),
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
        "scores": {
            "train": score_predictions(train[target], model.predict(train[FEATURE_COLUMNS])),
            "valid": score_predictions(valid[target], model.predict(valid[FEATURE_COLUMNS])),
            "test": score_predictions(test[target], model.predict(test[FEATURE_COLUMNS])),
        },
    }

    predictions = build_predictions_frame(model, clean_dataset, target)
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    model.booster_.save_model(str(model_path))
    predictions.to_csv(predictions_path, index=False)
    importance.to_csv(importance_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def score_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float((np.sign(y_true.to_numpy()) == np.sign(y_pred)).mean()),
    }


def split_date_range(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty:
        return {"start": None, "end": None}
    dates = pd.to_datetime(frame["trade_date"])
    return {"start": str(dates.min().date()), "end": str(dates.max().date())}


def build_predictions_frame(
    model: lgb.LGBMRegressor,
    dataset: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    result = dataset[["symbol", "ts_code", "trade_date", "adjust", "dataset_split", target]].copy()
    result["prediction"] = model.predict(dataset[FEATURE_COLUMNS])
    result["actual_direction"] = np.sign(result[target])
    result["predicted_direction"] = np.sign(result["prediction"])
    result["direction_correct"] = result["actual_direction"] == result["predicted_direction"]
    return result
