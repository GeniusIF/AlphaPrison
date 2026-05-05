from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

from src.models.cleaning import clean_training_frame
from src.models.dataset import FEATURE_COLUMNS
from src.models.train_lgbm import score_predictions, split_date_range
from src.utils.config import project_path


def rolling_validate_lgbm(
    dataset: pd.DataFrame,
    target: str,
    model_config: dict[str, Any],
    report_dir: str | Path,
    train_window: int = 252,
    test_window: int = 63,
    step: int = 63,
    min_train_rows: int = 200,
) -> dict[str, Any]:
    if dataset.empty:
        raise ValueError("Training dataset is empty. Run build-training-dataset first.")
    frame = clean_training_frame(dataset, target=target).dropna(subset=[target]).copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    dates = sorted(frame["trade_date"].unique())
    folds = build_rolling_folds(dates, train_window=train_window, test_window=test_window, step=step)
    if not folds:
        raise ValueError("Not enough dates for rolling validation. Use smaller windows or collect more history.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"rolling_lgbm_{target}_{timestamp}_metrics.json"
    predictions_path = output_dir / f"rolling_lgbm_{target}_{timestamp}_predictions.csv"

    params = dict(model_config)
    params.setdefault("objective", "regression")
    fold_reports = []
    prediction_frames = []
    for fold_index, fold in enumerate(folds, start=1):
        train = frame[frame["trade_date"].isin(fold["train_dates"])].copy()
        test = frame[frame["trade_date"].isin(fold["test_dates"])].copy()
        if len(train) < min_train_rows or test.empty:
            continue

        model = lgb.LGBMRegressor(**params)
        model.fit(train[FEATURE_COLUMNS], train[target], callbacks=[lgb.log_evaluation(period=0)])
        y_pred = model.predict(test[FEATURE_COLUMNS])
        fold_report = {
            "fold": fold_index,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_date_range": split_date_range(train),
            "test_date_range": split_date_range(test),
            "scores": score_predictions(test[target], y_pred),
        }
        fold_reports.append(fold_report)
        prediction_frames.append(build_fold_predictions(test, target=target, prediction=y_pred, fold=fold_index))

    if not fold_reports:
        raise ValueError("No rolling folds had enough rows to train.")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    aggregate_scores = score_predictions(predictions[target], predictions["prediction"].to_numpy())
    report = {
        "target": target,
        "train_window": train_window,
        "test_window": test_window,
        "step": step,
        "min_train_rows": min_train_rows,
        "folds": fold_reports,
        "aggregate_scores": aggregate_scores,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
    }
    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_rolling_folds(
    dates: list[object],
    train_window: int,
    test_window: int,
    step: int,
) -> list[dict[str, list[object]]]:
    folds = []
    start = 0
    while start + train_window + test_window <= len(dates):
        train_dates = dates[start : start + train_window]
        test_dates = dates[start + train_window : start + train_window + test_window]
        folds.append({"train_dates": train_dates, "test_dates": test_dates})
        start += step
    return folds


def build_fold_predictions(
    frame: pd.DataFrame,
    target: str,
    prediction,
    fold: int,
) -> pd.DataFrame:
    result = frame[["symbol", "ts_code", "trade_date", "adjust", target]].copy()
    result["fold"] = fold
    result["prediction"] = prediction
    result["actual_direction"] = result[target].map(sign)
    result["predicted_direction"] = result["prediction"].map(sign)
    result["direction_correct"] = result["actual_direction"] == result["predicted_direction"]
    return result


def sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
