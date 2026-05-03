from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import project_path


def load_json_report(path: str | Path) -> dict[str, Any]:
    return json.loads(project_path(path).read_text(encoding="utf-8"))


def list_json_reports(report_dir: str | Path) -> list[dict[str, Any]]:
    directory = project_path(report_dir)
    if not directory.exists():
        return []

    reports = []
    for path in sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = load_json_report(path)
        except json.JSONDecodeError:
            continue
        reports.append(
            {
                "name": path.name,
                "path": path,
                "type": infer_report_type(payload, path.name),
                "payload": payload,
            }
        )
    return reports


def infer_report_type(payload: dict[str, Any], filename: str = "") -> str:
    if "aggregate_scores" in payload and "folds" in payload:
        return "rolling_lgbm"
    if "models" in payload:
        return "baseline"
    if "scores" in payload:
        return "lgbm"
    if "top_rank_ic" in payload or filename.startswith("factor_analysis_"):
        return "factor_analysis"
    return "unknown"


def latest_report_by_type(report_dir: str | Path, report_type: str) -> dict[str, Any] | None:
    for report in list_json_reports(report_dir):
        if report["type"] == report_type:
            return report
    return None


def report_summary_tables(report: dict[str, Any]) -> dict[str, pd.DataFrame]:
    payload = report["payload"]
    report_type = report["type"]
    if report_type == "lgbm":
        return {"scores": score_table(payload.get("scores", {}))}
    if report_type == "baseline":
        return {"test_scores": baseline_test_table(payload.get("models", {}))}
    if report_type == "rolling_lgbm":
        return {
            "aggregate_scores": score_table({"aggregate": payload.get("aggregate_scores", {})}),
            "folds": rolling_fold_table(payload.get("folds", [])),
        }
    if report_type == "factor_analysis":
        return {"top_rank_ic": pd.DataFrame(payload.get("top_rank_ic", []))}
    return {}


def score_table(scores: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, score in scores.items():
        if isinstance(score, dict):
            rows.append({"name": name, **score})
    return pd.DataFrame(rows)


def baseline_test_table(models: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_name, splits in models.items():
        test_score = splits.get("test", {}) if isinstance(splits, dict) else {}
        rows.append({"model": model_name, **test_score})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def rolling_fold_table(folds: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for fold in folds:
        score = fold.get("scores", {})
        rows.append(
            {
                "fold": fold.get("fold"),
                "train_start": fold.get("train_date_range", {}).get("start"),
                "train_end": fold.get("train_date_range", {}).get("end"),
                "test_start": fold.get("test_date_range", {}).get("start"),
                "test_end": fold.get("test_date_range", {}).get("end"),
                "test_rows": fold.get("test_rows"),
                "mae": score.get("mae"),
                "rmse": score.get("rmse"),
                "r2": score.get("r2"),
                "directional_accuracy": score.get("directional_accuracy"),
            }
        )
    return pd.DataFrame(rows)
