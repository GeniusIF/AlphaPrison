from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.dataset import FEATURE_COLUMNS
from src.utils.config import project_path


def analyze_factors(
    dataset: pd.DataFrame,
    target: str,
    report_dir: str | Path,
    quantiles: int = 5,
    min_stocks_per_date: int = 5,
) -> dict[str, Any]:
    if dataset.empty:
        raise ValueError("Training dataset is empty. Run build-training-dataset first.")
    if target not in dataset.columns:
        raise ValueError(f"Target column not found: {target}")

    frame = dataset.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_factor_summary(frame, target=target, min_stocks_per_date=min_stocks_per_date)
    quantile_returns = build_quantile_returns(frame, target=target, quantiles=quantiles)

    summary_path = output_dir / f"factor_analysis_{target}_{timestamp}_summary.csv"
    quantile_path = output_dir / f"factor_analysis_{target}_{timestamp}_quantiles.csv"
    report_path = output_dir / f"factor_analysis_{target}_{timestamp}.json"

    summary.to_csv(summary_path, index=False)
    quantile_returns.to_csv(quantile_path, index=False)

    report = {
        "target": target,
        "rows": int(len(frame)),
        "date_range": {
            "start": str(frame["trade_date"].min()),
            "end": str(frame["trade_date"].max()),
        },
        "summary_path": str(summary_path),
        "quantile_path": str(quantile_path),
        "report_path": str(report_path),
        "top_rank_ic": summary.head(10).to_dict(orient="records"),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_factor_summary(
    dataset: pd.DataFrame,
    target: str,
    min_stocks_per_date: int = 5,
) -> pd.DataFrame:
    rows = []
    for feature in FEATURE_COLUMNS:
        if feature not in dataset.columns:
            continue
        pair = dataset[[feature, target, "trade_date"]].dropna()
        if pair.empty or pair[feature].nunique() < 2:
            continue

        daily_ic = build_daily_ic(pair, feature=feature, target=target, min_stocks_per_date=min_stocks_per_date)
        overall_ic = pair[feature].astype(float).corr(pair[target].astype(float), method="pearson")
        overall_rank_ic = pair[feature].astype(float).corr(pair[target].astype(float), method="spearman")
        rows.append(
            {
                "feature": feature,
                "rows": int(len(pair)),
                "overall_ic": safe_float(overall_ic),
                "overall_rank_ic": safe_float(overall_rank_ic),
                "daily_ic_mean": safe_float(daily_ic["ic"].mean()) if not daily_ic.empty else None,
                "daily_ic_std": safe_float(daily_ic["ic"].std()) if not daily_ic.empty else None,
                "daily_rank_ic_mean": safe_float(daily_ic["rank_ic"].mean()) if not daily_ic.empty else None,
                "daily_rank_ic_std": safe_float(daily_ic["rank_ic"].std()) if not daily_ic.empty else None,
                "daily_rank_ic_positive_rate": safe_float((daily_ic["rank_ic"] > 0).mean()) if not daily_ic.empty else None,
                "daily_count": int(len(daily_ic)),
                "abs_daily_rank_ic_mean": abs(safe_float(daily_ic["rank_ic"].mean()) or 0),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    max_daily_count = max(int(result["daily_count"].max()), 1)
    reliable_threshold = max(5, int(max_daily_count * 0.2))
    result["daily_coverage"] = result["daily_count"] / max_daily_count
    result["is_reliable"] = result["daily_count"] >= reliable_threshold
    return result.sort_values(
        ["is_reliable", "abs_daily_rank_ic_mean", "daily_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_daily_ic(
    dataset: pd.DataFrame,
    feature: str,
    target: str,
    min_stocks_per_date: int = 5,
) -> pd.DataFrame:
    rows = []
    for trade_date, group in dataset.groupby("trade_date"):
        group = group[[feature, target]].dropna()
        if len(group) < min_stocks_per_date or group[feature].nunique() < 2 or group[target].nunique() < 2:
            continue
        rows.append(
            {
                "trade_date": trade_date,
                "feature": feature,
                "ic": safe_float(group[feature].astype(float).corr(group[target].astype(float), method="pearson")),
                "rank_ic": safe_float(group[feature].astype(float).corr(group[target].astype(float), method="spearman")),
                "stocks": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def build_quantile_returns(
    dataset: pd.DataFrame,
    target: str,
    quantiles: int = 5,
) -> pd.DataFrame:
    rows = []
    for feature in FEATURE_COLUMNS:
        if feature not in dataset.columns:
            continue
        pieces = []
        for trade_date, group in dataset[["trade_date", feature, target]].dropna().groupby("trade_date"):
            bucketed = assign_quantiles(group, feature=feature, quantiles=quantiles)
            if bucketed.empty:
                continue
            bucketed["trade_date"] = trade_date
            pieces.append(bucketed)
        if not pieces:
            continue
        assigned = pd.concat(pieces, ignore_index=True)
        grouped = assigned.groupby("quantile", as_index=False)[target].agg(["mean", "count"]).reset_index()
        top = grouped.loc[grouped["quantile"].idxmax(), "mean"]
        bottom = grouped.loc[grouped["quantile"].idxmin(), "mean"]
        for row in grouped.itertuples(index=False):
            rows.append(
                {
                    "feature": feature,
                    "quantile": int(row.quantile),
                    "mean_forward_return": float(row.mean),
                    "count": int(row.count),
                    "top_minus_bottom": float(top - bottom),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "quantile"]).reset_index(drop=True) if rows else pd.DataFrame()


def assign_quantiles(group: pd.DataFrame, feature: str, quantiles: int) -> pd.DataFrame:
    unique_values = group[feature].nunique()
    bucket_count = min(quantiles, unique_values, len(group))
    if bucket_count < 2:
        return pd.DataFrame()

    result = group.copy()
    ranks = result[feature].rank(method="first")
    result["quantile"] = pd.qcut(ranks, q=bucket_count, labels=False, duplicates="drop") + 1
    return result


def safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
