from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.cleaning import clean_training_frame
from src.models.dataset import FEATURE_COLUMNS
from src.models.factor_analysis import build_factor_summary
from src.utils.config import project_path


def backtest_single_factors(
    dataset: pd.DataFrame,
    target: str,
    report_dir: str | Path,
    quantile: float = 0.2,
    top_n: int | None = None,
    rebalance_step: int = 5,
    min_stocks_per_date: int = 5,
    cost_rate: float = 0.00161,
) -> dict[str, Any]:
    if dataset.empty:
        raise ValueError("Training dataset is empty. Run build-training-dataset first.")
    if target not in dataset.columns:
        raise ValueError(f"Target column not found: {target}")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1].")
    if rebalance_step < 1:
        raise ValueError("rebalance_step must be >= 1.")

    frame = clean_training_frame(dataset, target=target).copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    direction_source = "train" if "dataset_split" in frame.columns and (frame["dataset_split"] == "train").any() else "all"
    direction_frame = frame[frame["dataset_split"] == "train"] if direction_source == "train" else frame
    factor_summary = build_factor_summary(direction_frame, target=target, min_stocks_per_date=min_stocks_per_date)
    directions = {
        row.feature: {
            "factor_direction": row.factor_direction,
            "direction_multiplier": int(row.direction_multiplier),
            "daily_rank_ic_mean": row.daily_rank_ic_mean,
            "daily_rank_ic_ir": row.daily_rank_ic_ir,
        }
        for row in factor_summary.itertuples(index=False)
    }

    summaries: list[dict[str, Any]] = []
    detail_pieces: list[pd.DataFrame] = []
    for feature in FEATURE_COLUMNS:
        direction = directions.get(feature)
        if not direction or direction["direction_multiplier"] == 0:
            continue
        summary, detail = backtest_one_factor(
            frame,
            feature=feature,
            target=target,
            direction_multiplier=direction["direction_multiplier"],
            quantile=quantile,
            top_n=top_n,
            rebalance_step=rebalance_step,
            min_stocks_per_date=min_stocks_per_date,
            cost_rate=cost_rate,
        )
        if not summary:
            continue
        summary.update(direction)
        summaries.append(summary)
        detail_pieces.append(detail)

    summary_frame = pd.DataFrame(summaries)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(
            ["cumulative_return", "max_drawdown", "periods"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    detail_frame = pd.concat(detail_pieces, ignore_index=True) if detail_pieces else pd.DataFrame()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"single_factor_backtest_{target}_{timestamp}_summary.csv"
    detail_path = output_dir / f"single_factor_backtest_{target}_{timestamp}_detail.csv"
    report_path = output_dir / f"single_factor_backtest_{target}_{timestamp}.json"
    summary_frame.to_csv(summary_path, index=False)
    detail_frame.to_csv(detail_path, index=False)

    report = {
        "target": target,
        "rows": int(len(frame)),
        "direction_source": direction_source,
        "quantile": quantile,
        "top_n": top_n,
        "rebalance_step": rebalance_step,
        "min_stocks_per_date": min_stocks_per_date,
        "cost_rate": cost_rate,
        "date_range": {
            "start": str(frame["trade_date"].min()),
            "end": str(frame["trade_date"].max()),
        },
        "summary_path": str(summary_path),
        "detail_path": str(detail_path),
        "report_path": str(report_path),
        "top_factors": summary_frame.head(10).to_dict(orient="records"),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def backtest_one_factor(
    dataset: pd.DataFrame,
    feature: str,
    target: str,
    direction_multiplier: int,
    quantile: float,
    top_n: int | None,
    rebalance_step: int,
    min_stocks_per_date: int,
    cost_rate: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    trade_dates = sorted(dataset["trade_date"].unique())[::rebalance_step]
    for trade_date in trade_dates:
        group = dataset.loc[dataset["trade_date"] == trade_date].copy()
        group = group.dropna(subset=[feature, target])
        if "is_suspended" in group.columns:
            group = group[~group["is_suspended"].fillna(False).astype(bool)]
        if "is_limit_up" in group.columns:
            group = group[~group["is_limit_up"].fillna(False).astype(bool)]
        if "is_limit_down" in group.columns:
            group = group[~group["is_limit_down"].fillna(False).astype(bool)]
        if len(group) < min_stocks_per_date:
            continue

        group["_factor_score"] = pd.to_numeric(group[feature], errors="coerce") * direction_multiplier
        group = group.dropna(subset=["_factor_score"]).sort_values("_factor_score", ascending=False)
        selected_count = top_n or max(1, int(len(group) * quantile))
        selected = group.head(min(selected_count, len(group)))
        if selected.empty:
            continue
        gross_return = float(selected[target].astype(float).mean())
        period_return = gross_return - cost_rate
        rows.append(
            {
                "trade_date": trade_date,
                "feature": feature,
                "selected_count": int(len(selected)),
                "gross_return": gross_return,
                "cost_rate": cost_rate,
                "period_return": period_return,
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {}, detail
    equity = (1 + detail["period_return"]).cumprod()
    drawdown = equity / equity.cummax() - 1
    summary = {
        "feature": feature,
        "periods": int(len(detail)),
        "avg_selected_count": float(detail["selected_count"].mean()),
        "mean_period_return": float(detail["period_return"].mean()),
        "cumulative_return": float(equity.iloc[-1] - 1),
        "annualized_return": annualize_return(float(equity.iloc[-1] - 1), len(detail), rebalance_step),
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((detail["period_return"] > 0).mean()),
    }
    return summary, detail


def annualize_return(cumulative_return: float, periods: int, rebalance_step: int) -> float | None:
    if periods <= 0:
        return None
    years = periods * rebalance_step / 252
    if years <= 0 or cumulative_return <= -1:
        return None
    return float((1 + cumulative_return) ** (1 / years) - 1)
