from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.factor_backtest import annualize_return
from src.models.cleaning import clean_training_frame
from src.models.factor_analysis import build_factor_summary
from src.utils.config import project_path


EXCLUDED_SCORING_FEATURES = {"close", "is_limit_up", "is_limit_down", "is_suspended"}


def backtest_multifactor_strategy(
    dataset: pd.DataFrame,
    target: str,
    report_dir: str | Path,
    quantile: float = 0.2,
    top_n: int | None = None,
    rebalance_step: int = 5,
    min_stocks_per_date: int = 5,
    cost_rate: float = 0.00212,
    max_factors: int = 5,
    min_abs_rank_ic: float = 0.02,
    evaluation_split: str = "test",
    weighting: str = "rank_ic",
) -> dict[str, Any]:
    if dataset.empty:
        raise ValueError("Training dataset is empty. Run build-training-dataset first.")
    if target not in dataset.columns:
        raise ValueError(f"Target column not found: {target}")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1].")
    if rebalance_step < 1:
        raise ValueError("rebalance_step must be >= 1.")
    if max_factors < 1:
        raise ValueError("max_factors must be >= 1.")
    if weighting not in {"equal", "rank_ic"}:
        raise ValueError("weighting must be 'equal' or 'rank_ic'.")

    frame = clean_training_frame(dataset, target=target).copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    direction_frame, direction_source = select_direction_frame(frame)
    factor_summary = build_factor_summary(direction_frame, target=target, min_stocks_per_date=min_stocks_per_date)
    selected_factors = select_scoring_factors(
        factor_summary,
        max_factors=max_factors,
        min_abs_rank_ic=min_abs_rank_ic,
    )
    evaluation_frame = select_evaluation_frame(frame, evaluation_split=evaluation_split)

    detail, signals = build_multifactor_portfolio(
        evaluation_frame,
        selected_factors=selected_factors,
        target=target,
        quantile=quantile,
        top_n=top_n,
        rebalance_step=rebalance_step,
        min_stocks_per_date=min_stocks_per_date,
        cost_rate=cost_rate,
        weighting=weighting,
    )
    summary = summarize_portfolio(detail, rebalance_step=rebalance_step)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"multifactor_backtest_{target}_{timestamp}_summary.csv"
    detail_path = output_dir / f"multifactor_backtest_{target}_{timestamp}_detail.csv"
    signals_path = output_dir / f"multifactor_backtest_{target}_{timestamp}_signals.csv"
    report_path = output_dir / f"multifactor_backtest_{target}_{timestamp}.json"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    signals.to_csv(signals_path, index=False)

    report = {
        "target": target,
        "rows": int(len(frame)),
        "direction_source": direction_source,
        "evaluation_split": evaluation_split,
        "quantile": quantile,
        "top_n": top_n,
        "rebalance_step": rebalance_step,
        "min_stocks_per_date": min_stocks_per_date,
        "cost_rate": cost_rate,
        "max_factors": max_factors,
        "min_abs_rank_ic": min_abs_rank_ic,
        "weighting": weighting,
        "selected_factors": selected_factors.to_dict(orient="records"),
        "summary": summary,
        "date_range": date_range(evaluation_frame),
        "summary_path": str(summary_path),
        "detail_path": str(detail_path),
        "signals_path": str(signals_path),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def rolling_backtest_multifactor_strategy(
    dataset: pd.DataFrame,
    target: str,
    report_dir: str | Path,
    quantile: float = 0.2,
    top_n: int | None = None,
    rebalance_step: int = 5,
    min_stocks_per_date: int = 5,
    cost_rate: float = 0.00212,
    max_factors: int = 5,
    min_abs_rank_ic: float = 0.02,
    weighting: str = "rank_ic",
    train_window: int = 252,
    test_window: int = 63,
    step: int = 63,
    embargo_days: int = 5,
    min_train_rows: int = 200,
) -> dict[str, Any]:
    if dataset.empty:
        raise ValueError("Training dataset is empty. Run build-training-dataset first.")
    if target not in dataset.columns:
        raise ValueError(f"Target column not found: {target}")
    if train_window < 2 or test_window < 1 or step < 1:
        raise ValueError("train_window, test_window and step must be positive enough.")
    if embargo_days < 0:
        raise ValueError("embargo_days must be >= 0.")

    frame = clean_training_frame(dataset, target=target).copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    dates = sorted(frame["trade_date"].unique())
    fold_reports: list[dict[str, Any]] = []
    detail_pieces: list[pd.DataFrame] = []
    signal_pieces: list[pd.DataFrame] = []

    fold_index = 0
    for train_start_index in range(0, len(dates), step):
        train_end_index = train_start_index + train_window
        test_start_index = train_end_index + embargo_days
        test_end_index = test_start_index + test_window
        if test_start_index >= len(dates):
            break

        train_dates = set(dates[train_start_index:train_end_index])
        test_dates = set(dates[test_start_index:min(test_end_index, len(dates))])
        train_frame = frame[frame["trade_date"].isin(train_dates)].copy()
        test_frame = frame[frame["trade_date"].isin(test_dates)].copy()
        if len(train_frame) < min_train_rows or test_frame.empty:
            continue

        factor_summary = build_factor_summary(train_frame, target=target, min_stocks_per_date=min_stocks_per_date)
        selected_factors = select_scoring_factors(
            factor_summary,
            max_factors=max_factors,
            min_abs_rank_ic=min_abs_rank_ic,
        )
        detail, signals = build_multifactor_portfolio(
            test_frame,
            selected_factors=selected_factors,
            target=target,
            quantile=quantile,
            top_n=top_n,
            rebalance_step=rebalance_step,
            min_stocks_per_date=min_stocks_per_date,
            cost_rate=cost_rate,
            weighting=weighting,
        )
        fold_summary = summarize_portfolio(detail, rebalance_step=rebalance_step)
        fold_record = {
            "fold": fold_index,
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "train_date_range": date_range(train_frame),
            "test_date_range": date_range(test_frame),
            "selected_features": selected_factors["feature"].tolist() if not selected_factors.empty else [],
            "summary": fold_summary,
        }
        fold_reports.append(fold_record)
        if not detail.empty:
            detail = detail.copy()
            detail["fold"] = fold_index
            detail_pieces.append(detail)
        if not signals.empty:
            signals = signals.copy()
            signals["fold"] = fold_index
            signal_pieces.append(signals)
        fold_index += 1

    combined_detail = pd.concat(detail_pieces, ignore_index=True) if detail_pieces else pd.DataFrame()
    combined_signals = pd.concat(signal_pieces, ignore_index=True) if signal_pieces else pd.DataFrame()
    if not combined_detail.empty:
        combined_detail = combined_detail.sort_values(["trade_date", "fold"]).reset_index(drop=True)
    aggregate_summary = summarize_portfolio(combined_detail, rebalance_step=rebalance_step)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"rolling_multifactor_backtest_{target}_{timestamp}_summary.csv"
    detail_path = output_dir / f"rolling_multifactor_backtest_{target}_{timestamp}_detail.csv"
    signals_path = output_dir / f"rolling_multifactor_backtest_{target}_{timestamp}_signals.csv"
    report_path = output_dir / f"rolling_multifactor_backtest_{target}_{timestamp}.json"
    pd.DataFrame([aggregate_summary]).to_csv(summary_path, index=False)
    combined_detail.to_csv(detail_path, index=False)
    combined_signals.to_csv(signals_path, index=False)

    report = {
        "target": target,
        "rows": int(len(frame)),
        "fold_count": len(fold_reports),
        "quantile": quantile,
        "top_n": top_n,
        "rebalance_step": rebalance_step,
        "min_stocks_per_date": min_stocks_per_date,
        "cost_rate": cost_rate,
        "max_factors": max_factors,
        "min_abs_rank_ic": min_abs_rank_ic,
        "weighting": weighting,
        "train_window": train_window,
        "test_window": test_window,
        "step": step,
        "embargo_days": embargo_days,
        "min_train_rows": min_train_rows,
        "aggregate_summary": aggregate_summary,
        "folds": fold_reports,
        "date_range": date_range(frame),
        "summary_path": str(summary_path),
        "detail_path": str(detail_path),
        "signals_path": str(signals_path),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def select_direction_frame(dataset: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "dataset_split" in dataset.columns and (dataset["dataset_split"] == "train").any():
        return dataset[dataset["dataset_split"] == "train"].copy(), "train"
    return dataset.copy(), "all"


def select_scoring_factors(
    factor_summary: pd.DataFrame,
    max_factors: int,
    min_abs_rank_ic: float,
) -> pd.DataFrame:
    if factor_summary.empty:
        return pd.DataFrame()
    result = factor_summary.copy()
    result = result[
        (result["direction_multiplier"] != 0)
        & (result["abs_daily_rank_ic_mean"] >= min_abs_rank_ic)
        & (~result["feature"].isin(EXCLUDED_SCORING_FEATURES))
    ].copy()
    if "is_reliable" in result.columns:
        result = result[result["is_reliable"]]
    if result.empty:
        return result
    return result.sort_values(
        ["abs_daily_rank_ic_mean", "daily_count"],
        ascending=[False, False],
    ).head(max_factors).reset_index(drop=True)


def select_evaluation_frame(dataset: pd.DataFrame, evaluation_split: str) -> pd.DataFrame:
    if evaluation_split == "all" or "dataset_split" not in dataset.columns:
        return dataset.copy()
    if evaluation_split == "valid_test":
        mask = dataset["dataset_split"].isin(["valid", "test"])
    else:
        mask = dataset["dataset_split"] == evaluation_split
    selected = dataset[mask].copy()
    return selected if not selected.empty else dataset.copy()


def build_multifactor_portfolio(
    dataset: pd.DataFrame,
    selected_factors: pd.DataFrame,
    target: str,
    quantile: float,
    top_n: int | None,
    rebalance_step: int,
    min_stocks_per_date: int,
    cost_rate: float,
    weighting: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset.empty or selected_factors.empty:
        return pd.DataFrame(), pd.DataFrame()

    features = selected_factors["feature"].tolist()
    weights = factor_weights(selected_factors, weighting=weighting)
    detail_rows: list[dict[str, Any]] = []
    signal_pieces: list[pd.DataFrame] = []

    for trade_date in sorted(dataset["trade_date"].unique())[::rebalance_step]:
        group = tradable_group(dataset[dataset["trade_date"] == trade_date].copy())
        group = group.dropna(subset=[target, *features])
        if len(group) < min_stocks_per_date:
            continue

        group["_factor_score"] = 0.0
        for row in selected_factors.itertuples(index=False):
            transformed = pd.to_numeric(group[row.feature], errors="coerce") * int(row.direction_multiplier)
            component = transformed.rank(pct=True, method="average")
            group[f"score_{row.feature}"] = component
            group["_factor_score"] += component * weights[row.feature]

        group = group.dropna(subset=["_factor_score"]).sort_values("_factor_score", ascending=False)
        selected_count = top_n or max(1, int(len(group) * quantile))
        selected = group.head(min(selected_count, len(group))).copy()
        if selected.empty:
            continue

        gross_return = float(selected[target].astype(float).mean())
        universe_return = float(group[target].astype(float).mean())
        period_return = gross_return - cost_rate
        benchmark_period_return = universe_return - cost_rate
        selected_symbols = ",".join(selected["symbol"].astype(str).tolist())
        detail_rows.append(
            {
                "trade_date": trade_date,
                "selected_count": int(len(selected)),
                "gross_return": gross_return,
                "period_return": period_return,
                "universe_return": universe_return,
                "benchmark_period_return": benchmark_period_return,
                "excess_return": period_return - benchmark_period_return,
                "avg_factor_score": float(selected["_factor_score"].mean()),
                "selected_symbols": selected_symbols,
            }
        )

        signal_columns = ["symbol", "ts_code", "trade_date", target, "_factor_score", *features]
        signal_frame = group[signal_columns].copy()
        signal_frame = signal_frame.rename(columns={"_factor_score": "factor_score", target: "forward_return"})
        signal_frame["rank"] = range(1, len(signal_frame) + 1)
        signal_frame["selected"] = signal_frame["symbol"].isin(selected["symbol"])
        signal_pieces.append(signal_frame)

    detail = pd.DataFrame(detail_rows)
    signals = pd.concat(signal_pieces, ignore_index=True) if signal_pieces else pd.DataFrame()
    return detail, signals


def factor_weights(selected_factors: pd.DataFrame, weighting: str) -> dict[str, float]:
    if selected_factors.empty:
        return {}
    if weighting == "equal":
        equal_weight = 1 / len(selected_factors)
        return {feature: equal_weight for feature in selected_factors["feature"]}
    raw_weights = selected_factors.set_index("feature")["abs_daily_rank_ic_mean"].astype(float)
    weight_sum = float(raw_weights.sum())
    if weight_sum <= 0:
        equal_weight = 1 / len(selected_factors)
        return {feature: equal_weight for feature in selected_factors["feature"]}
    return (raw_weights / weight_sum).to_dict()


def tradable_group(group: pd.DataFrame) -> pd.DataFrame:
    result = group.copy()
    if "is_suspended" in result.columns:
        result = result[~result["is_suspended"].fillna(False).astype(bool)]
    if "is_limit_up" in result.columns:
        result = result[~result["is_limit_up"].fillna(False).astype(bool)]
    if "is_limit_down" in result.columns:
        result = result[~result["is_limit_down"].fillna(False).astype(bool)]
    return result


def summarize_portfolio(detail: pd.DataFrame, rebalance_step: int) -> dict[str, Any]:
    if detail.empty:
        return {
            "periods": 0,
            "avg_selected_count": None,
            "mean_period_return": None,
            "cumulative_return": None,
            "annualized_return": None,
            "max_drawdown": None,
            "win_rate": None,
            "benchmark_cumulative_return": None,
            "excess_cumulative_return": None,
            "excess_win_rate": None,
        }

    equity = (1 + detail["period_return"]).cumprod()
    benchmark_equity = (1 + detail["benchmark_period_return"]).cumprod()
    drawdown = equity / equity.cummax() - 1
    cumulative_return = float(equity.iloc[-1] - 1)
    benchmark_cumulative_return = float(benchmark_equity.iloc[-1] - 1)
    return {
        "periods": int(len(detail)),
        "avg_selected_count": float(detail["selected_count"].mean()),
        "mean_period_return": float(detail["period_return"].mean()),
        "cumulative_return": cumulative_return,
        "annualized_return": annualize_return(cumulative_return, len(detail), rebalance_step),
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((detail["period_return"] > 0).mean()),
        "benchmark_cumulative_return": benchmark_cumulative_return,
        "excess_cumulative_return": cumulative_return - benchmark_cumulative_return,
        "excess_win_rate": float((detail["excess_return"] > 0).mean()),
    }


def date_range(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty:
        return {"start": None, "end": None}
    return {"start": str(frame["trade_date"].min()), "end": str(frame["trade_date"].max())}
