import pandas as pd

from src.backtest.multifactor_backtest import backtest_multifactor_strategy, rolling_backtest_multifactor_strategy
from src.models.dataset import FEATURE_COLUMNS


def sample_multifactor_dataset() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for date_index, trade_date in enumerate(dates):
        split = "train" if date_index < 6 else "test"
        for stock_index in range(6):
            row = {
                "symbol": f"00000{stock_index}",
                "ts_code": f"00000{stock_index}.SZ",
                "trade_date": trade_date.date(),
                "adjust": "qfq",
                "dataset_split": split,
                "future_return_5d": (5 - stock_index) * 0.01,
            }
            for feature in FEATURE_COLUMNS:
                row[feature] = float(stock_index) if feature in {"return_5d", "volatility_5"} else 1.0
            row["is_limit_up"] = False
            row["is_limit_down"] = False
            row["is_suspended"] = False
            rows.append(row)
    return pd.DataFrame(rows)


def test_backtest_multifactor_strategy_uses_test_split(tmp_path) -> None:
    report = backtest_multifactor_strategy(
        dataset=sample_multifactor_dataset(),
        target="future_return_5d",
        report_dir=tmp_path,
        quantile=0.2,
        rebalance_step=1,
        min_stocks_per_date=5,
        cost_rate=0,
        max_factors=2,
        min_abs_rank_ic=0.01,
        evaluation_split="test",
    )

    assert report["direction_source"] == "train"
    assert report["evaluation_split"] == "test"
    assert report["selected_factors"][0]["factor_direction"] == "negative"
    assert report["summary"]["periods"] == 4
    assert report["summary"]["cumulative_return"] > report["summary"]["benchmark_cumulative_return"]


def test_rolling_backtest_multifactor_strategy_builds_folds(tmp_path) -> None:
    report = rolling_backtest_multifactor_strategy(
        dataset=sample_multifactor_dataset(),
        target="future_return_5d",
        report_dir=tmp_path,
        quantile=0.2,
        rebalance_step=1,
        min_stocks_per_date=5,
        cost_rate=0,
        max_factors=2,
        min_abs_rank_ic=0.01,
        train_window=5,
        test_window=1,
        step=2,
        embargo_days=0,
        min_train_rows=20,
    )

    assert report["fold_count"] == 3
    assert report["aggregate_summary"]["periods"] == 3
    assert report["folds"][0]["selected_features"]
