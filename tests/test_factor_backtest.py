import pandas as pd

from src.backtest.factor_backtest import backtest_single_factors
from src.models.dataset import FEATURE_COLUMNS


def sample_negative_factor_dataset() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    rows = []
    for date_index, trade_date in enumerate(dates):
        split = "train" if date_index < 5 else "test"
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
                row[feature] = float(stock_index) if feature == "return_5d" else 1.0
            row["is_limit_up"] = False
            row["is_limit_down"] = False
            row["is_suspended"] = False
            rows.append(row)
    return pd.DataFrame(rows)


def test_backtest_single_factors_uses_train_direction(tmp_path) -> None:
    report = backtest_single_factors(
        dataset=sample_negative_factor_dataset(),
        target="future_return_5d",
        report_dir=tmp_path,
        quantile=0.2,
        rebalance_step=1,
        min_stocks_per_date=5,
        cost_rate=0,
    )

    top_factor = report["top_factors"][0]
    assert top_factor["feature"] == "return_5d"
    assert top_factor["factor_direction"] == "negative"
    assert top_factor["cumulative_return"] > 0
