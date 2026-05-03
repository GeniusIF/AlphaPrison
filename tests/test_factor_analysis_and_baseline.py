import pandas as pd

from src.models.baseline import train_baseline_models
from src.models.factor_analysis import build_factor_summary, build_quantile_returns
from src.models.dataset import FEATURE_COLUMNS


def sample_model_dataset() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    rows = []
    for date_index, trade_date in enumerate(dates):
        split = "train" if date_index < 3 else "valid" if date_index < 5 else "test"
        for stock_index in range(6):
            forward_return = stock_index * 0.01 + date_index * 0.001
            row = {
                "symbol": f"00000{stock_index}",
                "ts_code": f"00000{stock_index}.SZ",
                "trade_date": trade_date.date(),
                "adjust": "qfq",
                "dataset_split": split,
                "future_return_5d": forward_return,
            }
            for feature in FEATURE_COLUMNS:
                row[feature] = float(stock_index) if feature == "return_5d" else 1.0
            row["is_limit_up"] = False
            row["is_limit_down"] = False
            row["is_suspended"] = False
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_factor_summary() -> None:
    summary = build_factor_summary(sample_model_dataset(), target="future_return_5d", min_stocks_per_date=5)

    top_feature = summary.iloc[0]["feature"]
    assert top_feature == "return_5d"
    assert summary.iloc[0]["daily_count"] == 6


def test_build_quantile_returns() -> None:
    quantiles = build_quantile_returns(sample_model_dataset(), target="future_return_5d", quantiles=3)

    return_5d = quantiles[quantiles["feature"] == "return_5d"]
    assert len(return_5d) == 3
    assert return_5d["top_minus_bottom"].iloc[0] > 0


def test_train_baseline_models(tmp_path) -> None:
    metrics = train_baseline_models(
        dataset=sample_model_dataset(),
        target="future_return_5d",
        report_dir=tmp_path,
    )

    assert "ridge" in metrics["models"]
    assert "test" in metrics["models"]["ridge"]
    assert metrics["rows"]["train"] == 18
