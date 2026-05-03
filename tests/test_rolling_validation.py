import pandas as pd

from src.models.dataset import FEATURE_COLUMNS
from src.models.rolling_validation import build_rolling_folds, rolling_validate_lgbm


def sample_rolling_dataset() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    for date_index, trade_date in enumerate(dates):
        for stock_index in range(5):
            row = {
                "symbol": f"00000{stock_index}",
                "ts_code": f"00000{stock_index}.SZ",
                "trade_date": trade_date.date(),
                "adjust": "qfq",
                "future_return_5d": stock_index * 0.01 + date_index * 0.001,
            }
            for feature in FEATURE_COLUMNS:
                row[feature] = float(stock_index + date_index)
            row["is_limit_up"] = False
            row["is_limit_down"] = False
            row["is_suspended"] = False
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_rolling_folds() -> None:
    dates = list(range(10))
    folds = build_rolling_folds(dates, train_window=4, test_window=2, step=2)

    assert len(folds) == 3
    assert folds[0]["train_dates"] == [0, 1, 2, 3]
    assert folds[0]["test_dates"] == [4, 5]


def test_rolling_validate_lgbm(tmp_path) -> None:
    report = rolling_validate_lgbm(
        dataset=sample_rolling_dataset(),
        target="future_return_5d",
        model_config={"n_estimators": 5, "learning_rate": 0.1, "num_leaves": 3, "random_state": 42},
        report_dir=tmp_path,
        train_window=8,
        test_window=4,
        step=4,
        min_train_rows=10,
    )

    assert len(report["folds"]) >= 2
    assert "directional_accuracy" in report["aggregate_scores"]
