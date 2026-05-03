import pandas as pd

from src.models.dataset import build_model_training_dataset


def make_features_and_labels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    features = pd.DataFrame(
        {
            "symbol": ["000001"] * 10,
            "ts_code": ["000001.SZ"] * 10,
            "trade_date": [date.date() for date in dates],
            "adjust": ["qfq"] * 10,
            "close": range(10, 20),
            "return_1d": [0.01] * 10,
            "return_5d": [0.02] * 10,
            "return_20d": [0.03] * 10,
            "ma_5": [10.0] * 10,
            "ma_10": [10.0] * 10,
            "ma_20": [10.0] * 10,
            "ma_60": [10.0] * 10,
            "ma_5_ratio": [0.1] * 10,
            "ma_20_ratio": [0.1] * 10,
            "volatility_5": [0.1] * 10,
            "volatility_20": [0.1] * 10,
            "volume_ma_5": [1000.0] * 10,
            "volume_ma_20": [1000.0] * 10,
            "volume_ratio_5": [1.0] * 10,
            "rsi_14": [50.0] * 10,
            "macd": [0.1] * 10,
            "macd_signal": [0.1] * 10,
            "macd_hist": [0.0] * 10,
            "updated_at": pd.Timestamp("2024-01-01"),
        }
    )
    labels = pd.DataFrame(
        {
            "symbol": ["000001"] * 10,
            "ts_code": ["000001.SZ"] * 10,
            "trade_date": [date.date() for date in dates],
            "adjust": ["qfq"] * 10,
            "future_return_1d": [0.01] * 10,
            "future_return_5d": [0.05] * 9 + [None],
            "future_return_20d": [0.2] * 10,
            "future_max_drawdown_5d": [-0.01] * 10,
            "future_max_drawdown_20d": [-0.02] * 10,
            "label_up_5d": [True] * 9 + [None],
            "label_up_20d": [True] * 10,
            "updated_at": pd.Timestamp("2024-01-01"),
        }
    )
    limits = pd.DataFrame(
        {
            "symbol": ["000001"],
            "trade_date": [dates[0].date()],
            "is_limit_up": [True],
            "is_limit_down": [False],
        }
    )
    return features, labels, limits


def test_build_model_training_dataset_assigns_time_splits() -> None:
    features, labels, limits = make_features_and_labels()

    dataset = build_model_training_dataset(
        technical_features=features,
        training_labels=labels,
        limit_status=limits,
        target="future_return_5d",
        train_ratio=0.6,
        valid_ratio=0.2,
    )

    assert len(dataset) == 9
    assert dataset["dataset_split"].tolist() == [
        "train",
        "train",
        "train",
        "train",
        "train",
        "valid",
        "valid",
        "test",
        "test",
    ]
    assert dataset.loc[0, "is_limit_up"]
    assert not dataset.loc[1, "is_limit_up"]
    assert "is_suspended" in dataset.columns
