import pandas as pd

from src.features.labels import build_training_labels
from src.features.technical import build_technical_features


def sample_daily_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rows = []
    for index, trade_date in enumerate(dates, start=1):
        rows.append(
            {
                "symbol": "000001",
                "ts_code": "000001.SZ",
                "trade_date": trade_date.date(),
                "close": float(index),
                "volume": float(index * 100),
                "amount": float(index * 1000),
                "turnover_rate": float(index) / 10,
                "amplitude": float(index) / 100,
            }
        )
    return pd.DataFrame(rows)


def test_build_technical_features() -> None:
    features = build_technical_features(sample_daily_prices())

    assert len(features) == 30
    assert features.loc[4, "ma_5"] == 3.0
    assert "rsi_14" in features.columns
    assert "macd_hist" in features.columns
    assert features.loc[4, "amount_ma_5"] == 3000.0
    assert round(features.loc[4, "turnover_ma_5"], 6) == 0.3
    assert round(features.loc[4, "amplitude"], 6) == 0.05


def test_build_training_labels() -> None:
    labels = build_training_labels(sample_daily_prices())

    assert len(labels) == 30
    assert labels.loc[0, "future_return_1d"] == 1.0
    assert round(labels.loc[0, "future_return_5d"], 6) == 5.0
    assert pd.isna(labels.loc[29, "future_return_1d"])
    assert pd.isna(labels.loc[29, "label_up_5d"])


def test_build_training_labels_handles_zero_close() -> None:
    prices = sample_daily_prices()
    prices.loc[0, "close"] = 0.0

    labels = build_training_labels(prices)

    assert pd.isna(labels.loc[0, "future_return_1d"])
    assert pd.isna(labels.loc[0, "future_max_drawdown_5d"])
