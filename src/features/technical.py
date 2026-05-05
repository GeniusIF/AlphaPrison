from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


TECHNICAL_FEATURE_TABLE_COLUMNS = [
    "symbol",
    "ts_code",
    "trade_date",
    "adjust",
    "close",
    "return_1d",
    "return_5d",
    "return_20d",
    "ma_5",
    "ma_10",
    "ma_20",
    "ma_60",
    "ma_5_ratio",
    "ma_20_ratio",
    "volatility_5",
    "volatility_20",
    "volume_ma_5",
    "volume_ma_20",
    "volume_ratio_5",
    "amount_ma_5",
    "amount_ma_20",
    "amount_ratio_5",
    "turnover_rate",
    "turnover_ma_5",
    "turnover_ma_20",
    "turnover_ratio_5",
    "amplitude",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "updated_at",
]


def build_technical_features(daily_prices: pd.DataFrame, adjust: str = "qfq") -> pd.DataFrame:
    """Build basic technical features from one or many stocks' daily prices."""
    if daily_prices.empty:
        return pd.DataFrame(columns=TECHNICAL_FEATURE_TABLE_COLUMNS)

    data = daily_prices.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    pieces = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for _, group in data.groupby("symbol", sort=False):
        feature = group.copy()
        close = feature["close"].astype(float)
        volume = feature["volume"].astype(float)
        amount = numeric_column(feature, "amount")
        turnover = numeric_column(feature, "turnover_rate")
        amplitude = numeric_column(feature, "amplitude")
        returns = close.pct_change()

        feature["return_1d"] = returns
        feature["return_5d"] = close.pct_change(5)
        feature["return_20d"] = close.pct_change(20)
        feature["ma_5"] = close.rolling(5, min_periods=5).mean()
        feature["ma_10"] = close.rolling(10, min_periods=10).mean()
        feature["ma_20"] = close.rolling(20, min_periods=20).mean()
        feature["ma_60"] = close.rolling(60, min_periods=60).mean()
        feature["ma_5_ratio"] = close / feature["ma_5"] - 1
        feature["ma_20_ratio"] = close / feature["ma_20"] - 1
        feature["volatility_5"] = returns.rolling(5, min_periods=5).std()
        feature["volatility_20"] = returns.rolling(20, min_periods=20).std()
        feature["volume_ma_5"] = volume.rolling(5, min_periods=5).mean()
        feature["volume_ma_20"] = volume.rolling(20, min_periods=20).mean()
        feature["volume_ratio_5"] = volume / feature["volume_ma_5"]
        feature["amount_ma_5"] = amount.rolling(5, min_periods=5).mean()
        feature["amount_ma_20"] = amount.rolling(20, min_periods=20).mean()
        feature["amount_ratio_5"] = amount / feature["amount_ma_5"]
        feature["turnover_rate"] = turnover
        feature["turnover_ma_5"] = turnover.rolling(5, min_periods=5).mean()
        feature["turnover_ma_20"] = turnover.rolling(20, min_periods=20).mean()
        feature["turnover_ratio_5"] = turnover / feature["turnover_ma_5"]
        feature["amplitude"] = amplitude

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = gain / loss
        feature["rsi_14"] = 100 - (100 / (1 + rs))

        ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        feature["macd"] = ema_12 - ema_26
        feature["macd_signal"] = feature["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
        feature["macd_hist"] = feature["macd"] - feature["macd_signal"]
        feature["adjust"] = adjust
        feature["updated_at"] = now
        pieces.append(feature)

    result = pd.concat(pieces, ignore_index=True)
    result["trade_date"] = result["trade_date"].dt.date
    result[TECHNICAL_FEATURE_TABLE_COLUMNS] = result[TECHNICAL_FEATURE_TABLE_COLUMNS].replace([np.inf, -np.inf], pd.NA)
    return result[TECHNICAL_FEATURE_TABLE_COLUMNS]


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")
