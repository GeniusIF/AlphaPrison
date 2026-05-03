from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def build_training_labels(daily_prices: pd.DataFrame, adjust: str = "qfq") -> pd.DataFrame:
    columns = [
        "symbol",
        "ts_code",
        "trade_date",
        "adjust",
        "future_return_1d",
        "future_return_5d",
        "future_return_20d",
        "future_max_drawdown_5d",
        "future_max_drawdown_20d",
        "label_up_5d",
        "label_up_20d",
        "updated_at",
    ]
    if daily_prices.empty:
        return pd.DataFrame(columns=columns)

    data = daily_prices.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    pieces = []
    for _, group in data.groupby("symbol", sort=False):
        label = group[["symbol", "ts_code", "trade_date", "close"]].copy()
        close = label["close"].astype(float)
        label["future_return_1d"] = close.shift(-1) / close - 1
        label["future_return_5d"] = close.shift(-5) / close - 1
        label["future_return_20d"] = close.shift(-20) / close - 1
        label["future_max_drawdown_5d"] = _future_max_drawdown(close, 5).to_numpy()
        label["future_max_drawdown_20d"] = _future_max_drawdown(close, 20).to_numpy()
        label["label_up_5d"] = (label["future_return_5d"] > 0).where(label["future_return_5d"].notna(), pd.NA)
        label["label_up_20d"] = (label["future_return_20d"] > 0).where(label["future_return_20d"].notna(), pd.NA)
        pieces.append(label)

    result = pd.concat(pieces, ignore_index=True)
    result["adjust"] = adjust
    result["updated_at"] = now
    result["trade_date"] = result["trade_date"].dt.date
    return result[columns]


def _future_max_drawdown(close: pd.Series, window: int) -> pd.Series:
    values = []
    close = close.reset_index(drop=True)
    for index, current in close.items():
        future = close.iloc[index + 1 : index + window + 1]
        if future.empty or pd.isna(current):
            values.append(pd.NA)
            continue
        values.append((future.min() / current) - 1)
    return pd.Series(values, index=close.index, dtype="Float64")
