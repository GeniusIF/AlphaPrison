from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


FEATURE_COLUMNS = [
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
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "is_limit_up",
    "is_limit_down",
    "is_suspended",
]

LABEL_COLUMNS = [
    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "future_max_drawdown_5d",
    "future_max_drawdown_20d",
    "label_up_5d",
    "label_up_20d",
]

DATASET_COLUMNS = [
    "symbol",
    "ts_code",
    "trade_date",
    "adjust",
    "dataset_split",
    *FEATURE_COLUMNS,
    *LABEL_COLUMNS,
    "updated_at",
]


def build_model_training_dataset(
    technical_features: pd.DataFrame,
    training_labels: pd.DataFrame,
    limit_status: pd.DataFrame | None = None,
    suspensions: pd.DataFrame | None = None,
    target: str = "future_return_5d",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> pd.DataFrame:
    if technical_features.empty or training_labels.empty:
        return pd.DataFrame(columns=DATASET_COLUMNS)
    if target not in LABEL_COLUMNS:
        raise ValueError(f"Unsupported target: {target}")

    features = technical_features.copy()
    labels = training_labels.copy()
    features["trade_date"] = pd.to_datetime(features["trade_date"]).dt.date
    labels["trade_date"] = pd.to_datetime(labels["trade_date"]).dt.date

    dataset = features.merge(
        labels[["symbol", "trade_date", "adjust", *LABEL_COLUMNS]],
        on=["symbol", "trade_date", "adjust"],
        how="inner",
    )
    dataset = add_limit_status(dataset, limit_status)
    dataset = add_suspension_status(dataset, suspensions)
    dataset = dataset.dropna(subset=[target]).copy()
    dataset = assign_time_splits(dataset, train_ratio=train_ratio, valid_ratio=valid_ratio)
    dataset["updated_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    return dataset[DATASET_COLUMNS].sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def assign_time_splits(
    dataset: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> pd.DataFrame:
    if dataset.empty:
        return dataset.assign(dataset_split=pd.Series(dtype="object"))
    if train_ratio <= 0 or valid_ratio < 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio and valid_ratio must leave room for a test split")

    result = dataset.copy()
    dates = sorted(result["trade_date"].unique())
    train_end = max(1, int(len(dates) * train_ratio))
    valid_end = max(train_end + 1, int(len(dates) * (train_ratio + valid_ratio)))
    valid_end = min(valid_end, len(dates) - 1)
    train_dates = set(dates[:train_end])
    valid_dates = set(dates[train_end:valid_end])

    result["dataset_split"] = "test"
    result.loc[result["trade_date"].isin(train_dates), "dataset_split"] = "train"
    result.loc[result["trade_date"].isin(valid_dates), "dataset_split"] = "valid"
    return result


def add_limit_status(dataset: pd.DataFrame, limit_status: pd.DataFrame | None) -> pd.DataFrame:
    result = dataset.copy()
    if limit_status is None or limit_status.empty:
        result["is_limit_up"] = False
        result["is_limit_down"] = False
        return result

    status = limit_status[["symbol", "trade_date", "is_limit_up", "is_limit_down"]].copy()
    status["trade_date"] = pd.to_datetime(status["trade_date"]).dt.date
    result = result.merge(status, on=["symbol", "trade_date"], how="left")
    result["is_limit_up"] = result["is_limit_up"].map(lambda value: bool(value) if pd.notna(value) else False)
    result["is_limit_down"] = result["is_limit_down"].map(lambda value: bool(value) if pd.notna(value) else False)
    return result


def add_suspension_status(dataset: pd.DataFrame, suspensions: pd.DataFrame | None) -> pd.DataFrame:
    result = dataset.copy()
    if suspensions is None or suspensions.empty:
        result["is_suspended"] = False
        return result

    status = suspensions[["symbol", "trade_date", "is_suspended"]].copy()
    status["trade_date"] = pd.to_datetime(status["trade_date"]).dt.date
    status = status.groupby(["symbol", "trade_date"], as_index=False)["is_suspended"].max()
    result = result.merge(status, on=["symbol", "trade_date"], how="left")
    result["is_suspended"] = result["is_suspended"].map(lambda value: bool(value) if pd.notna(value) else False)
    return result
