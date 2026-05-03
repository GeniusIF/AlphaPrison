from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from src.utils.stocks import infer_exchange, normalize_symbol, to_ts_code


class AkshareCollector:
    def __init__(self, request_interval_seconds: float = 0.25) -> None:
        self.request_interval_seconds = request_interval_seconds

    @staticmethod
    def _akshare():
        import akshare as ak

        return ak

    def build_stock_basic_from_pool(self, pool: Iterable[dict[str, str]]) -> pd.DataFrame:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        rows = []
        for item in pool:
            symbol = normalize_symbol(item["symbol"])
            rows.append(
                {
                    "symbol": symbol,
                    "ts_code": to_ts_code(symbol),
                    "name": clean_stock_name(item.get("name", "")),
                    "exchange": infer_exchange(symbol),
                    "list_status": "L",
                    "source": "config",
                    "updated_at": now,
                }
            )
        return pd.DataFrame(rows)

    def fetch_stock_basic(self, pool: Iterable[dict[str, str]]) -> pd.DataFrame:
        fallback = self.build_stock_basic_from_pool(pool)
        pool_symbols = set(fallback["symbol"].tolist())
        try:
            ak = self._akshare()
            source_frame = ak.stock_info_a_code_name()
        except Exception:
            return fallback

        if source_frame.empty:
            return fallback

        code_col = "code" if "code" in source_frame.columns else "代码"
        name_col = "name" if "name" in source_frame.columns else "名称"
        source_frame[code_col] = source_frame[code_col].map(normalize_symbol)
        source_frame = source_frame[source_frame[code_col].isin(pool_symbols)]

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        rows = []
        configured_names = dict(zip(fallback["symbol"], fallback["name"]))
        for _, row in source_frame.iterrows():
            symbol = normalize_symbol(row[code_col])
            rows.append(
                {
                    "symbol": symbol,
                    "ts_code": to_ts_code(symbol),
                    "name": clean_stock_name(row.get(name_col) or configured_names.get(symbol, "")),
                    "exchange": infer_exchange(symbol),
                    "list_status": "L",
                    "source": "akshare",
                    "updated_at": now,
                }
            )
        if not rows:
            return fallback
        return pd.DataFrame(rows)

    def fetch_daily_price(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        ak = self._akshare()
        normalized = normalize_symbol(symbol)
        frame = ak.stock_zh_a_hist(
            symbol=normalized,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        time.sleep(self.request_interval_seconds)
        return normalize_daily_price_frame(frame, normalized, adjust)


def normalize_daily_price_frame(frame: pd.DataFrame, symbol: str, adjust: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "amplitude",
                "pct_change",
                "price_change",
                "turnover_rate",
                "adjust",
                "source",
                "ingested_at",
            ]
        )

    rename_map = {
        "日期": "trade_date",
        "股票代码": "symbol",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "price_change",
        "换手率": "turnover_rate",
    }
    normalized_symbol = normalize_symbol(symbol)
    data = frame.rename(columns=rename_map).copy()
    if "symbol" not in data.columns:
        data["symbol"] = normalized_symbol
    data["symbol"] = data["symbol"].fillna(normalized_symbol).map(normalize_symbol)
    data["ts_code"] = data["symbol"].map(to_ts_code)
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.date
    data["adjust"] = adjust
    data["source"] = "akshare"
    data["ingested_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

    columns = [
        "symbol",
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "price_change",
        "turnover_rate",
        "adjust",
        "source",
        "ingested_at",
    ]
    for column in columns:
        if column not in data.columns:
            data[column] = None
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "price_change",
        "turnover_rate",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data[columns].sort_values("trade_date").reset_index(drop=True)


def clean_stock_name(name: object) -> str:
    if name is None:
        return ""
    return "".join(str(name).split())
