from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.utils.stocks import infer_exchange, normalize_symbol, to_ts_code


def normalize_stock_pool_frame(frame: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    columns = ["symbol", "ts_code", "name", "exchange", "list_status", "source", "updated_at"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    code_col = "code" if "code" in frame.columns else "代码"
    name_col = "name" if "name" in frame.columns else "名称"
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    data = frame[[code_col, name_col]].copy()
    data = data.rename(columns={code_col: "symbol", name_col: "name"})
    data["symbol"] = data["symbol"].map(normalize_symbol)
    data["name"] = data["name"].map(clean_stock_name)
    data = data[data.apply(lambda row: is_common_a_share(row["symbol"], row["name"]), axis=1)]
    data = data.drop_duplicates(subset=["symbol"]).sort_values("symbol")
    if limit:
        data = data.head(limit)
    data["ts_code"] = data["symbol"].map(to_ts_code)
    data["exchange"] = data["symbol"].map(infer_exchange)
    data["list_status"] = "L"
    data["source"] = "akshare_pool"
    data["updated_at"] = now
    return data[columns].reset_index(drop=True)


def dataframe_to_pool(frame: pd.DataFrame) -> list[dict[str, str]]:
    if frame.empty:
        return []
    return [
        {"symbol": str(row.symbol), "name": str(row.name or "")}
        for row in frame[["symbol", "name"]].itertuples(index=False)
    ]


def is_common_a_share(symbol: str, name: str = "") -> bool:
    code = normalize_symbol(symbol)
    clean_name = clean_stock_name(name).upper()
    if "ST" in clean_name:
        return False
    return code.startswith(("000", "001", "002", "003", "300", "600", "601", "603", "605", "688"))


def clean_stock_name(name: object) -> str:
    if name is None:
        return ""
    return "".join(str(name).split())
