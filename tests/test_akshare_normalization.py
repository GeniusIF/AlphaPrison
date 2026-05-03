import pandas as pd

from src.collectors.akshare_client import normalize_daily_price_frame


def test_normalize_daily_price_frame() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-02"],
            "股票代码": ["000001"],
            "开盘": [10.0],
            "收盘": [10.2],
            "最高": [10.3],
            "最低": [9.9],
            "成交量": [1000],
            "成交额": [10200],
            "振幅": [4.0],
            "涨跌幅": [2.0],
            "涨跌额": [0.2],
            "换手率": [1.1],
        }
    )

    frame = normalize_daily_price_frame(raw, "000001", "qfq")

    assert frame.loc[0, "symbol"] == "000001"
    assert frame.loc[0, "ts_code"] == "000001.SZ"
    assert frame.loc[0, "close"] == 10.2
    assert frame.loc[0, "adjust"] == "qfq"
