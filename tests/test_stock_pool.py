import pandas as pd

from src.collectors.stock_pool import normalize_stock_pool_frame


def test_normalize_stock_pool_frame_filters_common_a_shares() -> None:
    raw = pd.DataFrame(
        {
            "code": ["000001", "600519", "830000", "900001"],
            "name": ["平安银行", "贵州茅台", "北交样例", "B股样例"],
        }
    )

    pool = normalize_stock_pool_frame(raw, limit=10)

    assert pool["symbol"].tolist() == ["000001", "600519"]
    assert pool.loc[0, "ts_code"] == "000001.SZ"
    assert pool.loc[1, "ts_code"] == "600519.SH"


def test_normalize_stock_pool_frame_excludes_st() -> None:
    raw = pd.DataFrame(
        {
            "code": ["000004", "000006"],
            "name": ["*ST国华", "深振业Ａ"],
        }
    )

    pool = normalize_stock_pool_frame(raw, limit=10)

    assert pool["symbol"].tolist() == ["000006"]
