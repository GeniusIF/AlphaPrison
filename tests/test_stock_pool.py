import pandas as pd

from src.collectors.stock_pool import normalize_stock_pool_frame


def test_normalize_stock_pool_frame_filters_common_a_shares() -> None:
    raw = pd.DataFrame(
        {
            "code": ["000001", "300001", "600519", "688001", "830000", "900001"],
            "name": ["平安银行", "创业样例", "贵州茅台", "科创样例", "北交样例", "B股样例"],
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


def test_normalize_stock_pool_frame_balances_shenzhen_and_shanghai() -> None:
    raw = pd.DataFrame(
        {
            "code": ["000001", "000002", "000003", "600000", "600001", "600002"],
            "name": ["深A1", "深A2", "深A3", "沪A1", "沪A2", "沪A3"],
        }
    )

    pool = normalize_stock_pool_frame(raw, limit=4)

    assert pool["symbol"].tolist() == ["000001", "000002", "600000", "600001"]
