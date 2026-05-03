from src.utils.stocks import infer_exchange, normalize_symbol, to_ts_code


def test_normalize_symbol() -> None:
    assert normalize_symbol("1") == "000001"
    assert normalize_symbol("000001.SZ") == "000001"


def test_infer_exchange() -> None:
    assert infer_exchange("600519") == "SH"
    assert infer_exchange("000001") == "SZ"
    assert infer_exchange("300750") == "SZ"


def test_to_ts_code() -> None:
    assert to_ts_code("600519") == "600519.SH"
    assert to_ts_code("000001") == "000001.SZ"
