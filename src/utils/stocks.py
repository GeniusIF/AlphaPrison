from __future__ import annotations


def normalize_symbol(symbol: str) -> str:
    clean = str(symbol).strip().upper()
    if "." in clean:
        clean = clean.split(".", 1)[0]
    return clean.zfill(6)


def infer_exchange(symbol: str) -> str:
    code = normalize_symbol(symbol)
    if code.startswith(("5", "6", "9")):
        return "SH"
    if code.startswith(("0", "1", "2", "3")):
        return "SZ"
    if code.startswith(("4", "8")):
        return "BJ"
    return "UNKNOWN"


def to_ts_code(symbol: str) -> str:
    code = normalize_symbol(symbol)
    exchange = infer_exchange(code)
    return f"{code}.{exchange}" if exchange != "UNKNOWN" else code
