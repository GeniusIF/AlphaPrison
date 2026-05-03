from __future__ import annotations


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stock_basic (
    symbol VARCHAR PRIMARY KEY,
    ts_code VARCHAR NOT NULL,
    name VARCHAR,
    exchange VARCHAR,
    list_status VARCHAR DEFAULT 'L',
    source VARCHAR NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_price (
    symbol VARCHAR NOT NULL,
    ts_code VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    amount DOUBLE,
    amplitude DOUBLE,
    pct_change DOUBLE,
    price_change DOUBLE,
    turnover_rate DOUBLE,
    adjust VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, trade_date, adjust)
);

CREATE INDEX IF NOT EXISTS idx_daily_price_symbol_date
ON daily_price(symbol, trade_date);
"""
