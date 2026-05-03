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

CREATE TABLE IF NOT EXISTS trade_calendar (
    trade_date DATE PRIMARY KEY,
    is_open BOOLEAN NOT NULL,
    source VARCHAR NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS stock_suspension (
    symbol VARCHAR NOT NULL,
    ts_code VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    is_suspended BOOLEAN NOT NULL,
    reason VARCHAR,
    source VARCHAR NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, trade_date, source)
);

CREATE INDEX IF NOT EXISTS idx_stock_suspension_symbol_date
ON stock_suspension(symbol, trade_date);

CREATE TABLE IF NOT EXISTS daily_limit_status (
    symbol VARCHAR NOT NULL,
    ts_code VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    close DOUBLE,
    pct_change DOUBLE,
    limit_up_threshold DOUBLE NOT NULL,
    limit_down_threshold DOUBLE NOT NULL,
    is_limit_up BOOLEAN NOT NULL,
    is_limit_down BOOLEAN NOT NULL,
    limit_type VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_limit_status_symbol_date
ON daily_limit_status(symbol, trade_date);

CREATE TABLE IF NOT EXISTS technical_features (
    symbol VARCHAR NOT NULL,
    ts_code VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    adjust VARCHAR NOT NULL,
    close DOUBLE,
    return_1d DOUBLE,
    return_5d DOUBLE,
    return_20d DOUBLE,
    ma_5 DOUBLE,
    ma_10 DOUBLE,
    ma_20 DOUBLE,
    ma_60 DOUBLE,
    ma_5_ratio DOUBLE,
    ma_20_ratio DOUBLE,
    volatility_5 DOUBLE,
    volatility_20 DOUBLE,
    volume_ma_5 DOUBLE,
    volume_ma_20 DOUBLE,
    volume_ratio_5 DOUBLE,
    rsi_14 DOUBLE,
    macd DOUBLE,
    macd_signal DOUBLE,
    macd_hist DOUBLE,
    updated_at TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, trade_date, adjust)
);

CREATE INDEX IF NOT EXISTS idx_technical_features_symbol_date
ON technical_features(symbol, trade_date);

CREATE TABLE IF NOT EXISTS training_labels (
    symbol VARCHAR NOT NULL,
    ts_code VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    adjust VARCHAR NOT NULL,
    future_return_1d DOUBLE,
    future_return_5d DOUBLE,
    future_return_20d DOUBLE,
    future_max_drawdown_5d DOUBLE,
    future_max_drawdown_20d DOUBLE,
    label_up_5d BOOLEAN,
    label_up_20d BOOLEAN,
    updated_at TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, trade_date, adjust)
);

CREATE INDEX IF NOT EXISTS idx_training_labels_symbol_date
ON training_labels(symbol, trade_date);
"""
