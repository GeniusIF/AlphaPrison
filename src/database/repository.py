from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.database.schema import SCHEMA_SQL
from src.utils.config import project_path
from src.utils.stocks import normalize_symbol


class MarketDataRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = project_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path), read_only=read_only)

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(SCHEMA_SQL)

    def upsert_stock_basic(self, stock_basic: pd.DataFrame) -> int:
        if stock_basic.empty:
            return 0
        self.init_schema()
        frame = stock_basic.copy()
        frame["symbol"] = frame["symbol"].map(normalize_symbol)
        symbols = frame["symbol"].tolist()
        with self.connect() as conn:
            conn.register("stock_basic_frame", frame)
            conn.execute("DELETE FROM stock_basic WHERE symbol IN (SELECT symbol FROM stock_basic_frame)")
            conn.execute(
                """
                INSERT INTO stock_basic (
                    symbol, ts_code, name, exchange, list_status, source, updated_at
                )
                SELECT symbol, ts_code, name, exchange, list_status, source, updated_at
                FROM stock_basic_frame
                """
            )
            conn.unregister("stock_basic_frame")
        return len(symbols)

    def upsert_daily_prices(self, daily_prices: pd.DataFrame) -> int:
        if daily_prices.empty:
            return 0
        self.init_schema()
        frame = daily_prices.copy()
        frame["symbol"] = frame["symbol"].map(normalize_symbol)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        with self.connect() as conn:
            conn.register("daily_price_frame", frame)
            conn.execute(
                """
                DELETE FROM daily_price
                USING daily_price_frame
                WHERE daily_price.symbol = daily_price_frame.symbol
                  AND daily_price.trade_date = daily_price_frame.trade_date
                  AND daily_price.adjust = daily_price_frame.adjust
                """
            )
            conn.execute(
                """
                INSERT INTO daily_price (
                    symbol, ts_code, trade_date, open, high, low, close, volume, amount,
                    amplitude, pct_change, price_change, turnover_rate, adjust, source, ingested_at
                )
                SELECT
                    symbol, ts_code, trade_date, open, high, low, close, volume, amount,
                    amplitude, pct_change, price_change, turnover_rate, adjust, source, ingested_at
                FROM daily_price_frame
                """
            )
            conn.unregister("daily_price_frame")
        return len(frame)

    def list_stocks(self) -> pd.DataFrame:
        if not self.db_path.exists():
            self.init_schema()
        with self.connect(read_only=True) as conn:
            return conn.execute(
                """
                SELECT symbol, ts_code, name, exchange, source, updated_at
                FROM stock_basic
                ORDER BY symbol
                """
            ).fetchdf()

    def query_daily_prices(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        adjust: str = "qfq",
        limit: int | None = None,
    ) -> pd.DataFrame:
        if not self.db_path.exists():
            self.init_schema()
        filters = ["symbol = ?", "adjust = ?"]
        params: list[object] = [normalize_symbol(symbol), adjust]
        if start_date:
            filters.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            filters.append("trade_date <= ?")
            params.append(end_date)
        limit_sql = f"LIMIT {int(limit)}" if limit else ""
        sql = f"""
            SELECT *
            FROM daily_price
            WHERE {' AND '.join(filters)}
            ORDER BY trade_date
            {limit_sql}
        """
        with self.connect(read_only=True) as conn:
            return conn.execute(sql, params).fetchdf()

    def latest_daily_prices(self, adjust: str = "qfq") -> pd.DataFrame:
        if not self.db_path.exists():
            self.init_schema()
        with self.connect(read_only=True) as conn:
            return conn.execute(
                """
                WITH ranked AS (
                    SELECT
                        dp.*,
                        sb.name,
                        ROW_NUMBER() OVER (
                            PARTITION BY dp.symbol, dp.adjust
                            ORDER BY dp.trade_date DESC
                        ) AS rn
                    FROM daily_price dp
                    LEFT JOIN stock_basic sb USING (symbol)
                    WHERE dp.adjust = ?
                )
                SELECT
                    symbol, ts_code, trade_date, open, high, low, close, volume, amount,
                    amplitude, pct_change, price_change, turnover_rate, adjust, source,
                    ingested_at, name
                FROM ranked
                WHERE rn = 1
                ORDER BY symbol
                """,
                [adjust],
            ).fetchdf()
