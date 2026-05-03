from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.database.schema import SCHEMA_SQL
from src.features.labels import build_training_labels
from src.features.technical import build_technical_features
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

    def upsert_trade_calendar(self, trade_calendar: pd.DataFrame) -> int:
        if trade_calendar.empty:
            return 0
        self.init_schema()
        frame = trade_calendar.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        with self.connect() as conn:
            conn.register("trade_calendar_frame", frame)
            conn.execute(
                """
                DELETE FROM trade_calendar
                USING trade_calendar_frame
                WHERE trade_calendar.trade_date = trade_calendar_frame.trade_date
                """
            )
            conn.execute(
                """
                INSERT INTO trade_calendar (trade_date, is_open, source, updated_at)
                SELECT trade_date, is_open, source, updated_at
                FROM trade_calendar_frame
                """
            )
            conn.unregister("trade_calendar_frame")
        return len(frame)

    def upsert_stock_suspensions(self, suspensions: pd.DataFrame) -> int:
        if suspensions.empty:
            return 0
        self.init_schema()
        frame = suspensions.copy()
        frame["symbol"] = frame["symbol"].map(normalize_symbol)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        with self.connect() as conn:
            conn.register("stock_suspension_frame", frame)
            conn.execute(
                """
                DELETE FROM stock_suspension
                USING stock_suspension_frame
                WHERE stock_suspension.symbol = stock_suspension_frame.symbol
                  AND stock_suspension.trade_date = stock_suspension_frame.trade_date
                  AND stock_suspension.source = stock_suspension_frame.source
                """
            )
            conn.execute(
                """
                INSERT INTO stock_suspension (
                    symbol, ts_code, trade_date, is_suspended, reason, source, updated_at
                )
                SELECT symbol, ts_code, trade_date, is_suspended, reason, source, updated_at
                FROM stock_suspension_frame
                """
            )
            conn.unregister("stock_suspension_frame")
        return len(frame)

    def upsert_daily_limit_status(self, limit_status: pd.DataFrame) -> int:
        if limit_status.empty:
            return 0
        self.init_schema()
        frame = limit_status.copy()
        frame["symbol"] = frame["symbol"].map(normalize_symbol)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        with self.connect() as conn:
            conn.register("daily_limit_status_frame", frame)
            conn.execute(
                """
                DELETE FROM daily_limit_status
                USING daily_limit_status_frame
                WHERE daily_limit_status.symbol = daily_limit_status_frame.symbol
                  AND daily_limit_status.trade_date = daily_limit_status_frame.trade_date
                """
            )
            conn.execute(
                """
                INSERT INTO daily_limit_status (
                    symbol, ts_code, trade_date, close, pct_change, limit_up_threshold,
                    limit_down_threshold, is_limit_up, is_limit_down, limit_type, source, updated_at
                )
                SELECT
                    symbol, ts_code, trade_date, close, pct_change, limit_up_threshold,
                    limit_down_threshold, is_limit_up, is_limit_down, limit_type, source, updated_at
                FROM daily_limit_status_frame
                """
            )
            conn.unregister("daily_limit_status_frame")
        return len(frame)

    def upsert_technical_features(self, features: pd.DataFrame) -> int:
        if features.empty:
            return 0
        self.init_schema()
        frame = features.copy()
        frame["symbol"] = frame["symbol"].map(normalize_symbol)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        with self.connect() as conn:
            conn.register("technical_features_frame", frame)
            conn.execute(
                """
                DELETE FROM technical_features
                USING technical_features_frame
                WHERE technical_features.symbol = technical_features_frame.symbol
                  AND technical_features.trade_date = technical_features_frame.trade_date
                  AND technical_features.adjust = technical_features_frame.adjust
                """
            )
            conn.execute(
                """
                INSERT INTO technical_features
                SELECT * FROM technical_features_frame
                """
            )
            conn.unregister("technical_features_frame")
        return len(frame)

    def upsert_training_labels(self, labels: pd.DataFrame) -> int:
        if labels.empty:
            return 0
        self.init_schema()
        frame = labels.copy()
        frame["symbol"] = frame["symbol"].map(normalize_symbol)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        with self.connect() as conn:
            conn.register("training_labels_frame", frame)
            conn.execute(
                """
                DELETE FROM training_labels
                USING training_labels_frame
                WHERE training_labels.symbol = training_labels_frame.symbol
                  AND training_labels.trade_date = training_labels_frame.trade_date
                  AND training_labels.adjust = training_labels_frame.adjust
                """
            )
            conn.execute(
                """
                INSERT INTO training_labels
                SELECT * FROM training_labels_frame
                """
            )
            conn.unregister("training_labels_frame")
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

    def query_trade_calendar(self, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        if not self.db_path.exists():
            self.init_schema()
        filters = ["is_open = TRUE"]
        params: list[object] = []
        if start_date:
            filters.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            filters.append("trade_date <= ?")
            params.append(end_date)
        with self.connect(read_only=True) as conn:
            return conn.execute(
                f"""
                SELECT *
                FROM trade_calendar
                WHERE {' AND '.join(filters)}
                ORDER BY trade_date
                """,
                params,
            ).fetchdf()

    def query_all_daily_prices(self, adjust: str = "qfq") -> pd.DataFrame:
        if not self.db_path.exists():
            self.init_schema()
        with self.connect(read_only=True) as conn:
            return conn.execute(
                """
                SELECT *
                FROM daily_price
                WHERE adjust = ?
                ORDER BY symbol, trade_date
                """,
                [adjust],
            ).fetchdf()

    def query_technical_features(self, symbol: str | None = None, adjust: str = "qfq", limit: int | None = None) -> pd.DataFrame:
        if not self.db_path.exists():
            self.init_schema()
        filters = ["adjust = ?"]
        params: list[object] = [adjust]
        if symbol:
            filters.append("symbol = ?")
            params.append(normalize_symbol(symbol))
        limit_sql = f"LIMIT {int(limit)}" if limit else ""
        with self.connect(read_only=True) as conn:
            return conn.execute(
                f"""
                SELECT *
                FROM technical_features
                WHERE {' AND '.join(filters)}
                ORDER BY symbol, trade_date
                {limit_sql}
                """,
                params,
            ).fetchdf()

    def query_training_labels(self, symbol: str | None = None, adjust: str = "qfq", limit: int | None = None) -> pd.DataFrame:
        if not self.db_path.exists():
            self.init_schema()
        filters = ["adjust = ?"]
        params: list[object] = [adjust]
        if symbol:
            filters.append("symbol = ?")
            params.append(normalize_symbol(symbol))
        limit_sql = f"LIMIT {int(limit)}" if limit else ""
        with self.connect(read_only=True) as conn:
            return conn.execute(
                f"""
                SELECT *
                FROM training_labels
                WHERE {' AND '.join(filters)}
                ORDER BY symbol, trade_date
                {limit_sql}
                """,
                params,
            ).fetchdf()

    def derive_missing_suspensions(self, adjust: str = "qfq") -> pd.DataFrame:
        if not self.db_path.exists():
            self.init_schema()
        with self.connect(read_only=True) as conn:
            return conn.execute(
                """
                SELECT
                    sb.symbol,
                    sb.ts_code,
                    tc.trade_date,
                    TRUE AS is_suspended,
                    'No daily price on open trade date' AS reason,
                    'derived_missing_daily_price' AS source,
                    CURRENT_TIMESTAMP AS updated_at
                FROM stock_basic sb
                CROSS JOIN trade_calendar tc
                LEFT JOIN daily_price dp
                    ON dp.symbol = sb.symbol
                   AND dp.trade_date = tc.trade_date
                   AND dp.adjust = ?
                WHERE tc.is_open = TRUE
                  AND dp.symbol IS NULL
                ORDER BY sb.symbol, tc.trade_date
                """,
                [adjust],
            ).fetchdf()

    def build_daily_limit_status(self, adjust: str = "qfq") -> pd.DataFrame:
        prices = self.query_all_daily_prices(adjust=adjust)
        if prices.empty:
            return pd.DataFrame()
        now = pd.Timestamp.utcnow().tz_convert(None)
        frame = prices[["symbol", "ts_code", "trade_date", "close", "pct_change"]].copy()
        frame["symbol"] = frame["symbol"].map(normalize_symbol)
        frame["pct_change"] = pd.to_numeric(frame["pct_change"], errors="coerce")
        frame["limit_up_threshold"] = frame["symbol"].map(limit_up_threshold_for_symbol)
        frame["limit_down_threshold"] = frame["symbol"].map(limit_down_threshold_for_symbol)
        frame["is_limit_up"] = frame["pct_change"] >= frame["limit_up_threshold"]
        frame["is_limit_down"] = frame["pct_change"] <= frame["limit_down_threshold"]
        frame["limit_type"] = "derived_from_pct_change"
        frame["source"] = f"daily_price_{adjust or 'raw'}"
        frame["updated_at"] = now
        return frame

    def build_and_store_technical_features(self, adjust: str = "qfq") -> int:
        prices = self.query_all_daily_prices(adjust=adjust)
        features = build_technical_features(prices, adjust=adjust)
        return self.upsert_technical_features(features)

    def build_and_store_training_labels(self, adjust: str = "qfq") -> int:
        prices = self.query_all_daily_prices(adjust=adjust)
        labels = build_training_labels(prices, adjust=adjust)
        return self.upsert_training_labels(labels)

    def table_counts(self) -> pd.DataFrame:
        if not self.db_path.exists():
            self.init_schema()
        tables = [
            "stock_basic",
            "daily_price",
            "trade_calendar",
            "stock_suspension",
            "daily_limit_status",
            "technical_features",
            "training_labels",
        ]
        rows = []
        with self.connect(read_only=True) as conn:
            for table in tables:
                rows.append({"table": table, "rows": conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]})
        return pd.DataFrame(rows)


def limit_up_threshold_for_symbol(symbol: str) -> float:
    code = normalize_symbol(symbol)
    if code.startswith(("300", "688")):
        return 19.8
    if code.startswith(("4", "8")):
        return 29.8
    return 9.8


def limit_down_threshold_for_symbol(symbol: str) -> float:
    code = normalize_symbol(symbol)
    if code.startswith(("300", "688")):
        return -19.8
    if code.startswith(("4", "8")):
        return -29.8
    return -9.8
