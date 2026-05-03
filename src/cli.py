from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from src.collectors.akshare_client import AkshareCollector
from src.database.repository import MarketDataRepository
from src.utils.config import load_yaml


DATA_SOURCE_CONFIG = "config/data_source.yaml"


@dataclass(frozen=True)
class AppContext:
    config: dict
    repository: MarketDataRepository
    collector: AkshareCollector


def build_context() -> AppContext:
    config = load_yaml(DATA_SOURCE_CONFIG)
    db_path = config["database"]["path"]
    interval = float(config.get("akshare", {}).get("request_interval_seconds", 0.25))
    return AppContext(
        config=config,
        repository=MarketDataRepository(db_path),
        collector=AkshareCollector(request_interval_seconds=interval),
    )


def select_pool(config: dict, limit: int | None) -> list[dict[str, str]]:
    pool = list(config.get("debug_stock_pool", []))
    if limit:
        return pool[:limit]
    return pool


def cmd_init_db(_: argparse.Namespace) -> None:
    context = build_context()
    context.repository.init_schema()
    print(f"Initialized DuckDB at {context.repository.db_path}")


def cmd_collect(args: argparse.Namespace) -> None:
    context = build_context()
    ak_config = context.config.get("akshare", {})
    pool = select_pool(context.config, args.limit)
    start_date = args.start_date or ak_config.get("default_start_date")
    end_date = args.end_date or ak_config.get("default_end_date")
    adjust = args.adjust or ak_config.get("default_adjust", "qfq")

    stock_basic = context.collector.fetch_stock_basic(pool)
    stock_count = context.repository.upsert_stock_basic(stock_basic)
    print(f"Upserted {stock_count} stock_basic rows")

    total_rows = 0
    failures: list[tuple[str, str]] = []
    for item in pool:
        symbol = item["symbol"]
        try:
            daily = context.collector.fetch_daily_price(symbol, start_date, end_date, adjust)
            row_count = context.repository.upsert_daily_prices(daily)
            total_rows += row_count
            print(f"{symbol}: upserted {row_count} daily rows")
        except Exception as exc:
            failures.append((symbol, str(exc)))
            print(f"{symbol}: failed - {exc}")

    print(f"Finished. Upserted {total_rows} daily_price rows.")
    if failures:
        print("Failures:")
        for symbol, error in failures:
            print(f"- {symbol}: {error}")


def cmd_query(args: argparse.Namespace) -> None:
    context = build_context()
    frame = context.repository.query_daily_prices(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        adjust=args.adjust,
    )
    if args.tail:
        frame = frame.tail(args.tail)
    print(format_frame(frame))


def cmd_list_stocks(_: argparse.Namespace) -> None:
    context = build_context()
    print(format_frame(context.repository.list_stocks()))


def cmd_latest(args: argparse.Namespace) -> None:
    context = build_context()
    print(format_frame(context.repository.latest_daily_prices(adjust=args.adjust)))


def format_frame(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    return frame.to_string(index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A 股数据采集、入库和查询工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db = subparsers.add_parser("init-db", help="初始化 DuckDB schema")
    init_db.set_defaults(func=cmd_init_db)

    collect = subparsers.add_parser("collect", help="采集调试股票池日线数据并入库")
    collect.add_argument("--start-date", help="开始日期，格式 YYYYMMDD")
    collect.add_argument("--end-date", help="结束日期，格式 YYYYMMDD")
    collect.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    collect.add_argument("--limit", type=int, default=10, help="调试股票数量")
    collect.set_defaults(func=cmd_collect)

    query = subparsers.add_parser("query", help="查询某只股票日线")
    query.add_argument("--symbol", required=True, help="股票代码，例如 000001")
    query.add_argument("--start-date", help="开始日期，格式 YYYY-MM-DD 或 YYYYMMDD")
    query.add_argument("--end-date", help="结束日期，格式 YYYY-MM-DD 或 YYYYMMDD")
    query.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    query.add_argument("--tail", type=int, help="只显示最后 N 行")
    query.set_defaults(func=cmd_query)

    latest = subparsers.add_parser("latest", help="查询每只股票最新一条日线")
    latest.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    latest.set_defaults(func=cmd_latest)

    list_stocks = subparsers.add_parser("list-stocks", help="列出本地股票基础信息")
    list_stocks.set_defaults(func=cmd_list_stocks)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
