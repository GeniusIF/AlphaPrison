from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import pandas as pd

from src.collectors.akshare_client import AkshareCollector
from src.database.repository import MarketDataRepository
from src.models.baseline import train_baseline_models
from src.models.dataset import LABEL_COLUMNS
from src.models.factor_analysis import analyze_factors
from src.models.train_lgbm import train_lgbm_regressor
from src.utils.config import load_yaml


DATA_SOURCE_CONFIG = "config/data_source.yaml"
MODEL_CONFIG = "config/model.yaml"


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


def cmd_collect_calendar(args: argparse.Namespace) -> None:
    context = build_context()
    ak_config = context.config.get("akshare", {})
    start_date = args.start_date or ak_config.get("default_start_date")
    end_date = args.end_date or ak_config.get("default_end_date")
    calendar = context.collector.fetch_trade_calendar(start_date=start_date, end_date=end_date)
    row_count = context.repository.upsert_trade_calendar(calendar)
    print(f"Upserted {row_count} trade_calendar rows")


def cmd_collect_suspensions(args: argparse.Namespace) -> None:
    context = build_context()
    pool = select_pool(context.config, args.limit)
    pool_symbols = [item["symbol"] for item in pool]
    calendar = context.repository.query_trade_calendar(start_date=args.start_date, end_date=args.end_date)
    if calendar.empty:
        print("No trade calendar rows found. Run collect-calendar first.")
        return

    total_rows = 0
    failures: list[tuple[str, str]] = []
    for row in calendar.itertuples():
        trade_date = pd.Timestamp(row.trade_date).strftime("%Y%m%d")
        try:
            suspensions = context.collector.fetch_suspensions_by_date(trade_date, pool_symbols=pool_symbols)
            row_count = context.repository.upsert_stock_suspensions(suspensions)
            total_rows += row_count
            if row_count:
                print(f"{trade_date}: upserted {row_count} suspension rows")
        except Exception as exc:
            failures.append((trade_date, str(exc)))
            print(f"{trade_date}: failed - {exc}")
    print(f"Finished. Upserted {total_rows} stock_suspension rows.")
    if failures:
        print("Failures:")
        for trade_date, error in failures:
            print(f"- {trade_date}: {error}")


def cmd_derive_suspensions(args: argparse.Namespace) -> None:
    context = build_context()
    suspensions = context.repository.derive_missing_suspensions(adjust=args.adjust)
    row_count = context.repository.upsert_stock_suspensions(suspensions)
    print(f"Upserted {row_count} derived stock_suspension rows")


def cmd_build_limit_status(args: argparse.Namespace) -> None:
    context = build_context()
    limit_status = context.repository.build_daily_limit_status(adjust=args.adjust)
    row_count = context.repository.upsert_daily_limit_status(limit_status)
    print(f"Upserted {row_count} daily_limit_status rows")


def cmd_build_features(args: argparse.Namespace) -> None:
    context = build_context()
    row_count = context.repository.build_and_store_technical_features(adjust=args.adjust)
    print(f"Upserted {row_count} technical_features rows")


def cmd_build_labels(args: argparse.Namespace) -> None:
    context = build_context()
    row_count = context.repository.build_and_store_training_labels(adjust=args.adjust)
    print(f"Upserted {row_count} training_labels rows")


def cmd_build_dataset(args: argparse.Namespace) -> None:
    context = build_context()
    limit_rows = context.repository.upsert_daily_limit_status(
        context.repository.build_daily_limit_status(adjust=args.adjust)
    )
    suspension_rows = context.repository.upsert_stock_suspensions(
        context.repository.derive_missing_suspensions(adjust=args.adjust)
    )
    feature_rows = context.repository.build_and_store_technical_features(adjust=args.adjust)
    label_rows = context.repository.build_and_store_training_labels(adjust=args.adjust)
    print(f"Upserted {limit_rows} daily_limit_status rows")
    print(f"Upserted {suspension_rows} derived stock_suspension rows")
    print(f"Upserted {feature_rows} technical_features rows")
    print(f"Upserted {label_rows} training_labels rows")


def cmd_build_training_dataset(args: argparse.Namespace) -> None:
    context = build_context()
    model_config = load_yaml(MODEL_CONFIG)
    dataset_config = model_config.get("dataset", {})
    target = args.target or dataset_config.get("target", "future_return_5d")
    train_ratio = args.train_ratio or float(dataset_config.get("train_ratio", 0.7))
    valid_ratio = args.valid_ratio or float(dataset_config.get("valid_ratio", 0.15))
    validate_target(target)
    row_count = context.repository.build_and_store_model_training_dataset(
        adjust=args.adjust,
        target=target,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )
    print(f"Upserted {row_count} model_training_dataset rows")


def cmd_train_lgbm(args: argparse.Namespace) -> None:
    context = build_context()
    model_config = load_yaml(MODEL_CONFIG)
    dataset_config = model_config.get("dataset", {})
    artifact_config = model_config.get("artifacts", {})
    target = args.target or dataset_config.get("target", "future_return_5d")
    validate_target(target)

    dataset = context.repository.query_model_training_dataset(
        adjust=args.adjust,
        target=target,
    )
    if dataset.empty:
        print("No training dataset found. Run build-training-dataset first.")
        return

    metrics = train_lgbm_regressor(
        dataset=dataset,
        target=target,
        model_config=model_config.get("lightgbm", {}),
        model_dir=artifact_config.get("model_dir", "artifacts/models"),
        report_dir=artifact_config.get("report_dir", "artifacts/reports"),
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def cmd_analyze_factors(args: argparse.Namespace) -> None:
    context = build_context()
    model_config = load_yaml(MODEL_CONFIG)
    dataset_config = model_config.get("dataset", {})
    factor_config = model_config.get("factor_analysis", {})
    artifact_config = model_config.get("artifacts", {})
    target = args.target or dataset_config.get("target", "future_return_5d")
    validate_target(target)

    dataset = context.repository.query_model_training_dataset(
        adjust=args.adjust,
        target=target,
    )
    if dataset.empty:
        print("No training dataset found. Run build-training-dataset first.")
        return

    report = analyze_factors(
        dataset=dataset,
        target=target,
        report_dir=artifact_config.get("report_dir", "artifacts/reports"),
        quantiles=args.quantiles or int(factor_config.get("quantiles", 5)),
        min_stocks_per_date=args.min_stocks_per_date or int(factor_config.get("min_stocks_per_date", 5)),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_train_baseline(args: argparse.Namespace) -> None:
    context = build_context()
    model_config = load_yaml(MODEL_CONFIG)
    dataset_config = model_config.get("dataset", {})
    baseline_config = model_config.get("baseline", {})
    artifact_config = model_config.get("artifacts", {})
    target = args.target or dataset_config.get("target", "future_return_5d")
    validate_target(target)

    dataset = context.repository.query_model_training_dataset(
        adjust=args.adjust,
        target=target,
    )
    if dataset.empty:
        print("No training dataset found. Run build-training-dataset first.")
        return

    metrics = train_baseline_models(
        dataset=dataset,
        target=target,
        report_dir=artifact_config.get("report_dir", "artifacts/reports"),
        ridge_alpha=args.ridge_alpha or float(baseline_config.get("ridge_alpha", 1.0)),
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def cmd_counts(_: argparse.Namespace) -> None:
    context = build_context()
    print(format_frame(context.repository.table_counts()))


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


def cmd_query_features(args: argparse.Namespace) -> None:
    context = build_context()
    frame = context.repository.query_technical_features(symbol=args.symbol, adjust=args.adjust)
    if args.tail:
        frame = frame.tail(args.tail)
    print(format_frame(frame))


def cmd_query_labels(args: argparse.Namespace) -> None:
    context = build_context()
    frame = context.repository.query_training_labels(symbol=args.symbol, adjust=args.adjust)
    if args.tail:
        frame = frame.tail(args.tail)
    print(format_frame(frame))


def cmd_query_training_dataset(args: argparse.Namespace) -> None:
    context = build_context()
    target = args.target
    if target:
        validate_target(target)
    frame = context.repository.query_model_training_dataset(
        adjust=args.adjust,
        target=target,
        split=args.split,
    )
    if args.tail:
        frame = frame.tail(args.tail)
    print(format_frame(frame))


def validate_target(target: str) -> None:
    if target not in LABEL_COLUMNS:
        raise ValueError(f"Unsupported target: {target}. Choose one of: {', '.join(LABEL_COLUMNS)}")


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

    collect_calendar = subparsers.add_parser("collect-calendar", help="采集交易日历并入库")
    collect_calendar.add_argument("--start-date", help="开始日期，格式 YYYYMMDD")
    collect_calendar.add_argument("--end-date", help="结束日期，格式 YYYYMMDD")
    collect_calendar.set_defaults(func=cmd_collect_calendar)

    collect_suspensions = subparsers.add_parser("collect-suspensions", help="按交易日采集停复牌公告并入库")
    collect_suspensions.add_argument("--start-date", help="开始日期，格式 YYYY-MM-DD 或 YYYYMMDD")
    collect_suspensions.add_argument("--end-date", help="结束日期，格式 YYYY-MM-DD 或 YYYYMMDD")
    collect_suspensions.add_argument("--limit", type=int, default=10, help="调试股票数量")
    collect_suspensions.set_defaults(func=cmd_collect_suspensions)

    derive_suspensions = subparsers.add_parser("derive-suspensions", help="根据交易日有无日线数据推导停牌/缺失状态")
    derive_suspensions.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    derive_suspensions.set_defaults(func=cmd_derive_suspensions)

    build_limit_status = subparsers.add_parser("build-limit-status", help="根据日线涨跌幅生成涨跌停标记")
    build_limit_status.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    build_limit_status.set_defaults(func=cmd_build_limit_status)

    build_features = subparsers.add_parser("build-features", help="生成基础技术因子")
    build_features.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    build_features.set_defaults(func=cmd_build_features)

    build_labels = subparsers.add_parser("build-labels", help="生成训练标签")
    build_labels.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    build_labels.set_defaults(func=cmd_build_labels)

    build_dataset = subparsers.add_parser("build-dataset", help="生成涨跌停、推导停牌、技术因子和训练标签")
    build_dataset.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    build_dataset.set_defaults(func=cmd_build_dataset)

    build_training_dataset = subparsers.add_parser("build-training-dataset", help="合并技术因子和训练标签，生成模型训练表")
    build_training_dataset.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    build_training_dataset.add_argument("--target", help="目标标签，例如 future_return_5d")
    build_training_dataset.add_argument("--train-ratio", type=float, help="训练集时间占比")
    build_training_dataset.add_argument("--valid-ratio", type=float, help="验证集时间占比")
    build_training_dataset.set_defaults(func=cmd_build_training_dataset)

    train_lgbm = subparsers.add_parser("train-lgbm", help="训练 LightGBM 回归模型")
    train_lgbm.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    train_lgbm.add_argument("--target", help="目标标签，例如 future_return_5d")
    train_lgbm.set_defaults(func=cmd_train_lgbm)

    analyze_factors_parser = subparsers.add_parser("analyze-factors", help="分析因子 IC、Rank IC 和分层收益")
    analyze_factors_parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    analyze_factors_parser.add_argument("--target", help="目标标签，例如 future_return_5d")
    analyze_factors_parser.add_argument("--quantiles", type=int, help="分层数量")
    analyze_factors_parser.add_argument("--min-stocks-per-date", type=int, help="每个交易日至少多少只股票才计算横截面 IC")
    analyze_factors_parser.set_defaults(func=cmd_analyze_factors)

    train_baseline = subparsers.add_parser("train-baseline", help="训练和比较简单 baseline 模型")
    train_baseline.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    train_baseline.add_argument("--target", help="目标标签，例如 future_return_5d")
    train_baseline.add_argument("--ridge-alpha", type=float, help="Ridge 正则强度")
    train_baseline.set_defaults(func=cmd_train_baseline)

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

    query_features = subparsers.add_parser("query-features", help="查询技术因子")
    query_features.add_argument("--symbol", help="股票代码，例如 000001")
    query_features.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    query_features.add_argument("--tail", type=int, help="只显示最后 N 行")
    query_features.set_defaults(func=cmd_query_features)

    query_labels = subparsers.add_parser("query-labels", help="查询训练标签")
    query_labels.add_argument("--symbol", help="股票代码，例如 000001")
    query_labels.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    query_labels.add_argument("--tail", type=int, help="只显示最后 N 行")
    query_labels.set_defaults(func=cmd_query_labels)

    query_training_dataset = subparsers.add_parser("query-training-dataset", help="查询模型训练表")
    query_training_dataset.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    query_training_dataset.add_argument("--target", help="只保留目标标签非空的样本")
    query_training_dataset.add_argument("--split", choices=["train", "valid", "test"], help="只查看某个时间切分")
    query_training_dataset.add_argument("--tail", type=int, help="只显示最后 N 行")
    query_training_dataset.set_defaults(func=cmd_query_training_dataset)

    counts = subparsers.add_parser("counts", help="查看核心数据表行数")
    counts.set_defaults(func=cmd_counts)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
