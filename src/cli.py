from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import pandas as pd

from src.backtest.factor_backtest import backtest_single_factors
from src.backtest.multifactor_backtest import backtest_multifactor_strategy, rolling_backtest_multifactor_strategy
from src.collectors.akshare_client import AkshareCollector
from src.database.repository import MarketDataRepository
from src.models.baseline import train_baseline_models
from src.models.dataset import LABEL_COLUMNS
from src.models.factor_analysis import analyze_factors
from src.models.reports import list_json_reports, report_summary_tables
from src.models.rolling_validation import rolling_validate_lgbm
from src.models.train_lgbm import train_lgbm_regressor
from src.utils.config import load_yaml


DATA_SOURCE_CONFIG = "config/data_source.yaml"
MODEL_CONFIG = "config/model.yaml"
TRADING_COST_CONFIG = "config/trading_cost.yaml"


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
    pool = select_collection_pool(context, source=args.pool_source, limit=args.limit)
    start_date = args.start_date or ak_config.get("default_start_date")
    end_date = args.end_date or ak_config.get("default_end_date")
    adjust = args.adjust or ak_config.get("default_adjust", "qfq")

    stock_basic = context.collector.fetch_stock_basic(pool)
    stock_count = context.repository.upsert_stock_basic(stock_basic)
    print(f"Upserted {stock_count} stock_basic rows")

    total_rows = 0
    failures: list[tuple[str, str]] = []
    progress_every = max(1, int(args.progress_every or 1))
    for index, item in enumerate(pool, start=1):
        symbol = item["symbol"]
        try:
            daily = context.collector.fetch_daily_price(symbol, start_date, end_date, adjust)
            row_count = context.repository.upsert_daily_prices(daily)
            total_rows += row_count
            if not args.quiet and (index == 1 or index % progress_every == 0 or index == len(pool)):
                print(f"{index}/{len(pool)} {symbol}: upserted {row_count} daily rows, total {total_rows}")
        except Exception as exc:
            failures.append((symbol, str(exc)))
            print(f"{symbol}: failed - {exc}")

    print(f"Finished. Upserted {total_rows} daily_price rows.")
    if failures:
        print("Failures:")
        for symbol, error in failures:
            print(f"- {symbol}: {error}")


def cmd_collect_stock_pool(args: argparse.Namespace) -> None:
    context = build_context()
    stock_pool = context.collector.fetch_a_stock_pool(limit=args.limit)
    row_count = context.repository.replace_stock_basic(stock_pool)
    print(f"Replaced stock_basic with {row_count} filtered stock pool rows")
    if args.show:
        print(format_frame(stock_pool.head(args.show)))


def cmd_prune_market_data(_: argparse.Namespace) -> None:
    context = build_context()
    print(format_frame(context.repository.prune_market_data_to_stock_basic()))


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


def cmd_rolling_validate_lgbm(args: argparse.Namespace) -> None:
    context = build_context()
    model_config = load_yaml(MODEL_CONFIG)
    dataset_config = model_config.get("dataset", {})
    rolling_config = model_config.get("rolling_validation", {})
    artifact_config = model_config.get("artifacts", {})
    target = args.target or dataset_config.get("target", "future_return_5d")
    validate_target(target)

    dataset = context.repository.query_model_training_dataset(adjust=args.adjust, target=target)
    if dataset.empty:
        print("No training dataset found. Run build-training-dataset first.")
        return

    report = rolling_validate_lgbm(
        dataset=dataset,
        target=target,
        model_config=model_config.get("lightgbm", {}),
        report_dir=artifact_config.get("report_dir", "artifacts/reports"),
        train_window=args.train_window or int(rolling_config.get("train_window", 252)),
        test_window=args.test_window or int(rolling_config.get("test_window", 63)),
        step=args.step or int(rolling_config.get("step", 63)),
        min_train_rows=args.min_train_rows or int(rolling_config.get("min_train_rows", 200)),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


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


def cmd_backtest_factors(args: argparse.Namespace) -> None:
    context = build_context()
    model_config = load_yaml(MODEL_CONFIG)
    trading_cost_config = load_yaml(TRADING_COST_CONFIG)
    dataset_config = model_config.get("dataset", {})
    factor_config = model_config.get("factor_analysis", {})
    backtest_config = model_config.get("factor_backtest", {})
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

    cost_rate = args.cost_rate
    if cost_rate is None:
        cost_rate = estimate_round_trip_cost_rate(trading_cost_config)

    report = backtest_single_factors(
        dataset=dataset,
        target=target,
        report_dir=artifact_config.get("report_dir", "artifacts/reports"),
        quantile=args.quantile or float(backtest_config.get("quantile", 0.2)),
        top_n=args.top_n or backtest_config.get("top_n"),
        rebalance_step=args.rebalance_step or int(backtest_config.get("rebalance_step", 5)),
        min_stocks_per_date=args.min_stocks_per_date or int(factor_config.get("min_stocks_per_date", 5)),
        cost_rate=float(cost_rate),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_backtest_multifactor(args: argparse.Namespace) -> None:
    context = build_context()
    model_config = load_yaml(MODEL_CONFIG)
    trading_cost_config = load_yaml(TRADING_COST_CONFIG)
    dataset_config = model_config.get("dataset", {})
    factor_config = model_config.get("factor_analysis", {})
    backtest_config = model_config.get("multifactor_backtest", {})
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

    cost_rate = args.cost_rate
    if cost_rate is None:
        cost_rate = estimate_round_trip_cost_rate(trading_cost_config)

    report = backtest_multifactor_strategy(
        dataset=dataset,
        target=target,
        report_dir=artifact_config.get("report_dir", "artifacts/reports"),
        quantile=args.quantile or float(backtest_config.get("quantile", 0.2)),
        top_n=args.top_n or backtest_config.get("top_n"),
        rebalance_step=args.rebalance_step or int(backtest_config.get("rebalance_step", 5)),
        min_stocks_per_date=args.min_stocks_per_date or int(factor_config.get("min_stocks_per_date", 5)),
        cost_rate=float(cost_rate),
        max_factors=args.max_factors or int(backtest_config.get("max_factors", 5)),
        min_abs_rank_ic=args.min_abs_rank_ic or float(backtest_config.get("min_abs_rank_ic", 0.02)),
        evaluation_split=args.evaluation_split or str(backtest_config.get("evaluation_split", "test")),
        weighting=args.weighting or str(backtest_config.get("weighting", "rank_ic")),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_rolling_backtest_multifactor(args: argparse.Namespace) -> None:
    context = build_context()
    model_config = load_yaml(MODEL_CONFIG)
    trading_cost_config = load_yaml(TRADING_COST_CONFIG)
    dataset_config = model_config.get("dataset", {})
    factor_config = model_config.get("factor_analysis", {})
    backtest_config = model_config.get("multifactor_backtest", {})
    rolling_config = model_config.get("rolling_multifactor_backtest", {})
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

    cost_rate = args.cost_rate
    if cost_rate is None:
        cost_rate = estimate_round_trip_cost_rate(trading_cost_config)

    report = rolling_backtest_multifactor_strategy(
        dataset=dataset,
        target=target,
        report_dir=artifact_config.get("report_dir", "artifacts/reports"),
        quantile=args.quantile or float(backtest_config.get("quantile", 0.2)),
        top_n=args.top_n or backtest_config.get("top_n"),
        rebalance_step=args.rebalance_step or int(backtest_config.get("rebalance_step", 5)),
        min_stocks_per_date=args.min_stocks_per_date or int(factor_config.get("min_stocks_per_date", 5)),
        cost_rate=float(cost_rate),
        max_factors=args.max_factors or int(backtest_config.get("max_factors", 5)),
        min_abs_rank_ic=args.min_abs_rank_ic or float(backtest_config.get("min_abs_rank_ic", 0.02)),
        weighting=args.weighting or str(backtest_config.get("weighting", "rank_ic")),
        train_window=args.train_window or int(rolling_config.get("train_window", 252)),
        test_window=args.test_window or int(rolling_config.get("test_window", 63)),
        step=args.step or int(rolling_config.get("step", 63)),
        embargo_days=args.embargo_days if args.embargo_days is not None else int(rolling_config.get("embargo_days", 5)),
        min_train_rows=args.min_train_rows or int(rolling_config.get("min_train_rows", 200)),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_run_research(args: argparse.Namespace) -> None:
    context = build_context()
    model_config = load_yaml(MODEL_CONFIG)
    trading_cost_config = load_yaml(TRADING_COST_CONFIG)
    dataset_config = model_config.get("dataset", {})
    factor_config = model_config.get("factor_analysis", {})
    factor_backtest_config = model_config.get("factor_backtest", {})
    multifactor_config = model_config.get("multifactor_backtest", {})
    rolling_multifactor_config = model_config.get("rolling_multifactor_backtest", {})
    rolling_lgbm_config = model_config.get("rolling_validation", {})
    baseline_config = model_config.get("baseline", {})
    artifact_config = model_config.get("artifacts", {})
    target = args.target or dataset_config.get("target", "future_return_5d")
    validate_target(target)

    adjust = args.adjust
    report_dir = artifact_config.get("report_dir", "artifacts/reports")
    model_dir = artifact_config.get("model_dir", "artifacts/models")
    cost_rate = estimate_round_trip_cost_rate(trading_cost_config)

    if args.rebuild_dataset:
        print("[1/9] Rebuilding derived dataset")
        limit_rows = context.repository.upsert_daily_limit_status(
            context.repository.build_daily_limit_status(adjust=adjust)
        )
        suspension_rows = context.repository.upsert_stock_suspensions(
            context.repository.derive_missing_suspensions(adjust=adjust)
        )
        feature_rows = context.repository.build_and_store_technical_features(adjust=adjust)
        label_rows = context.repository.build_and_store_training_labels(adjust=adjust)
        training_rows = context.repository.build_and_store_model_training_dataset(
            adjust=adjust,
            target=target,
            train_ratio=float(dataset_config.get("train_ratio", 0.7)),
            valid_ratio=float(dataset_config.get("valid_ratio", 0.15)),
        )
        print(
            f"built rows: limit={limit_rows}, suspensions={suspension_rows}, "
            f"features={feature_rows}, labels={label_rows}, training={training_rows}"
        )
    else:
        print("[1/9] Reusing existing model_training_dataset")

    dataset = context.repository.query_model_training_dataset(adjust=adjust, target=target)
    if dataset.empty:
        print("No training dataset found. Run build-training-dataset first.")
        return

    step_no = 2
    print(f"[{step_no}/9] analyze-factors")
    factor_report = analyze_factors(
        dataset=dataset,
        target=target,
        report_dir=report_dir,
        quantiles=int(factor_config.get("quantiles", 5)),
        min_stocks_per_date=int(factor_config.get("min_stocks_per_date", 5)),
    )
    print_report_path("factor_analysis", factor_report)

    step_no += 1
    print(f"[{step_no}/9] backtest-factors")
    single_factor_report = backtest_single_factors(
        dataset=dataset,
        target=target,
        report_dir=report_dir,
        quantile=float(factor_backtest_config.get("quantile", 0.2)),
        top_n=factor_backtest_config.get("top_n"),
        rebalance_step=int(factor_backtest_config.get("rebalance_step", 5)),
        min_stocks_per_date=int(factor_config.get("min_stocks_per_date", 5)),
        cost_rate=cost_rate,
    )
    print_report_path("factor_backtest", single_factor_report)

    step_no += 1
    print(f"[{step_no}/9] backtest-multifactor")
    multifactor_report = backtest_multifactor_strategy(
        dataset=dataset,
        target=target,
        report_dir=report_dir,
        quantile=float(multifactor_config.get("quantile", 0.2)),
        top_n=multifactor_config.get("top_n"),
        rebalance_step=int(multifactor_config.get("rebalance_step", 5)),
        min_stocks_per_date=int(factor_config.get("min_stocks_per_date", 5)),
        cost_rate=cost_rate,
        max_factors=int(multifactor_config.get("max_factors", 5)),
        min_abs_rank_ic=float(multifactor_config.get("min_abs_rank_ic", 0.02)),
        evaluation_split=str(multifactor_config.get("evaluation_split", "test")),
        weighting=str(multifactor_config.get("weighting", "rank_ic")),
    )
    print_report_path("multifactor_backtest", multifactor_report)

    step_no += 1
    print(f"[{step_no}/9] rolling-backtest-multifactor")
    rolling_multifactor_report = rolling_backtest_multifactor_strategy(
        dataset=dataset,
        target=target,
        report_dir=report_dir,
        quantile=float(multifactor_config.get("quantile", 0.2)),
        top_n=multifactor_config.get("top_n"),
        rebalance_step=int(multifactor_config.get("rebalance_step", 5)),
        min_stocks_per_date=int(factor_config.get("min_stocks_per_date", 5)),
        cost_rate=cost_rate,
        max_factors=int(multifactor_config.get("max_factors", 5)),
        min_abs_rank_ic=float(multifactor_config.get("min_abs_rank_ic", 0.02)),
        weighting=str(multifactor_config.get("weighting", "rank_ic")),
        train_window=int(rolling_multifactor_config.get("train_window", 252)),
        test_window=int(rolling_multifactor_config.get("test_window", 63)),
        step=int(rolling_multifactor_config.get("step", 63)),
        embargo_days=int(rolling_multifactor_config.get("embargo_days", 5)),
        min_train_rows=int(rolling_multifactor_config.get("min_train_rows", 200)),
    )
    print_report_path("rolling_multifactor_backtest", rolling_multifactor_report)

    step_no += 1
    print(f"[{step_no}/9] train-baseline")
    baseline_report = train_baseline_models(
        dataset=dataset,
        target=target,
        report_dir=report_dir,
        ridge_alpha=float(baseline_config.get("ridge_alpha", 1.0)),
    )
    print_report_path("baseline", baseline_report, path_key="metrics_path")

    if args.skip_lgbm:
        print("[7/9] train-lgbm skipped")
        print("[8/9] rolling-validate-lgbm skipped")
    else:
        step_no += 1
        print(f"[{step_no}/9] train-lgbm")
        lgbm_report = train_lgbm_regressor(
            dataset=dataset,
            target=target,
            model_config=model_config.get("lightgbm", {}),
            model_dir=model_dir,
            report_dir=report_dir,
        )
        print_report_path("lgbm", lgbm_report, path_key="metrics_path")

        step_no += 1
        print(f"[{step_no}/9] rolling-validate-lgbm")
        rolling_lgbm_report = rolling_validate_lgbm(
            dataset=dataset,
            target=target,
            model_config=model_config.get("lightgbm", {}),
            report_dir=report_dir,
            train_window=int(rolling_lgbm_config.get("train_window", 252)),
            test_window=int(rolling_lgbm_config.get("test_window", 63)),
            step=int(rolling_lgbm_config.get("step", 63)),
            min_train_rows=int(rolling_lgbm_config.get("min_train_rows", 200)),
        )
        print_report_path("rolling_lgbm", rolling_lgbm_report, path_key="metrics_path")

    print("[9/9] done")


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


def cmd_report_summary(args: argparse.Namespace) -> None:
    model_config = load_yaml(MODEL_CONFIG)
    artifact_config = model_config.get("artifacts", {})
    reports = list_json_reports(artifact_config.get("report_dir", "artifacts/reports"))
    if args.type:
        reports = [report for report in reports if report["type"] == args.type]
    if not reports:
        print("No reports found.")
        return

    selected_reports = reports[: args.limit]
    for report in selected_reports:
        print(f"\n[{report['type']}] {report['name']}")
        print(f"path: {report['path']}")
        tables = report_summary_tables(report)
        if not tables:
            print(json.dumps(report["payload"], indent=2, ensure_ascii=False))
            continue
        for title, table in tables.items():
            print(f"\n{title}:")
            print(format_frame(table))


def select_collection_pool(context: AppContext, source: str, limit: int | None) -> list[dict[str, str]]:
    if source == "config":
        return select_pool(context.config, limit)
    pool = context.repository.get_stock_pool(limit=limit)
    if not pool:
        raise ValueError("No stock_basic rows found. Run collect-stock-pool first or use --pool-source config.")
    return pool


def validate_target(target: str) -> None:
    if target not in LABEL_COLUMNS:
        raise ValueError(f"Unsupported target: {target}. Choose one of: {', '.join(LABEL_COLUMNS)}")


def estimate_round_trip_cost_rate(config: dict) -> float:
    commission = float(config.get("commission_rate", 0))
    stamp_tax_sell = float(config.get("stamp_tax_sell_rate", 0))
    transfer_fee = float(config.get("transfer_fee_rate", 0))
    slippage = float(config.get("slippage_rate", 0))
    return commission * 2 + stamp_tax_sell + transfer_fee * 2 + slippage * 2


def print_report_path(name: str, report: dict, path_key: str = "report_path") -> None:
    path = report.get(path_key) or report.get("metrics_path")
    print(f"{name}: {path}")


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
    collect.add_argument("--pool-source", default="config", choices=["config", "db"], help="股票池来源")
    collect.add_argument("--quiet", action="store_true", help="减少采集过程中的逐股票输出")
    collect.add_argument("--progress-every", type=int, default=25, help="每采集多少只股票打印一次进度")
    collect.set_defaults(func=cmd_collect)

    collect_stock_pool = subparsers.add_parser("collect-stock-pool", help="从 AKShare 获取 A 股股票池并写入 stock_basic")
    collect_stock_pool.add_argument("--limit", type=int, default=100, help="股票池数量")
    collect_stock_pool.add_argument("--show", type=int, default=10, help="显示前 N 行")
    collect_stock_pool.set_defaults(func=cmd_collect_stock_pool)

    prune_market_data = subparsers.add_parser("prune-market-data", help="删除当前股票池之外的本地行情、特征、标签和训练集")
    prune_market_data.set_defaults(func=cmd_prune_market_data)

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

    rolling_lgbm = subparsers.add_parser("rolling-validate-lgbm", help="LightGBM 滚动时间窗口验证")
    rolling_lgbm.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    rolling_lgbm.add_argument("--target", help="目标标签，例如 future_return_5d")
    rolling_lgbm.add_argument("--train-window", type=int, help="训练窗口交易日数量")
    rolling_lgbm.add_argument("--test-window", type=int, help="测试窗口交易日数量")
    rolling_lgbm.add_argument("--step", type=int, help="滚动步长交易日数量")
    rolling_lgbm.add_argument("--min-train-rows", type=int, help="每折最少训练样本数")
    rolling_lgbm.set_defaults(func=cmd_rolling_validate_lgbm)

    analyze_factors_parser = subparsers.add_parser("analyze-factors", help="分析因子 IC、Rank IC 和分层收益")
    analyze_factors_parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    analyze_factors_parser.add_argument("--target", help="目标标签，例如 future_return_5d")
    analyze_factors_parser.add_argument("--quantiles", type=int, help="分层数量")
    analyze_factors_parser.add_argument("--min-stocks-per-date", type=int, help="每个交易日至少多少只股票才计算横截面 IC")
    analyze_factors_parser.set_defaults(func=cmd_analyze_factors)

    backtest_factors = subparsers.add_parser("backtest-factors", help="按因子方向做单因子组合回测")
    backtest_factors.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    backtest_factors.add_argument("--target", help="目标标签，例如 future_return_5d")
    backtest_factors.add_argument("--quantile", type=float, help="每次选取得分最高的比例，例如 0.2")
    backtest_factors.add_argument("--top-n", type=int, help="每次固定选择前 N 只股票；设置后优先于 quantile")
    backtest_factors.add_argument("--rebalance-step", type=int, help="每隔多少个交易日调仓一次")
    backtest_factors.add_argument("--min-stocks-per-date", type=int, help="每个调仓日至少多少只可选股票")
    backtest_factors.add_argument("--cost-rate", type=float, help="单次买入再卖出的估算总成本率")
    backtest_factors.set_defaults(func=cmd_backtest_factors)

    backtest_multifactor = subparsers.add_parser("backtest-multifactor", help="按训练集因子方向做多因子组合回测")
    backtest_multifactor.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    backtest_multifactor.add_argument("--target", help="目标标签，例如 future_return_5d")
    backtest_multifactor.add_argument("--quantile", type=float, help="每次选取得分最高的比例，例如 0.2")
    backtest_multifactor.add_argument("--top-n", type=int, help="每次固定选择前 N 只股票；设置后优先于 quantile")
    backtest_multifactor.add_argument("--rebalance-step", type=int, help="每隔多少个交易日调仓一次")
    backtest_multifactor.add_argument("--min-stocks-per-date", type=int, help="每个调仓日至少多少只可选股票")
    backtest_multifactor.add_argument("--cost-rate", type=float, help="单次买入再卖出的估算总成本率")
    backtest_multifactor.add_argument("--max-factors", type=int, help="最多使用 Rank IC 靠前的多少个因子")
    backtest_multifactor.add_argument("--min-abs-rank-ic", type=float, help="因子入选所需的最小绝对 Rank IC")
    backtest_multifactor.add_argument("--evaluation-split", choices=["train", "valid", "test", "valid_test", "all"], help="评估区间")
    backtest_multifactor.add_argument("--weighting", choices=["rank_ic", "equal"], help="因子权重方式")
    backtest_multifactor.set_defaults(func=cmd_backtest_multifactor)

    rolling_multifactor = subparsers.add_parser("rolling-backtest-multifactor", help="滚动多因子样本外回测")
    rolling_multifactor.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    rolling_multifactor.add_argument("--target", help="目标标签，例如 future_return_5d")
    rolling_multifactor.add_argument("--quantile", type=float, help="每次选取得分最高的比例，例如 0.2")
    rolling_multifactor.add_argument("--top-n", type=int, help="每次固定选择前 N 只股票；设置后优先于 quantile")
    rolling_multifactor.add_argument("--rebalance-step", type=int, help="每隔多少个交易日调仓一次")
    rolling_multifactor.add_argument("--min-stocks-per-date", type=int, help="每个调仓日至少多少只可选股票")
    rolling_multifactor.add_argument("--cost-rate", type=float, help="单次买入再卖出的估算总成本率")
    rolling_multifactor.add_argument("--max-factors", type=int, help="最多使用 Rank IC 靠前的多少个因子")
    rolling_multifactor.add_argument("--min-abs-rank-ic", type=float, help="因子入选所需的最小绝对 Rank IC")
    rolling_multifactor.add_argument("--weighting", choices=["rank_ic", "equal"], help="因子权重方式")
    rolling_multifactor.add_argument("--train-window", type=int, help="训练窗口交易日数量")
    rolling_multifactor.add_argument("--test-window", type=int, help="测试窗口交易日数量")
    rolling_multifactor.add_argument("--step", type=int, help="滚动步长交易日数量")
    rolling_multifactor.add_argument("--embargo-days", type=int, help="训练窗口和测试窗口之间空出多少个交易日，降低标签穿越")
    rolling_multifactor.add_argument("--min-train-rows", type=int, help="每折最少训练样本数")
    rolling_multifactor.set_defaults(func=cmd_rolling_backtest_multifactor)

    run_research = subparsers.add_parser("run-research", help="重建训练集并一键运行当前所有研究报告")
    run_research.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权类型")
    run_research.add_argument("--target", help="目标标签，例如 future_return_5d")
    run_research.add_argument("--rebuild-dataset", action=argparse.BooleanOptionalAction, default=True, help="是否先重建衍生数据集")
    run_research.add_argument("--skip-lgbm", action="store_true", help="跳过 LightGBM 训练和滚动验证")
    run_research.set_defaults(func=cmd_run_research)

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

    report_summary = subparsers.add_parser("report-summary", help="查看本地 artifacts/reports 里的报告摘要")
    report_summary.add_argument("--type", choices=["lgbm", "baseline", "rolling_lgbm", "factor_analysis", "factor_backtest", "multifactor_backtest", "rolling_multifactor_backtest", "unknown"], help="只查看某类报告")
    report_summary.add_argument("--limit", type=int, default=3, help="显示最近 N 个报告")
    report_summary.set_defaults(func=cmd_report_summary)

    counts = subparsers.add_parser("counts", help="查看核心数据表行数")
    counts.set_defaults(func=cmd_counts)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
