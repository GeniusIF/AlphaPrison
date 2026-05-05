from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.repository import MarketDataRepository
from src.models.reports import list_json_reports, report_summary_tables
from src.utils.config import load_yaml


st.set_page_config(page_title="赚钱的牢A", page_icon="📈", layout="wide")


@st.cache_resource
def get_repository() -> MarketDataRepository:
    config = load_yaml("config/data_source.yaml")
    return MarketDataRepository(config["database"]["path"])


repo = get_repository()
st.title("赚钱的牢A")
st.caption("个人 A 股 AI 投研台 · 数据底座 MVP")

stocks = repo.list_stocks()
latest = repo.latest_daily_prices()
counts = repo.table_counts()

left, mid, right = st.columns(3)
left.metric("本地股票数", len(stocks))
mid.metric("最新行情数", len(latest))
right.metric("数据库", str(repo.db_path.name))

tab_market, tab_stock, tab_dataset, tab_model = st.tabs(["市场数据", "股票详情", "训练集", "模型训练"])

with tab_market:
    st.subheader("核心表行数")
    st.dataframe(counts, use_container_width=True, hide_index=True)

    if latest.empty:
        st.info("还没有本地数据。先运行：python -m src.cli collect --start-date 20240101 --end-date 20260503")
    else:
        st.subheader("最新行情")
        st.dataframe(
            latest[
                [
                    "symbol",
                    "name",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "pct_change",
                    "turnover_rate",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

with tab_stock:
    if stocks.empty:
        st.info("还没有股票基础信息。")
    else:
        options = [f"{row.symbol} {row.name or ''}".strip() for row in stocks.itertuples()]
        selected = st.selectbox("股票", options)
        symbol = selected.split()[0]

        history = repo.query_daily_prices(symbol=symbol)
        st.subheader(f"{selected} 日线")
        if history.empty:
            st.warning("这只股票还没有日线数据。")
        else:
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=history["trade_date"],
                        open=history["open"],
                        high=history["high"],
                        low=history["low"],
                        close=history["close"],
                        name=symbol,
                    )
                ]
            )
            fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("原始日线数据")
            st.dataframe(history.tail(120), use_container_width=True, hide_index=True)

with tab_dataset:
    if stocks.empty:
        st.info("先采集行情数据，再生成特征和标签。")
    else:
        symbol = st.selectbox(
            "查看训练样例",
            [row.symbol for row in stocks.itertuples()],
            key="dataset_symbol",
        )
        features = repo.query_technical_features(symbol=symbol).tail(60)
        labels = repo.query_training_labels(symbol=symbol).tail(60)
        st.subheader("技术因子")
        st.dataframe(features, use_container_width=True, hide_index=True)
        st.subheader("训练标签")
        st.dataframe(labels, use_container_width=True, hide_index=True)

with tab_model:
    st.subheader("训练表")
    training_rows = counts.loc[counts["table"] == "model_training_dataset", "rows"]
    st.metric("训练样本数", int(training_rows.iloc[0]) if not training_rows.empty else 0)

    report_dir = ROOT / "artifacts" / "reports"
    reports = list_json_reports(report_dir)
    if not reports:
        st.info("还没有模型报告。先运行：python -m src.cli train-lgbm")
    else:
        options = [f"{report['type']} · {report['name']}" for report in reports]
        selected_index = st.selectbox("报告", range(len(options)), format_func=lambda index: options[index])
        report = reports[selected_index]
        payload = report["payload"]
        tables = report_summary_tables(report)
        st.caption(str(report["path"]))

        if report["type"] == "lgbm":
            st.subheader("LightGBM 单次切分")
            scores = payload.get("scores", {})
            cols = st.columns(3)
            for col, split in zip(cols, ["train", "valid", "test"]):
                score = scores.get(split, {})
                col.metric(f"{split} RMSE", f"{score.get('rmse', 0):.4f}")
                col.metric(f"{split} 方向准确率", f"{score.get('directional_accuracy', 0):.2%}")
        elif report["type"] == "baseline":
            st.subheader("Baseline 测试集对比")
            st.dataframe(tables.get("test_scores"), use_container_width=True, hide_index=True)
        elif report["type"] == "rolling_lgbm":
            st.subheader("LightGBM 滚动验证")
            aggregate = payload.get("aggregate_scores", {})
            cols = st.columns(3)
            cols[0].metric("整体 RMSE", f"{aggregate.get('rmse', 0):.4f}")
            cols[1].metric("整体方向准确率", f"{aggregate.get('directional_accuracy', 0):.2%}")
            cols[2].metric("折数", len(payload.get("folds", [])))
            st.dataframe(tables.get("folds"), use_container_width=True, hide_index=True)
        elif report["type"] == "rolling_multifactor_backtest":
            st.subheader("滚动多因子样本外回测")
            aggregate_summary = payload.get("aggregate_summary", {})
            cols = st.columns(4)
            cols[0].metric("累计收益", f"{aggregate_summary.get('cumulative_return', 0):.2%}")
            cols[1].metric("超额收益", f"{aggregate_summary.get('excess_cumulative_return', 0):.2%}")
            cols[2].metric("最大回撤", f"{aggregate_summary.get('max_drawdown', 0):.2%}")
            cols[3].metric("折数", payload.get("fold_count", 0))
            st.dataframe(tables.get("folds"), use_container_width=True, hide_index=True)
        elif report["type"] == "factor_analysis":
            st.subheader("因子 Rank IC")
            top_rank_ic = tables.get("top_rank_ic")
            display_columns = [
                "feature",
                "daily_rank_ic_mean",
                "daily_rank_ic_ir",
                "factor_direction",
                "recommended_transform",
                "daily_rank_ic_positive_rate",
                "daily_count",
                "daily_coverage",
                "is_reliable",
            ]
            if top_rank_ic is not None and not top_rank_ic.empty:
                st.dataframe(
                    top_rank_ic[[column for column in display_columns if column in top_rank_ic.columns]],
                    use_container_width=True,
                    hide_index=True,
                )
        elif report["type"] == "factor_backtest":
            st.subheader("单因子组合回测")
            top_factors = tables.get("top_factors")
            display_columns = [
                "feature",
                "factor_direction",
                "daily_rank_ic_mean",
                "cumulative_return",
                "annualized_return",
                "max_drawdown",
                "win_rate",
                "periods",
            ]
            if top_factors is not None and not top_factors.empty:
                st.dataframe(
                    top_factors[[column for column in display_columns if column in top_factors.columns]],
                    use_container_width=True,
                    hide_index=True,
                )
        elif report["type"] == "multifactor_backtest":
            st.subheader("多因子组合回测")
            summary = tables.get("summary")
            if summary is not None and not summary.empty:
                score = summary.iloc[0]
                cols = st.columns(4)
                cols[0].metric("累计收益", f"{score.get('cumulative_return', 0):.2%}")
                cols[1].metric("超额收益", f"{score.get('excess_cumulative_return', 0):.2%}")
                cols[2].metric("最大回撤", f"{score.get('max_drawdown', 0):.2%}")
                cols[3].metric("胜率", f"{score.get('win_rate', 0):.2%}")
            st.subheader("入选因子")
            selected_factors = tables.get("selected_factors")
            display_columns = [
                "feature",
                "factor_direction",
                "daily_rank_ic_mean",
                "daily_rank_ic_ir",
                "abs_daily_rank_ic_mean",
                "daily_count",
            ]
            if selected_factors is not None and not selected_factors.empty:
                st.dataframe(
                    selected_factors[[column for column in display_columns if column in selected_factors.columns]],
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning("暂不认识这种报告结构，下面展示原始 JSON。")

        st.subheader("完整报告")
        st.json(payload)
