from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.repository import MarketDataRepository
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

tab_market, tab_stock, tab_dataset = st.tabs(["市场数据", "股票详情", "训练集"])

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
