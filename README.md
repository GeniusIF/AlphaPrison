# 赚钱的牢A

个人 A 股 AI 投研台。当前版本先完成最小可用的数据工程底座：

- 从 AKShare 采集 A 股日线数据
- 将股票基础信息和日线行情写入 DuckDB
- 通过 CLI 查询本地数据
- 提供一个 Streamlit 看板入口

> 说明：本项目用于个人研究和交易决策辅助，不构成投资建议，也不应直接用于自动下单。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.cli init-db
python -m src.cli collect --start-date 20240101 --end-date 20260503 --limit 10
python -m src.cli query --symbol 000001 --tail 5
streamlit run app/streamlit_app.py
```

默认数据库路径是 `data/duckdb/a_stock.duckdb`，默认调试股票池在 `config/data_source.yaml`。

## 常用命令

初始化数据库：

```bash
python -m src.cli init-db
```

采集默认 10 支股票的前复权日线：

```bash
python -m src.cli collect --start-date 20240101 --end-date 20260503 --adjust qfq
```

查询某只股票最近 10 条：

```bash
python -m src.cli query --symbol 600519 --tail 10
```

列出本地股票：

```bash
python -m src.cli list-stocks
```

## 项目结构

```text
app/                     Streamlit UI
config/                  数据源、数据库、交易成本等配置
data/duckdb/             DuckDB 本地数据库
src/collectors/          数据采集器
src/database/            Schema 与查询/入库 Repository
src/features/            特征工程，后续扩展
src/models/              模型训练，后续扩展
src/backtest/            回测模块，后续扩展
src/portfolio/           持仓管理，后续扩展
tests/                   单元测试
```

## 下一步

建议按这个顺序继续：

1. 加入交易日历、停复牌、涨跌停数据。
2. 做基础技术因子和训练集标签。
3. 接 LightGBM 训练和滚动验证。
4. 做真实化回测，包括费用、滑点、T+1、涨跌停不能成交。
5. 接入公告/新闻的大模型事件分析信号。
