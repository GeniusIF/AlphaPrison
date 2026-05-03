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
python -m src.cli collect-calendar --start-date 20240101 --end-date 20260504
python -m src.cli build-dataset
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

查看核心表行数：

```bash
python -m src.cli counts
```

## 数据集生成

训练模型不能直接拿原始 K 线就开跑。这个项目会先把原始行情加工成几类更适合回测和训练的数据。

采集交易日历：

```bash
python -m src.cli collect-calendar --start-date 20240101 --end-date 20260504
```

交易日历回答的是：“A 股哪些日期开市？”  
它后面会用于判断某只股票在交易日没有行情时，是不是可能停牌或数据缺失。

采集东方财富停复牌公告：

```bash
python -m src.cli collect-suspensions --start-date 2026-04-01 --end-date 2026-05-04 --limit 10
```

这个接口按日期查询，跑很长时间范围会发很多请求。调试阶段建议先查最近一小段。

根据交易日和日线行情推导停牌/缺失：

```bash
python -m src.cli derive-suspensions
```

逻辑是：如果某天是交易日，但某只股票没有日线行情，就记一条 `derived_missing_daily_price`。  
注意：这只是工程上的推导，真实生产里还要结合交易所公告和数据源校验。

生成涨跌停标记：

```bash
python -m src.cli build-limit-status
```

当前 MVP 用日线 `pct_change` 推导涨跌停，例如普通 A 股约 `+9.8%/-9.8%`，创业板/科创板约 `+19.8%/-19.8%`。  
这足够支持第一版回测约束，但不是交易所级别的精确涨跌停价计算。

生成基础技术因子：

```bash
python -m src.cli build-features
```

目前会生成：

- `return_1d`、`return_5d`、`return_20d`
- `ma_5`、`ma_10`、`ma_20`、`ma_60`
- `ma_5_ratio`、`ma_20_ratio`
- `volatility_5`、`volatility_20`
- `volume_ma_5`、`volume_ma_20`、`volume_ratio_5`
- `rsi_14`
- `macd`、`macd_signal`、`macd_hist`

生成训练标签：

```bash
python -m src.cli build-labels
```

目前会生成：

- `future_return_1d`
- `future_return_5d`
- `future_return_20d`
- `future_max_drawdown_5d`
- `future_max_drawdown_20d`
- `label_up_5d`
- `label_up_20d`

一键生成涨跌停、推导停牌、技术因子和训练标签：

```bash
python -m src.cli build-dataset
```

查看某只股票的特征和标签：

```bash
python -m src.cli query-features --symbol 600519 --tail 5
python -m src.cli query-labels --symbol 600519 --tail 5
```

这里有个重要概念：`features` 是模型在当下能看到的信息，`labels` 是未来才知道的答案。训练时要用历史样本让模型学习“当时的特征”和“后来发生了什么”之间的关系，绝不能把未来信息混进特征里。

## LightGBM 训练

训练前先把特征和标签合并成 `model_training_dataset`：

```bash
python -m src.cli build-training-dataset
```

这一步会按下面的键合并：

```text
symbol + trade_date + adjust
```

并按时间顺序切成三段：

- `train`：较早的 70% 日期
- `valid`：中间的 15% 日期
- `test`：最后的 15% 日期

注意这里不是随机切分。股票模型必须尽量模拟真实场景：只能用过去训练，然后看未来表现。

查看训练表：

```bash
python -m src.cli query-training-dataset --target future_return_5d --tail 5
python -m src.cli query-training-dataset --split test --target future_return_5d --tail 5
```

训练 LightGBM：

```bash
python -m src.cli train-lgbm
```

默认目标是 `future_return_5d`，也就是预测未来 5 个交易日收益率。配置在：

```text
config/model.yaml
```

训练完成后会在本地生成：

```text
artifacts/models/   模型文件
artifacts/reports/  指标、预测结果、特征重要性
```

这些是运行产物，默认不会提交到 GitHub。

第一次模型结果不要只看训练集。如果出现：

```text
train 很好，valid/test 很差
```

这通常叫过拟合。它说明模型记住了过去的小样本，但没有学到稳定的未来规律。后面要靠更多股票、更长时间、更好的因子、更严格回测来改善。

## 因子分析与 Baseline

在继续调复杂模型之前，先回答两个问题：

```text
因子本身有没有一点预测信息？
简单模型能做到什么水平？
```

运行因子分析：

```bash
python -m src.cli analyze-factors
```

它会生成：

```text
artifacts/reports/factor_analysis_*_summary.csv
artifacts/reports/factor_analysis_*_quantiles.csv
artifacts/reports/factor_analysis_*.json
```

重点看这些字段：

- `daily_rank_ic_mean`：每天横截面 Rank IC 的平均值，越远离 0 越可能有信号。
- `daily_rank_ic_positive_rate`：Rank IC 为正的日期占比。
- `daily_count`：有多少个交易日能计算这个因子。
- `daily_coverage`：覆盖率。
- `is_reliable`：覆盖日期太少的因子会被标成不可靠。

IC 可以粗略理解为“因子排序”和“未来收益排序”的相关性。  
如果某个因子 IC 很高但 `daily_count` 很小，通常不要急着相信它，它可能只是偶然好看。

运行 baseline 对比：

```bash
python -m src.cli train-baseline
```

当前 baseline 包括：

- `zero`：永远预测 0。
- `train_mean`：永远预测训练集平均未来收益。
- `momentum_5d`：直接用过去 5 日收益当预测。
- `linear_regression`：线性回归。
- `ridge`：带正则的线性回归。

它会生成：

```text
artifacts/reports/baseline_*_metrics.json
artifacts/reports/baseline_*_predictions.csv
```

baseline 的作用不是为了最终上线，而是给 LightGBM 一个参照。  
如果 LightGBM 在测试集上还打不过简单 baseline，优先怀疑过拟合、样本太少、因子太弱，而不是继续盲目调复杂模型。

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
2. 做 LightGBM 滚动验证和预测信号表。
3. 做真实化回测，包括费用、滑点、T+1、涨跌停不能成交。
4. 接入公告/新闻的大模型事件分析信号。
5. 接入手动持仓和组合风险看板。
