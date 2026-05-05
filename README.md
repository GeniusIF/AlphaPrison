# 赚钱的牢A

个人 A 股 AI 投研台。目标是把 A 股数据采集、特征工程、模型训练、因子分析、滚动验证、回测和持仓看板逐步做成一个本地可运行的研究系统。

> 本项目仅用于个人学习、研究和交易决策辅助，不构成投资建议，也不应直接用于自动下单。

## 当前状态

已经完成：

- AKShare A 股股票池、日线行情、交易日历、停复牌公告采集。
- DuckDB 本地入库和查询。
- 涨跌停标记、停牌/缺失推导。
- 基础技术因子和训练标签。
- 模型训练表 `model_training_dataset`。
- LightGBM 回归模型，默认预测 `future_return_5d`。
- 因子 IC / Rank IC / 分层收益分析，并自动判断因子方向。
- 单因子组合回测，默认用训练集判断方向，再在全样本做 5 日调仓验证。
- 多因子打分回测，默认用训练集筛因子，只在测试集验证。
- 滚动多因子样本外回测，每折只用过去窗口筛因子，再验证未来窗口。
- baseline 模型对比。
- LightGBM 滚动验证。
- Streamlit 本地看板和报告查看入口。

当前研究结论要保守但有价值：扩到 800 只沪深主板股票、2018-01-01 到 2026-05-04 后，数据规模约为 158 万行日线、157 万行训练样本。LightGBM 滚动验证方向准确率约 50.43%，R2 约 -0.10，说明“直接预测未来 5 日收益”的复杂模型暂时没有稳定优势；但滚动多因子样本外回测累计收益约 202.54%，等权基准约 63.60%，累计超额约 138.94%，年化约 16.61%，最大回撤约 -25.97%。这说明“低成交额 / 低换手 / 低波动 / 短期反转”这类截面择股信号在扩大样本后仍然有研究价值。

下一步不应只盯着 LightGBM 调参，而应该分两条线推进：一条继续挖掘更稳定、更有经济含义的因子；另一条研究怎样把已验证的单因子组合成更有效的多因子模型。同时，要逐步把交易成本、涨跌停、停牌、仓位和行业约束加入回测，否则研究收益会高估。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

默认数据库：

```text
data/duckdb/a_stock.duckdb
```

默认配置：

```text
config/data_source.yaml
config/model.yaml
config/trading_cost.yaml
```

本地数据库、模型文件、报告、虚拟环境都不会提交到 GitHub。

## 最短流程

调试 10 支股票：

```bash
python -m src.cli init-db
python -m src.cli collect --pool-source config --limit 10 --start-date 20240101 --end-date 20260504
python -m src.cli collect-calendar --start-date 20240101 --end-date 20260504
python -m src.cli build-dataset
python -m src.cli build-training-dataset
python -m src.cli analyze-factors
python -m src.cli backtest-factors
python -m src.cli backtest-multifactor
python -m src.cli rolling-backtest-multifactor
python -m src.cli train-baseline
python -m src.cli train-lgbm
python -m src.cli rolling-validate-lgbm
```

扩到 300 只沪深主板股票：

```bash
python -m src.cli collect-stock-pool --limit 300 --show 0
python -m src.cli prune-market-data
python -m src.cli collect-calendar --start-date 20240101 --end-date 20260504
python -m src.cli collect --pool-source db --limit 300 --start-date 20240101 --end-date 20260504 --quiet --progress-every 50
python -m src.cli run-research
```

`collect-stock-pool` 默认保留沪深主板，排除 ST / *ST、创业板、科创板、北交所、新股标记和退市整理类名称。`prune-market-data` 会删除当前股票池之外的旧行情、特征、标签和训练集，确保后续分析真的只基于当前筛选池。

为后续神经网络 / 强化学习准备更大训练数据时，可以先跑 800 只、2018 年以来的数据：

```bash
python -m src.cli collect-stock-pool --limit 800 --show 0
python -m src.cli prune-market-data
python -m src.cli collect-calendar --start-date 20180101 --end-date 20260504
python -m src.cli collect --pool-source db --limit 800 --start-date 20180101 --end-date 20260504 --quiet --progress-every 50
python -m src.cli run-research
python -m src.cli counts
```

如果机器时间充裕，再把 `--limit 800` 提高到 `1200` 或更多。第一次扩大数据建议不要同时改模型参数，先保持同一套分析流程，方便比较结论是否稳定。

这轮 800 股票研究对应的最新本地报告已保留在 `artifacts/reports/`，旧的重复报告可以随时删掉，因为都能通过 `python -m src.cli run-research` 重新生成。`artifacts/` 和 `data/` 默认不会提交到 GitHub。

## 看板

启动本地 UI：

```bash
streamlit run app/streamlit_app.py
```

浏览器打开：

```text
http://127.0.0.1:8501
```

看板页签：

- `市场数据`：核心表行数和最新行情。
- `股票详情`：单票 K 线和原始日线。
- `训练集`：技术因子和训练标签样例。
- `模型训练`：查看 LightGBM、baseline、滚动验证、因子分析报告。

## 报告

训练和分析报告会保存在：

```text
artifacts/reports/
```

模型文件会保存在：

```text
artifacts/models/
```

终端查看报告摘要：

```bash
python -m src.cli report-summary
python -m src.cli report-summary --type factor_analysis --limit 1
python -m src.cli report-summary --type factor_backtest --limit 1
python -m src.cli report-summary --type multifactor_backtest --limit 1
python -m src.cli report-summary --type rolling_multifactor_backtest --limit 1
python -m src.cli report-summary --type baseline --limit 1
python -m src.cli report-summary --type lgbm --limit 1
python -m src.cli report-summary --type rolling_lgbm --limit 1
```

报告类型：

- `factor_analysis`：因子 IC、Rank IC、方向判断和分层收益。
- `factor_backtest`：单因子组合回测。
- `multifactor_backtest`：多因子打分组合回测。
- `rolling_multifactor_backtest`：滚动多因子样本外回测。
- `baseline`：零预测、均值预测、动量、线性回归、Ridge。
- `lgbm`：单次 train/valid/test 切分训练。
- `rolling_lgbm`：滚动时间窗口验证。

重点看：

- `directional_accuracy`：方向准确率。
- `rmse`：预测误差。
- `r2`：相对均值预测的解释能力。
- `daily_rank_ic_mean`：因子横截面排序和未来收益排序的平均相关性。
- `daily_count` / `daily_coverage`：因子统计覆盖是否足够。
- `factor_direction`：`positive` 表示因子越大越好，`negative` 表示因子越小越好。
- `cumulative_return` / `max_drawdown`：单因子组合累计收益和最大回撤。
- `excess_cumulative_return`：多因子组合相对等权股票池的累计超额收益。
- `folds`：滚动验证每个历史窗口的独立表现。

如果训练集很好，但 valid/test 或 rolling 很差，通常说明模型过拟合。此时应优先扩大样本、改善因子、降低复杂度，而不是继续调参。

当前单因子和多因子回测只是研究工具，不等于真实可交易策略。它们已经开始考虑估算交易成本，但还没有完整处理真实撮合、持仓重叠、行业暴露、仓位上限、涨停买不进、跌停卖不出、停牌不能交易、T+1、滑点、止损/止盈和完整交易流水。

## 常用命令

数据查看：

```bash
python -m src.cli counts
python -m src.cli list-stocks
python -m src.cli prune-market-data
python -m src.cli latest
python -m src.cli query --symbol 600519 --tail 10
```

特征和标签：

```bash
python -m src.cli query-features --symbol 600519 --tail 5
python -m src.cli query-labels --symbol 600519 --tail 5
python -m src.cli query-training-dataset --target future_return_5d --tail 5
```

单独构建数据集：

```bash
python -m src.cli build-limit-status
python -m src.cli derive-suspensions
python -m src.cli build-features
python -m src.cli build-labels
python -m src.cli build-dataset
python -m src.cli build-training-dataset
```

模型研究：

```bash
python -m src.cli run-research
python -m src.cli analyze-factors
python -m src.cli backtest-factors
python -m src.cli backtest-multifactor
python -m src.cli rolling-backtest-multifactor
python -m src.cli train-baseline
python -m src.cli train-lgbm
python -m src.cli rolling-validate-lgbm
```

## 数据表

核心表：

- `stock_basic`：股票基础信息。
- `daily_price`：日线行情。
- `trade_calendar`：交易日历。
- `stock_suspension`：停牌/缺失信息。
- `daily_limit_status`：涨跌停标记。
- `technical_features`：技术因子。
- `training_labels`：未来收益和风险标签。
- `model_training_dataset`：模型训练表。

## 重要概念

`features` 是模型在当下能看到的信息。  
`labels` 是未来才知道的答案。

训练时要用历史样本学习：

```text
当时的特征 -> 后来发生了什么
```

不能把未来信息混进特征里，否则会出现“回测很好，实盘失效”的未来函数问题。

模型当前默认预测：

```text
future_return_5d
```

也就是未来 5 个交易日收益率。模型输出不是买卖建议，只是研究信号。后续需要通过回测、交易成本、仓位和风控规则把预测转成可执行策略。

## 项目结构

```text
app/                     Streamlit UI
config/                  数据源、模型、交易成本配置
data/duckdb/             DuckDB 本地数据库
artifacts/models/        本地模型文件
artifacts/reports/       本地研究报告
src/collectors/          数据采集和股票池
src/database/            Schema、入库、查询
src/features/            技术因子、流动性因子和训练标签
src/models/              数据集、模型、因子分析、报告
src/backtest/            单因子回测、多因子打分回测、滚动多因子回测
src/portfolio/           持仓管理，待实现
tests/                   单元测试
```

## 已完成

1. **数据底座**
   AKShare 采集、DuckDB 入库、CLI 查询、Streamlit 数据看板。

2. **市场约束数据**
   交易日历、停复牌公告、停牌/缺失推导、涨跌停标记。

3. **训练数据**
   技术因子、成交额/换手率/振幅等流动性因子、未来收益标签、未来回撤标签、模型训练表。

4. **模型研究**
   LightGBM、baseline、因子方向分析、单因子组合回测、多因子测试集回测、滚动多因子样本外回测、报告查看。

5. **工程化**
   配置文件、测试、GitHub 备份、运行产物忽略规则。

## 下一步

建议接下来分两条主线推进，再补一条交易真实化主线。

1. **继续挖掘因子**
   当前最有效的是低成交额、低换手、低波动、短期反转一类因子，但它们还不够丰富。下一步应补截面因子、市值/估值/质量/成长因子、行业相对强弱、市场宽度、指数状态、主力资金流、融资融券、公告事件和政策情绪。主力资金可以先研究 AKShare / 东方财富等公开数据源，腾讯自选股如果没有稳定公开接口，就先作为参考，不把它作为唯一数据源。政策情绪可以先做成按发布日期入库的事件表，后续再接新闻、政策 API 或大模型摘要，重点避免把未来新闻倒灌到历史样本。

2. **优化多因子组合模型**
   现在的现象是“单因子有价值，但直接用 LightGBM 做收益回归不稳定”。下一步可以先做更朴素但稳健的组合方法，比如因子方向统一、横截面 rank/z-score、去极值、行业/市值中性化、因子相关性过滤、Ridge/Lasso/ElasticNet、排序学习、LightGBM Ranker 和小型 MLP。神经网络或强化学习可以做，但应放在交易环境和样本外验证更完善之后；否则模型很容易学到回测噪声。

3. **交易真实化和组合风控**
   当前滚动回测已经能看样本外表现，但还不是真实交易。接下来需要加入佣金、印花税、滑点、涨跌停买卖限制、停牌限制、T+1、单票仓位上限、行业暴露、最大持仓数、调仓换手、止损/止盈和完整交易流水。

4. **预测信号表**
   新增 `model_signal`，保存每次模型对股票的预测分数、排名、建议动作和解释字段，供 UI、回测和持仓看板共用。

5. **持仓管理**
   手动录入持仓，展示组合仓位、浮盈浮亏、行业集中度、模型信号冲突和止损/止盈提醒。

6. **大模型事件分析**
   接公告/新闻，提取事件类型、情绪、风险和摘要，作为事件因子加入模型与 UI 解释。

最近一到两个开发步骤最推荐先做 **资金/基本面/市场状态因子 + 更真实的多因子回测约束**。这样能先判断当前有效因子是不是在交易成本和约束后仍然有效，再决定是否值得上神经网络、强化学习或更复杂的大模型信号。
