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
- 因子 IC / Rank IC / 分层收益分析。
- baseline 模型对比。
- LightGBM 滚动验证。
- Streamlit 本地看板和报告查看入口。

当前研究结论要保守：扩到 100 支股票后，LightGBM 在滚动验证里的方向准确率大约仍在 50% 附近，说明当前技术因子还没有形成稳定预测力。下一步不建议直接美化收益曲线，而是先做真实化回测框架和更强的数据/因子。

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
python -m src.cli train-baseline
python -m src.cli train-lgbm
python -m src.cli rolling-validate-lgbm
```

扩到 100 支股票：

```bash
python -m src.cli collect-stock-pool --limit 100
python -m src.cli collect --pool-source db --limit 100 --start-date 20240101 --end-date 20260504
python -m src.cli collect-calendar --start-date 20240101 --end-date 20260504
python -m src.cli build-dataset
python -m src.cli build-training-dataset
python -m src.cli analyze-factors
python -m src.cli train-baseline
python -m src.cli train-lgbm
python -m src.cli rolling-validate-lgbm
```

`collect-stock-pool` 默认保留常见 A 股代码段，并排除 ST / *ST 股票。ST 股票涨跌停规则不同，早期研究阶段先避开更清爽。

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
python -m src.cli report-summary --type baseline --limit 1
python -m src.cli report-summary --type lgbm --limit 1
python -m src.cli report-summary --type rolling_lgbm --limit 1
```

报告类型：

- `factor_analysis`：因子 IC、Rank IC 和分层收益。
- `baseline`：零预测、均值预测、动量、线性回归、Ridge。
- `lgbm`：单次 train/valid/test 切分训练。
- `rolling_lgbm`：滚动时间窗口验证。

重点看：

- `directional_accuracy`：方向准确率。
- `rmse`：预测误差。
- `r2`：相对均值预测的解释能力。
- `daily_rank_ic_mean`：因子横截面排序和未来收益排序的平均相关性。
- `daily_count` / `daily_coverage`：因子统计覆盖是否足够。

如果训练集很好，但 valid/test 或 rolling 很差，通常说明模型过拟合。此时应优先扩大样本、改善因子、降低复杂度，而不是继续调参。

## 常用命令

数据查看：

```bash
python -m src.cli counts
python -m src.cli list-stocks
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
python -m src.cli analyze-factors
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
src/features/            技术因子和训练标签
src/models/              数据集、模型、因子分析、报告
src/backtest/            回测模块，待实现
src/portfolio/           持仓管理，待实现
tests/                   单元测试
```

## 已完成

1. **数据底座**
   AKShare 采集、DuckDB 入库、CLI 查询、Streamlit 数据看板。

2. **市场约束数据**
   交易日历、停复牌公告、停牌/缺失推导、涨跌停标记。

3. **训练数据**
   技术因子、未来收益标签、未来回撤标签、模型训练表。

4. **模型研究**
   LightGBM、baseline、因子分析、滚动验证、报告查看。

5. **工程化**
   配置文件、测试、GitHub 备份、运行产物忽略规则。

## 下一步

建议接下来按这个顺序做：

1. **真实化回测引擎**
   用模型预测值生成每周调仓 Top N 策略，加入手续费、印花税、滑点、T+1、涨跌停不能成交、停牌不能成交、单票仓位上限。

2. **预测信号表**
   新增 `model_signal`，保存每次模型对股票的预测分数、排名、建议动作和解释字段，供 UI 和回测共用。

3. **更强因子**
   加入估值、市值、换手率、行业、指数状态、市场宽度等数据。当前纯技术因子偏弱。

4. **滚动训练与预测流水线**
   每个滚动窗口训练模型，并只对未来窗口预测，避免未来函数。

5. **持仓管理**
   手动录入持仓，展示组合仓位、浮盈浮亏、行业集中度、模型信号冲突和止损/止盈提醒。

6. **大模型事件分析**
   接公告/新闻，提取事件类型、情绪、风险和摘要，作为事件因子加入模型与 UI 解释。

下一步最推荐先做 **真实化回测引擎 + model_signal 表**。这会让系统从“模型研究”进入“策略验证”。
