from src.models.reports import infer_report_type, report_summary_tables


def test_infer_report_type() -> None:
    assert infer_report_type({"scores": {}}) == "lgbm"
    assert infer_report_type({"models": {}}) == "baseline"
    assert infer_report_type({"aggregate_scores": {}, "folds": []}) == "rolling_lgbm"
    assert infer_report_type({"top_rank_ic": []}) == "factor_analysis"
    assert infer_report_type({"top_factors": []}) == "factor_backtest"
    assert infer_report_type({"selected_factors": [], "summary": {}}) == "multifactor_backtest"
    assert infer_report_type({"aggregate_summary": {}, "folds": []}) == "rolling_multifactor_backtest"


def test_report_summary_tables_for_baseline() -> None:
    report = {
        "type": "baseline",
        "payload": {
            "models": {
                "zero": {"test": {"rmse": 0.2, "directional_accuracy": 0.4}},
                "ridge": {"test": {"rmse": 0.1, "directional_accuracy": 0.5}},
            }
        },
    }

    table = report_summary_tables(report)["test_scores"]

    assert table.iloc[0]["model"] == "ridge"
