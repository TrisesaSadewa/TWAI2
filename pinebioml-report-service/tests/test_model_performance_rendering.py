from core.report.report_engine import ReportEngine


def test_model_performance_uses_report_accuracy_when_model_test_accuracy_missing():
    engine = ReportEngine()
    html = engine._render_model_performance_table(
        [
            {
                "model": "ElasticLogit",
                "train_accuracy": 0.9771,
                "cv_accuracy": 0.9578,
                "test_auc": 0.9982,
                "test_f1": 0.969,
                "test_specificity": 0.98,
                "test_mcc": 0.9512,
            }
        ],
        {"accuracy": 0.9543},
        "classification",
    )

    assert "Test Accuracy:" in html
    assert "95.4%" in html
    assert "0.0%" not in html
