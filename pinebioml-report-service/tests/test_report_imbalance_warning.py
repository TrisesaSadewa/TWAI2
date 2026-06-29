import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.report.report_engine import ReportEngine


def test_accuracy_warning_when_imbalance_hides_poor_specificity():
    engine = ReportEngine()

    warning = engine._build_imbalance_warning(
        metrics={
            "accuracy": 0.9067,
            "specificity": "5.08%",
            "recall": 0.5254,
            "MCC": "0.2147",
        },
        per_class=[
            {"class": "0", "precision": 1.0, "recall": 0.0508, "f1": 0.0968, "support": 59},
            {"class": "1", "precision": 0.9062, "recall": 1.0, "f1": 0.9508, "support": 541},
        ],
        all_models=[],
    )

    assert warning["severity"] == "high"
    assert warning["imbalance_ratio"] == 9.17
    assert warning["minority_class"] == "0"
    assert "accuracy alone" in warning["message"].lower()
    assert "specificity is only 5.1%" in warning["message"]


def test_no_accuracy_warning_for_balanced_reasonable_metrics():
    engine = ReportEngine()

    warning = engine._build_imbalance_warning(
        metrics={
            "accuracy": 0.82,
            "specificity": 0.78,
            "recall": 0.80,
            "MCC": 0.61,
        },
        per_class=[
            {"class": "0", "precision": 0.80, "recall": 0.78, "f1": 0.79, "support": 100},
            {"class": "1", "precision": 0.83, "recall": 0.80, "f1": 0.81, "support": 110},
        ],
        all_models=[],
    )

    assert warning == {}
