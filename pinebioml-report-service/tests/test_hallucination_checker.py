import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.report.narrative_generator import NarrativeGenerator, HallucinationError

def test_hallucination_checker_pass():
    generator = NarrativeGenerator()
    
    # 3-sentence data simulation
    metrics = {
        "accuracy": "85.5%",
        "ROC-AUC": "0.9200",
        "recall": "80.0%"
    }
    
    # LLM responds correctly without making up numbers
    narrative = {
        "expert": {
            "findings": "Hello. The model is performing well. The accuracy achieved is 85.5%. The AUC is 92.0%, and recall is 80.0%."
        }
    }
    
    # Should pass without raising an error
    try:
        generator._validate_no_hallucinations(narrative, metrics)
    except HallucinationError:
        pytest.fail("Hallucination checker flagged correct data as a hallucination.")

def test_hallucination_checker_fail():
    generator = NarrativeGenerator()
    
    metrics = {
        "accuracy": "85.5%",
        "ROC-AUC": "0.9200",
        "recall": "80.0%"
    }
    
    # LLM hallucinates 100% accuracy (not in metrics)
    narrative = {
        "expert": {
            "findings": "Hello! The model is absolutely perfect. The accuracy is 100% and we caught everything."
        }
    }
    
    # Should raise HallucinationError
    with pytest.raises(HallucinationError) as exc_info:
        generator._validate_no_hallucinations(narrative, metrics)
        
    assert "100" in str(exc_info.value)


def test_hallucination_checker_rejects_unsupported_feature_names():
    generator = NarrativeGenerator()

    metrics = {
        "accuracy": 0.9067,
        "ROC-AUC": "0.7278",
        "recall": 0.5254,
    }
    narrative = {
        "expert": {
            "findings": (
                "Accuracy is about 90%. The top driving features include age, "
                "blood pressure, and cholesterol levels."
            )
        }
    }
    feature_names = [
        "cell_radius",
        "cell_texture",
        "cell_perimeter",
        "noise_metric_1",
        "noise_metric_2",
    ]

    with pytest.raises(HallucinationError) as exc_info:
        generator._validate_no_hallucinations(
            narrative,
            metrics,
            selected_features=feature_names,
        )

    assert "unsupported feature mention" in str(exc_info.value)
