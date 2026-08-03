import json
from core.report.narrative_generator import NarrativeGenerator
from core.config import get_deployment_writer_model
import asyncio

def test_narrative():
    generator = NarrativeGenerator()
    manifest_dict = {
        "dataset_name": "TestDataset.csv",
        "task_type": "classification",
        "metrics": {"accuracy": 0.95, "ROC-AUC": 0.96},
        "artifacts": {},
        "feature_names": ["f1", "f2"],
        "additional_context": "Disease: Heart Disease Cleveland. Goal: Risk classification.",
        "settings": {}
    }
    model = get_deployment_writer_model()
    
    print("Generating narrative...")
    res = generator.generate_narrative(
        dataset_name="TestDataset.csv",
        task_type="classification",
        metrics={"accuracy": 0.95, "ROC-AUC": 0.96},
        visuals_summary={"roc_curve": "shows AUC 0.96", "confusion_matrix": "mostly correct", "feature_importance": "f1 is top", "shap": "f1 is top"},
        shap_features=[{"feature": "f1", "importance": 0.8}],
        anomaly_flags=[],
        models={"analysis": model},
        report_id="test_id",
        manifest=manifest_dict
    )
    
    with open("test_narrative_output2.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Done. Output written to test_narrative_output2.json")
    
if __name__ == "__main__":
    test_narrative()
