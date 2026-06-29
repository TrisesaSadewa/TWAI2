from core.report.narrative_generator import NarrativeGenerator


def test_data_block_forbids_inferring_imbalance_when_metadata_absent():
    data_block = NarrativeGenerator()._build_data_block(
        dataset_name="breast.csv",
        task_type="classification",
        formatted_metrics="accuracy: 0.9543",
        shap_features=[],
        anomaly_flags=[],
        per_class=[],
        overfit_analysis=None,
        selected_features=[],
        visual_descriptions={},
        imbalance_metadata=None,
        imbalance_warning=None,
    )

    assert "CLASS IMBALANCE METADATA: not recorded" in data_block
    assert "Do not claim the data are balanced" in data_block
    assert "do not claim that SMOTE" in data_block


def test_data_block_states_actual_class_weight_handling():
    data_block = NarrativeGenerator()._build_data_block(
        dataset_name="breast.csv",
        task_type="classification",
        formatted_metrics="accuracy: 0.9543",
        shap_features=[],
        anomaly_flags=[],
        per_class=[],
        overfit_analysis=None,
        selected_features=[],
        visual_descriptions={},
        imbalance_metadata={
            "class_distribution": {"0": "10.0%", "1": "90.0%"},
            "minority_class_percentage": 10.0,
            "class_weight_applied": True,
            "tool_used": "Stratified Split + Cost-Sensitive class_weight='balanced'",
        },
        imbalance_warning=None,
    )

    assert "class_weight='balanced'" in data_block
    assert "do NOT claim SMOTE/oversampling" in data_block
    assert "MINORITY class ('0')" in data_block
