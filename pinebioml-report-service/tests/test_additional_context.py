import pytest
from core.report.narrative_generator import NarrativeGenerator

def test_clinical_context_combines_auto_and_user_context():
    gen = NarrativeGenerator()
    
    # Case 1: Auto-detected dataset name + user additional context
    context = gen._build_clinical_context(
        dataset_name="breast_cancer_wisconsin",
        task_type="binary_classification",
        selected_features=["radius_mean", "texture_mean"],
        additional_context="• Research Goal: Predict 5-year survival in stage II patients\n• Cohort: 120 patients from NCU Hospital"
    )
    
    assert "AUTO-DETECTED CLINICAL DOMAIN:" in context
    assert "breast tumor malignancy" in context
    assert "USER-PROVIDED STUDY METADATA & OBJECTIVES:" in context
    assert "Predict 5-year survival" in context

def test_clinical_context_only_auto_detected_when_no_user_context():
    gen = NarrativeGenerator()
    
    context = gen._build_clinical_context(
        dataset_name="Breast Cancer Wisconsin",
        task_type="binary_classification",
        selected_features=["radius_mean", "texture_mean"],
        additional_context=""
    )
    
    assert "AUTO-DETECTED CLINICAL DOMAIN:" in context
    assert "USER-PROVIDED STUDY METADATA & OBJECTIVES:" not in context

def test_clinical_context_only_user_context_when_unrecognized_dataset():
    gen = NarrativeGenerator()
    
    context = gen._build_clinical_context(
        dataset_name="custom_genomics_xyz.csv",
        task_type="binary_classification",
        selected_features=["gene_A", "gene_B"],
        additional_context="• Disease: Acute Myeloid Leukemia\n• Objective: Identify gene markers"
    )
    
    assert "AUTO-DETECTED CLINICAL DOMAIN:" in context
    assert "USER-PROVIDED STUDY METADATA & OBJECTIVES:" in context
    assert "Acute Myeloid Leukemia" in context
