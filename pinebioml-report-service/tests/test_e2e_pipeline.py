import pytest
import sys
import os
import pandas as pd
import numpy as np
import json
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from workers.ml_pipeline_runner import run_dynamic_pipeline

@pytest.fixture
def dirty_dataset(tmp_path):
    """
    Creates a highly imbalanced dataset with missing values and noise.
    Returns the path to the CSV file.
    """
    np.random.seed(42)
    
    # 50 rows
    n_samples = 50
    
    # Highly imbalanced target: 95% class 0, 5% class 1
    target = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    # Meaningful features
    f1 = target * 5 + np.random.normal(0, 1, n_samples)
    f2 = target * -3 + np.random.normal(0, 2, n_samples)
    
    # Noise features
    noise1 = np.random.uniform(0, 100, n_samples)
    noise2 = np.random.normal(50, 10, n_samples)
    
    df = pd.DataFrame({
        'feature_1': f1,
        'feature_2': f2,
        'noise_1': noise1,
        'noise_2': noise2,
        'target_col': target
    })
    
    # Introduce NaNs (Missing values)
    nan_indices = np.random.choice(df.index, size=5, replace=False)
    df.loc[nan_indices, 'feature_1'] = np.nan
    
    nan_indices2 = np.random.choice(df.index, size=3, replace=False)
    df.loc[nan_indices2, 'noise_1'] = np.nan
    
    file_path = tmp_path / "dirty_dataset.csv"
    df.to_csv(file_path, index=False)
    
    return str(file_path)

def test_ml_pipeline_edge_cases(dirty_dataset, tmp_path):
    """
    Test that the PineBioML pipeline handles edge cases gracefully:
    - Missing Values
    - Imbalanced Classes
    - Noise Features
    """
    report_id = "test_edge_case_report"
    output_dir = str(tmp_path / "output")
    
    settings = {
        "modeling_methods": ["lr"],
        "missing_value_methods": ["Mean"],
        "normalization_methods": ["StandardScaler"],
        "feature_selection_methods": ["SelectKBest"],
        "k_fold": 2  # Small fold for fast testing
    }
    
    # The pipeline should not crash
    result = run_dynamic_pipeline(
        report_id=report_id,
        dataset_path=dirty_dataset,
        target_col="target_col",
        settings=settings,
        output_dir=output_dir
    )
    
    assert result
    
    # Verify outputs are generated
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "classification_report.json"))
    assert os.path.exists(os.path.join(output_dir, "All-model-result.csv"))
    
    # Check that plots are generated
    assert os.path.exists(os.path.join(output_dir, "_Confusion Matrix.png"))
    assert os.path.exists(os.path.join(output_dir, "_ROC Curve.png"))
    
    # Ensure classification report contains proper metrics
    with open(os.path.join(output_dir, "classification_report.json"), "r") as f:
        cr = json.load(f)
        assert "accuracy" in cr
        assert "macro avg" in cr

@pytest.fixture
def regression_dataset(tmp_path):
    """
    Creates a regression dataset with continuous targets (many unique values).
    """
    np.random.seed(42)
    n_samples = 50
    f1 = np.random.normal(0, 1, n_samples)
    f2 = np.random.normal(0, 2, n_samples)
    # Ensure there are > 10 unique values so it's detected as regression
    target = 3.5 * f1 - 1.2 * f2 + np.random.normal(0, 0.5, n_samples)
    
    df = pd.DataFrame({
        'feature_1': f1,
        'feature_2': f2,
        'target_col': target
    })
    
    file_path = tmp_path / "regression_dataset.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_ml_pipeline_regression(regression_dataset, tmp_path):
    report_id = "test_regression_report"
    output_dir = str(tmp_path / "output_reg")
    
    settings = {
        "modeling_methods": ["lr"],
        "missing_value_methods": ["Mean"],
        "normalization_methods": ["StandardScaler"],
        "feature_selection_methods": ["SelectKBest"],
        "k_fold": 2
    }
    
    result = run_dynamic_pipeline(
        report_id=report_id,
        dataset_path=regression_dataset,
        target_col="target_col",
        settings=settings,
        output_dir=output_dir
    )
    
    assert result
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "regression_report.json"))
    assert os.path.exists(os.path.join(output_dir, "_True vs Predicted.png"))
    assert os.path.exists(os.path.join(output_dir, "_Residuals.png"))
    
    with open(os.path.join(output_dir, "regression_report.json"), "r") as f:
        rr = json.load(f)
        assert "MSE" in rr
        assert "R2" in rr
