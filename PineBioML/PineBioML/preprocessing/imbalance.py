import pandas as pd

def analyze_imbalance(y: pd.Series) -> dict:
    """Analyze class distribution and determine imbalance strategy."""
    class_counts = y.value_counts()
    class_proportions = y.value_counts(normalize=True)
    minority_pct = float(class_proportions.min() * 100)

    if minority_pct >= 45.0:
        strategy = "balanced"
        tool = "None (Standard Training)"
    elif minority_pct >= 20.0:
        strategy = "moderately_imbalanced"
        tool = "Stratified Split + Cost-Sensitive class_weight='balanced'"
    else:
        strategy = "severely_imbalanced"
        try:
            import imblearn
            tool = "Stratified Split + SMOTE Oversampling"
        except ImportError:
            tool = "Stratified Split + Cost-Sensitive class_weight='balanced' (SMOTE unavailable)"

    return {
        "class_distribution": {
            str(k): f"{v*100:.2f}% ({class_counts[k]} samples)"
            for k, v in class_proportions.items()
        },
        "minority_class_percentage": minority_pct,
        "imbalance_strategy": strategy,
        "tool_used": tool
    }
