import sys
import json
import asyncio
from core.report.narrative_generator import NarrativeGenerator

def main():
    gen = NarrativeGenerator()

    # Base parameters
    dataset_name = "custom_clinical_data.csv"
    task_type = "binary_classification"
    metrics = {
        "accuracy": 0.85, 
        "precision": 0.82,
        "recall": 0.88,
        "f1": 0.85,
        "ROC-AUC": 0.89
    }
    visuals_summary = {
        "roc_curve": {"description_fallback": "ROC curve showing good separation."},
        "confusion_matrix": {"description_fallback": "Confusion matrix showing balanced errors."}
    }
    selected_features = ["biomarker_a", "age", "blood_pressure", "gene_x"]
    
    # 1. Without context
    print("Generating report WITHOUT additional context...")
    try:
        res_no_context = gen.generate_narrative(
            dataset_name=dataset_name,
            task_type=task_type,
            metrics=metrics,
            visuals_summary=visuals_summary,
            selected_features=selected_features,
            additional_context=""
        )
    except Exception as e:
        print(f"Failed to generate without context: {e}")
        res_no_context = {}

    # 2. With context
    print("\nGenerating report WITH additional context...")
    context = "Research Goal: Predict 5-year survival in stage II patients. The most critical aspect is avoiding false negatives (maximizing recall) even at the cost of precision, because missed cases lead to mortality. 'biomarker_a' is an experimental novel protein we are testing."
    try:
        res_with_context = gen.generate_narrative(
            dataset_name=dataset_name,
            task_type=task_type,
            metrics=metrics,
            visuals_summary=visuals_summary,
            selected_features=selected_features,
            additional_context=context
        )
    except Exception as e:
        print(f"Failed to generate with context: {e}")
        res_with_context = {}

    # Save outputs
    with open("output_no_context.json", "w") as f:
        json.dump(res_no_context, f, indent=2)

    with open("output_with_context.json", "w") as f:
        json.dump(res_with_context, f, indent=2)

    print("\nDone. Saved to output_no_context.json and output_with_context.json.")
    
    # Print some comparisons
    print("\n--- COMPARISON ---")
    def print_section(title, no_ctx_dict, ctx_dict, section_key):
        print(f"\n[{title}]")
        expert_no = no_ctx_dict.get("expert", {}).get(section_key, "N/A")
        expert_yes = ctx_dict.get("expert", {}).get(section_key, "N/A")
        print("  >> WITHOUT CONTEXT:\n    " + expert_no.replace("\n", "\n    ")[:500] + ("..." if len(expert_no) > 500 else ""))
        print("  >> WITH CONTEXT:\n    " + expert_yes.replace("\n", "\n    ")[:500] + ("..." if len(expert_yes) > 500 else ""))

    print_section("Executive Summary", res_no_context, res_with_context, "executive_summary")
    print_section("Recommendations", res_no_context, res_with_context, "recommendations")
    print_section("Findings", res_no_context, res_with_context, "findings")

if __name__ == "__main__":
    main()
