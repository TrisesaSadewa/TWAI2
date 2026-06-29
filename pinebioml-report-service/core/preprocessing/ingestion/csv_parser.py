import pandas as pd
import os
from typing import Dict, Any

def parse_model_scores(path: str) -> Dict[str, Any]:
    """
    Parse model scores from all-model-result.csv (or similar).
    Extracts:
    - best model name (lowest sum of cv + test rank)
    - top 3 models
    - metric comparison table
    """
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}
        
    try:
        df = pd.read_csv(path)
        
        # Identify columns for cv rank and test rank
        cv_rank_col = next((col for col in df.columns if 'cv' in col.lower() and 'rank' in col.lower()), None)
        test_rank_col = next((col for col in df.columns if 'test' in col.lower() and 'rank' in col.lower()), None)
        
        # Identify model column
        model_col = next((col for col in df.columns if 'model' in col.lower()), df.columns[0])
        
        if cv_rank_col and test_rank_col:
            df['rank_sum'] = df[cv_rank_col] + df[test_rank_col]
            df_sorted = df.sort_values(by='rank_sum')
        elif cv_rank_col:
            df_sorted = df.sort_values(by=cv_rank_col)
        elif test_rank_col:
            df_sorted = df.sort_values(by=test_rank_col)
        else:
            # Fallback if no rank columns
            df_sorted = df

        best_model = df_sorted.iloc[0][model_col] if not df_sorted.empty else None
        top_3_models = df_sorted[model_col].head(3).tolist() if not df_sorted.empty else []
        
        # Clean up the dataframe before converting to dict (e.g. drop rank_sum if added)
        if 'rank_sum' in df.columns:
            df = df.drop(columns=['rank_sum'])
            
        metric_comparison = df.to_dict(orient='records')
        
        return {
            "best_model": best_model,
            "top_3_models": top_3_models,
            "metric_comparison": metric_comparison
        }
    except Exception as e:
        return {"error": str(e)}


def parse_features(path: str) -> Dict[str, Any]:
    """
    Parse features to extract top-ranked features per selection method.
    """
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}
        
    try:
        df = pd.read_csv(path)
        # Assume the first column is the feature name
        feature_col = df.columns[0]
        
        top_features_per_method = {}
        
        for col in df.columns[1:]:
            # Check if column represents a rank (lower is better) or score (higher is better)
            ascending = 'rank' in col.lower()
            
            # Sort by the method column and get top 10 features
            sorted_features = df.sort_values(by=col, ascending=ascending)
            top_10 = sorted_features[feature_col].head(10).tolist()
            
            top_features_per_method[col] = top_10
            
        return {
            "top_features": top_features_per_method
        }
    except Exception as e:
        return {"error": str(e)}


def parse_optimal_scores(path: str) -> Dict[str, Any]:
    """
    Extract precision, recall, F1, accuracy, sensitivity, specificity.
    """
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}
        
    try:
        df = pd.read_csv(path)
        
        target_metrics = ['precision', 'recall', 'f1', 'accuracy', 'sensitivity', 'specificity']
        extracted_data = {}
        
        model_col = next((col for col in df.columns if 'model' in col.lower()), None)
        
        for metric in target_metrics:
            # Find matching column for the metric
            col_match = next((col for col in df.columns if metric in col.lower()), None)
            
            if col_match:
                if model_col:
                    extracted_data[metric] = dict(zip(df[model_col], df[col_match]))
                else:
                    extracted_data[metric] = df[col_match].tolist()
                    
        # Fallback for long-format data (e.g. 'metric', 'value' columns)
        if not extracted_data:
            metric_col = next((col for col in df.columns if 'metric' in col.lower()), None)
            val_col = next((col for col in df.columns if 'value' in col.lower() or 'score' in col.lower()), None)
            
            if metric_col and val_col:
                for _, row in df.iterrows():
                    m = str(row[metric_col]).lower()
                    if any(target in m for target in target_metrics):
                        extracted_data[m] = row[val_col]
        
        # If we couldn't parse it specifically, just return all rows
        if not extracted_data:
            return {"raw_data": df.to_dict(orient='records')}
            
        return extracted_data
    except Exception as e:
        return {"error": str(e)}


def parse_shap_values(path: str) -> Dict[str, Any]:
    """
    Extract the top 5 positive and negative influencers per class.
    Integrate directionality (High/Low) into the prompt context.
    """
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}
        
    try:
        df = pd.read_csv(path)
        
        # Try to identify standard columns
        feature_col = next((col for col in df.columns if 'feature' in col.lower() or 'name' in col.lower()), df.columns[0])
        val_col = next((col for col in df.columns if 'shap' in col.lower() or 'value' in col.lower() or 'importance' in col.lower() or 'impact' in col.lower()), None)
        class_col = next((col for col in df.columns if 'class' in col.lower() or 'target' in col.lower()), None)
        
        # If no explicit value column found, assume the second column is the value
        if not val_col and len(df.columns) > 1:
            val_col = df.columns[1]
            
        shap_data = {}
        
        if class_col:
            for cls in df[class_col].unique():
                cls_df = df[df[class_col] == cls]
                
                pos_df = cls_df[cls_df[val_col] > 0].sort_values(by=val_col, ascending=False).head(5)
                neg_df = cls_df[cls_df[val_col] < 0].sort_values(by=val_col, ascending=True).head(5)
                
                shap_data[str(cls)] = {
                    "top_positive_influencers": [{"feature": row[feature_col], "impact_score": row[val_col], "direction": "High"} for _, row in pos_df.iterrows()],
                    "top_negative_influencers": [{"feature": row[feature_col], "impact_score": row[val_col], "direction": "Low"} for _, row in neg_df.iterrows()]
                }
        else:
            pos_df = df[df[val_col] > 0].sort_values(by=val_col, ascending=False).head(5)
            neg_df = df[df[val_col] < 0].sort_values(by=val_col, ascending=True).head(5)
            
            shap_data["overall"] = {
                "top_positive_influencers": [{"feature": row[feature_col], "impact_score": row[val_col], "direction": "High"} for _, row in pos_df.iterrows()],
                "top_negative_influencers": [{"feature": row[feature_col], "impact_score": row[val_col], "direction": "Low"} for _, row in neg_df.iterrows()]
            }
            
        return shap_data
    except Exception as e:
        return {"error": str(e)}
