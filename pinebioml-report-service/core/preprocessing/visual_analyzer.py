import os
import base64
import logging
import requests
from core.config import settings

logger = logging.getLogger(__name__)

class VisualAnalyzer:
    """
    Handles extracting visual artifacts, base64 encoding them, 
    and generating automated text descriptions for model plots.
    Supports optional Ollama vision-model (llava/moondream) for richer analysis.
    """
    
    # Plot types that benefit from a specific SHAP/importance-aware description
    _RICH_DESCRIPTIONS = {
        "confusion_matrix_png": (
            "A Confusion Matrix showing the distribution of true positive, true negative, "
            "false positive, and false negative predictions. It displays the classification performance, "
            "indicating whether the model has high sensitivity and specificity, and reveals any specific "
            "class imbalances or misclassifications."
        ),
        "roc_curve_png": (
            "A Receiver Operating Characteristic (ROC) curve plotting the True Positive Rate (Sensitivity) "
            "against the False Positive Rate (1 - Specificity) across different thresholds. The Area Under the "
            "Curve (AUC) represents the model's overall discriminative power, where 0.5 is random chance and 1.0 is perfect classification."
        ),
        "feature_importance_png": (
            "A ranked bar chart of the top predictive features identified by the ML pipeline. "
            "Each bar represents how much a given feature contributes to the model's predictions — "
            "longer bars indicate stronger predictive power. Features at the top are the primary drivers "
            "of the classification decision and are the most clinically actionable for follow-up investigation."
        ),
        "shap_plot_png": (
            "A SHAP (SHapley Additive exPlanations) summary plot illustrating the impact of each feature "
            "on individual predictions. Each point represents one sample; the horizontal position shows "
            "whether the feature pushed the prediction higher (positive, right) or lower (negative, left). "
            "Color indicates the feature value (red = high, blue = low), revealing non-linear interaction effects "
            "between feature magnitude and predictive direction."
        ),
        "pca_plot_png": (
            "A 2D Principal Component Analysis (PCA) projection of the high-dimensional biological/clinical data. "
            "It displays cluster separation along the two main components (PC1 and PC2) which account for the highest variance, "
            "illustrating the natural separability of the sample groups."
        ),
        "umap_plot_png": (
            "A Uniform Manifold Approximation and Projection (UMAP) visualization. "
            "This non-linear dimensionality reduction technique preserves local structures and highlights clear clusters "
            "and phenotypic sub-groups within the high-dimensional dataset."
        ),
        "pls_plot_png": (
            "A Partial Least Squares (PLS) score plot, showing how well the supervised dimensionality reduction "
            "separates sample groups along the latent variables designed to maximize class covariance."
        ),
        "corr_heatmap_png": (
            "A correlation heatmap illustrating pairwise relationships between high-impact features. "
            "Strong positive correlations are colored in deep warm hues, while strong negative correlations are in cool hues, "
            "helping identify redundant features and multi-collinearity."
        )
    }

    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """Helper to read and encode an image to base64."""
        if not os.path.exists(image_path):
            logger.warning(f"Artifact image not found: {image_path}")
            return ""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to base64 encode {image_path}: {e}")
            return ""

    def _describe_via_vision_model(self, b64_str: str, plot_key: str, progress_callback=None) -> str:
        """
        Query the local Ollama vision model to generate a
        clinically-relevant description of the plot image.
        Returns empty string if the call fails or Ollama is unavailable.
        """
        if not b64_str:
            return ""
        if not settings.ENABLE_VISION_ANALYSIS:
            return ""
        try:
            vision_model = settings.VISION_MODEL  
            
            # Check if model is locally installed first to prevent download/loading hangs
            tags_url = f"{settings.OLLAMA_BASE_URL}/api/tags"
            try:
                tags_resp = requests.get(tags_url, timeout=3)
                if tags_resp.status_code == 200:
                    models = tags_resp.json().get("models", [])
                    model_names = [m.get("name") for m in models]
                    # Check both exact name and base name
                    found = False
                    for name in model_names:
                        if name == vision_model or name.split(":")[0] == vision_model.split(":")[0]:
                            found = True
                            break
                    if not found:
                        logger.info(f"Vision model '{vision_model}' not found in local Ollama tags. Skipping vision enhancement.")
                        return ""
            except Exception as tags_err:
                logger.warning(f"Could not connect to Ollama tags endpoint: {tags_err}. Skipping vision enhancement.")
                return ""

            if progress_callback:
                plot_name = plot_key.replace("_png", "").replace("_", " ").title()
                progress_callback(30, f"Analyzing {plot_name} chart...", current_model=vision_model)

            ollama_url = f"{settings.OLLAMA_BASE_URL}/api/generate"
            plot_name = plot_key.replace("_png", "").replace("_", " ").title()
            prompt = (
                f"You are a clinical data scientist analyzing a machine learning validation report. "
                f"This image is a '{plot_name}' plot from an ML pipeline. "
                f"Describe in 2-4 precise sentences: what patterns, values, or anomalies are visible, "
                f"what they mean for model quality, and any clinical implications. "
                f"Be specific — mention actual visible numbers, class labels, or curve shapes if present."
            )
            payload = {
                "model": vision_model,
                "prompt": prompt,
                "images": [b64_str],
                "stream": False
            }
            resp = requests.post(
                ollama_url,
                json=payload,
                timeout=settings.VISION_ANALYSIS_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            resp_text = resp.json().get("response", "").strip()

            # Clean up repeating markdown code block loops and duplicate paragraphs
            paragraphs = []
            seen_paras = set()
            backtick_consec = 0

            for p in resp_text.split('\n\n'):
                p_strip = p.strip()
                if not p_strip:
                    continue

                # Deduplicate repeating markdown blocks or loops
                if all(c == '`' for c in p_strip) or p_strip.startswith('```'):
                    backtick_consec += 1
                    if backtick_consec > 2:
                        break
                else:
                    backtick_consec = 0

                # Near-duplicate paragraph check using normalized suffix
                p_norm = "".join(c for c in p_strip.lower() if c.isalnum())
                p_norm = p_norm.replace("ampx27", "").replace("x27", "")
                
                suffix = p_norm[-50:] if len(p_norm) >= 50 else p_norm
                is_dup = False
                if suffix:
                    for s in seen_paras:
                        if suffix in s:
                            is_dup = True
                            break
                if is_dup:
                    continue

                seen_paras.add(p_norm)
                paragraphs.append(p_strip)

            return '\n\n'.join(paragraphs).strip()
        except Exception as e:
            logger.warning(f"Vision model analysis failed for {plot_key}: {e}")
            return ""

    def analyze_plots(self, artifacts: dict, progress_callback=None) -> dict:
        """
        Analyze and describe visual artifacts from the job run.
        Uses rich rule-based descriptions for known plot types, and optionally
        queries the Ollama vision model for feature importance and SHAP plots.
        """
        results = {}
        
        vision_enhance_keys = {
            "confusion_matrix_png",
            "roc_curve_png",
            "feature_importance_png",
            "shap_plot_png",
            "pca_plot_png",
            "umap_plot_png",
            "pls_plot_png",
            "corr_heatmap_png"
        }

        for key, path in artifacts.items():
            if not path or not (path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg")):
                continue
                
            # Base64 encode
            b64_str = self.encode_image_to_base64(path)

            # Start with the rich rule-based description
            description = self._rich_descriptions.get(key) or self._get_plot_description_fallback(key)

            # For feature importance and SHAP, try to enrich with vision model
            if key in vision_enhance_keys and b64_str:
                vision_desc = self._describe_via_vision_model(b64_str, key, progress_callback=progress_callback)
                if vision_desc:
                    description = vision_desc
            
            results[key] = {
                "path": path,
                "base64": b64_str[:100] + "..." if b64_str else "",
                "description_fallback": description
            }
            
        return results

    @property
    def _rich_descriptions(self):
        return self._RICH_DESCRIPTIONS

    def _get_plot_description_fallback(self, plot_type: str) -> str:
        """
        Provides structural descriptions for standard classification/regression plots.
        """
        return self._RICH_DESCRIPTIONS.get(
            plot_type,
            "A model validation plot showing experiment results. "
            "Inspect the chart for outliers, class separation, and performance indicators."
        )

