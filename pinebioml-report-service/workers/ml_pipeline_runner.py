import os
import sys
import pandas as pd
import structlog
import json
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# pyrefly: ignore [missing-import]
from PineBioML.model.utils import Pine, sklearn_esitimator_wrapper
from core.config import settings as app_settings

# Monkeypatch PineBioML's sklearn_esitimator_wrapper to properly support regression
try:
    def _patched_dir(self):
        d = set(dir(type(self)) + list(self.__dict__.keys()))
        if hasattr(self, 'is_regression') and self.is_regression() and 'predict_proba' in d:
            d.remove('predict_proba')
        return list(d)
        
    sklearn_esitimator_wrapper.__dir__ = _patched_dir
except Exception:
    pass

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    SGDClassifier, ElasticNet, Lasso,
)
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler,
    Normalizer, PowerTransformer,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from catboost import CatBoostClassifier, CatBoostRegressor
from PineBioML.selection.classification import ensemble_selector as cls_ensemble_selector
from PineBioML.selection.regression import ensemble_selector as reg_ensemble_selector
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    SelectFromModel,
)
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score

logger = structlog.get_logger(__name__)

def run_dynamic_pipeline(
    report_id: str,
    dataset_path: str,
    target_col: str,
    settings: dict,
    output_dir: str
):
    """
    Run PineBioML pipeline with given settings and generate output plots.
    """
    logger.info(f"Starting dynamic pipeline for {report_id} with dataset {dataset_path}")
    
    # Load the dataset
    if dataset_path == "pd" or dataset_path.endswith("pd"):
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df[target_col] = data.target
    elif dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.tsv') or dataset_path.endswith('.txt'):
        df = pd.read_csv(dataset_path, sep='\t')
    elif dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
        df = pd.read_excel(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")
        
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
        
    # Drop rows where target is NaN
    df = df.dropna(subset=[target_col])
    
    train_y = df[target_col]
    train_x = df.drop(columns=[target_col])
    # Drop completely empty columns (e.g. from trailing commas in CSV) to prevent downstream pipeline crashes
    train_x = train_x.dropna(axis=1, how='all')
    
    train_y_orig = train_y.copy()
    
    # Auto-detect task type
    is_regression = pd.api.types.is_numeric_dtype(train_y) and train_y.nunique() > 10
    logger.info(f"Task type detected as: {'Regression' if is_regression else 'Classification'}")

    target_le = None
    if not is_regression and (train_y.dtype == 'object' or train_y.dtype.name == 'category' or train_y.dtype == 'string'):
        target_le = LabelEncoder()
        train_y = pd.Series(target_le.fit_transform(train_y.astype(str)), index=train_y.index, name=train_y.name)
        
    for col in train_x.columns:
        if train_x[col].dtype == 'object' or train_x[col].dtype.name == 'category' or train_x[col].dtype == 'string':
            le_x = LabelEncoder()
            mask = train_x[col].notna()
            train_x.loc[mask, col] = le_x.fit_transform(train_x.loc[mask, col].astype(str))
            train_x[col] = pd.to_numeric(train_x[col])
    
    # Pre-select numeric columns for plotting
    numeric_x = train_x.select_dtypes(include=['number'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate exploratory plots (Pre-training)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA
        from sklearn.cross_decomposition import PLSRegression
        from umap import UMAP
        import plotly.express as px
        import numpy as np
        
        def _save_fig(fig, filename: str):
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=160)
            plt.close(fig)
        
        # Correlation Heatmap. Limit very wide datasets to the most informative
        # features so labels and color values remain readable in the result page.
        corr_source = numeric_x.dropna(axis=1, how='all')
        corr_source = corr_source.loc[:, corr_source.nunique(dropna=True) > 1]
        if corr_source.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.text(0.5, 0.5, "Correlation heatmap not available\nNo numeric feature columns with variance.", ha="center", va="center")
            ax.axis("off")
            _save_fig(fig, "_Correlation Heatmap.png")
        else:
            max_heatmap_features = 24
            if corr_source.shape[1] > max_heatmap_features:
                corr_abs = corr_source.corr().abs()
                np.fill_diagonal(corr_abs.values, 0)
                ranked_cols = corr_abs.sum().sort_values(ascending=False).head(max_heatmap_features).index
                corr_source = corr_source[ranked_cols]
            corr_matrix = corr_source.corr()
            fig_size = max(8, min(16, 0.45 * len(corr_matrix.columns) + 4))
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
            sns.heatmap(
                corr_matrix,
                vmin=-1,
                vmax=1,
                center=0,
                cmap='RdBu_r',
                square=True,
                linewidths=0.25,
                linecolor="white",
                cbar_kws={"label": "Pearson correlation"},
                ax=ax
            )
            title_suffix = " (top correlated features)" if numeric_x.shape[1] > max_heatmap_features else ""
            ax.set_title(f"Correlation Heatmap{title_suffix}")
            ax.tick_params(axis='x', labelrotation=45, labelsize=7)
            ax.tick_params(axis='y', labelrotation=0, labelsize=7)
            _save_fig(fig, "_Correlation Heatmap.png")
        
        # PCA
        numeric_x_clean = numeric_x.dropna(axis=1, how='all')
        numeric_x_imputed = numeric_x_clean.fillna(numeric_x_clean.mean())
        pca = PCA(n_components=2)
        scaled_x = (numeric_x_imputed - numeric_x_imputed.mean()) / (numeric_x_imputed.std() + 1e-4)
        pcs = pca.fit_transform(scaled_x)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=train_y_orig.astype(str), s=34, alpha=0.85, edgecolor="none", ax=ax)
        ax.set_title("PCA scatter plot")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
        ax.grid(alpha=0.2)
        ax.legend(title=str(target_col), loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "_PCA.png"), bbox_inches='tight', dpi=160)
        plt.close(fig)
        
        fig_pca = px.scatter(x=pcs[:,0], y=pcs[:,1], color=train_y_orig.astype(str), title="PCA Plot")
        fig_pca.write_html(os.path.join(output_dir, "pca_plot.html"))
        
        # UMAP
        reducer = UMAP(n_neighbors=max(2, round(np.log2(numeric_x_imputed.shape[0]))), n_components=2, random_state=42)
        umap_emb = reducer.fit_transform(scaled_x)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=umap_emb[:,0], y=umap_emb[:,1], hue=train_y_orig.astype(str), s=34, alpha=0.85, edgecolor="none", ax=ax)
        ax.set_title("UMAP scatter plot")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(alpha=0.2)
        ax.legend(title=str(target_col), loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "_UMAP.png"), bbox_inches='tight', dpi=160)
        plt.close(fig)
        
        fig_umap = px.scatter(x=umap_emb[:,0], y=umap_emb[:,1], color=train_y_orig.astype(str), title="UMAP Plot")
        fig_umap.write_html(os.path.join(output_dir, "umap_plot.html"))
        
        # PLS
        train_y_numeric = train_y.to_numpy().reshape(-1, 1) if is_regression else pd.get_dummies(train_y).values
        pls_components = min(2, numeric_x_imputed.shape[1], max(1, train_y_numeric.shape[1]))
        pls = PLSRegression(n_components=pls_components)
        pls.fit(numeric_x_imputed, train_y_numeric)
        pls_emb = pls.transform(numeric_x_imputed)
        if pls_emb.shape[1] == 1:
            pls_emb = np.column_stack([pls_emb[:, 0], np.zeros(len(pls_emb))])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=pls_emb[:,0], y=pls_emb[:,1], hue=train_y_orig.astype(str), s=34, alpha=0.85, edgecolor="none", ax=ax)
        ax.set_title("PLS scatter plot")
        ax.set_xlabel("PLS component 1")
        ax.set_ylabel("PLS component 2")
        ax.grid(alpha=0.2)
        ax.legend(title=str(target_col), loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "_PLS.png"), bbox_inches='tight', dpi=160)
        plt.close(fig)
        
        fig_pls = px.scatter(x=pls_emb[:,0], y=pls_emb[:,1], color=train_y_orig.astype(str), title="PLS Plot")
        fig_pls.write_html(os.path.join(output_dir, "pls_plot.html"))
        
    except Exception as e:
        logger.warning(f"Error generating exploratory plots: {e}")
        
    # Configure PineBioML Experiment Models.
    # NOTE: the UI (setting.html) submits each method's display name as the form
    # value (e.g. "RandomForest", "Support Vector Machine(SVM)"), and main.py
    # passes them through unchanged. So every check here must match the UI's
    # exact value= string — NOT a lowercase code. The dict key becomes the
    # method name recorded in Pine's result rows (shown in the optimal-model
    # table), so using the display name keeps the report human-readable.
    modeling_methods = settings.get('modeling_methods', [])
    model_dict = {}

    # methods shared across task types (class/reg variants selected below)
    has = lambda *keys: any(k in modeling_methods for k in keys) or "modeling_all" in modeling_methods

    # ── Class-imbalance detection (classification only) ──────────────────────
    # Reuse PineBioML's analyzer. When the minority class is <45% of the data,
    # pass class_weight='balanced' to estimators that support it (RF, Logit, SVM,
    # DecisionTree, AdaBoost, LightGBM) and scale_pos_weight to XGBoost. This
    # stops accuracy from looking good while the minority class is ignored.
    imbalance_metadata = None
    cw = None            # class_weight arg (None when balanced/no imbalance)
    xgb_spw = None       # XGBoost scale_pos_weight
    if not is_regression:
        try:
            from PineBioML.preprocessing.imbalance import analyze_imbalance
            imbalance_metadata = analyze_imbalance(train_y)
        except Exception as _e:
            logger.warning(f"analyze_imbalance unavailable ({_e}); proceeding without imbalance handling.")
            imbalance_metadata = None
        if imbalance_metadata and imbalance_metadata.get("minority_class_percentage", 100) < 45:
            cw = "balanced"
            _vc = train_y.value_counts()
            _pos = _vc.min()
            _neg = _vc.max()
            if _pos and _neg:
                xgb_spw = _neg / _pos
            logger.info(f"Imbalance detected ({imbalance_metadata.get('imbalance_strategy')}): "
                        f"applying class_weight='balanced'{', scale_pos_weight=' + str(round(xgb_spw, 2)) if xgb_spw else ''}.")

    try:
        validation_method = settings.get("validation_method", "k-fold cross validation")
        evaluate_ncv = -1 if validation_method == "Leave-one-out cross validation" else int(settings.get("k_fold", "5"))
    except ValueError:
        evaluate_ncv = 5

    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, ParameterGrid

    tuning_strategy = settings.get("tuning_strategy", "RandomizedSearchCV")
    tuning_n_iter = settings.get("tuning_n_iter", 10)
    
    def tune_model(estimator, param_grid):
        if not param_grid or tuning_strategy == "None":
            return sklearn_esitimator_wrapper(estimator)
        
        cv_folds = evaluate_ncv if evaluate_ncv != -1 else 3 # Fallback to 3 if LOOCV is selected to prevent freezing
        
        if tuning_strategy == "GridSearchCV":
            search = GridSearchCV(estimator, param_grid, cv=cv_folds, n_jobs=-1)
        elif tuning_strategy == "BayesianOptimization":
            from skopt import BayesSearchCV
            # Convert list to tuple to ensure skopt treats it as categorical choices rather than trying to infer mixed numerical ranges
            skopt_grid = {k: tuple(v) if isinstance(v, list) else v for k, v in param_grid.items()}
            search = BayesSearchCV(estimator, skopt_grid, n_iter=tuning_n_iter, cv=cv_folds, n_jobs=-1, random_state=42)
        else: # RandomizedSearchCV
            grid_size = len(list(ParameterGrid(param_grid)))
            n_iter = min(tuning_n_iter, grid_size)
            search = RandomizedSearchCV(estimator, param_grid, n_iter=n_iter, cv=cv_folds, n_jobs=-1, random_state=42)
        
        return sklearn_esitimator_wrapper(search)

    if is_regression:
        if has("RandomForest"):
            model_dict["RandomForest"] = tune_model(RandomForestRegressor(random_state=42), {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]})
        if has("LogisticRegression"):
            model_dict["LogisticRegression"] = tune_model(LinearRegression(), {})
        if has("ElasticLogit"):
            model_dict["ElasticLogit"] = tune_model(ElasticNet(max_iter=5000), {"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]})
        if has("Support Vector Machine(SVM)"):
            model_dict["Support Vector Machine(SVM)"] = tune_model(SVR(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]})
        if has("KNN"):
            model_dict["KNN"] = tune_model(KNeighborsRegressor(), {"n_neighbors": [3, 5, 7, 9]})
        if has("MLP"):
            model_dict["MLP"] = tune_model(MLPRegressor(max_iter=1000), {"hidden_layer_sizes": [(50,), (100,), (50,50)], "alpha": [0.0001, 0.001]})
        if has("XGBoost"):
            model_dict["XGBoost"] = tune_model(XGBRegressor(random_state=42), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]})
        if has("LightGBM"):
            model_dict["LightGBM"] = tune_model(LGBMRegressor(verbose=-1, random_state=42), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]})
        if has("AdaBoost"):
            model_dict["AdaBoost"] = tune_model(AdaBoostRegressor(random_state=42), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]})
        if has("CatBoost"):
            model_dict["CatBoost"] = tune_model(CatBoostRegressor(verbose=False, random_state=42), {"iterations": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]})
        if has("DecisionTree"):
            model_dict["DecisionTree"] = tune_model(DecisionTreeRegressor(random_state=42), {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]})

        if not model_dict:
            model_dict["RandomForest"] = tune_model(RandomForestRegressor(random_state=42), {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]})
    else:
        if has("RandomForest"):
            model_dict["RandomForest"] = tune_model(RandomForestClassifier(class_weight=cw, random_state=42), {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]})
        if has("LogisticRegression"):
            model_dict["LogisticRegression"] = tune_model(LogisticRegression(max_iter=1000, class_weight=cw), {"C": [0.01, 0.1, 1, 10]})
        if has("ElasticLogit"):
            model_dict["ElasticLogit"] = tune_model(SGDClassifier(loss="log_loss", penalty="elasticnet", max_iter=1000, class_weight=cw), {"alpha": [0.0001, 0.001, 0.01], "l1_ratio": [0.2, 0.5, 0.8]})
        if has("Support Vector Machine(SVM)"):
            model_dict["Support Vector Machine(SVM)"] = tune_model(SVC(probability=True, class_weight=cw), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]})
        if has("KNN"):
            model_dict["KNN"] = tune_model(KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9]})
        if has("MLP"):
            model_dict["MLP"] = tune_model(MLPClassifier(max_iter=1000), {"hidden_layer_sizes": [(50,), (100,), (50,50)], "alpha": [0.0001, 0.001]})
        if has("XGBoost"):
            model_dict["XGBoost"] = tune_model(XGBClassifier(eval_metric="logloss", scale_pos_weight=xgb_spw, random_state=42), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]})
        if has("LightGBM"):
            model_dict["LightGBM"] = tune_model(LGBMClassifier(verbose=-1, class_weight=cw, random_state=42), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]})
        if has("AdaBoost"):
            model_dict["AdaBoost"] = tune_model(AdaBoostClassifier(random_state=42), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]})
        if has("CatBoost"):
            cb_kwargs = {"verbose": False, "random_state": 42}
            if cw == "balanced":
                cb_kwargs["auto_class_weights"] = "Balanced"
            model_dict["CatBoost"] = tune_model(CatBoostClassifier(**cb_kwargs), {"iterations": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]})
        if has("DecisionTree"):
            model_dict["DecisionTree"] = tune_model(DecisionTreeClassifier(class_weight=cw, random_state=42), {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]})

        if not model_dict:
            # Default to RF if nothing valid selected
            model_dict["RandomForest"] = tune_model(RandomForestClassifier(class_weight=cw, random_state=42), {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]})
        
    class PandasTransformerWrapper:
        def __init__(self, transformer):
            self.transformer = transformer

        @staticmethod
        def _ensure_finite(X):
            # Many downstream transformers (SelectKBest, Lasso, LinearSVC,
            # SelectFromModel, scalers) reject NaN. When the user picked
            # missing=None (do-not-impute) NaNs flow through to the next stage
            # and crash the whole experiment. Defensively fill any remaining
            # NaNs with a column median so a transformer never sees NaN
            # regardless of whether imputation ran upstream. (Purely a
            # safety net; when imputation did run, this is a no-op.)
            import pandas as pd
            if hasattr(X, "isna") and X.isna().any().any():
                fill = X.median(numeric_only=True)
                X = X.fillna(fill)
                # Any all-NaN columns (median is NaN) -> fill with 0
                if hasattr(X, "isna") and X.isna().any().any():
                    X = X.fillna(0)
            return X

        def fit_transform(self, X, y=None):
            import pandas as pd
            X = self._ensure_finite(X)
            if y is not None:
                try:
                    res = self.transformer.fit_transform(X, y)
                except TypeError:
                    res = self.transformer.fit_transform(X)
            else:
                res = self.transformer.fit_transform(X)
            if isinstance(res, pd.DataFrame):
                return res
            if hasattr(self.transformer, "get_feature_names_out"):
                try:
                    cols = self.transformer.get_feature_names_out(X.columns)
                except Exception:
                    cols = [f"feat_{i}" for i in range(res.shape[1])]
            else:
                cols = [f"feat_{i}" for i in range(res.shape[1])]
            return pd.DataFrame(res, index=X.index, columns=cols)
        def transform(self, X):
            import pandas as pd
            X = self._ensure_finite(X)
            res = self.transformer.transform(X)
            if isinstance(res, pd.DataFrame):
                return res
            if hasattr(self.transformer, "get_feature_names_out"):
                try:
                    cols = self.transformer.get_feature_names_out(X.columns)
                except Exception:
                    cols = [f"feat_{i}" for i in range(res.shape[1])]
            else:
                cols = [f"feat_{i}" for i in range(res.shape[1])]
            return pd.DataFrame(res, index=X.index, columns=cols)

    class _PassThrough:
        """Identity transformer: returns X unchanged. Used for "None" options
        (do-not-impute / do-not-normalize) so they fork and appear in results."""
        def fit_transform(self, X, y=None):
            return X
        def transform(self, X):
            return X
        def get_feature_names_out(self, input_features=None):
            return input_features
            
    missing_methods = settings.get('missing_value_methods', [])

    missing_dict = {}
    if "Mean" in missing_methods or "missing_all" in missing_methods:
        missing_dict["Mean"] = PandasTransformerWrapper(SimpleImputer(strategy="mean"))
    if "Median" in missing_methods or "missing_all" in missing_methods:
        missing_dict["Median"] = PandasTransformerWrapper(SimpleImputer(strategy="median"))
    if "Constant" in missing_methods or "missing_all" in missing_methods:
        # "Constant" in this service maps to most_frequent (the most common value
        # per column) — a sensible constant fill for mixed feature types.
        missing_dict["Constant"] = PandasTransformerWrapper(SimpleImputer(strategy="most_frequent"))
    if "KNN" in missing_methods or "missing_all" in missing_methods:
        missing_dict["KNN"] = PandasTransformerWrapper(KNNImputer())
    if "Iterative" in missing_methods or "missing_all" in missing_methods:
        missing_dict["Iterative"] = PandasTransformerWrapper(IterativeImputer(random_state=42))
    if "None" in missing_methods or "missing_all" in missing_methods:
        missing_dict["None"] = PandasTransformerWrapper(_PassThrough())
        
    norm_methods = settings.get('normalization_methods', [])
    norm_dict = {}
    if "StandardScaler" in norm_methods or "norm_all" in norm_methods:
        norm_dict["StandardScaler"] = PandasTransformerWrapper(StandardScaler())
    if "MinMaxScaler" in norm_methods or "norm_all" in norm_methods:
        norm_dict["MinMaxScaler"] = PandasTransformerWrapper(MinMaxScaler())
    if "RobustScaler" in norm_methods or "norm_all" in norm_methods:
        norm_dict["RobustScaler"] = PandasTransformerWrapper(RobustScaler())
    if "Normalizer" in norm_methods or "norm_all" in norm_methods:
        # Normalizer scales each ROW (sample) to unit norm, not each column.
        norm_dict["Normalizer"] = PandasTransformerWrapper(Normalizer())
    if "PowerTransformer" in norm_methods or "norm_all" in norm_methods:
        # Yeo-Johnson transform; maps data toward normality. Default is Yeo-Johnson.
        norm_dict["PowerTransformer"] = PandasTransformerWrapper(PowerTransformer())
    if "None" in norm_methods or "norm_all" in norm_methods:
        # "None" = do not normalize; pass raw data through as a trackable fork.
        norm_dict["None"] = PandasTransformerWrapper(_PassThrough())
        
    fs_methods = settings.get('feature_selection_methods', [])
    fs_dict = {}
    # PCA / SelectKBest are valid dimensionality-reduction transformers; kept even
    # though the current UI doesn't offer them (harmless, and feature_all expands
    # to them). SelectKBest with k='all' keeps all features but scores them.
    if "PCA" in fs_methods or "feature_all" in fs_methods:
        fs_dict["PCA"] = PandasTransformerWrapper(PCA(n_components=0.95))
    if "SelectKBest" in fs_methods or "feature_all" in fs_methods:
        score_fn = f_regression if is_regression else f_classif
        fs_dict["SelectKBest"] = PandasTransformerWrapper(SelectKBest(score_func=score_fn, k='all'))

    # Model-based selectors. These are estimators, not transformers, so wrap each
    # in SelectFromModel (which fits the estimator then masks features by
    # importance/coef). The base estimator must match the task type.
    if "Lasso_selection" in fs_methods or "feature_all" in fs_methods:
        base = Lasso(alpha=0.001, max_iter=5000) if is_regression else SGDClassifier(loss="log_loss", penalty="l1", alpha=0.001, max_iter=5000)
        fs_dict["Lasso_selection"] = PandasTransformerWrapper(SelectFromModel(base, threshold="median"))
    if "Multi-task Lasso" in fs_methods or "feature_all" in fs_methods:
        # MultiTaskLasso requires a 2-D (multi-target) y and crashes on the 1-D
        # single-target series Pine passes. Use plain Lasso (L1) as the selector
        # base for regression; SGDClassifier(log_loss + l1) for classification.
        # The fork keeps the UI's "Multi-task Lasso" label so it shows in results.
        if is_regression:
            base = Lasso(alpha=0.001, max_iter=5000)
        else:
            base = SGDClassifier(loss="log_loss", penalty="l1", alpha=0.001, max_iter=5000)
        fs_dict["Multi-task Lasso"] = PandasTransformerWrapper(SelectFromModel(base, threshold="median"))
    if "Support Vector Machine(SVM)" in fs_methods or "feature_all" in fs_methods:
        # LinearSVC exposes coef_; SVC(kernel=linear) does too but is slower.
        if is_regression:
            fs_dict["Support Vector Machine(SVM)"] = PandasTransformerWrapper(SelectFromModel(SVR(kernel="linear"), threshold="median"))
        else:
            fs_dict["Support Vector Machine(SVM)"] = PandasTransformerWrapper(SelectFromModel(LinearSVC(C=0.1, max_iter=5000, dual="auto"), threshold="median"))
    if "RandomForest" in fs_methods or "feature_all" in fs_methods:
        base = RandomForestRegressor(n_estimators=100) if is_regression else RandomForestClassifier(n_estimators=100)
        fs_dict["RandomForest"] = PandasTransformerWrapper(SelectFromModel(base, threshold="median"))
    if "Decision Stump" in fs_methods or "feature_all" in fs_methods:
        # A "stump" is a depth-1 decision tree.
        base = DecisionTreeRegressor(max_depth=1) if is_regression else DecisionTreeClassifier(max_depth=1)
        fs_dict["Decision Stump"] = PandasTransformerWrapper(SelectFromModel(base, threshold="median"))
    if "AdaBoost" in fs_methods or "feature_all" in fs_methods:
        base = AdaBoostRegressor() if is_regression else AdaBoostClassifier()
        fs_dict["AdaBoost"] = PandasTransformerWrapper(SelectFromModel(base, threshold="median"))
    if "Elastic Net_selection" in fs_methods or "feature_all" in fs_methods:
        base = ElasticNet() if is_regression else SGDClassifier(loss="log_loss", penalty="elasticnet", max_iter=5000)
        fs_dict["Elastic Net_selection"] = PandasTransformerWrapper(SelectFromModel(base, threshold="median"))
    if "Ensemble" in fs_methods or "feature_all" in fs_methods:
        base = reg_ensemble_selector() if is_regression else cls_ensemble_selector()
        fs_dict["Ensemble"] = PandasTransformerWrapper(base)
    if "None" in fs_methods or "feature_all" in fs_methods:
        # "None" = do not select; pass through as a trackable fork.
        fs_dict["None"] = PandasTransformerWrapper(_PassThrough())

    experiment = []
    if missing_dict:
        experiment.append(("missing", missing_dict))
    if norm_dict:
        experiment.append(("normalization", norm_dict))
    if fs_dict:
        experiment.append(("feature_selection", fs_dict))
    
    experiment.append(("model", model_dict))
    
            # Export hook removed since report generation runs in the local background task.
    
    # target_label = the MINORITY class (rarest label). classification_scorer
    # defines sensitivity/specificity/AUC relative to target_label, so setting it
    # to the minority class makes those metrics track the class that matters for
    # screening — instead of the majority class, which inflates accuracy while
    # the minority is ignored. (Regression is unaffected; target_label is only
    # used by classification_scorer.)
    if not is_regression:
        target_label = train_y.value_counts().idxmin()
        logger.info(f"target_label (minority class) = {target_label}; "
                    f"distribution = {train_y.value_counts().to_dict()}")
    else:
        target_label = train_y.unique()[0]
    
    validation_method = settings.get("validation_method", "k-fold cross validation")
    if validation_method == "Leave-one-out cross validation":
        evaluate_ncv = -1
    else:
        try:
            evaluate_ncv = int(settings.get("k_fold", "5"))
        except ValueError:
            evaluate_ncv = 5
    
    pine_model = Pine(experiment=experiment, target_label=target_label, cv_result=True, evaluate_ncv=evaluate_ncv)
    
    # Intercept the results list to extract best_params_ dynamically during the loop
    class TrackingList(list):
        def append(self, item):
            model_name = item.get('model')
            if model_name and model_name in model_dict:
                kernel = model_dict[model_name].kernel
                if hasattr(kernel, 'best_params_'):
                    # Convert to standard dict for clean string representation
                    params_dict = {k: v for k, v in kernel.best_params_.items()}
                    item['best_hyperparameters'] = str(params_dict)
                else:
                    item['best_hyperparameters'] = "Default (No Tuning)"
            super().append(item)
            
    pine_model.result = TrackingList()
    
    logger.info(f"Starting PineBioML Experiment for {report_id}...")
    pine_model.do_experiment(train_x, train_y)
    
    # Generate post-training plots
    try:
        # We find the best model to generate ROC and CM
        best_idx = 0
        best_val = -float('inf')
        for i, r in enumerate(pine_model.result):
            score = r.get("cv_accuracy", r.get("test_accuracy", r.get("train_accuracy", 0)))
            if score > best_val:
                best_val = score
                best_idx = i
                
        if pine_model.result:
            best = pine_model.result[best_idx]
            best_model_name = best.get("model", "Unknown")
            best_model_obj = model_dict.get(best_model_name)
            model = best_model_obj.estimator if hasattr(best_model_obj, 'estimator') else best_model_obj
            
            # Export all model results to CSV
            pd.DataFrame(pine_model.result).to_csv(os.path.join(output_dir, "All-model-result.csv"), index=False)
            
            # Use training predictions for plotting since we didn't pass a separate test set
            import numpy as np
            pred_raw = np.array(pine_model.train_pred[best_idx])
            if pred_raw.ndim == 2 and pred_raw.shape[1] > 1:
                pred_y = np.argmax(pred_raw, axis=1)
            else:
                pred_y = pred_raw.ravel()
            train_y_flat = np.array(train_y).ravel()
            
            if len(pred_y) != len(train_y_flat):
                # Try to use the model's predict method if it's fitted
                try:
                    pred_y = model.predict(train_x)
                except Exception:
                    # Truncate as a fallback
                    if len(pred_y) > len(train_y_flat):
                        pred_y = pred_y[:len(train_y_flat)]
                    else:
                        train_y_flat = train_y_flat[:len(pred_y)]
            
            if not is_regression:
                if target_le is not None:
                    pred_y_labels = target_le.inverse_transform(pred_y.astype(int)).astype(str)
                    train_y_labels = target_le.inverse_transform(train_y_flat.astype(int)).astype(str)
                else:
                    pred_y_labels = pred_y.astype(str)
                    train_y_labels = train_y_flat.astype(str)
                
                # Save classification report to JSON
                report_dict = classification_report(train_y_labels, pred_y_labels, output_dict=True)
                with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
                    json.dump(report_dict, f)

                # ── Threshold tuning (binary classification, imbalanced only) ──────
                # argmax (above) picks the majority class on imbalanced data, so
                # accuracy looks good while minority recall/specificity collapse.
                # Search a decision threshold on out-of-fold minority-class
                # probabilities, maximizing the configured metric (default F1),
                # then re-derive predictions/labels/report at that threshold so
                # every downstream artifact (CM, ROC, narrative) reflects it.
                tuned_threshold = None
                threshold_metric = None
                n_classes = len(np.unique(train_y_flat))
                if (not is_regression) and n_classes == 2 and imbalance_metadata and cw:
                    try:
                        from sklearn.model_selection import StratifiedKFold
                        from sklearn.metrics import f1_score, recall_score, matthews_corrcoef
                        metric_name = getattr(app_settings, "THRESHOLD_OPTIMIZE_METRIC", "f1") or "f1"
                        # Resolve the best model's predict_proba on OOF folds.
                        skf = StratifiedKFold(n_splits=min(5, max(2, int(settings.get("k_fold", "5")) or 5)),
                                              shuffle=True, random_state=42)
                        # Reconstruct the best fork's preprocessing chain (same as
                        # the SHAP block) so predict_proba sees the trained space.
                        best_row = pine_model.result[best_idx]
                        stage_method_map = {s: best_row[s] for (s, _) in experiment[:-1] if s in best_row}
                        proc_x = numeric_x
                        for (s, methods) in experiment[:-1]:
                            mname = stage_method_map.get(s)
                            if not mname or mname not in methods:
                                continue
                            import copy as _copy
                            proc_x = _copy.deepcopy(methods[mname]).fit_transform(proc_x, train_y)
                        # minority = target_label; find its column index in classes_.
                        classes = np.asarray(getattr(model, "classes_", np.sort(train_y.unique())))
                        pos_idx = int(np.where(classes == target_label)[0][0]) if target_label in classes else 1

                        oof_prob = np.zeros((len(proc_x), len(classes)))
                        for tr_idx, va_idx in skf.split(proc_x, train_y):
                            import copy as _copy
                            # fit a fresh clone on the fold
                            fold_kernel = _copy.deepcopy(model.kernel if hasattr(model, 'kernel') else model)
                            fold_kernel.fit(proc_x.iloc[tr_idx] if hasattr(proc_x, 'iloc') else proc_x[tr_idx],
                                            train_y.iloc[tr_idx] if hasattr(train_y, 'iloc') else train_y[tr_idx])
                            oof_prob[va_idx] = fold_kernel.predict_proba(
                                proc_x.iloc[va_idx] if hasattr(proc_x, 'iloc') else proc_x[va_idx])

                        pos_prob = oof_prob[:, pos_idx]
                        y_bin = (np.array(train_y) == target_label).astype(int)

                        best_score, best_t = -1.0, 0.5
                        for t in np.arange(0.05, 0.951, 0.05):
                            yhat = (pos_prob >= t).astype(int)
                            if metric_name == "sensitivity":
                                sc = recall_score(y_bin, yhat, pos_label=1, zero_division=0)
                            elif metric_name == "mcc":
                                sc = matthews_corrcoef(y_bin, yhat) if len(np.unique(yhat)) > 1 else -1
                            else:  # f1
                                sc = f1_score(y_bin, yhat, zero_division=0)
                            if sc > best_score:
                                best_score, best_t = sc, float(t)

                        # Re-derive predictions at the tuned threshold.
                        tuned_threshold = round(best_t, 4)
                        threshold_metric = metric_name
                        tuned_pred = (pos_prob >= best_t).astype(int)
                        # Map back to original class labels (minority / other).
                        other_label = next((c for c in classes if c != target_label), target_label)
                        tuned_labels = np.where(tuned_pred == 1, target_label, other_label)
                        if target_le is not None:
                            # labels were possibly label-encoded; keep encoding consistent
                            tuned_labels = tuned_labels.astype(train_y_flat.dtype)

                        pred_y = tuned_pred  # downstream CM/plots use these
                        if target_le is not None:
                            pred_y_labels = target_le.inverse_transform(tuned_pred.astype(int)).astype(str)
                        else:
                            pred_y_labels = tuned_labels.astype(str)

                        # Rewrite classification_report.json with tuned predictions.
                        report_dict = classification_report(train_y_labels, pred_y_labels, output_dict=True)
                        with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
                            json.dump(report_dict, f)

                        logger.info(f"Threshold tuned to {tuned_threshold} (maximizing {metric_name}="
                                    f"{best_score:.4f} for minority class '{target_label}').")
                    except Exception as _te:
                        logger.warning(f"Threshold tuning skipped: {_te}")
                        tuned_threshold = None

                # Attach imbalance + threshold info for the report engine/narrative.
                if imbalance_metadata:
                    imbalance_metadata["tuned_threshold"] = tuned_threshold
                    imbalance_metadata["threshold_metric"] = threshold_metric
                    imbalance_metadata["class_weight_applied"] = cw is not None
                    # Override analyze_imbalance's RECOMMENDED tool with what was
                    # ACTUALLY applied, so the report doesn't claim SMOTE was used
                    # when only class_weight='balanced' was (a hallucination seed).
                    if cw:
                        imbalance_metadata["tool_used"] = (
                            "Stratified Split + Cost-Sensitive class_weight='balanced'"
                            + (" + threshold tuning" if tuned_threshold else "")
                        )
                    # Write where report_engine picks up imbalance_metadata (manifest).
                    try:
                        with open(os.path.join(output_dir, "imbalance_metadata.json"), "w") as f:
                            json.dump(imbalance_metadata, f, indent=2, default=str)
                    except Exception:
                        pass
                
                import matplotlib.pyplot as plt
                import sklearn.metrics as metrics
                import plotly.express as px
                import shap
                
                # Confusion Matrix
                fig, ax = plt.subplots(figsize=(7, 6))
                metrics.ConfusionMatrixDisplay.from_predictions(
                    train_y_labels,
                    pred_y_labels,
                    xticks_rotation="vertical",
                    cmap="Blues",
                    ax=ax
                )
                ax.set_title("Confusion Matrix")
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, "_Confusion Matrix.png"), bbox_inches='tight', dpi=160)
                plt.close(fig)
                
                # ROC Curve
                try:
                    import copy

                    best_row = pine_model.result[best_idx] if pine_model.result else {}
                    stage_method_map = {
                        stage_name: best_row[stage_name]
                        for (stage_name, _) in experiment[:-1]
                        if stage_name in best_row
                    }

                    roc_x = numeric_x
                    for (stage_name, methods) in experiment[:-1]:
                        method_name = stage_method_map.get(stage_name)
                        if not method_name or method_name not in methods:
                            continue
                        transformer = copy.deepcopy(methods[method_name])
                        roc_x = transformer.fit_transform(roc_x, train_y)

                    roc_wrapper = copy.deepcopy(best_model_obj)
                    roc_wrapper.fit(roc_x, train_y)
                    roc_model = roc_wrapper.kernel if hasattr(roc_wrapper, 'kernel') else roc_wrapper

                    if hasattr(roc_wrapper, "predict_proba"):
                        probas = np.asarray(roc_wrapper.predict_proba(roc_x))
                        classes = np.asarray(getattr(roc_model, "classes_", np.sort(train_y.unique())))
                    elif hasattr(roc_model, "decision_function"):
                        scores = np.asarray(roc_model.decision_function(roc_x))
                        classes = np.asarray(getattr(roc_model, "classes_", np.sort(train_y.unique())))
                        if scores.ndim == 1:
                            probas = np.column_stack([-scores, scores])
                        else:
                            probas = scores
                    else:
                        raise RuntimeError("Best model does not expose predict_proba or decision_function")

                    fig, ax = plt.subplots(figsize=(7, 6))
                    if len(classes) <= 2:
                        pos_idx = 1 if probas.ndim == 2 and probas.shape[1] > 1 else 0
                        pos_label = classes[pos_idx] if len(classes) > pos_idx else np.sort(train_y.unique())[-1]
                        y_score = probas[:, pos_idx] if probas.ndim == 2 else probas.ravel()
                        fpr, tpr, _ = metrics.roc_curve(train_y, y_score, pos_label=pos_label)
                        roc_auc = metrics.auc(fpr, tpr)
                        label_str = target_le.inverse_transform([int(pos_label)])[0] if target_le is not None else str(pos_label)
                        ax.plot(fpr, tpr, linewidth=2, label=f'{label_str} AUC = {roc_auc:0.3f}')
                    else:
                        for i, label in enumerate(classes):
                            if probas.ndim != 2 or i >= probas.shape[1]:
                                continue
                            fpr, tpr, _ = metrics.roc_curve(train_y == label, probas[:, i])
                            roc_auc = metrics.auc(fpr, tpr)
                            label_str = target_le.inverse_transform([int(label)])[0] if target_le is not None else str(label)
                            ax.plot(fpr, tpr, linewidth=2, label=str(label_str) + ' (AUC=%0.3f)' % roc_auc)
                    ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label="Random baseline")
                    ax.set_xlim([0, 1])
                    ax.set_ylim([0, 1])
                    ax.set_ylabel('True Positive Rate')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_title('ROC curve')
                    ax.grid(alpha=0.25)
                    ax.legend(loc='lower right')
                    fig.tight_layout()
                    fig.savefig(os.path.join(output_dir, "_ROC Curve.png"), bbox_inches='tight', dpi=160)
                    plt.close(fig)
                except Exception as ex:
                    logger.warning(f"Failed to plot ROC: {ex}")
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.text(0.5, 0.5, 'ROC Not Available', ha='center', va='center')
                    ax.axis("off")
                    fig.tight_layout()
                    fig.savefig(os.path.join(output_dir, "_ROC Curve.png"), bbox_inches='tight', dpi=160)
                    plt.close(fig)
            else:
                import matplotlib.pyplot as plt
                import plotly.express as px
                import shap
                
                # Regression Report
                report_dict = {
                    "MSE": float(mean_squared_error(train_y_flat, pred_y)),
                    "MAE": float(mean_absolute_error(train_y_flat, pred_y)),
                    "R2": float(r2_score(train_y_flat, pred_y))
                }
                with open(os.path.join(output_dir, "regression_report.json"), "w") as f:
                    json.dump(report_dict, f)
                
                # True vs Predicted
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.scatter(train_y_flat, pred_y, alpha=0.6, s=32, edgecolor="none")
                ax.plot([train_y_flat.min(), train_y_flat.max()], [train_y_flat.min(), train_y_flat.max()], 'r--', linewidth=1.5)
                ax.set_xlabel("True Values")
                ax.set_ylabel("Predictions")
                ax.set_title("True vs Predicted")
                ax.grid(alpha=0.25)
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, "_True vs Predicted.png"), bbox_inches='tight', dpi=160)
                plt.close(fig)
                
                # Residuals
                residuals = train_y_flat - pred_y
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.scatter(pred_y, residuals, alpha=0.6, s=32, edgecolor="none")
                ax.hlines(0, pred_y.min(), pred_y.max(), colors='r', linestyles='--', linewidth=1.5)
                ax.set_xlabel("Predictions")
                ax.set_ylabel("Residuals")
                ax.set_title("Residuals Plot")
                ax.grid(alpha=0.25)
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, "_Residuals.png"), bbox_inches='tight', dpi=160)
                plt.close(fig)
            
            # Extract underlying sklearn model from PineBioML wrapper
            underlying_model = model.kernel if hasattr(model, 'kernel') else model
            
            # Align features with what the model actually expects (handles dropped variance/NaN columns)
            if hasattr(underlying_model, "feature_names_in_"):
                eval_x = numeric_x[underlying_model.feature_names_in_]
            else:
                eval_x = numeric_x
                
            # Feature Importance
            try:
                if is_regression:
                    raise ValueError("ensemble_selector is for classification only")

                from PineBioML.selection.classification import ensemble_selector
                selector = ensemble_selector()
                # eval_x is raw numeric_x (not the experiment's preprocessed space),
                # so it may still contain NaNs (e.g. when missing=None was chosen or
                # the raw data has gaps). ensemble_selector runs LassoLars / SVM /
                # RandomForest internally; LassoLars rejects NaN and would abort the
                # whole feature-importance chart. Defensively median-fill any
                # remaining NaNs so the ensemble scoring always runs.
                fi_x = eval_x
                if fi_x.isna().any().any():
                    fi_x = fi_x.fillna(fi_x.median(numeric_only=True))
                    # all-NaN columns (median is NaN) -> fill with 0
                    fi_x = fi_x.fillna(0)
                selector.fit(fi_x, train_y)
                
                scores = selector.scores_.copy()
                scores['Feature'] = scores.index
                scores = scores.sort_values('ensemble', ascending=True).tail(24)
                
                # Standardize scores like in Plotting()
                z_scores = (scores.drop(['ensemble', 'Feature'], axis=1) - scores.drop(['ensemble', 'Feature'], axis=1).mean()) / (scores.drop(['ensemble', 'Feature'], axis=1).std() + 1e-4)
                esemble_score = scores['ensemble']
                z_scores.columns = ['DT_score_c45', 'RandomForest_gini', 'LassoLars', 'multi_Lasso', 'SVM']
                
                import plotly.graph_objects as go
                fig_fi = go.Figure()
                
                colors = ['#FFC107', '#673AB7', '#4CAF50', '#FF5722', '#2196F3']
                for i, col in enumerate(z_scores.columns):
                    fig_fi.add_trace(go.Bar(
                        y=scores['Feature'],
                        x=z_scores[col],
                        name=col,
                        orientation='h',
                        marker_color=colors[i]
                    ))
                
                fig_fi.add_trace(go.Scatter(
                    y=scores['Feature'],
                    x=esemble_score,
                    mode='markers',
                    name='Ensemble score',
                    marker=dict(symbol='star', size=12, color='black', line=dict(width=1, color='white'))
                ))
                
                fig_fi.update_layout(
                    barmode='relative', 
                    title='Feature Importance (Ensemble)',
                    height=800,
                    plot_bgcolor='white'
                )
                fig_fi.update_xaxes(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='black')
                fig_fi.update_yaxes(showgrid=False)
                fig_fi.write_html(os.path.join(output_dir, "feature_importance.html"))
            except Exception as e:
                logger.warning(f"Failed to generate ensemble feature importance: {e}")
                if hasattr(underlying_model, 'feature_importances_'):
                    importances = underlying_model.feature_importances_
                    df_fi = pd.DataFrame({'Feature': eval_x.columns, 'Importance': importances}).sort_values('Importance', ascending=True)
                    fig_fi = px.bar(df_fi, y='Feature', x='Importance', orientation='h', title="Feature Importance")
                    fig_fi.update_layout(height=800)
                    fig_fi.write_html(os.path.join(output_dir, "feature_importance.html"))
                elif hasattr(underlying_model, 'coef_'):
                    importances = underlying_model.coef_[0] if len(underlying_model.coef_.shape) > 1 else underlying_model.coef_
                    df_fi = pd.DataFrame({'Feature': eval_x.columns, 'Importance': importances}).sort_values('Importance', ascending=True)
                    fig_fi = px.bar(df_fi, y='Feature', x='Importance', orientation='h', title="Feature Coefficients")
                    fig_fi.update_layout(height=800)
                    fig_fi.write_html(os.path.join(output_dir, "feature_importance.html"))
                else:
                    with open(os.path.join(output_dir, "feature_importance.html"), "w") as f:
                        f.write("<html><body><h3>Feature Importance not available for this model</h3></body></html>")
            
            # SHAP — reconstruct the best fork's preprocessing chain so the explainer
            # operates in the SAME transformed feature space the kernel was trained on.
            #
            # Pine.do_stage() does a depth-first traversal that forks across every
            # preprocessing x model combination, reusing the same mutable transformer /
            # wrapper objects. When it returns, those objects are left in the state of
            # the LAST fork, not the best-scoring one. Using them directly for SHAP
            # (as the old code did) explains the wrong feature space and produces
            # degenerate (near-zero) SHAP values. Instead we rebuild the exact chain
            # recorded in pine_model.result[best_idx], re-fit a fresh cloned kernel on
            # that space, sample there, and explain there.
            try:
                import io, base64, copy
                import matplotlib.pyplot as plt

                def _write_shap_html(fig, note: str = ""):
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight', dpi=160)
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    note_html = f'<p class="note">{note}</p>' if note else ""
                    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {{
    margin: 0;
    padding: 0;
    background: #fff;
    font-family: Arial, sans-serif;
}}
.wrap {{
    width: 100%;
    box-sizing: border-box;
    padding: 12px;
    text-align: center;
}}
img {{
    display: block;
    width: 100%;
    max-width: 1100px;
    height: auto;
    margin: 0 auto;
}}
.note {{
    max-width: 900px;
    margin: 10px auto 0;
    color: #555;
    font-size: 13px;
    line-height: 1.45;
}}
</style>
</head>
<body>
<div class="wrap">
<img src="data:image/png;base64,{img_str}" alt="SHAP feature contribution plot">
{note_html}
</div>
</body>
</html>"""
                    with open(os.path.join(output_dir, "shap_plot.html"), "w", encoding="utf-8") as f:
                        f.write(html)
                    plt.close(fig)

                def _write_feature_contribution_fallback(reason: str):
                    kernel = shap_kernel.kernel if hasattr(shap_kernel, 'kernel') else shap_kernel
                    feature_names = list(processed_x.columns)
                    scores = None

                    if hasattr(kernel, "feature_importances_"):
                        scores = np.asarray(kernel.feature_importances_, dtype=float)
                    elif hasattr(kernel, "coef_"):
                        coef = np.asarray(kernel.coef_, dtype=float)
                        scores = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

                    if scores is None or len(scores) != len(feature_names):
                        baseline = np.asarray(shap_kernel.predict(sample_x)).ravel() if hasattr(shap_kernel, 'is_regression') and shap_kernel.is_regression() else np.asarray(shap_kernel.predict_proba(sample_x))[:, -1]
                        scores = []
                        for col in feature_names:
                            permuted = sample_x.copy()
                            permuted[col] = permuted[col].sample(frac=1.0, random_state=42).to_numpy()
                            changed = np.asarray(shap_kernel.predict(permuted)).ravel() if hasattr(shap_kernel, 'is_regression') and shap_kernel.is_regression() else np.asarray(shap_kernel.predict_proba(permuted))[:, -1]
                            scores.append(float(np.mean(np.abs(baseline - changed))))
                        scores = np.asarray(scores, dtype=float)

                    if len(scores) != len(feature_names) or not np.isfinite(scores).any() or float(np.nanmax(np.abs(scores))) <= 1e-12:
                        numeric_var = processed_x.var(numeric_only=True).fillna(0)
                        feature_names = numeric_var.index.tolist()
                        scores = numeric_var.to_numpy(dtype=float)

                    if len(scores) == 0 or not np.isfinite(scores).any() or float(np.nanmax(np.abs(scores))) <= 1e-12:
                        raise ValueError(f"Could not build SHAP fallback: {reason}")

                    scores = np.nan_to_num(np.abs(scores), nan=0.0, posinf=0.0, neginf=0.0)
                    top_idx = np.argsort(scores)[-20:]
                    top_scores = scores[top_idx]
                    top_features = [feature_names[i] for i in top_idx]

                    fig_height = max(4.5, 0.35 * len(top_features) + 1.5)
                    fig, ax = plt.subplots(figsize=(10, fig_height))
                    ax.barh(top_features, top_scores, color="#2563eb")
                    ax.set_title("Feature Contribution Fallback")
                    ax.set_xlabel("Relative contribution")
                    ax.grid(axis="x", linestyle="--", alpha=0.25)
                    fig.tight_layout()
                    _write_shap_html(
                        fig,
                        "SHAP values were not visually informative for this run, so this fallback ranks features by model importance or prediction sensitivity."
                    )
                    logger.warning(f"Generated SHAP fallback for {report_id}: {reason}")

                best_row = pine_model.result[best_idx] if pine_model.result else {}

                # Map each non-model stage -> the method name this fork used.
                stage_method_map = {
                    stage_name: best_row[stage_name]
                    for (stage_name, _) in experiment[:-1]
                    if stage_name in best_row
                }
                # stage_method_map now holds e.g. {"missing": "Mean", "normalization": "StandardScaler", "feature_selection": "PCA"}

                # Rebuild & fit the preprocessing chain on the full training set.
                processed_x = numeric_x
                fitted_transformers = []
                for (stage_name, methods) in experiment[:-1]:
                    method_name = stage_method_map.get(stage_name)
                    if not method_name or method_name not in methods:
                        continue
                    # Clone so we don't mutate the shared objects Pine left behind.
                    transformer = copy.deepcopy(methods[method_name])
                    processed_x = transformer.fit_transform(processed_x, train_y)
                    fitted_transformers.append(transformer)

                # Fresh clone of the best kernel, fit on the true processed space.
                best_model_name = best_row.get("model", best.get("model", "Unknown"))
                best_wrapper = model_dict.get(best_model_name)
                if best_wrapper is None:
                    raise RuntimeError(f"best model '{best_model_name}' not in model_dict")
                shap_kernel = copy.deepcopy(best_wrapper)
                shap_kernel.fit(processed_x, train_y)

                # Sample in the transformed space; SHAP perturbs these columns directly.
                # Keep this bounded because Kernel/Permutation SHAP can become slow
                # during the interactive training flow.
                sample_x = processed_x.sample(min(40, len(processed_x)), random_state=42)

                # Use is_regression() rather than hasattr(predict_proba): the wrapper
                # always defines predict_proba (it raises NotImplementedError for
                # regression kernels), so hasattr would mis-route regressors.
                if hasattr(shap_kernel, 'is_regression') and shap_kernel.is_regression():
                    def _predict(x_arr):
                        xdf = pd.DataFrame(x_arr, columns=sample_x.columns)
                        return np.asarray(shap_kernel.predict(xdf)).ravel()
                    explainer = shap.Explainer(_predict, sample_x)
                else:
                    def _proba1(x_arr):
                        xdf = pd.DataFrame(x_arr, columns=sample_x.columns)
                        return np.asarray(shap_kernel.predict_proba(xdf))[:, 1]
                    explainer = shap.Explainer(_proba1, sample_x)

                shap_values = explainer(sample_x)
                vals = shap_values.values
                if isinstance(shap_values, list):
                    vals = shap_values[1].values if len(shap_values) > 1 else shap_values[0].values
                elif len(vals.shape) == 3:
                    vals = vals[:, :, 1] if vals.shape[2] > 1 else vals[:, :, 0]
                vals = np.asarray(vals, dtype=float)

                if vals.ndim != 2:
                    _write_feature_contribution_fallback(f"unexpected SHAP shape {vals.shape}")
                elif vals.shape[1] != len(sample_x.columns):
                    _write_feature_contribution_fallback(f"SHAP feature count {vals.shape[1]} did not match sample columns {len(sample_x.columns)}")
                elif vals.shape[1] == 0 or not np.isfinite(vals).any():
                    _write_feature_contribution_fallback("SHAP returned no finite feature values")
                elif float(np.nanmax(np.abs(vals))) <= 1e-12:
                    _write_feature_contribution_fallback("SHAP values were all near zero")
                else:
                    logger.info(
                        f"SHAP for {report_id}: best='{best_model_name}', "
                        f"fork={stage_method_map}, sample_x.shape={sample_x.shape}, "
                        f"max|shap|={float(np.max(np.abs(vals))):.4g}, mean|shap|={float(np.mean(np.abs(vals))):.4g}"
                    )

                    plt.close("all")
                    n_features = min(20, len(sample_x.columns))
                    fig_height = max(4.5, 0.35 * n_features + 1.5)
                    fig = plt.figure(figsize=(10, fig_height))
                    shap.summary_plot(vals, sample_x, show=False, max_display=n_features)
                    _write_shap_html(plt.gcf())
            except Exception as e:
                logger.warning(f"Error generating SHAP plot: {e}")
                with open(os.path.join(output_dir, "shap_plot.html"), "w", encoding="utf-8") as f:
                    f.write("""<!doctype html>
<html><body style="font-family:Arial,sans-serif;text-align:center;padding:24px;color:#555;">
<h3>SHAP plot not available</h3>
<p>The model finished training, but SHAP could not produce a readable plot for this run.</p>
</body></html>""")
                    
    except Exception as e:
        logger.warning(f"Error generating evaluation plots: {e}")
        
    logger.info(f"Finished PineBioML Experiment for {report_id}")
    return {"task_type": "regression" if is_regression else "classification"}
