import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from PineBioML.model.utils import Pine, sklearn_esitimator_wrapper

print("Loading dummy dataset...")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
train_x = df
train_y = pd.Series(data.target, name="target")

os.environ["FASTAPI_EXPORT_URL"] = "http://127.0.0.1:8001/report/generate"
os.environ["PINEBIOML_API_KEY"] = "pinebioml_default_key_change_me"

print("Starting PineBioML Auto-ML Training Pipeline...")
experiment = [
    ("model", {"RandomForest": sklearn_esitimator_wrapper(RandomForestClassifier(n_estimators=10))})
]
pine_model = Pine(experiment=experiment, target_label=1)
pine_model.do_experiment(train_x, train_y)

print("Training finished! The pipeline should have automatically sent the metrics to the FastAPI service.")
