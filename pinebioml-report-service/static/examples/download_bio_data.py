import pandas as pd
import urllib.request
import os

output_dir = r"c:\Users\Trisesa S\Documents\TRS\ITS\IIPP\TWAI2\pinebioml-report-service\static\examples"
os.makedirs(output_dir, exist_ok=True)

print("Downloading Heart Disease...")
url_heart = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
cols_heart = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df_heart = pd.read_csv(url_heart, names=cols_heart, na_values="?")
df_heart.to_csv(os.path.join(output_dir, "heart_disease_cleveland.csv"), index=False)

print("Downloading Parkinson's...")
url_park = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df_park = pd.read_csv(url_park)
df_park.to_csv(os.path.join(output_dir, "parkinsons_disease.tsv"), sep="\t", index=False)

print("Generating Diabetes...")
from sklearn.datasets import load_diabetes
diabetes = load_diabetes(as_frame=True)
df_diabetes = diabetes.frame
df_diabetes.to_excel(os.path.join(output_dir, "diabetes_disease_progression.xlsx"), index=False)

print("Done.")
