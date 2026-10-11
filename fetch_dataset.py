"""
fetch_dataset.py  ─  Alternative: download Algerian Forest Fires CSV directly
from the UCI Machine Learning Repository (no Kaggle credentials needed).

Run:  python fetch_dataset.py
"""
import urllib.request, pathlib, sys, os

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

TARGET = DATA_DIR / "Algerian_forest_fires_dataset_UPDATE.csv"

if TARGET.exists():
    print(f"Already exists: {TARGET}"); sys.exit(0)

# UCI ML Repository URL
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00547/Algerian_forest_fires_dataset_UPDATE.csv"
)

print(f"Downloading from UCI: {UCI_URL}")
try:
    urllib.request.urlretrieve(UCI_URL, TARGET)
    print(f"Saved to {TARGET}")
except Exception as e:
    print(f"UCI download failed: {e}")
    print("Please manually download the CSV from Kaggle:")
    print("  https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset")
    print(f"and place it in: {DATA_DIR.resolve()}")
    sys.exit(1)
