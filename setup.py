"""
Setup script: downloads the Algerian Forest Fires dataset from Kaggle.
Run once:  python setup.py

Requires kaggle package + kaggle.json token in ~/.kaggle/
OR set env vars KAGGLE_USERNAME and KAGGLE_KEY.

If download fails, manually copy the CSV to:
    data/Algerian_forest_fires_dataset_UPDATE.csv
"""
import os, subprocess, sys
from pathlib import Path

DATA = Path("data"); DATA.mkdir(exist_ok=True)

def try_dl(dataset):
    r = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", dataset, "--unzip", "-p", str(DATA)],
        capture_output=True, text=True)
    return r.returncode == 0

csv_candidates = list(DATA.glob("*.csv"))
if csv_candidates:
    print(f"Dataset already present: {csv_candidates[0]}"); sys.exit(0)

datasets = [
    "nitinchoudhary012/algerian-forest-fires-dataset",
    "abhinav099802/algerian-forest-fires",
]
for ds in datasets:
    print(f"Trying: {ds}")
    if try_dl(ds):
        print("Download successful.")
        print("\nFiles in data/:")
        [print(f"  {f.name}") for f in DATA.iterdir()]
        sys.exit(0)

print("\nAll downloads failed. Please manually download from Kaggle and place the CSV in data/")
sys.exit(1)
