# 🌲 Algerian Forest Fires — Machine Learning Analysis & Modeling

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning study predicting fire behavior and risk in northern Algeria using meteorology and Canadian Forest Fire Weather Index (FWI) components. This project tackles both **Regression** (predicting the continuous Fire Weather Index metric) and **Classification** (identifying Fire vs. Not-Fire regimes) across two distinct regions: **Bejaia** and **Sidi Bel-Abbes** (June–September 2012).

---

## 📌 Project Highlights & Objectives

- **Dual-Task Architecture**:
  1. **Regression Task**: Predict the continuous **Fire Weather Index (FWI)**.
  2. **Classification Task**: Classify the binary status (**Fire** vs. **Not-Fire**).
- **Mathematical Foundations & Custom Implementations**:
  - Closed-form **Ordinary Least Squares (OLS)** via Normal Equations ($\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$).
  - **Bayesian Logistic Regression** with Laplace approximation for parameter uncertainty and predictive credible intervals.
  - Linear Basis Functions, Polynomial feature expansions, Regularization (Ridge/Huber/Kernel Ridge), and Maximum Margin Classifiers (SVM with Linear, Polynomial, and RBF kernels).
- **Rigorous Data Science Methodology**:
  - Data leakage prevention: Standardized scaling fitted strictly within training cross-validation folds.
  - **Experiment A vs. Experiment B**: Demonstrating data leakage by comparing raw meteorological drivers against derived index calculations.
  - Comprehensive statistical EDA: Kernel Density Estimations (KDE), Q-Q plots, Chi-square contingency tests, and correlation analysis.

---

## 📂 Repository Structure

```text
├── algerian_forest_fires_PBL1.ipynb   # Main end-to-end Jupyter Notebook (all 6 rubric steps)
├── data/
│   └── Algerian_forest_fires_dataset_UPDATE.csv  # Cleaned Algerian Forest Fires dataset
├── plots/                             # 55+ exported high-resolution figures & diagnostics (200 DPI)
├── results/
│   ├── regression_summary.csv         # Comparative benchmark metrics for regression models
│   └── classification_summary.csv     # Comprehensive evaluation metrics for classification models
├── build_notebook.py                  # Script to assemble the notebook programmatically
├── patch_notebook.py                  # Notebook templating and patching pipeline
├── setup.py                           # Dataset initialization script
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## 📊 Dataset Overview

The dataset covers daily observations from two ecological regions in northern Algeria between **June 2012 and September 2012** (244 total instances, 122 per region):

| Attribute | Type | Description |
|---|---|---|
| **Date** | Temporal | Day, month, year of observation |
| **Temperature** | Meteorological | Noon temperature in Celsius (22°C to 42°C) |
| **RH** | Meteorological | Relative Humidity in % (21% to 90%) |
| **Ws** | Meteorological | Wind speed in km/h (6 to 29 km/h) |
| **Rain** | Meteorological | Total daily rainfall in mm (0 to 16.8 mm) |
| **FFMC** | FWI System | Fine Fuel Moisture Code (moisture content of litter) |
| **DMC** | FWI System | Duff Moisture Code (moisture content of moderate organic layers) |
| **DC** | FWI System | Drought Code (moisture rating of deep organic layers) |
| **ISI** | FWI System | Initial Spread Index (rate of fire spread combining FFMC and wind) |
| **BUI** | FWI System | Buildup Index (fuel available for combustion combining DMC and DC) |
| **FWI** | Target (Reg.) | Fire Weather Index (general fire intensity indicator) |
| **Classes** | Target (Clf.) | Binary label: `fire` or `not fire` |
| **Region** | Categorical | `0` = Bejaia (Northeastern coast), `1` = Sidi Bel-Abbes (Northwestern inland) |

---

## 🔬 Benchmark Results

### 1. Regression Models (Predicting FWI)

*Trained on weather variables (`Temperature`, `RH`, `Ws`, `Rain`, `Region`, `Month`)*

| Model | Test RMSE | Test MAE | Test $R^2$ | 5-Fold CV RMSE | Key Parameters |
|---|:---:|:---:|:---:|:---:|---|
| **Kernel Ridge (RBF)** 🏆 | **2.65** | **2.07** | **0.769** | **4.16 ± 0.68** | $\alpha=0.1, \gamma=0.1$ |
| **Polynomial Regression (deg=2)** | 3.65 | 2.61 | 0.560 | 4.87 ± 0.57 | Degree = 2 |
| **Huber Regressor** | 4.06 | 3.34 | 0.455 | 5.23 ± 0.54 | $\epsilon=1.35$ |
| **Ridge Regression** | 4.24 | 3.42 | 0.406 | 5.18 ± 0.49 | $\alpha=13.22$ |
| **Bayesian Ridge** | 4.27 | 3.43 | 0.397 | 5.19 ± 0.49 | Prior posteriors |
| **OLS (Normal Equations)** | 4.32 | 3.46 | 0.384 | 5.19 ± 0.48 | Closed-form analytic solution |
| *OLS (Exp B — with FWI indices)* | *0.60* | *0.43* | *0.988* | — | *Demonstrates data leakage* |

> **Key Finding**: Non-linear models (Kernel Ridge with RBF kernel) significantly outperform linear approximations on raw weather indicators ($R^2$ jumps from $0.38$ to $0.77$).

---

### 2. Classification Models (Predicting Fire vs. Not-Fire)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score | Log Loss |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SVM (RBF Kernel)** 🏆 | **83.67%** | 81.25% | **92.86%** | **0.867** | 0.864 | 0.124 | 0.416 |
| **Gaussian Naive Bayes** | 81.63% | 78.79% | **92.86%** | 0.852 | 0.913 | 0.138 | 1.100 |
| **Logistic Regression** | 81.63% | 80.65% | 89.29% | 0.847 | **0.930** | **0.117** | 0.558 |
| **Bayesian Logistic (Laplace)** | 81.63% | 80.65% | 89.29% | 0.847 | **0.930** | 0.119 | **0.411** |
| **SVM (Linear)** | 81.63% | 80.65% | 89.29% | 0.847 | 0.915 | 0.124 | 0.512 |
| **SVM (Polynomial, deg=2)** | 79.59% | 76.47% | **92.86%** | 0.839 | 0.896 | 0.138 | 0.443 |
| **LDA (Linear Discriminant)** | 79.59% | 78.12% | 89.29% | 0.833 | 0.900 | 0.137 | 0.417 |

> **Classification Takeaway**: All classifiers achieve strong recall (>89%), critical in wildfire monitoring where false negatives (missing an active fire) carry severe environmental and societal cost.

---

## 🧪 The Data Leakage Experiment (Experiment A vs. B)

A core pedagogical experiment in this project highlights the pitfall of feature leakage:
- **Experiment A (Operational Deployment)**: Uses solely accessible atmospheric parameters (`Temperature`, `RH`, `Ws`, `Rain`, `Region`, `Month`). Yields an honest test $R^2$ of **0.384** (linear) to **0.769** (RBF).
- **Experiment B (Index Leakage)**: Incorporates FWI sub-indices (`FFMC`, `DMC`, `DC`, `ISI`, `BUI`). Results shoot up artificially to $R^2 = \mathbf{0.988}$ and $\text{RMSE} = \mathbf{0.60}$.
- **Insight**: In the Canadian FWI system, FWI is an analytical mathematical calculation derived directly from ISI and BUI. Training an ML model with those indices as features is mathematically redundant and unfeasible in early-warning systems that lack sub-index sensors.

---

## 📈 Visualizations & Diagnostics

All plots are generated with clean styling and saved in [`plots/`](./plots/):
- **Exploratory Data Analysis**: Histograms with KDE, Box plots by region, Correlation & Covariance matrices, Pairplots, and temporal series.
- **Regression Diagnostics**: Residuals vs. Fitted plots, Q-Q residual normality, learning curves, and polynomial bias-variance trade-offs.
- **Classification Diagnostics**: ROC curves, Precision-Recall curves, confusion matrices, probability calibration curves, and 2D decision boundary projections.
- **Feature Importance**: Ridge standardized coefficient profiles and Logistic regression log-odds weights.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/samhitastuti/algerian-forest-fires.git
cd algerian-forest-fires
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Project
Open and run the notebook interactively:
```bash
jupyter notebook algerian_forest_fires_PBL1.ipynb
```
Or execute all cells top to bottom via `nbconvert`:
```bash
jupyter nbconvert --to notebook --execute --inplace algerian_forest_fires_PBL1.ipynb
```

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3.10+
- **Data Manipulation**: `numpy`, `pandas`
- **Machine Learning & Modeling**: `scikit-learn`, `scipy`
- **Visualization**: `matplotlib`, `seaborn`
- **Environment**: `jupyter`, `ipykernel`

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
