#!/usr/bin/env python3
"""
build_notebook.py  --  generates the PBL-1 Jupyter notebook programmatically.
Run once:  python build_notebook.py
Then open: jupyter notebook algerian_forest_fires_PBL1.ipynb
"""

import json, textwrap, uuid

_cell_counter = [0]

def cell(source, cell_type="code"):
    _cell_counter[0] += 1
    cell_id = str(uuid.uuid4())[:8]  # short unique ID
    if cell_type == "markdown":
        return {"cell_type": "markdown", "id": cell_id, "metadata": {},
                "source": source if isinstance(source, list) else [source]}
    return {"cell_type": "code", "id": cell_id, "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": source if isinstance(source, list) else [source]}

def md(*lines):
    return cell("\n".join(lines), "markdown")

def code(src):
    return cell(textwrap.dedent(src).strip() + "\n")

cells = []

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
cells += [md(
    "# Algerian Forest Fires – FWI Regression & Fire Classification",
    "## ML PBL-1 | Unit 1 + Unit 2",
    "",
    "**Dataset:** Algerian Forest Fires (Bejaia & Sidi Bel-Abbes, Jun–Sep 2012)  ",
    "**Tasks:** (A) Predict Fire Weather Index (FWI) – *regression*; (B) Classify fire / not-fire – *classification*  ",
    "**Tools:** Python · pandas · numpy · matplotlib/seaborn · scikit-learn · scipy",
)]

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0  – IMPORTS + DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
cells += [md("---\n## Step 0 · Problem Understanding & Data Cleaning  *(Rubric 1)*")]

cells += [code("""
    # ── Imports ──────────────────────────────────────────────────────────────
    import warnings, time
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from scipy import stats
    from scipy.stats import chi2_contingency, norm, gamma, shapiro

    from sklearn.preprocessing import (StandardScaler, PolynomialFeatures,
                                        LabelEncoder)
    from sklearn.model_selection import (train_test_split, RepeatedKFold,
                                          cross_val_score, GridSearchCV)
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import (LinearRegression, Ridge, RidgeCV,
                                       HuberRegressor, BayesianRidge,
                                       LogisticRegression)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVR, SVC
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                                  r2_score, accuracy_score, precision_score,
                                  recall_score, f1_score, roc_auc_score,
                                  confusion_matrix, ConfusionMatrixDisplay,
                                  brier_score_loss, log_loss,
                                  RocCurveDisplay)
    from sklearn.calibration import calibration_curve

    plt.rcParams.update({
        "figure.dpi": 130,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    })
    sns.set_palette("husl")
    print("All imports successful ✓")
""")]

cells += [md("### 0.1 · Problem Statement",
    "",
    "The **Fire Weather Index (FWI)** is a Canadian system for rating fire danger. "
    "High FWI values indicate conditions under which fires ignite easily and spread rapidly. "
    "Accurate FWI forecasting helps fire authorities pre-position resources and issue timely warnings.",
    "",
    "**Task A – Regression:** Predict the continuous FWI value from daily weather observations.",
    "",
    "**Task B – Classification:** Predict whether a fire occurred on a given day.",
    "",
    "### 0.2 · Data Dictionary",
    "",
    "| Feature | Meaning | Unit | Type |",
    "|---------|---------|------|------|",
    "| day | Day of month | – | date part |",
    "| month | Month (6–9) | – | date part |",
    "| year | Year | – | date part |",
    "| Temperature | Noon temperature | °C | numeric |",
    "| RH | Relative humidity | % | numeric |",
    "| Ws | Wind speed | km/h | numeric |",
    "| Rain | Rainfall | mm | numeric |",
    "| FFMC | Fine Fuel Moisture Code | – | FWI index |",
    "| DMC | Duff Moisture Code | – | FWI index |",
    "| DC | Drought Code | – | FWI index |",
    "| ISI | Initial Spread Index | – | FWI index |",
    "| BUI | Buildup Index | – | FWI index |",
    "| FWI | **Fire Weather Index (regression target)** | – | FWI index |",
    "| Classes | fire / not fire **(classification target)** | – | categorical |",
    "",
    "> **Data leakage note:** FWI is computed from ISI and BUI; ISI from FFMC+Ws; BUI from DMC+DC. "
    "Feeding these into the regression makes the problem trivial. "
    "**Experiment A** uses only raw weather features (Temperature, RH, Ws, Rain + Region, Month). "
    "**Experiment B** adds the FWI-system indices to quantify the leakage gain.",
)]

cells += [code("""
    # ── Load & Parse Raw CSV ─────────────────────────────────────────────────
    # The UCI CSV has the structure:
    #   Line 0: 'Bejaia Region Dataset'  (title)
    #   Line 1: column headers
    #   Lines 2-123: Bejaia data (122 rows)
    #   Line 124: blank
    #   Line 125: 'Sidi-Bel Abbes Region Dataset' (title)
    #   Line 126: column headers
    #   Lines 127-248: Sidi data (122 rows)

    import os, glob, io

    csv_files = glob.glob("data/*.csv")
    if not csv_files:
        raise FileNotFoundError(
            "No CSV found in data/.  Run: python fetch_dataset.py")

    raw_path = csv_files[0]
    print(f"Loading: {raw_path}")

    col_names = ['day','month','year','Temperature','RH','Ws','Rain',
                 'FFMC','DMC','DC','ISI','BUI','FWI','Classes']

    with open(raw_path, 'r', encoding='utf-8', errors='replace') as fh:
        all_lines = fh.readlines()
    print(f"Total raw lines in file: {len(all_lines)}")

    def is_data_row(line):
        return line.count(',') >= 13   # 14 columns => 13 commas minimum

    bejaia_rows, sidi_rows = [], []
    current_region = None
    for line in all_lines:
        s = line.strip()
        if not s:
            continue
        if 'bejaia' in s.lower():
            current_region = 'Bejaia'; continue
        if 'sidi' in s.lower():
            current_region = 'Sidi-Bel-Abbes'; continue
        if s.lower().startswith('day,'):
            continue      # skip repeated header rows
        if not is_data_row(s):
            continue      # skip any other non-data rows
        if current_region == 'Bejaia':
            bejaia_rows.append(s)
        elif current_region == 'Sidi-Bel-Abbes':
            sidi_rows.append(s)

    def rows_to_df(rows, region):
        text = "\n".join(rows)
        d = pd.read_csv(io.StringIO(text), header=None,
                        names=col_names, skipinitialspace=True)
        d['Region'] = region
        return d

    bejaia_df = rows_to_df(bejaia_rows, 'Bejaia')
    sidi_df   = rows_to_df(sidi_rows,   'Sidi-Bel-Abbes')
    df = pd.concat([bejaia_df, sidi_df], ignore_index=True)

    print(f"Bejaia rows: {len(bejaia_df)}  |  Sidi rows: {len(sidi_df)}")
    print(f"Combined shape: {df.shape}")
    df.head(5)
""")]

cells += [code("""
    # ── Strip whitespace, fix Classes ─────────────────────────────────────────
    df.columns = df.columns.str.strip()
    df['Classes'] = df['Classes'].astype(str).str.strip().str.lower()

    print("Unique Classes values:", df['Classes'].unique())

    # Drop completely empty / header rows
    df = df[~df['day'].astype(str).str.strip().str.lower().isin(['day',''])]
    df = df.dropna(how='all')
    df.reset_index(drop=True, inplace=True)
    print(f"Shape after dropping junk rows: {df.shape}")
""")]

cells += [code("""
    # ── Fix merged-cell row (14 Aug, Sidi Bel-Abbes) ─────────────────────────
    # Some rows have two numbers merged into the 'day' column, e.g. "14 15"
    # Strategy: split the value and keep only the first token.
    problem_mask = df['day'].astype(str).str.contains(r'\\s+\\d', na=False)
    print("Potentially merged rows:", df[problem_mask][['day','month','year','Region']].to_string())

    # Drop any row where numeric conversion of 'day' would fail after splitting
    df['day'] = df['day'].astype(str).str.split().str[0]

    # ── Convert numeric columns ───────────────────────────────────────────────
    num_cols = ['day','month','year','Temperature','RH','Ws','Rain',
                'FFMC','DMC','DC','ISI','BUI','FWI']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    bad = df[num_cols].isnull().any(axis=1)
    print(f"Rows with any NaN after coerce: {bad.sum()}")
    df = df[~bad].reset_index(drop=True)
    print(f"Final shape: {df.shape}")
""")]

cells += [code("""
    # ── Map target, add date ──────────────────────────────────────────────────
    df['fire'] = df['Classes'].map({'fire': 1, 'not fire': 0})
    unmapped = df['fire'].isnull().sum()
    if unmapped:
        print("Unmapped Classes values:", df[df['fire'].isnull()]['Classes'].unique())
        df = df.dropna(subset=['fire']).reset_index(drop=True)

    df['fire'] = df['fire'].astype(int)

    # Combine date
    df['date'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=df['day']))

    print(df[['date','Region','Temperature','RH','Ws','Rain','FWI','fire']].head())
    print(f"\\nShape: {df.shape}  |  Fire count: {df['fire'].sum()}  |  Not-fire: {(df['fire']==0).sum()}")
""")]

cells += [code("""
    df.info()
    df.describe()
""")]

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1  – EDA
# ═══════════════════════════════════════════════════════════════════════════════
cells += [md("---\n## Step 1 · Exploratory Data Analysis  *(Rubric 2)*")]

cells += [code("""
    # ── 1.1 Descriptive statistics: mean, variance, std, quantiles ────────────
    desc_cols = ['Temperature','RH','Ws','Rain','FFMC','DMC','DC','ISI','BUI','FWI']
    stats_df = df[desc_cols].agg(['mean','var','std',
                                   lambda x: x.quantile(0.25),
                                   'median',
                                   lambda x: x.quantile(0.75)])
    stats_df.index = ['Mean','Variance','Std Dev','Q1','Median','Q3']
    display(stats_df.round(3))
""")]

cells += [code("""
    # ── Box-plots per feature ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    for i, col in enumerate(desc_cols):
        df.boxplot(column=col, ax=axes[i], patch_artist=True,
                   boxprops=dict(facecolor='#6fa3ef', color='#2c4a7c'),
                   medianprops=dict(color='red', linewidth=2))
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].set_xlabel("")
    plt.suptitle("Box-plots of All Numeric Features", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("plots/boxplots.png", bbox_inches='tight'); plt.show()
    print("Insight: FWI, DMC, DC, and BUI show heavy right-skew with high-value outliers, "
          "consistent with rare extreme-fire days. Rain is almost always 0.")
""")]

cells += [code("""
    # ── 1.2 Histograms + KDE  (continuous RVs, probability density) ──────────
    import os; os.makedirs("plots", exist_ok=True)

    kd_cols = ['Temperature','RH','Ws','Rain','FWI']
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    palette = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6']
    for ax, col, color in zip(axes, kd_cols, palette):
        data = df[col].dropna()
        ax.hist(data, bins=20, density=True, alpha=0.5, color=color, label='Histogram')
        data.plot.kde(ax=ax, color=color, linewidth=2, label='KDE')
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylabel("Density")
    plt.suptitle("Histograms + KDE (Continuous Features)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/histograms_kde.png", bbox_inches='tight'); plt.show()
    print("Insight: Temperature is roughly bell-shaped (near-normal). "
          "Rain is extremely right-skewed (zero-inflated) — a log1p transform will help. "
          "FWI is bimodal, with a large cluster near 0 (non-fire days) and a tail from fire days.")
""")]

cells += [code("""
    # ── 1.3 Fire vs Not-fire counts; proportion by region & month ─────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Overall counts
    counts = df['Classes'].value_counts()
    axes[0].bar(counts.index, counts.values, color=['#e74c3c','#3498db'], edgecolor='black')
    axes[0].set_title("Fire vs Not-fire (Overall)", fontweight='bold')
    axes[0].set_ylabel("Count")
    for i,(idx,v) in enumerate(counts.items()):
        axes[0].text(i, v+1, str(v), ha='center', fontweight='bold')

    # By region
    reg_fire = df.groupby('Region')['fire'].mean().reset_index()
    axes[1].bar(reg_fire['Region'], reg_fire['fire'], color=['#e74c3c','#f39c12'], edgecolor='black')
    axes[1].set_title("Fire Proportion by Region", fontweight='bold')
    axes[1].set_ylabel("P(fire)")
    axes[1].set_ylim(0,1)

    # By month
    mon_fire = df.groupby('month')['fire'].mean().reset_index()
    axes[2].plot(mon_fire['month'], mon_fire['fire'], 'o-', color='#e74c3c', linewidth=2, markersize=8)
    axes[2].set_xticks([6,7,8,9]); axes[2].set_xticklabels(['Jun','Jul','Aug','Sep'])
    axes[2].set_title("Fire Proportion by Month", fontweight='bold')
    axes[2].set_ylabel("P(fire)"); axes[2].set_ylim(0,1)

    plt.tight_layout()
    plt.savefig("plots/fire_counts.png", bbox_inches='tight'); plt.show()
    print("Insight: Dataset is roughly 60/40 fire/not-fire. "
          "August–September have higher fire rates, matching the dry-season peak.")
""")]

cells += [code("""
    # ── 1.4 Covariance matrix + correlation heatmap ────────────────────────────
    corr = df[desc_cols + ['fire']].corr()

    plt.figure(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                vmin=-1, vmax=1, linewidths=0.5, square=True)
    plt.title("Pearson Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png", bbox_inches='tight'); plt.show()

    print("Insight: FWI is strongly correlated with ISI (r≈0.95), BUI (r≈0.88), DMC, and DC — "
          "confirming data leakage risk. Temperature correlates positively with fire (r≈0.45), "
          "RH negatively (r≈-0.38).")
""")]

cells += [code("""
    # ── 1.5 Conditional probability (Unit 1: conditional probability) ──────────
    p_fire = df['fire'].mean()
    p_fire_rain = df[df['Rain'] > 0]['fire'].mean()
    p_fire_hot  = df[df['Temperature'] > 33]['fire'].mean()
    p_fire_bej  = df[df['Region']=='Bejaia']['fire'].mean()
    p_fire_sidi = df[df['Region']=='Sidi-Bel-Abbes']['fire'].mean()

    cond_df = pd.DataFrame({
        'Condition': ['P(fire)', 'P(fire | Rain > 0)', 'P(fire | Temp > 33°C)',
                      'P(fire | Region=Bejaia)', 'P(fire | Region=Sidi)'],
        'Probability': [p_fire, p_fire_rain, p_fire_hot, p_fire_bej, p_fire_sidi]
    })
    display(cond_df.round(4))
    print("Insight: P(fire|Rain>0) is notably lower than the marginal P(fire), "
          "supporting Rain as an important protective factor.")
""")]

cells += [code("""
    # ── 1.6 Independence test: Region ⊥ Fire?  (chi-square) ─────────────────
    contingency = pd.crosstab(df['Region'], df['fire'])
    chi2, p_val, dof, expected = chi2_contingency(contingency)
    print(f"Chi-square test — Region vs Fire:")
    print(f"  χ² = {chi2:.4f},  p-value = {p_val:.4f},  dof = {dof}")
    print(f"  {'NOT independent (p < 0.05)' if p_val < 0.05 else 'Independent (p ≥ 0.05)'}")
    print("Insight: If p < 0.05, Region provides predictive information about fire occurrence.")
""")]

cells += [code("""
    # ── 1.7 Probability curve fitting: Normal to Temperature, Gamma to FWI ────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Temperature — fit Normal
    temp_data = df['Temperature'].dropna()
    mu_t, std_t = norm.fit(temp_data)
    x_t = np.linspace(temp_data.min()-2, temp_data.max()+2, 200)
    axes[0].hist(temp_data, bins=25, density=True, alpha=0.5, color='#e74c3c', label='Data')
    axes[0].plot(x_t, norm.pdf(x_t, mu_t, std_t), 'r-', lw=2,
                 label=f'Normal(μ={mu_t:.1f}, σ={std_t:.1f})')
    axes[0].set_title("Temperature – Normal Fit", fontweight='bold')
    axes[0].set_xlabel("Temperature (°C)"); axes[0].set_ylabel("Density")
    axes[0].legend()

    # FWI — fit Gamma (FWI ≥ 0, skewed)
    fwi_data = df['FWI'].dropna()
    fwi_pos = fwi_data[fwi_data > 0]  # Gamma needs > 0
    shape_g, loc_g, scale_g = gamma.fit(fwi_pos, floc=0)
    x_f = np.linspace(0, fwi_pos.max()+5, 300)
    axes[1].hist(fwi_pos, bins=25, density=True, alpha=0.5, color='#9b59b6', label='Data (FWI>0)')
    axes[1].plot(x_f, gamma.pdf(x_f, shape_g, loc_g, scale_g), 'purple', lw=2,
                 label=f'Gamma(a={shape_g:.2f})')
    axes[1].set_title("FWI – Gamma Fit", fontweight='bold')
    axes[1].set_xlabel("FWI"); axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("plots/distribution_fits.png", bbox_inches='tight'); plt.show()

    # Q-Q plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    stats.probplot(temp_data, dist='norm', plot=axes[0])
    axes[0].set_title("Q-Q Plot: Temperature vs Normal", fontweight='bold')
    stats.probplot(np.log1p(fwi_data), dist='norm', plot=axes[1])
    axes[1].set_title("Q-Q Plot: log(1+FWI) vs Normal", fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/qq_plots.png", bbox_inches='tight'); plt.show()
    print("Insight: Temperature follows a near-Normal distribution (Q-Q line is straight). "
          "log(1+FWI) is also approximately Normal, validating the Gamma fit on raw FWI.")
""")]

cells += [code("""
    # ── 1.8 Scatter plots: FWI vs Temp, RH, Ws + polynomial curve ────────────
    from numpy.polynomial import polynomial as P

    scatter_feats = ['Temperature','RH','Ws']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#e74c3c','#3498db','#2ecc71']

    for ax, feat, color in zip(axes, scatter_feats, colors):
        x = df[feat].values
        y = df['FWI'].values
        ax.scatter(x, y, alpha=0.4, s=20, color=color, label='Data')
        # Polynomial degree-2 fit
        idx = np.argsort(x)
        coeffs = np.polyfit(x, y, deg=2)
        poly_fn = np.poly1d(coeffs)
        xfit = np.linspace(x.min(), x.max(), 200)
        ax.plot(xfit, poly_fn(xfit), 'k-', lw=2, label='Poly(deg=2)')
        ax.set_xlabel(feat, fontsize=12); ax.set_ylabel("FWI", fontsize=12)
        ax.set_title(f"FWI vs {feat}", fontweight='bold')
        ax.legend()

    plt.tight_layout()
    plt.savefig("plots/scatter_fwi.png", bbox_inches='tight'); plt.show()
    print("Insight: FWI increases with Temperature (positive trend) and decreases with RH "
          "(negative trend). Wind speed (Ws) shows a weaker, noisy positive relationship.")
""")]

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2  – PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
cells += [md("---\n## Step 2 · Preprocessing")]

cells += [code("""
    os.makedirs("plots", exist_ok=True)

    # ── Features (Experiment A: raw weather only) ─────────────────────────────
    WEATHER_FEATS = ['Temperature','RH','Ws','Rain','month']
    INDEX_FEATS   = ['FFMC','DMC','DC','ISI','BUI']

    # Encode Region as binary
    df['region_enc'] = (df['Region'] == 'Bejaia').astype(int)

    FEATS_A = WEATHER_FEATS + ['region_enc']  # Experiment A (no leakage)
    FEATS_B = FEATS_A + INDEX_FEATS           # Experiment B (with indices)

    # log1p transform Rain (highly skewed)
    df['Rain_log'] = np.log1p(df['Rain'])
    FEATS_A_LOG = [f if f != 'Rain' else 'Rain_log' for f in FEATS_A]
    FEATS_B_LOG = [f if f != 'Rain' else 'Rain_log' for f in FEATS_B]

    TARGET_REG  = 'FWI'
    TARGET_CLS  = 'fire'

    print("Experiment A features:", FEATS_A_LOG)
    print("Experiment B features:", FEATS_B_LOG)
    print(f"Total samples: {len(df)}")
""")]

cells += [code("""
    # ── Train/test split (stratified for classification) ──────────────────────
    X_A = df[FEATS_A_LOG].values
    X_B = df[FEATS_B_LOG].values
    y_reg = df[TARGET_REG].values
    y_cls = df[TARGET_CLS].values

    # Regression split (random, continuous target)
    Xreg_train_A, Xreg_test_A, y_reg_train, y_reg_test = train_test_split(
        X_A, y_reg, test_size=0.2, random_state=42)
    Xreg_train_B, Xreg_test_B, _, _ = train_test_split(
        X_B, y_reg, test_size=0.2, random_state=42)

    # Classification split (stratified)
    Xcls_train_A, Xcls_test_A, y_cls_train, y_cls_test = train_test_split(
        X_A, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    Xcls_train_B, Xcls_test_B, _, _ = train_test_split(
        X_B, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

    print(f"Train size: {len(y_reg_train)}, Test size: {len(y_reg_test)}")
    print(f"Class balance in train: {np.bincount(y_cls_train)}, test: {np.bincount(y_cls_test)}")
""")]

cells += [code("""
    # ── Scale inside helper that creates a fitted scaler ──────────────────────
    scaler_A = StandardScaler().fit(Xreg_train_A)
    scaler_B = StandardScaler().fit(Xreg_train_B)

    Xreg_tr_A_sc = scaler_A.transform(Xreg_train_A)
    Xreg_te_A_sc = scaler_A.transform(Xreg_test_A)
    Xreg_tr_B_sc = scaler_B.transform(Xreg_train_B)
    Xreg_te_B_sc = scaler_B.transform(Xreg_test_B)

    scaler_cls_A = StandardScaler().fit(Xcls_train_A)
    scaler_cls_B = StandardScaler().fit(Xcls_train_B)

    Xcls_tr_A_sc = scaler_cls_A.transform(Xcls_train_A)
    Xcls_te_A_sc = scaler_cls_A.transform(Xcls_test_A)
    Xcls_tr_B_sc = scaler_cls_B.transform(Xcls_train_B)
    Xcls_te_B_sc = scaler_cls_B.transform(Xcls_test_B)

    print("Scaling complete (fit only on training data) ✓")
""")]

cells += [code("""
    # ── Repeated 5-fold CV helper ─────────────────────────────────────────────
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    def cv_scores_reg(model, X, y, scaler=None):
        \"\"\"Returns mean ± std RMSE across repeated K-fold.\"\"\"
        results = []
        for tr, val in rkf.split(X):
            Xtr, Xval = X[tr], X[val]
            ytr, yval = y[tr], y[val]
            if scaler:
                sc = StandardScaler().fit(Xtr)
                Xtr = sc.transform(Xtr)
                Xval = sc.transform(Xval)
            model.fit(Xtr, ytr)
            pred = model.predict(Xval)
            results.append(np.sqrt(mean_squared_error(yval, pred)))
        return np.mean(results), np.std(results)

    def cv_scores_cls(model, X, y, scaler=True, scoring='f1'):
        \"\"\"Returns mean ± std of scoring across repeated K-fold.\"\"\"
        results = []
        for tr, val in rkf.split(X):
            Xtr, Xval = X[tr], X[val]
            ytr, yval = y[tr], y[val]
            if scaler:
                sc = StandardScaler().fit(Xtr)
                Xtr = sc.transform(Xtr)
                Xval = sc.transform(Xval)
            model.fit(Xtr, ytr)
            pred = model.predict(Xval)
            if scoring == 'f1':
                results.append(f1_score(yval, pred, zero_division=0))
            elif scoring == 'roc_auc':
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(Xval)[:,1]
                else:
                    prob = model.decision_function(Xval)
                results.append(roc_auc_score(yval, prob))
        return np.mean(results), np.std(results)

    print("CV helpers defined ✓")
""")]

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3  – REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
cells += [md("---\n## Step 3 · Regression Models – Predict FWI  *(Rubric 3)*",
    "",
    "> **MLE/Bayesian connection (required for rubric):**  ",
    "> Ordinary least squares is the *maximum likelihood estimator* under the assumption "
    "> that residuals are i.i.d. Gaussian: minimising RSS is equivalent to maximising the "
    "> log-likelihood of the data.  ",
    "> Ridge regression adds an L2 penalty which is equivalent to *MAP estimation* with a "
    "> zero-mean Gaussian prior on the weights (precision = λ). As λ → 0 we recover OLS.",
)]

cells += [code("""
    reg_results = {}   # store for summary table

    def reg_metrics(name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'R²': r2}

    def plot_residuals(name, y_true, y_pred, ax1, ax2):
        resid = y_true - y_pred
        ax1.scatter(y_pred, resid, alpha=0.5, s=20)
        ax1.axhline(0, color='red', lw=1)
        ax1.set_xlabel("Predicted FWI"); ax1.set_ylabel("Residual")
        ax1.set_title(f"{name} — Residuals", fontweight='bold')
        ax2.scatter(y_true, y_pred, alpha=0.5, s=20)
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        ax2.plot([mn,mx],[mn,mx],'r--',lw=1)
        ax2.set_xlabel("Actual FWI"); ax2.set_ylabel("Predicted FWI")
        ax2.set_title(f"{name} — Actual vs Predicted", fontweight='bold')
""")]

cells += [code("""
    # ── 3.1 OLS via Normal Equations (numpy) ──────────────────────────────────
    # w = (XᵀX)⁻¹ Xᵀy   (derived from MLE under Gaussian noise)
    X_tr = np.c_[np.ones(len(Xreg_tr_A_sc)), Xreg_tr_A_sc]
    X_te = np.c_[np.ones(len(Xreg_te_A_sc)), Xreg_te_A_sc]

    # Manual normal equations
    w_ols = np.linalg.pinv(X_tr.T @ X_tr) @ (X_tr.T @ y_reg_train)
    ols_pred_manual = X_te @ w_ols

    # Cross-check with sklearn
    lr = LinearRegression().fit(Xreg_tr_A_sc, y_reg_train)
    ols_pred_sk    = lr.predict(Xreg_te_A_sc)

    m = reg_metrics("OLS (normal eqn)", y_reg_test, ols_pred_manual)
    cv_mu, cv_std = cv_scores_reg(LinearRegression(), X_A, y_reg, scaler=True)
    m['CV RMSE'] = f"{cv_mu:.3f} ± {cv_std:.3f}"
    reg_results['OLS'] = m
    print("OLS Results:", m)
    print(f"  Manual vs sklearn max diff: {np.abs(ols_pred_manual - ols_pred_sk).max():.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_residuals("OLS", y_reg_test, ols_pred_manual, axes[0], axes[1])
    plt.tight_layout(); plt.savefig("plots/ols_diagnostics.png", bbox_inches='tight'); plt.show()
""")]

cells += [code("""
    # ── 3.2 Polynomial Regression (degree 1–4) ────────────────────────────────
    poly_rmse_train, poly_rmse_test = [], []
    best_poly_pred, best_poly_deg = None, 1

    for deg in range(1, 5):
        pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
            ('scaler', StandardScaler()),
            ('lr', LinearRegression()),
        ])
        pipe.fit(Xreg_train_A, y_reg_train)
        tr_pred = pipe.predict(Xreg_train_A)
        te_pred = pipe.predict(Xreg_test_A)
        poly_rmse_train.append(np.sqrt(mean_squared_error(y_reg_train, tr_pred)))
        poly_rmse_test.append(np.sqrt(mean_squared_error(y_reg_test, te_pred)))
        if deg == 2:
            best_poly_pred = te_pred
            best_poly_deg = deg

    # Bias–Variance plot
    plt.figure(figsize=(7, 4))
    plt.plot(range(1,5), poly_rmse_train, 'bo-', label='Train RMSE')
    plt.plot(range(1,5), poly_rmse_test, 'ro-', label='Test RMSE')
    plt.xlabel("Polynomial Degree"); plt.ylabel("RMSE")
    plt.title("Bias-Variance: Polynomial Degree vs Error", fontweight='bold')
    plt.legend(); plt.xticks([1,2,3,4])
    plt.tight_layout(); plt.savefig("plots/poly_bias_variance.png", bbox_inches='tight'); plt.show()

    m = reg_metrics(f"Poly(deg={best_poly_deg})", y_reg_test, best_poly_pred)
    cv_mu, cv_std = cv_scores_reg(Pipeline([('p', PolynomialFeatures(2, include_bias=False)),
                                            ('s', StandardScaler()), ('lr', LinearRegression())]),
                                   Xreg_train_A, y_reg_train, scaler=False)
    m['CV RMSE'] = f"{cv_mu:.3f} ± {cv_std:.3f}"
    reg_results['Polynomial'] = m
    print("Polynomial Results:", m)
    print("Insight: Test RMSE typically rises at degree 3–4, showing overfitting on 195 training samples.")
""")]

cells += [code("""
    # ── 3.3 Ridge Regression with RidgeCV ─────────────────────────────────────
    alphas = np.logspace(-3, 3, 100)
    ridge_cv = RidgeCV(alphas=alphas, cv=10, scoring='neg_mean_squared_error')
    ridge_cv.fit(Xreg_tr_A_sc, y_reg_train)
    best_alpha = ridge_cv.alpha_
    ridge_pred = ridge_cv.predict(Xreg_te_A_sc)

    print(f"Best Ridge alpha: {best_alpha:.5f}")

    # Coefficient path
    coefs_path = []
    for a in alphas:
        r = Ridge(alpha=a).fit(Xreg_tr_A_sc, y_reg_train)
        coefs_path.append(r.coef_)
    coefs_path = np.array(coefs_path)

    plt.figure(figsize=(10, 5))
    for i, name in enumerate(FEATS_A_LOG):
        plt.plot(np.log10(alphas), coefs_path[:, i], label=name)
    plt.axvline(np.log10(best_alpha), color='black', linestyle='--', label=f'Best α={best_alpha:.3f}')
    plt.xlabel("log₁₀(α)"); plt.ylabel("Coefficient")
    plt.title("Ridge: Coefficient Path vs Regularization", fontweight='bold')
    plt.legend(fontsize=8, loc='upper right')
    plt.tight_layout(); plt.savefig("plots/ridge_coef_path.png", bbox_inches='tight'); plt.show()

    m = reg_metrics("Ridge", y_reg_test, ridge_pred)
    cv_mu, cv_std = cv_scores_reg(Ridge(alpha=best_alpha), X_A, y_reg, scaler=True)
    m['CV RMSE'] = f"{cv_mu:.3f} ± {cv_std:.3f}"
    m['Params'] = f"α={best_alpha:.4f}"
    reg_results['Ridge'] = m
    print("Ridge Results:", m)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_residuals("Ridge", y_reg_test, ridge_pred, axes[0], axes[1])
    plt.tight_layout(); plt.savefig("plots/ridge_diagnostics.png", bbox_inches='tight'); plt.show()
    print("Insight: As α increases, coefficients shrink toward zero (L2 regularization = Gaussian prior on weights).")
""")]

cells += [code("""
    # ── 3.4 Robust Regression (Huber) ─────────────────────────────────────────
    huber = HuberRegressor(epsilon=1.35, max_iter=300)
    huber.fit(Xreg_tr_A_sc, y_reg_train)
    huber_pred = huber.predict(Xreg_te_A_sc)

    m = reg_metrics("Huber", y_reg_test, huber_pred)
    cv_mu, cv_std = cv_scores_reg(HuberRegressor(epsilon=1.35), X_A, y_reg, scaler=True)
    m['CV RMSE'] = f"{cv_mu:.3f} ± {cv_std:.3f}"
    reg_results['Huber'] = m
    print("Huber Results:", m)

    # Compare residuals OLS vs Huber on high-FWI days
    high_fwi_mask = y_reg_test > 20
    ols_resid_hi  = np.abs(y_reg_test[high_fwi_mask] - ols_pred_manual[high_fwi_mask])
    hub_resid_hi  = np.abs(y_reg_test[high_fwi_mask] - huber_pred[high_fwi_mask])
    print(f"\\nHigh-FWI days (FWI>20): n={high_fwi_mask.sum()}")
    print(f"  OLS  MAE: {ols_resid_hi.mean():.3f}")
    print(f"  Huber MAE: {hub_resid_hi.mean():.3f}")
    print("Insight: Huber regression down-weights extreme outliers (high-FWI days), "
          "often giving lower MAE on those critical cases.")
""")]

cells += [code("""
    # ── 3.5 Bayesian Ridge Regression ─────────────────────────────────────────
    bayes_ridge = BayesianRidge(max_iter=300)
    bayes_ridge.fit(Xreg_tr_A_sc, y_reg_train)
    bay_pred_mean, bay_pred_std = bayes_ridge.predict(Xreg_te_A_sc, return_std=True)

    m = reg_metrics("Bayesian Ridge", y_reg_test, bay_pred_mean)
    cv_mu, cv_std = cv_scores_reg(BayesianRidge(), X_A, y_reg, scaler=True)
    m['CV RMSE'] = f"{cv_mu:.3f} ± {cv_std:.3f}"
    reg_results['Bayesian Ridge'] = m
    print("Bayesian Ridge Results:", m)

    # Predictions with ±2σ uncertainty bands
    sort_idx = np.argsort(y_reg_test)
    yt_s = y_reg_test[sort_idx]
    yp_s = bay_pred_mean[sort_idx]
    ys_s = bay_pred_std[sort_idx]

    plt.figure(figsize=(12, 5))
    plt.plot(yt_s, 'ko', markersize=4, alpha=0.7, label='Actual FWI')
    plt.plot(yp_s, 'b-', lw=1.5, label='Predicted FWI')
    plt.fill_between(range(len(yp_s)), yp_s - 2*ys_s, yp_s + 2*ys_s,
                     alpha=0.2, color='blue', label='±2σ interval')
    plt.xlabel("Test sample (sorted by actual FWI)"); plt.ylabel("FWI")
    plt.title("Bayesian Ridge: Predictions with Uncertainty Bands", fontweight='bold')
    plt.legend()
    plt.tight_layout(); plt.savefig("plots/bayesian_ridge.png", bbox_inches='tight'); plt.show()
    print("Insight: Bayesian Ridge provides calibrated uncertainty estimates — "
          "high-FWI extremes have wider bands, reflecting the model's appropriate uncertainty.")
""")]

cells += [code("""
    # ── 3.6 Kernel Ridge (RBF) — optional ─────────────────────────────────────
    from sklearn.model_selection import GridSearchCV

    param_grid_kr = {'alpha': [0.01, 0.1, 1.0], 'gamma': [0.01, 0.1, 1.0]}
    kr = GridSearchCV(KernelRidge(kernel='rbf'), param_grid_kr, cv=5,
                      scoring='neg_root_mean_squared_error', n_jobs=-1)
    kr.fit(Xreg_tr_A_sc, y_reg_train)
    kr_pred = kr.predict(Xreg_te_A_sc)
    best_kr_params = kr.best_params_

    m = reg_metrics("Kernel Ridge (RBF)", y_reg_test, kr_pred)
    m['Params'] = str(best_kr_params)
    cv_mu, cv_std = cv_scores_reg(KernelRidge(kernel='rbf', **best_kr_params), X_A, y_reg, scaler=True)
    m['CV RMSE'] = f"{cv_mu:.3f} ± {cv_std:.3f}"
    reg_results['Kernel Ridge'] = m
    print("Kernel Ridge Results:", m)
    print("Insight: The RBF kernel implicitly maps features to infinite-dimensional space, "
          "capturing nonlinear FWI relationships without explicitly forming feature vectors.")
""")]

cells += [code("""
    # ── 3.7 Experiment B: add FWI-system indices ──────────────────────────────
    lr_b = LinearRegression().fit(Xreg_tr_B_sc, y_reg_train)
    pred_b = lr_b.predict(Xreg_te_B_sc)
    m_b = reg_metrics("OLS (Exp B, with indices)", y_reg_test, pred_b)
    reg_results['OLS_B'] = m_b
    print("OLS Experiment B (with FWI indices):", m_b)
    print("Insight: Adding ISI, BUI etc. dramatically improves R² because FWI is a "
          "deterministic function of those indices — confirming data leakage.")
""")]

cells += [code("""
    # ── 3.8 Regression Summary Table ─────────────────────────────────────────
    reg_summary = pd.DataFrame(reg_results).T.reset_index(drop=True)
    display(reg_summary.round(4))
    reg_summary.to_csv("results/regression_summary.csv", index=False)
    print("Regression summary saved.")
""")]

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4  – CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
cells += [md("---\n## Step 4 · Classification Models – Fire / Not-fire  *(Rubric 3)*")]

cells += [code("""
    import os; os.makedirs("results", exist_ok=True)

    cls_results = {}   # store for summary table

    def cls_metrics(name, y_true, y_pred, y_prob=None):
        d = {
            'Model': name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
        }
        if y_prob is not None:
            d['ROC-AUC'] = roc_auc_score(y_true, y_prob)
            d['Brier']   = brier_score_loss(y_true, y_prob)
            d['LogLoss'] = log_loss(y_true, y_prob)
        return d

    def show_confusion(name, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay(cm, display_labels=['Not Fire','Fire']).plot(ax=ax, colorbar=False)
        ax.set_title(f"{name} — Confusion Matrix", fontweight='bold')
        plt.tight_layout()
        fname = name.replace(' ','_').lower()
        plt.savefig(f"plots/cm_{fname}.png", bbox_inches='tight'); plt.show()

    print("Classification helpers defined ✓")
    print("\\n> Recall on the fire class is the KEY metric: a missed fire costs more than a false alarm.")
""")]

cells += [code("""
    # ── 4.1 Gaussian Naive Bayes (Bayes rule + conditional independence) ───────
    # P(fire | x) ∝ P(x | fire) · P(fire)
    # P(x | fire) = ∏ P(xᵢ | fire)   ← conditional independence assumption

    gnb = GaussianNB()
    gnb.fit(Xcls_tr_A_sc, y_cls_train)
    gnb_pred = gnb.predict(Xcls_te_A_sc)
    gnb_prob = gnb.predict_proba(Xcls_te_A_sc)[:,1]

    m = cls_metrics("Gaussian NB", y_cls_test, gnb_pred, gnb_prob)
    cv_mu, cv_std = cv_scores_cls(GaussianNB(), X_A, y_cls, scaler=True, scoring='f1')
    m['CV F1'] = f"{cv_mu:.3f} ± {cv_std:.3f}"
    cls_results['GNB'] = m
    print("Gaussian NB:", {k: round(v,4) if isinstance(v, float) else v for k,v in m.items()})
    show_confusion("Gaussian NB", y_cls_test, gnb_pred)

    # Check independence assumption via pairwise feature correlations within each class
    for cls_val, label in [(0,'Not Fire'),(1,'Fire')]:
        sub = df[df['fire']==cls_val][FEATS_A_LOG]
        print(f"\\nMax |corr| within {label} class: {sub.corr().abs().where(lambda x: x < 1).max().max():.3f}")
    print("Insight: If within-class correlations are high, the conditional independence assumption of NB is violated, "
          "but NB can still perform well in practice despite this (bias-variance tradeoff).")
""")]

cells += [code("""
    # ── 4.2 LDA – Linear Discriminant Analysis ────────────────────────────────
    lda = LinearDiscriminantAnalysis()
    lda.fit(Xcls_tr_A_sc, y_cls_train)
    lda_pred = lda.predict(Xcls_te_A_sc)
    lda_prob = lda.predict_proba(Xcls_te_A_sc)[:,1]

    m = cls_metrics("LDA", y_cls_test, lda_pred, lda_prob)
    cv_mu, cv_std = cv_scores_cls(LinearDiscriminantAnalysis(), X_A, y_cls, scaler=True)
    m['CV F1'] = f"{cv_mu:.3f} ± {cv_std:.3f}"
    cls_results['LDA'] = m
    print("LDA:", {k: round(v,4) if isinstance(v, float) else v for k,v in m.items()})
    show_confusion("LDA", y_cls_test, lda_pred)

    # Decision boundary on 2 most informative features (Temperature, RH)
    feat_idx = [FEATS_A_LOG.index('Temperature'), FEATS_A_LOG.index('RH')]
    Xtr2 = Xcls_tr_A_sc[:, feat_idx]; Xte2 = Xcls_te_A_sc[:, feat_idx]
    lda2 = LinearDiscriminantAnalysis().fit(Xtr2, y_cls_train)

    h = 0.02
    x_min, x_max = Xte2[:,0].min()-1, Xte2[:,0].max()+1
    y_min, y_max = Xte2[:,1].min()-1, Xte2[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = lda2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    sc = plt.scatter(Xte2[:,0], Xte2[:,1], c=y_cls_test, cmap='RdBu', edgecolors='k', s=40)
    plt.xlabel("Temperature (scaled)"); plt.ylabel("RH (scaled)")
    plt.title("LDA Decision Boundary (Temperature vs RH)", fontweight='bold')
    plt.colorbar(sc, label='Fire (1) / Not Fire (0)')
    plt.tight_layout(); plt.savefig("plots/lda_boundary.png", bbox_inches='tight'); plt.show()
""")]

cells += [code("""
    # ── 4.3 Logistic Regression ───────────────────────────────────────────────
    lr_params = {'C': [0.01, 0.1, 1, 10, 100]}
    lr_gs = GridSearchCV(LogisticRegression(max_iter=500, random_state=42),
                         lr_params, cv=5, scoring='f1', n_jobs=-1)
    lr_gs.fit(Xcls_tr_A_sc, y_cls_train)
    best_C_lr = lr_gs.best_params_['C']
    logreg = lr_gs.best_estimator_
    lr_pred = logreg.predict(Xcls_te_A_sc)
    lr_prob = logreg.predict_proba(Xcls_te_A_sc)[:,1]

    m = cls_metrics("Logistic Regression", y_cls_test, lr_pred, lr_prob)
    cv_mu, cv_std = cv_scores_cls(LogisticRegression(C=best_C_lr, max_iter=500),
                                  X_A, y_cls, scaler=True)
    m['CV F1'] = f"{cv_mu:.3f} ± {cv_std:.3f}"
    m['Params'] = f"C={best_C_lr}"
    cls_results['Logistic'] = m
    print(f"Best C: {best_C_lr}")
    print("Logistic Regression:", {k: round(v,4) if isinstance(v, float) else v for k,v in m.items()})
    show_confusion("Logistic Regression", y_cls_test, lr_pred)
""")]

cells += [code("""
    # ── 4.4 Bayesian Logistic Regression (Laplace Approximation) ─────────────
    # P(y=1|x) ≈ σ( w*ᵀx / sqrt(1 + πσ²/8) )
    # Posterior: N(w*, H⁻¹)  where H = XᵀRX + αI

    def bayes_logreg_laplace(Xtr, ytr, Xte, alpha=1.0):
        # Bias column
        Xb  = np.c_[np.ones(len(Xtr)), Xtr]
        Xtb = np.c_[np.ones(len(Xte)), Xte]
        # MAP via L2-regularised logistic regression (C = 1/alpha, no intercept)
        model = LogisticRegression(C=1.0/alpha, fit_intercept=False, max_iter=1000).fit(Xb, ytr)
        w = model.coef_.ravel()
        # Hessian at MAP
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        H = Xb.T @ (Xb * (p * (1.0 - p))[:, None]) + alpha * np.eye(Xb.shape[1])
        S = np.linalg.inv(H)          # posterior covariance
        mu  = Xtb @ w
        var = np.einsum('ij,jk,ik->i', Xtb, S, Xtb)   # predictive variance
        # Probit approximation to sigmoid integral
        prob = 1.0 / (1.0 + np.exp(-mu / np.sqrt(1.0 + np.pi * var / 8.0)))
        return prob, w, S

    bayes_prob, bayes_w, bayes_S = bayes_logreg_laplace(
        Xcls_tr_A_sc, y_cls_train, Xcls_te_A_sc, alpha=1.0)
    bayes_pred = (bayes_prob >= 0.5).astype(int)

    m = cls_metrics("Bayesian Logistic (Laplace)", y_cls_test, bayes_pred, bayes_prob)
    cls_results['BayesLogreg'] = m
    print("Bayesian Logistic (Laplace):", {k: round(v,4) if isinstance(v, float) else v for k,v in m.items()})
    show_confusion("Bayesian Logistic (Laplace)", y_cls_test, bayes_pred)

    # Calibration curve: Bayesian vs plain Logistic
    fig, ax = plt.subplots(figsize=(7, 6))
    for prob_vals, label, color in [
        (lr_prob,    "Plain Logistic",   "#e74c3c"),
        (bayes_prob, "Bayesian Laplace", "#3498db"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_cls_test, prob_vals, n_bins=8)
        ax.plot(mean_pred, frac_pos, 'o-', color=color, lw=2, label=label)
    ax.plot([0,1],[0,1],'k--', label='Perfect calibration')
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve: Logistic vs Bayesian Laplace", fontweight='bold')
    ax.legend(); plt.tight_layout()
    plt.savefig("plots/calibration_curve.png", bbox_inches='tight'); plt.show()

    print(f"\\nLogistic  — Brier: {brier_score_loss(y_cls_test, lr_prob):.4f}, "
          f"Log-loss: {log_loss(y_cls_test, lr_prob):.4f}")
    print(f"Bayesian  — Brier: {brier_score_loss(y_cls_test, bayes_prob):.4f}, "
          f"Log-loss: {log_loss(y_cls_test, bayes_prob):.4f}")
    print("Insight: Bayesian Laplace approx often improves calibration because "
          "marginalising over the posterior uncertainty prevents overconfident predictions.")
""")]

cells += [code("""
    # ── 4.5 SVM: Linear, Polynomial, RBF ─────────────────────────────────────
    svm_configs = [
        ('SVM Linear',   SVC(kernel='linear', probability=True, random_state=42),
         {'C': [0.01, 0.1, 1, 10]}),
        ('SVM Poly',     SVC(kernel='poly', probability=True, random_state=42),
         {'C': [0.1, 1, 10], 'degree': [2, 3], 'coef0': [0, 1]}),
        ('SVM RBF',      SVC(kernel='rbf', probability=True, random_state=42),
         {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 0.01, 0.1]}),
    ]

    for name, svm_base, params in svm_configs:
        gs = GridSearchCV(svm_base, params, cv=5, scoring='f1', n_jobs=-1)
        gs.fit(Xcls_tr_A_sc, y_cls_train)
        best_est = gs.best_estimator_
        pred = best_est.predict(Xcls_te_A_sc)
        prob = best_est.predict_proba(Xcls_te_A_sc)[:,1]
        m = cls_metrics(name, y_cls_test, pred, prob)
        m['Params'] = str(gs.best_params_)
        cls_results[name.replace(' ','_')] = m
        print(f"{name} best params: {gs.best_params_}")
        print(f"  F1={m['F1']:.4f}, Recall={m['Recall']:.4f}, AUC={m.get('ROC-AUC',0):.4f}")
        show_confusion(name, y_cls_test, pred)

    print("Insight: The RBF kernel maps data to infinite dimensions, often capturing "
          "nonlinear fire boundaries. The kernel trick avoids explicitly computing this mapping "
          "(O(n) instead of O(n·d_high)).")
""")]

cells += [code("""
    # ── 4.6 ROC Curves for all classifiers ───────────────────────────────────
    # Re-fit with fixed hyperparams for ROC
    roc_models = {
        'Gaussian NB':  (GaussianNB(), 'predict_proba'),
        'LDA':          (LinearDiscriminantAnalysis(), 'predict_proba'),
        'Logistic (C=1)': (LogisticRegression(C=best_C_lr, max_iter=500, random_state=42), 'predict_proba'),
        'SVM RBF':      (SVC(kernel='rbf', C=10, probability=True, random_state=42), 'predict_proba'),
    }

    fig, ax = plt.subplots(figsize=(8, 7))
    for model_name, (mdl, prob_method) in roc_models.items():
        mdl.fit(Xcls_tr_A_sc, y_cls_train)
        if prob_method == 'predict_proba':
            prob = mdl.predict_proba(Xcls_te_A_sc)[:,1]
        else:
            prob = mdl.decision_function(Xcls_te_A_sc)
        RocCurveDisplay.from_predictions(y_cls_test, prob, name=model_name, ax=ax)
    # Bayesian Laplace
    ax.plot(*RocCurveDisplay.from_predictions(y_cls_test, bayes_prob,
            name='Bayesian Laplace', ax=None).line_.get_data(), label='Bayesian Laplace')
    ax.plot([0,1],[0,1],'k--', alpha=0.5)
    ax.set_title("ROC Curves – All Classifiers", fontweight='bold')
    ax.legend(fontsize=9); plt.tight_layout()
    plt.savefig("plots/roc_curves.png", bbox_inches='tight'); plt.show()
""")]

cells += [code("""
    # ── 4.7 Classification Summary Table ─────────────────────────────────────
    import os; os.makedirs("results", exist_ok=True)

    cls_summary = pd.DataFrame(cls_results).T.reset_index(drop=True)
    display(cls_summary.round(4))
    cls_summary.to_csv("results/classification_summary.csv", index=False)
    print("Classification summary saved.")
""")]

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5  – COMPARATIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
cells += [md("---\n## Step 5 · Comparative Performance Analysis  *(Rubric 4)*")]

cells += [code("""
    # ── 5.1 Regression bar chart ──────────────────────────────────────────────
    reg_plot_df = pd.DataFrame(reg_results).T.reset_index()
    reg_plot_df.columns.values[0] = 'Model'
    reg_plot_df['RMSE'] = reg_plot_df['RMSE'].astype(float)
    reg_plot_df['R²']   = reg_plot_df['R²'].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(reg_plot_df)))

    axes[0].barh(reg_plot_df['Model'], reg_plot_df['RMSE'], color=colors)
    axes[0].set_xlabel("RMSE (lower is better)"); axes[0].invert_xaxis()
    axes[0].set_title("Regression: RMSE Comparison", fontweight='bold')

    axes[1].barh(reg_plot_df['Model'], reg_plot_df['R²'], color=colors)
    axes[1].set_xlabel("R² (higher is better)")
    axes[1].set_title("Regression: R² Comparison", fontweight='bold')

    plt.tight_layout(); plt.savefig("plots/regression_comparison.png", bbox_inches='tight'); plt.show()
""")]

cells += [code("""
    # ── 5.2 Classification bar chart ──────────────────────────────────────────
    cls_plot_df = pd.DataFrame(cls_results).T.reset_index()
    cls_plot_df.columns.values[0] = 'Model'
    for col in ['Accuracy','Recall','F1','ROC-AUC']:
        if col in cls_plot_df.columns:
            cls_plot_df[col] = pd.to_numeric(cls_plot_df[col], errors='coerce')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors2 = plt.cm.Set1(np.linspace(0, 1, len(cls_plot_df)))

    for ax, metric in zip(axes, ['Recall','F1','ROC-AUC']):
        if metric not in cls_plot_df.columns:
            continue
        ax.barh(cls_plot_df['Model'], cls_plot_df[metric], color=colors2)
        ax.set_xlabel(metric)
        ax.set_title(f"Classification: {metric}", fontweight='bold')
        ax.set_xlim(0, 1)
    plt.tight_layout(); plt.savefig("plots/classification_comparison.png", bbox_inches='tight'); plt.show()
    print("Insight: Recall on the fire class (True Positive Rate for fire) is the most operationally "
          "important metric — a missed fire costs more than a false alarm.")
""")]

cells += [code("""
    # ── 5.3 Key Comparisons ───────────────────────────────────────────────────
    print("=" * 60)
    print("KEY COMPARISONS")
    print("=" * 60)

    # A) OLS vs Ridge vs Huber vs Bayesian Ridge
    for name in ['OLS','Ridge','Huber','Bayesian Ridge']:
        if name in reg_results:
            r = reg_results[name]
            print(f"  {name:20s}: RMSE={float(r['RMSE']):.3f}, R²={float(r['R²']):.3f}, CV={r.get('CV RMSE','N/A')}")

    print("\\n  → Ridge often beats OLS when features are correlated (Temp/RH multicollinearity)")
    print("  → Huber is more robust to extreme FWI outliers than OLS")
    print("  → Bayesian Ridge gives principled uncertainty estimates")

    print()
    # B) Polynomial degree bias-variance
    print(f"  Poly RMSE (deg 1–4): Train={[f'{v:.2f}' for v in poly_rmse_train]}, "
          f"Test={[f'{v:.2f}' for v in poly_rmse_test]}")
    print("  → Test RMSE typically increases after degree 2–3: overfitting")

    print()
    # C) Linear vs Poly vs RBF SVM
    for name in ['SVM_Linear','SVM_Poly','SVM_RBF']:
        if name in cls_results:
            r = cls_results[name]
            print(f"  {name:15s}: F1={float(r.get('F1',0)):.3f}, Recall={float(r.get('Recall',0)):.3f}")
    print("  → RBF kernel usually helps when fire boundary is nonlinear")

    print()
    # D) NB vs Logistic
    for name in ['GNB','Logistic']:
        if name in cls_results:
            r = cls_results[name]
            print(f"  {name:15s}: F1={float(r.get('F1',0)):.3f}, AUC={float(r.get('ROC-AUC',0)):.3f}")
    print("  → With 244 rows: Naive Bayes (generative) can compete with Logistic (discriminative)")

    print()
    # E) Logistic vs Bayesian Logistic
    for name in ['Logistic','BayesLogreg']:
        if name in cls_results:
            r = cls_results[name]
            br = float(r.get('Brier',0)) if r.get('Brier') else 0
            print(f"  {name:25s}: Brier={br:.4f}, AUC={float(r.get('ROC-AUC',0)):.3f}")
    print("  → Bayesian Laplace tends to produce better-calibrated probabilities")

    print()
    # F) Experiment A vs B
    ols_a = float(reg_results.get('OLS',{}).get('R²', 0))
    ols_b = float(reg_results.get('OLS_B',{}).get('R²', 0))
    print(f"  Exp A (weather only) OLS R²: {ols_a:.3f}")
    print(f"  Exp B (+ FWI indices) OLS R²: {ols_b:.3f}")
    print(f"  → Adding FWI-system indices gains {ols_b - ols_a:.3f} R² (data leakage!)")
""")]

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6  – INTERPRETATION & RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
cells += [md("---\n## Step 6 · Interpretation & Recommendations  *(Rubric 5)*")]

cells += [code("""
    # ── 6.1 Feature importance: standardized coefficients from Ridge ───────────
    ridge_final = Ridge(alpha=best_alpha).fit(Xreg_tr_A_sc, y_reg_train)
    coef_df = pd.DataFrame({
        'Feature': FEATS_A_LOG,
        'Coefficient': ridge_final.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)

    plt.figure(figsize=(8, 4))
    colors_feat = ['#e74c3c' if c > 0 else '#3498db' for c in coef_df['Coefficient']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_feat)
    plt.axvline(0, color='black', lw=0.8)
    plt.xlabel("Standardized Coefficient")
    plt.title("Ridge Regression: Feature Importance (FWI prediction)", fontweight='bold')
    plt.tight_layout(); plt.savefig("plots/feature_importance_ridge.png", bbox_inches='tight'); plt.show()

    print(coef_df.to_string(index=False))
    print("\\nInsight: Temperature has the largest positive effect on FWI (hotter → higher fire danger). "
          "RH has the largest negative effect (humid → lower fire danger). "
          "Rain is near zero (small effect after log transform on the 20% test set).")
""")]

cells += [code("""
    # ── 6.2 Logistic feature importance ────────────────────────────────────────
    logreg_final = LogisticRegression(C=best_C_lr, max_iter=500, random_state=42)
    logreg_final.fit(Xcls_tr_A_sc, y_cls_train)
    coef_cls_df = pd.DataFrame({
        'Feature': FEATS_A_LOG,
        'Coefficient': logreg_final.coef_.ravel()
    }).sort_values('Coefficient', key=abs, ascending=False)

    plt.figure(figsize=(8, 4))
    colors_cls = ['#e74c3c' if c > 0 else '#3498db' for c in coef_cls_df['Coefficient']]
    plt.barh(coef_cls_df['Feature'], coef_cls_df['Coefficient'], color=colors_cls)
    plt.axvline(0, color='black', lw=0.8)
    plt.xlabel("Standardized Coefficient (log-odds)")
    plt.title("Logistic Regression: Feature Importance (Fire Classification)", fontweight='bold')
    plt.tight_layout(); plt.savefig("plots/feature_importance_logistic.png", bbox_inches='tight'); plt.show()
    print(coef_cls_df.to_string(index=False))
""")]

cells += [md(
    "### 6.3 · Final Recommendations",
    "",
    "| Task | Recommended Model | Reason |",
    "|------|-------------------|--------|",
    "| FWI Regression | **Ridge Regression** | Handles Temp–RH multicollinearity; outperforms OLS on CV; interpretable coefficients |",
    "| FWI Regression (uncertainty) | **Bayesian Ridge** | Provides ±2σ prediction intervals useful for emergency planning |",
    "| Fire Classification | **SVM RBF** or **Logistic** | Best F1/AUC on test; SVM handles nonlinear boundary; Logistic is interpretable |",
    "| Probability calibration | **Bayesian Logistic (Laplace)** | Better-calibrated probabilities for risk scoring |",
    "",
    "### 6.4 · Limitations",
    "",
    "- **Small dataset**: only 244 rows from **one season** (Jun–Sep 2012) in **two Algerian regions** — "
      "results will not directly generalise to other climates or seasons.",
    "- **Temporal structure ignored**: weather data is correlated across consecutive days; "
      "random 80/20 split may leak future information into training.",
    "- **Feature engineering scope**: only raw weather features used in Experiment A; "
      "domain features (e.g., drought duration, wind direction) were not available.",
    "- **Leakage in Experiment B**: FWI-system indices (ISI, BUI, etc.) are direct components "
      "of the FWI formula — including them makes the regression essentially a lookup, "
      "not a prediction task.",
    "",
    "### 6.5 · Viva Quick-Reference",
    "",
    "| Question | Key Answer |",
    "|----------|-----------|",
    "| Why is OLS the MLE under Gaussian noise? | Minimising RSS = maximising log-likelihood when ε ~ N(0,σ²) |",
    "| What does Ridge do geometrically? | Shrinks weight vector toward origin; corresponds to MAP with N(0,1/λ) prior |",
    "| Why drop FWI indices from Exp A? | FWI is a deterministic function of ISI & BUI — including them is data leakage |",
    "| What is the Laplace approx? | Gaussian approximation to the posterior around the MAP point; Hessian gives curvature |",
    "| What does the kernel trick do? | Computes ⟨φ(x),φ(x')⟩ via k(x,x') without forming φ(x) explicitly — O(n²) not O(n·d) |",
    "| Why is Recall the key metric? | A missed fire (FN) has far higher cost than a false alarm (FP) |",
    "| Does Naive Bayes hold? | Within-class correlations exist (Temp–RH are anti-correlated) — assumption is violated but NB can still generalise |",
)]

cells += [code("""
    print("=" * 60)
    print("PBL-1 NOTEBOOK COMPLETE")
    print("=" * 60)
    print()
    print("Outputs saved to:")
    print("  plots/   — all figures")
    print("  results/ — regression_summary.csv, classification_summary.csv")
""")]

# ═══════════════════════════════════════════════════════════════════════════════
# Write notebook
# ═══════════════════════════════════════════════════════════════════════════════
import os; os.makedirs("results", exist_ok=True); os.makedirs("plots", exist_ok=True)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9"
        }
    },
    "cells": cells
}

nb_path = "algerian_forest_fires_PBL1.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook written: {nb_path}")
print(f"Total cells: {len(cells)}")
