#!/usr/bin/env python3
"""
patch_docx.py
Edits PBL1_Forest_Fire_Implementation_Plan.docx:
  1. Removes all "viva" mentions (section + inline text)
  2. Updates timeline Day 6
  3. Removes "good point for comparative analysis and viva"
  4. Adds "Visuals checklist" table
  5. Adds plot scheduling to timeline
"""

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy, re, os

DOCX_PATH = r"C:\Users\samhi\Downloads\PBL1_Forest_Fire_Implementation_Plan.docx"

doc = Document(DOCX_PATH)

# ─────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────

def para_text(para):
    return para.text.strip()

def para_contains(para, term):
    return term.lower() in para.text.lower()

def remove_para(para):
    p = para._element
    p.getparent().remove(p)

def replace_text_in_para(para, old, new):
    """Replace 'old' with 'new' in a paragraph, preserving runs as much as possible."""
    full = para.text
    if old.lower() not in full.lower():
        return False
    # Simple approach: rebuild the first run with the new text, clear the rest
    new_full = re.sub(re.escape(old), new, full, flags=re.IGNORECASE)
    for i, run in enumerate(para.runs):
        if i == 0:
            run.text = new_full
        else:
            run.text = ""
    return True

def add_table_after_para(doc, target_para, headers, rows_data):
    """Add a formatted table immediately after target_para."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    table = doc.add_table(rows=1 + len(rows_data), cols=len(headers))
    table.style = 'Table Grid'

    # Header row
    hdr_row = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr_row.cells[i]
        cell.text = h
        for run in cell.paragraphs[0].runs:
            run.bold = True

    # Data rows
    for r_idx, row_data in enumerate(rows_data):
        row = table.rows[r_idx + 1]
        for c_idx, val in enumerate(row_data):
            row.cells[c_idx].text = str(val)

    # Move table immediately after the target paragraph
    tbl_element = table._tbl
    target_para._element.addnext(tbl_element)
    return table

# ─────────────────────────────────────────────────────────────
# Step 1: Remove "viva" sections and mentions
# ─────────────────────────────────────────────────────────────

VIVA_REPLACEMENTS = {
    "good point for comparative analysis and viva": "good point for comparative analysis",
    "Report/PPT, viva prep": "Report/PPT, final checks",
    "report/ppt, viva prep": "Report/PPT, final checks",
}

paras_to_remove = []

for para in doc.paragraphs:
    pt = para_text(para)

    # Mark viva heading / section for removal
    if re.match(r'.*viva\s*(question|quick)', pt, re.IGNORECASE):
        paras_to_remove.append(para)
        continue

    # Check if this is a table or bullet that is pure-viva content
    if pt.lower().startswith('viva') and '?' in pt:
        paras_to_remove.append(para)
        continue

    # Replace inline viva phrases
    for old, new in VIVA_REPLACEMENTS.items():
        replace_text_in_para(para, old, new)

    # Generic: remove remaining standalone "viva" mentions (not in mid-sentence)
    if para_contains(para, 'viva'):
        # replace word 'viva' standalone
        new_text = re.sub(r'\bviva\b', '', para.text, flags=re.IGNORECASE).strip()
        new_text = re.sub(r'\s{2,}', ' ', new_text)
        if para.runs:
            para.runs[0].text = new_text
            for run in para.runs[1:]:
                run.text = ''

# Remove collected viva paragraphs
for para in paras_to_remove:
    try:
        remove_para(para)
    except Exception:
        pass

# Also check table cells
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                pt = para_text(para)
                for old, new in VIVA_REPLACEMENTS.items():
                    replace_text_in_para(para, old, new)
                if para_contains(para, 'viva') and para.runs:
                    new_text = re.sub(r'\bviva\b', '', para.text, flags=re.IGNORECASE).strip()
                    new_text = re.sub(r'\s{2,}', ' ', new_text)
                    para.runs[0].text = new_text
                    for run in para.runs[1:]:
                        run.text = ''

print("[OK] Viva references removed/replaced")

# ─────────────────────────────────────────────────────────────
# Step 2: Add "Visuals Checklist" table section
# ─────────────────────────────────────────────────────────────

VISUALS_CHECKLIST = [
    ("Fig 1.1",  "Box-plots of all features",                         "Step 1 – EDA"),
    ("Fig 1.2",  "Histograms + KDE",                                   "Step 1 – EDA"),
    ("Fig 1.3",  "Pair plot coloured by class (KDE diagonal)",         "Step 1 – EDA"),
    ("Fig 1.4",  "Temperature vs RH scatter (size=FWI, cov ellipse)",  "Step 1 – EDA"),
    ("Fig 1.5",  "Class-conditional KDEs P(x|fire) vs P(x|not fire)", "Step 1 – EDA"),
    ("Fig 1.6",  "Bayes-rule posterior curve P(fire|Temperature)",     "Step 1 – EDA"),
    ("Fig 1.7",  "P(fire|bin) bar chart with bootstrap CIs",           "Step 1 – EDA"),
    ("Fig 1.8",  "Chi-square: observed vs expected counts",            "Step 1 – EDA"),
    ("Fig 1.9",  "Q-Q, ECDF vs fitted CDF, MLE log-likelihood curve", "Step 1 – EDA"),
    ("Fig 1.10", "Before/after log1p(Rain) histograms",               "Step 1 – EDA"),
    ("Fig 1.11", "Daily time-series per region (rolling mean + fire)", "Step 1 – EDA"),
    ("Fig 1.12", "Month × Region heatmaps (fire prop., mean FWI)",    "Step 1 – EDA"),
    ("Fig 1.13", "Violin plots by class",                             "Step 1 – EDA"),
    ("Fig 1.14", "Clustered correlation & covariance heatmaps",        "Step 1 – EDA"),
    ("Fig 1.15", "Polynomial curve fitting FWI vs Temp (deg 1,3,6,9)","Step 1 – EDA"),
    ("Fig 1.16", "Leakage demo: FWI vs ISI & FWI vs BUI",            "Step 1 – EDA"),
    ("Fig 1.17", "Fire counts / proportion by region & month",        "Step 1 – EDA"),
    ("Fig 3.1",  "Basis function plots (poly, Gaussian, sigmoidal)",  "Step 3 – Regression"),
    ("Fig 3.2",  "Gaussian basis regression fit",                     "Step 3 – Regression"),
    ("Fig 3.3",  "OLS residual diagnostics + pred vs actual",         "Step 3 – Regression"),
    ("Fig 3.4",  "Train/val RMSE vs polynomial degree (CV error bars)","Step 3 – Regression"),
    ("Fig 3.5",  "Ridge coeff path vs α + CV error vs α",             "Step 3 – Regression"),
    ("Fig 3.6",  "Huber vs squared loss curve; OLS vs Huber lines",   "Step 3 – Regression"),
    ("Fig 3.7",  "Bayesian Ridge: ±2σ band, prior→posterior, weight CIs","Step 3 – Regression"),
    ("Fig 3.8",  "Kernel Ridge: 3 gamma values + RBF kernel matrix",  "Step 3 – Regression"),
    ("Fig 3.9",  "Learning curves: OLS, Ridge, Bayesian Ridge",       "Step 3 – Regression"),
    ("Fig 3.10", "Experiment A vs B predicted vs actual side-by-side", "Step 3 – Regression"),
    ("Fig 4.1",  "2-D decision boundaries for all classifiers",        "Step 4 – Classification"),
    ("Fig 4.2",  "NB class-conditional Gaussians + confusion matrix",  "Step 4 – Classification"),
    ("Fig 4.3",  "LDA discriminant score histogram + confusion matrix","Step 4 – Classification"),
    ("Fig 4.4",  "Logistic confusion matrix + probability histogram",  "Step 4 – Classification"),
    ("Fig 4.5",  "Bayesian Laplace: MAP vs moderated prob + weight unc","Step 4 – Classification"),
    ("Fig 4.6",  "SVM confusion matrices (Linear, Poly, RBF)",        "Step 4 – Classification"),
    ("Fig 4.7",  "SVM margins, C effect, C×γ grid-search heatmap",    "Step 4 – Classification"),
    ("Fig 4.8",  "Precision-Recall curves + threshold vs metrics",    "Step 4 – Classification"),
    ("Fig 4.9",  "Calibration curves: Logistic vs Bayesian Laplace",  "Step 4 – Classification"),
    ("Fig 4.10", "ROC curves for all classifiers",                    "Step 4 – Classification"),
    ("Fig 5.1",  "Regression CV RMSE dot plot + boxplots",            "Step 5 – Comparison"),
    ("Fig 5.2",  "Classifier metrics heatmap + radar chart",          "Step 5 – Comparison"),
    ("Fig 5.3",  "Training time vs F1 score",                         "Step 5 – Comparison"),
    ("Fig 5.4",  "Per-region grouped bar chart",                      "Step 5 – Comparison"),
    ("Fig 5.5",  "Experiment A vs B RMSE grouped bars",               "Step 5 – Comparison"),
    ("Fig 6.1",  "Ridge feature importance + bootstrap CIs",          "Step 6 – Interpretation"),
    ("Fig 6.2",  "Logistic feature importance + bootstrap CIs",       "Step 6 – Interpretation"),
    ("Fig 6.3",  "Summary dashboard (9 panels)",                      "Step 6 – Interpretation"),
]

# Find a good insertion point (end of document or after "Summary / Appendix" heading)
# Insert at end
doc.add_page_break()
h = doc.add_paragraph()
r = h.add_run("Visuals Checklist")
r.bold = True
r.font.size = Pt(16)

intro = doc.add_paragraph(
    "All figures produced by the notebook are listed below. "
    "Each is saved to plots/ at dpi=200 with a numbered title and Insight cell underneath."
)

# Add table
tbl = doc.add_table(rows=1 + len(VISUALS_CHECKLIST), cols=3)

headers = ["Figure", "Description", "Step"]
hdr = tbl.rows[0]
for i, h_text in enumerate(headers):
    cell = hdr.cells[i]
    cell.text = h_text
    cell.paragraphs[0].runs[0].bold = True

for r_idx, (fig, desc, step) in enumerate(VISUALS_CHECKLIST):
    row = tbl.rows[r_idx + 1]
    row.cells[0].text = fig
    row.cells[1].text = desc
    row.cells[2].text = step

print(f"[OK] Visuals Checklist added ({len(VISUALS_CHECKLIST)} figures)")

# ─────────────────────────────────────────────────────────────
# Step 3: Update timeline to schedule new plots
# ─────────────────────────────────────────────────────────────
TIMELINE_UPDATES = {
    "day 1": "Day 1: Data loading, cleaning, Step 0 complete; EDA statistics (Step 1.1–1.2)",
    "day 2": "Day 2: EDA visuals Figs 1.3–1.10 (pair plot, scatter, KDEs, Bayes posterior, chi-square, Q-Q)",
    "day 3": "Day 3: EDA visuals Figs 1.11–1.17 (time-series, heatmaps, violin, corr, poly fits, leakage demo); Step 2 preprocessing",
    "day 4": "Day 4: Step 3 regression models (OLS→Kernel Ridge); Figs 3.1–3.10",
    "day 5": "Day 5: Step 4 classification (NB, LDA, Logistic, SVM, Bayesian Laplace); Figs 4.1–4.10",
    "day 6": "Day 6: Step 5 comparison (Figs 5.1–5.5); Step 6 feature importance, dashboard (Figs 6.1–6.3); Report/PPT, final checks",
}

for para in doc.paragraphs:
    pt = para_text(para).lower()
    for day_key, new_text in TIMELINE_UPDATES.items():
        if pt.startswith(day_key + ':') or pt.startswith(day_key + ' '):
            if para.runs:
                para.runs[0].text = new_text
                for run in para.runs[1:]:
                    run.text = ''
            break

# Also update timeline in tables
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                pt = para_text(para).lower()
                for day_key, new_text in TIMELINE_UPDATES.items():
                    if pt.startswith(day_key + ':') or pt.startswith(day_key + ' '):
                        if para.runs:
                            para.runs[0].text = new_text
                            for run in para.runs[1:]:
                                run.text = ''
                        break

print("[OK] Timeline updated")

# ─────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────
doc.save(DOCX_PATH)
print(f"[OK] Document saved to {DOCX_PATH}")

# Final viva check
doc2 = Document(DOCX_PATH)
viva_remaining = []
for para in doc2.paragraphs:
    if 'viva' in para.text.lower():
        viva_remaining.append(para.text.strip()[:80])
for table in doc2.tables:
    for row in table.rows:
        for cell in row.cells:
            if 'viva' in cell.text.lower():
                viva_remaining.append(cell.text.strip()[:80])
if viva_remaining:
    print("[WARN]  Remaining 'viva' mentions (may be in table cells):")
    for v in viva_remaining:
        print("   ", v)
else:
    print("[OK] No 'viva' mentions remaining")
