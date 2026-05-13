# Student Performance Early Warning System
### OSEMN Pipeline · Random Forest · Flask Demo

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-83.3%25-22c55e?style=flat)](.)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset)

---

An end-to-end data science project that predicts whether a student is at risk of receiving a D/F grade **before the final exam**, using only midterm scores, assignment performance, and behavioral data (attendance, stress, sleep).

---

## Demo — Live Prediction Web App

An interactive Flask web application where you can input student data and get a real-time grade prediction with class probability distributions.

```bash
git clone https://github.com/TheK3R1M/student-grade-early-warning.git
cd student-grade-early-warning
pip install -r requirements.txt
python app.py
# Open http://127.0.0.1:5050
```

**Two prediction modes:**
- **Early Warning Mode** (Final not yet taken): Uses a separate model trained *without* `Final_Score` — no imputation, no false assumptions. Accuracy: ~35%
- **Full Prediction Mode** (Final available): Uses the complete model with all features. Accuracy: 83.3%

> The accuracy gap between modes is itself a finding: `Final_Score` alone carries 27.6% of predictive importance, making pre-final prediction inherently harder.

---

## Project Overview

| | |
|---|---|
| **Dataset** | [Students Grading Dataset](https://www.kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset) — Mahmoud Elhemaly (2025) |
| **Size** | 5,000 students · 23 raw columns |
| **Framework** | OSEMN (Obtain → Scrub → Explore → Model → iNterpret) |
| **Algorithm** | `RandomForestClassifier(n_estimators=100, random_state=42)` |
| **Target** | `Grade` → A / B / C / D / F |
| **Model Accuracy** | **83.3%** (train/test split: 80/20) |

---

## OSEMN Pipeline

### Obtain
- 5,000 rows, 23 columns; 4 identity columns dropped (`Student_ID`, `First_Name`, `Last_Name`, `Email`)
- Missing data: `Attendance (%)` — 516 rows (10.3%), `Assignments_Avg` — 482 rows (9.6%), `Parent_Education_Level` — 1,794 rows (35.9%)
- Root cause analysis: operational gaps, behavioral, and privacy-driven non-response

### Scrub
Five-step, order-sensitive pipeline (order matters — Grade distribution breaks if steps are swapped):

1. **Missing value imputation** — Median for numerical, Mode for categorical
2. **Outlier clipping (IQR)** — `clip(Q1 − 1.5×IQR, Q3 + 1.5×IQR)` on 5 key columns
3. **Weighted Total Score** — `Midterm×0.30 + Final×0.40 + Assignments×0.15 + Projects×0.15`
4. **Bell-curve grading** — Relative grading to fix class imbalance (90% C-clustering with fixed thresholds)
5. **StandardScaler** — Numerical features scaled to μ=0, σ=1

### Explore
Fixed thresholds (A≥90, B≥80) caused 90% of students to cluster at C — the model couldn't learn. Solution: bell-curve thresholds based on `mean=71.4, std=9.14`:

| Grade | Threshold | Students | Share |
|---|---|---|---|
| A | ≥ mean + 1.25×std (≥82.8) | 567 | 11.3% |
| B | ≥ mean + 0.50×std (≥75.9) | 1,073 | 21.5% |
| C | ≥ mean − 0.50×std (66.8–75.9) | 1,767 | 35.3% |
| D | ≥ mean − 1.25×std (60.0–66.8) | 1,009 | 20.2% |
| F | < mean − 1.25×std (<60.0) | 584 | 11.7% |

**Key behavioral finding:** Attendance rate differs by 25.6 percentage points between A and F groups (87.4% vs 61.8%) — strongest behavioral signal after academic scores.

### Model

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
```

| Grade | Actual | Correct | Accuracy |
|---|---|---|---|
| A | 107 | 84 | 78.5% |
| B | 207 | 163 | 78.7% |
| C | 359 | 338 | 94.2% |
| D | 214 | 154 | 72.0% |
| F | 113 | 94 | 83.2% |
| **Total** | **1,000** | **833** | **83.3%** |

> Most misclassifications occur between **neighboring classes (B↔C, C↔D)** — a structural boundary ambiguity, not random error. Students near decision boundaries have nearly identical feature profiles.

### iNterpret — Feature Importance

| Rank | Feature | Importance | Category |
|---|---|---|---|
| 1 | Final_Score | 27.6% | Academic |
| 2 | Midterm_Score | 18.8% | Academic |
| 3 | Projects_Score | 8.1% | Academic |
| 4 | Assignments_Avg | 7.5% | Academic |
| 5 | Participation_Score | 5.0% | Behavioral |
| 6 | Quizzes_Avg | 5.0% | Academic |
| 7 | Attendance (%) | 4.9% | Behavioral |
| 8 | Study_Hours_per_Week | 4.8% | Behavioral |
| 9 | Sleep_Hours_per_Night | 4.3% | Behavioral |
| 10 | Stress_Level (1-10) | 3.2% | Behavioral |
| 11–22 | Age, Gender, Department, Income, etc. (binary-encoded) | <2.6% each | Demographic/Socioeconomic |

> `pd.get_dummies()` expanded 17 raw features into 22 model features. The first 4 features account for 62% of total importance.

---

## Visualizations

| Plot | Description |
|---|---|
| `plots/eksik_veri_analizi.png` | Missing data bar chart by column |
| `plots/aykiri_deger_boxplot.png` | IQR outlier clipping — before/after distribution |
| `plots/cinsiyet_dagilimi.png` | Gender distribution countplot |
| `plots/yas_stres_iliskisi.png` | Age vs. average stress level |
| `plots/model_hata_matrisi.png` | Normalized confusion matrix (row-wise %) |
| `plots/basari_etkileyen_faktorler.png` | Feature importance bar chart (top 8) |

---

## Repository Structure

```
├── app.py                        # Flask web app — dual-model prediction API
├── osemn_pipeline.py             # Full OSEMN pipeline (EDA + Model + plots)
├── demo_tahmin.py                # CLI prediction demo
├── 01_Data_Preprocessing.py      # Standalone preprocessing script
├── requirements.txt
├── templates/
│   └── index.html                # Interactive prediction UI
├── data/
│   └── Students_Grading_Dataset_Biased.csv   # Raw dataset (Kaggle)
└── plots/                        # Generated visualizations
    ├── eksik_veri_analizi.png
    ├── aykiri_deger_boxplot.png
    ├── model_hata_matrisi.png
    └── basari_etkileyen_faktorler.png
```

---

## Tech Stack

- **Python 3.12** — pandas, numpy, scikit-learn, matplotlib, seaborn
- **Flask 3.0** — REST API + Jinja2-free frontend (fetch-based)
- **Random Forest** — ensemble of 100 decision trees

---

## Dataset Citation

Elhemaly, M. (2025). *Students Grading Dataset*. Kaggle.
[kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset](https://www.kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset)

---

*Built as part of a Data Analytics course project. Designed as an early-warning EduTech prototype.*
