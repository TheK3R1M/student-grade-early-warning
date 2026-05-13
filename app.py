# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ============================================================
# ORTAK VERİ HAZIRLAMA
# ============================================================
print("[...] Veri hazirlaniyor...")

df = pd.read_csv("data/Students_Grading_Dataset_Biased.csv")

num_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
cat_cols = df.select_dtypes(include=["object"]).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

for col in ["Midterm_Score", "Final_Score", "Assignments_Avg", "Projects_Score", "Attendance (%)"]:
    if col in df.columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

df["Total_Score"] = (
    df["Midterm_Score"]   * 0.30 +
    df["Final_Score"]     * 0.40 +
    df["Assignments_Avg"] * 0.15 +
    df["Projects_Score"]  * 0.15
)
mean_s, std_s = df["Total_Score"].mean(), df["Total_Score"].std()

def grade(score):
    if score >= mean_s + 1.25 * std_s: return "A"
    elif score >= mean_s + 0.5  * std_s: return "B"
    elif score >= mean_s - 0.5  * std_s: return "C"
    elif score >= mean_s - 1.25 * std_s: return "D"
    else: return "F"

df["Grade"] = df["Total_Score"].apply(grade)

# ============================================================
# MODEL 1: TAM MODEL (Final dahil)
# ============================================================
print("[1/2] Tam model egitiliyor (Final dahil)...")

scaler_full = StandardScaler()
scale_cols_full = ["Midterm_Score", "Final_Score", "Assignments_Avg", "Projects_Score"]
df_full = df.copy()
df_full[scale_cols_full] = scaler_full.fit_transform(df_full[scale_cols_full])

drop_base = ["Student_ID", "First_Name", "Last_Name", "Email", "Grade", "Total_Score"]
X_full = pd.get_dummies(df_full.drop(columns=[c for c in drop_base if c in df_full.columns]), drop_first=True)
y = df_full["Grade"]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_full, y, test_size=0.2, random_state=42)
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train_f, y_train_f)
acc_full = rf_full.score(X_test_f, y_test_f)
print(f"   Tam model dogruluk: %{acc_full*100:.1f}")

# ============================================================
# MODEL 2: ERKEN UYARI MODELİ (Final yok)
# ============================================================
print("[2/2] Erken uyari modeli egitiliyor (Final olmadan)...")

scaler_early = StandardScaler()
scale_cols_early = ["Midterm_Score", "Assignments_Avg", "Projects_Score"]
df_early = df.copy()
df_early[scale_cols_early] = scaler_early.fit_transform(df_early[scale_cols_early])

# Final_Score tamamen cikar
drop_early = drop_base + ["Final_Score"]
X_early = pd.get_dummies(df_early.drop(columns=[c for c in drop_early if c in df_early.columns]), drop_first=True)

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_early, y, test_size=0.2, random_state=42)
rf_early = RandomForestClassifier(n_estimators=100, random_state=42)
rf_early.fit(X_train_e, y_train_e)
acc_early = rf_early.score(X_test_e, y_test_e)
print(f"   Erken uyari modeli dogruluk: %{acc_early*100:.1f}")

print("Hazir!")

# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/info")
def info():
    return jsonify({
        "acc_full":  round(acc_full  * 100, 1),
        "acc_early": round(acc_early * 100, 1),
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    midterm   = float(data["midterm"])
    odv       = float(data["odv"])
    proje     = float(data["proje"])
    devam     = float(data["devam"])
    stres     = float(data["stres"])
    uyku      = float(data["uyku"])
    final_raw = data.get("final")

    if final_raw is not None:
        # --- TAM MODEL ---
        final_val = float(final_raw)
        mid_s = (midterm    - scaler_full.mean_[0]) / scaler_full.scale_[0]
        fin_s = (final_val  - scaler_full.mean_[1]) / scaler_full.scale_[1]
        odv_s = (odv        - scaler_full.mean_[2]) / scaler_full.scale_[2]
        prj_s = (proje      - scaler_full.mean_[3]) / scaler_full.scale_[3]

        sample = pd.DataFrame(0, index=[0], columns=X_full.columns)
        for col, val in [
            ("Midterm_Score", mid_s), ("Final_Score", fin_s),
            ("Assignments_Avg", odv_s), ("Projects_Score", prj_s),
            ("Attendance (%)", devam), ("Stress_Level (1-10)", stres),
            ("Sleep_Hours_per_Night", uyku),
        ]:
            if col in sample.columns:
                sample[col] = val

        tahmin      = rf_full.predict(sample)[0]
        olasiliklar = rf_full.predict_proba(sample)[0]
        siniflar    = rf_full.classes_
        mod_label   = f"Tam model — Final dahil ({int(final_val)} puan) | Dogruluk %{acc_full*100:.1f}"

    else:
        # --- ERKEN UYARI MODELİ (Final kullanilmiyor) ---
        mid_s = (midterm - scaler_early.mean_[0]) / scaler_early.scale_[0]
        odv_s = (odv     - scaler_early.mean_[1]) / scaler_early.scale_[1]
        prj_s = (proje   - scaler_early.mean_[2]) / scaler_early.scale_[2]

        sample = pd.DataFrame(0, index=[0], columns=X_early.columns)
        for col, val in [
            ("Midterm_Score", mid_s), ("Assignments_Avg", odv_s),
            ("Projects_Score", prj_s), ("Attendance (%)", devam),
            ("Stress_Level (1-10)", stres), ("Sleep_Hours_per_Night", uyku),
        ]:
            if col in sample.columns:
                sample[col] = val

        tahmin      = rf_early.predict(sample)[0]
        olasiliklar = rf_early.predict_proba(sample)[0]
        siniflar    = rf_early.classes_
        mod_label   = f"Erken uyari modeli — Final yok, varsayim yok | Dogruluk %{acc_early*100:.1f}"

    risk_map = {
        "A": ("Dusuk Risk",  "#22c55e"),
        "B": ("Dusuk Risk",  "#22c55e"),
        "C": ("Orta Risk",   "#f59e0b"),
        "D": ("Yuksek Risk", "#f97316"),
        "F": ("Kritik Risk", "#ef4444"),
    }
    risk_label, risk_color = risk_map[tahmin]
    probs = {s: round(float(p) * 100, 1) for s, p in zip(siniflar, olasiliklar)}

    return jsonify({
        "grade":      tahmin,
        "risk_label": risk_label,
        "risk_color": risk_color,
        "probs":      probs,
        "mod_label":  mod_label,
        "total":      round(midterm * 0.30 + odv * 0.15 + proje * 0.15, 1),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050)
