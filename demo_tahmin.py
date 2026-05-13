import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ============================================================
# 1. MODELİ EĞİT
# ============================================================
print("=" * 55)
print("  OGRENCI BASARI TAHMIN SISTEMI - Canli Demo")
print("=" * 55)
print("\n[...] Model egitiliyor, lutfen bekle...\n")


df = pd.read_csv("data/Students_Grading_Dataset_Biased.csv")

# Eksikleri doldur
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
cat_cols = df.select_dtypes(include=["object"]).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# IQR baskılama
for col in ["Midterm_Score", "Final_Score", "Assignments_Avg", "Projects_Score", "Attendance (%)"]:
    if col in df.columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# Total Score ve Grade (çan eğrisi)
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

# Standartlaştır
scaler = StandardScaler()
scale_cols = ["Midterm_Score", "Final_Score", "Assignments_Avg", "Projects_Score"]
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Model eğit
drop_cols = ["Student_ID", "First_Name", "Last_Name", "Email", "Grade", "Total_Score"]
df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])
df_model = pd.get_dummies(df_model, drop_first=True)
X, y = df_model, df["Grade"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

acc = rf.score(X_test, y_test)
print(f"✅ Model hazır! Test doğruluğu: %{acc*100:.1f}")
print(f"   (Sınıf ortalaması: {mean_s:.1f} | Standart sapma: {std_s:.1f})\n")

# ============================================================
# 2. KULLANICIDAN DEĞERLERİ AL
# ============================================================
RISK = {"A": "🟢 Düşük risk", "B": "🟢 Düşük risk",
        "C": "🟡 Orta risk", "D": "🔴 Yüksek risk", "F": "🔴 ÇOK YÜKSEK RİSK"}

while True:
    print("─" * 55)
    print("📝 Yeni öğrenci için değerleri gir (çıkmak için 'q'):")
    print("─" * 55)

    try:
        midterm = input("  Vize notu       (0–100): ").strip()
        if midterm.lower() == "q": break

        final    = float(input("  Final notu      (0–100): "))
        odv      = float(input("  Ödev ortalaması (0–100): "))
        proje    = float(input("  Proje notu      (0–100): "))
        devam    = float(input("  Devam (%)       (0–100): "))
        stres    = float(input("  Stres seviyesi  (1–10) : "))
        uyku     = float(input("  Uyku saati/gece (4–10) : "))
        midterm  = float(midterm)

        # Total score hesapla (orijinal ölçekte)
        total = midterm * 0.30 + final * 0.40 + odv * 0.15 + proje * 0.15

        # Standartlaştır
        mid_s   = (midterm - scaler.mean_[0]) / scaler.scale_[0]
        fin_s   = (final   - scaler.mean_[1]) / scaler.scale_[1]
        odv_s   = (odv     - scaler.mean_[2]) / scaler.scale_[2]
        prj_s   = (proje   - scaler.mean_[3]) / scaler.scale_[3]

        # Örnek DataFrame oluştur (model sütunlarıyla eşleştir)
        sample = pd.DataFrame(0, index=[0], columns=X.columns)

        # Bilinen sayısal sütunları doldur
        for col, val in [
            ("Midterm_Score", mid_s), ("Final_Score", fin_s),
            ("Assignments_Avg", odv_s), ("Projects_Score", prj_s),
            ("Attendance (%)", devam), ("Stress_Level (1-10)", stres),
            ("Sleep_Hours_per_Night", uyku),
        ]:
            if col in sample.columns:
                sample[col] = val

        tahmin     = rf.predict(sample)[0]
        olasiliklar = rf.predict_proba(sample)[0]
        siniflar    = rf.classes_

        print(f"\n{'='*55}")
        print(f"  📊 TAHMİN SONUCU")
        print(f"{'='*55}")
        print(f"  Hesaplanan toplam puan : {total:.1f}")
        print(f"  Modelin tahmini        : {tahmin}  →  {RISK[tahmin]}")
        print(f"\n  Olasılık dağılımı:")
        for sinif, olas in sorted(zip(siniflar, olasiliklar)):
            bar = "█" * int(olas * 20)
            print(f"    {sinif}: {bar:<20} %{olas*100:.0f}")
        print(f"{'='*55}\n")

    except ValueError:
        print("  ⚠️  Geçersiz değer — sadece sayı gir.\n")
    except Exception as e:
        print(f"  ⚠️  Hata: {e}\n")

print("\n✅ Demo tamamlandı. Sunum için başarılar!")
