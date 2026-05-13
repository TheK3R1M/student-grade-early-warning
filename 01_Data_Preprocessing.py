import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# ============================================================
# AŞAMA 1: VERİ TEMİZLEME VE ÖN İŞLEME (Data Preprocessing)
#
# Bu script, ham veriyi akademik standartlarda adım adım
# temizler, eksikleri doldurur, aykırı değerleri baskılar
# ve okunabilir bir temiz veri seti üretir.
#
# Çıktı: data/step1_cleaned_data.csv
# ============================================================

DATA_DIR   = './data'
RAW_PATH   = os.path.join(DATA_DIR, 'Students_Grading_Dataset_Biased.csv')
OUT_PATH   = os.path.join(DATA_DIR, 'step1_cleaned_data.csv')

print("=" * 60)
print("🔍 AŞAMA 1 — Veri Temizleme ve Ön İşleme")
print("=" * 60)

# ----------------------------------------------------------
# ADIM 1: Veriyi Yükle (Obtain)
# ----------------------------------------------------------
print("\n[1/5] Veri seti yükleniyor...")
df = pd.read_csv(RAW_PATH)
print(f"  ✔  Ham veri: {len(df)} satır, {len(df.columns)} sütun")

# Orijinal Grade ve Total_Score sütunları ham veride zaten var.
# Bunları koruyacağız; sadece eksikleri, aykırıları ve
# gereksiz kişisel sütunları temizleyeceğiz.

# ----------------------------------------------------------
# ADIM 2: Eksik Verileri Doldur (Scrub – Missing Values)
# ----------------------------------------------------------
print("\n[2/5] Eksik veriler tespit ediliyor ve dolduruluyor...")
missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) > 0:
    for col, cnt in missing.items():
        print(f"  ⚠  '{col}' sütununda {cnt} eksik değer bulundu")
    
    print()
    print("  💡 VERİ BİLİMCİSİ KARARI:")
    print("     Eksik satırları silmek istatistiksel güç kaybına yol açar.")
    print("     Sayısal sütunlar → Medyan (dağılımı bozmaz)")
    print("     Kategorik sütunlar → Mod (en yaygın sınıfa atar)")
    
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    
    print(f"  ✔  {len(missing)} sütundaki eksikler dolduruldu. Kalan eksik: {df.isnull().sum().sum()}")
else:
    print("  ✔  Eksik veri bulunamadı.")

# ----------------------------------------------------------
# ADIM 3: Aykırı Değerleri Baskıla (Outlier Clipping – IQR)
# ----------------------------------------------------------
print("\n[3/5] Aykırı değerler (Outliers) tespit ediliyor...")
score_cols = ['Midterm_Score', 'Final_Score', 'Assignments_Avg',
              'Projects_Score', 'Total_Score', 'Attendance (%)']
score_cols = [c for c in score_cols if c in df.columns]

total_clipped = 0
for col in score_cols:
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    cnt   = int(((df[col] < lower) | (df[col] > upper)).sum())
    total_clipped += cnt
    df[col] = np.clip(df[col], lower, upper)

print(f"  ✔  Toplamda {total_clipped} aykırı değer IQR yöntemiyle baskılandı (silme yapılmadı)")

# ----------------------------------------------------------
# ADIM 4: Gereksiz Kimlik Sütunlarını At
# ----------------------------------------------------------
print("\n[4/5] Kimlik bilgileri (gürültü) atılıyor...")
drop_cols = ['Student_ID', 'First_Name', 'Last_Name', 'Email']
before = len(df.columns)
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
print(f"  ✔  {before - len(df.columns)} sütun atıldı ({', '.join(drop_cols)})")
print(f"     Kalan sütun sayısı: {len(df.columns)}")

# ----------------------------------------------------------
# ADIM 5: Temizlenmiş Veriyi Kaydet
# (Standardizasyon EDA sonrasında, modelleme öncesinde yapılacak)
# ----------------------------------------------------------
print("\n[5/5] Temizlenmiş veri kaydediliyor...")
df.to_csv(OUT_PATH, index=False)
print(f"  ✔  Kayıt tamamlandı → {OUT_PATH}")
print(f"     Son veri boyutu: {len(df)} satır, {len(df.columns)} sütun")

print()
print("=" * 60)
print("✅ AŞAMA 1 TAMAMLANDI")
print("   Bir sonraki adım: 02_Exploratory_Data_Analysis.py")
print("=" * 60)
