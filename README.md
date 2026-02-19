# Prepocessing-data-pasien-Panic-Attack
preprocessing data mentah yang diperoleh melalui Kaggle dan di olah agar data Missing value diisi rata-rata, kemudian outlier dihapus memakai metode Z-score sebelum data disimpan kembali dalam file bersih
# Import library yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Membaca dataset (ganti path sesuai kebutuhan)
df = pd.read_csv("/content/data_preprocessed.csv", delimiter=';')

# 1. Eksplorasi Data
print("=== Informasi Dataset ===")
print(df.info())
print("\n=== 5 Baris Pertama ===")
print(df.head())
print("\n=== Statistik Deskriptif ===")
print(df.describe())
print("\n=== Jumlah Nilai Unik Tiap Kolom ===")
print(df.nunique())

# Visualisasi distribusi data
# Select numerical columns for histogram
numerical_cols = df.select_dtypes(include=np.number).columns
if len(numerical_cols) > 0:
    df[numerical_cols].hist(figsize=(15,10))
    plt.tight_layout()
    plt.show()
else:
    print("\nTidak ada kolom numerik untuk divisualisasikan histogramnya.")


# Korelasi antar variabel numerik
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Matriks Korelasi")
plt.show()

# 2. Deteksi dan Penanganan Missing Value
print("\n=== Jumlah Missing Value Tiap Kolom ===")
print(df.isnull().sum())

# Contoh: Mengisi missing value dengan mean (untuk numerik)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Jika ingin menghapus baris dengan missing value:
# df.dropna(inplace=True)

# 3. Deteksi dan Penghilangan Outlier
# Menggunakan Z-score
# Select numerical columns for Z-score calculation
numerical_cols_for_zscore = df.select_dtypes(include=np.number)
if not numerical_cols_for_zscore.empty:
    z_scores = np.abs(stats.zscore(numerical_cols_for_zscore))
    threshold = 3
    df_clean = df[(z_scores < threshold).all(axis=1)]

    print(f"\nJumlah data sebelum menghapus outlier: {df.shape[0]}")
    print(f"Jumlah data setelah menghapus outlier: {df_clean.shape[0]}")

    # Simpan data yang telah dibersihkan
    df_clean.to_csv("nama_dataset_clean.csv", index=False)
else:
    print("\nTidak ada kolom numerik untuk deteksi outlier.")
    df_clean = df.copy() # Keep the original data if no numerical columns
