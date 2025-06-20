# 📘 UAS PROJECT – Multilayer Perceptron (MLP) & CNN

Repositori ini merupakan hasil pengerjaan Ujian Akhir Semester (UAS) untuk topik **Deep Learning**, mencakup implementasi model MLP (Multilayer Perceptron) dan CNN (Convolutional Neural Network) pada tiga jenis data:

1. Prediksi data numerik (regresi)
2. Klasifikasi kategori (diskrit)
3. Klasifikasi berbasis data visual (gambar)

---

## 📂 Struktur Folder

| File/Folder | Deskripsi |
|-------------|-----------|
| `1. Regression MLP.ipynb` | Model MLP untuk prediksi numerik pada dataset `RegresiUTSTelkom.csv` (contoh: prediksi nilai berdasarkan fitur sensor/data time-series) |
| `2. Classification MLP.ipynb` | Model MLP untuk klasifikasi diskrit pada dataset `KlasifikasiUTS.csv` |
| `3. CNN Klasifikasi Ikan.ipynb` | Model CNN untuk klasifikasi citra ikan pada folder dataset gambar (`train/`, `val/`, `test/`) |

---

## 🎯 Tujuan Proyek

1. **Membuat pipeline end-to-end Deep Learning**:
   - Preprocessing data (Pandas, Augmentasi data gambar)
   - Feature Engineering (Transformasi, One-hot Encoding, Label Encoding)
   - Arsitektur MLP atau CNN (menggunakan TensorFlow / PyTorch)
   - Penyesuaian hyperparameter (learning rate, batch size, dropout, weight decay, optimizer modern, dll)

2. **Evaluasi Model**:
   - Regresi: RMSE, MSE, R²
   - Klasifikasi: Akurasi, Presisi, Recall, F1-Score, AUC-ROC
   - Visualisasi Confusion Matrix & perbandingan predicted vs actual value

3. **Analisis & Interpretasi**:
   - Menentukan model terbaik berdasarkan metrik evaluasi
   - Memberikan penjelasan dan interpretasi hasil

---

## ⚙️ Tools & Library

- Python 3.x
- TensorFlow / PyTorch
- Scikit-learn
- Pandas, NumPy, Matplotlib
- Jupyter Notebook / Google Colab

---

## 🗂 Dataset

| Dataset | Tipe | Ukuran |
|---------|------|--------|
| `RegresiUTSTelkom.csv` | Data numerik untuk regresi | ~423 MB |
| `KlasifikasiUTS.csv` | Data tabular klasifikasi | ~150 MB |
| Dataset Ikan (train/test/val folders) | Gambar, klasifikasi visual | Berbasis folder dan nama kelas |

---

## ✅ Petunjuk Eksekusi

1. Jalankan setiap notebook secara berurutan.
2. Pastikan semua dependensi telah diinstal (`pip install` atau `conda install`).
3. Untuk CNN, pastikan struktur folder gambar sudah benar (`train/class1/`, `train/class2/`, dst).

---

## ✍️ Penulis

Proyek ini disusun oleh:
- **[Nama Anda]**
- Sebagai bagian dari Ujian Akhir Semester - Deep Learning

