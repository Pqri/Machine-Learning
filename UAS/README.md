# 🧠 Hands-On Machine Learning Reproduction (UAS Project)

Selamat datang di folder `UAS` yang berisi hasil reproduksi dari buku **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition)** oleh **Aurélien Géron**. Proyek ini merupakan bagian dari tugas akhir (UAS) dengan tujuan untuk memperdalam pemahaman konseptual dan praktik Machine Learning (ML) melalui implementasi langsung dari setiap bab dalam buku.

---

## 📚 Daftar Isi dan Ringkasan Bab

### 🔹 Part I: The Fundamentals of Machine Learning

| Bab | Judul | Ringkasan |
|-----|-------|-----------|
| 1 | The Machine Learning Landscape | Pengantar konsep ML, tipe-tipe sistem ML (supervised, unsupervised, reinforcement), serta tantangan umum seperti overfitting dan data berkualitas rendah. |
| 2 | End-to-End Machine Learning Project | Proyek lengkap prediksi harga rumah menggunakan dataset California housing. Menjelaskan seluruh pipeline: eksplorasi data, preprocessing, training, evaluasi, hingga deployment. |
| 3 | Classification | Fokus pada klasifikasi menggunakan dataset MNIST. Diperkenalkan metrik evaluasi seperti confusion matrix, precision/recall, ROC curve, dan teknik multiclass classification. |
| 4 | Training Models | Pembahasan regresi linear, gradient descent, regularisasi (Ridge, Lasso, ElasticNet), hingga logistic regression dan softmax regression. |
| 5 | Support Vector Machines | Penjelasan tentang klasifikasi SVM linear dan non-linear, kernel trick, dan SVM untuk regresi. |
| 6 | Decision Trees | Membangun dan visualisasi decision tree, teori CART, overfitting, regularisasi, dan regresi dengan pohon keputusan. |
| 7 | Ensemble Learning and Random Forests | Teknik voting, bagging, boosting (AdaBoost, Gradient Boosting), dan stacking. |
| 8 | Dimensionality Reduction | Reduksi dimensi menggunakan PCA, kernel PCA, dan LLE. Bermanfaat untuk visualisasi dan preprocessing. |
| 9 | Unsupervised Learning Techniques | Clustering (K-Means, DBSCAN), Gaussian Mixture Models, dan anomaly detection. |

---

### 🔸 Part II: Neural Networks and Deep Learning

| Bab | Judul | Ringkasan |
|-----|-------|-----------|
| 10 | Introduction to Artificial Neural Networks with Keras | Arsitektur neuron buatan, implementasi MLP (regresi dan klasifikasi) dengan Keras, Functional API, Subclassing. |
| 11 | Training Deep Neural Networks | Masalah vanishing/exploding gradients, optimizers (Adam, RMSprop), regularisasi (Dropout), dan tips training jaringan dalam. |
| 12 | Custom Models and Training with TensorFlow | Membangun model custom, training loop, custom metrics, dan fungsi aktivasi menggunakan TensorFlow. |
| 13 | Loading and Preprocessing Data with TensorFlow | TF Data API, TFRecord, protobuffers, dan preprocessing fitur (one-hot, embedding). |
| 14 | Deep Computer Vision Using CNNs | CNN untuk visi komputer (LeNet, AlexNet, VGG, ResNet, Xception), pretrained model, YOLO, dan semantic segmentation. |
| 15 | Processing Sequences Using RNNs and CNNs | Pemrosesan urutan menggunakan RNN dan CNN (Time Series, LSTM, GRU). |
| 16 | NLP with RNNs and Attention | Text generation, sentiment analysis, encoder-decoder, attention, dan Transformer architecture. |
| 17 | Representation and Generative Learning (Autoencoders & GANs) | Autoencoder (variasi: denoising, variational), dan GAN (DCGAN, StyleGAN). |
| 18 | Reinforcement Learning | Konsep RL, Q-learning, TF-Agents, pembuatan agen dengan DQN dan variasinya. |
| 19 | Training and Deploying TensorFlow Models at Scale | TensorFlow Serving, deployment ke GCP, mobile, dan distribusi training. |

---

## 💼 Tujuan Repositori

Repositori ini bertujuan untuk:
- Meningkatkan pemahaman konsep dan implementasi Machine Learning.
- Mempersiapkan pemahaman praktikal menggunakan Scikit-Learn, Keras, dan TensorFlow.
- Menerapkan pipeline ML end-to-end dari preprocessing hingga deployment.

---

## ⚙️ Tools & Library

- Python 3.x
- Scikit-Learn
- TensorFlow 2.x + Keras API
- Matplotlib, NumPy, Pandas
- Jupyter Notebook / Google Colab

---

## 📁 Struktur Folder

Setiap folder/bab berisi:
- Notebook `.ipynb` yang mereproduksi kode dari buku
- Penjelasan teori di dalam notebook
- Eksperimen tambahan bila ada

---

## 📢 Catatan

Jika kamu ingin menjalankan notebook di Google Colab, pastikan:
- Runtime > Change runtime type > Hardware accelerator → GPU
- Install dependencies dengan `%pip install` jika diperlukan

---

## 🙌 Kontributor

Proyek ini dikerjakan oleh:  
**Fiqri Siraj Al Majeed**  
Sebagai bagian dari Ujian Akhir Semester (UAS) untuk mata kuliah Machine Learning.
