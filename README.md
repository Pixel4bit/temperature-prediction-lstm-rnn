# 📈 Penerapan Algoritma LSTM untuk Memprediksi Pola Kenaikan Suhu di Jakarta Pusat

Proyek ini adalah implementasi dari model Deep Learning menggunakan algoritma LSTM dan RNN untuk memprediksi suhu maksimum harian di Jakarta Pusat. Aplikasi ini dibangun menggunakan Streamlit untuk antarmuka interaktif dan menggunakan data iklim dari BMKG (2013–2024) sebagai sumber data pelatihan dan pengujian.

## 🚀 Fitur Utama

- Prediksi suhu maksimum harian menggunakan LSTM dan RNN.
- Visualisasi tren suhu, kelembapan, curah hujan, dan parameter cuaca lainnya.
- Evaluasi performa model dengan metrik MAE, RMSE, dan MAPE.
- Pengguna dapat memilih jumlah hari prediksi dan panjang data historis.
- Opsi update data otomatis hingga Juli 2024.
- Download dataset dalam format ZIP.

## 📊 Dataset

- **Sumber:** [BMKG Data Online](https://dataonline.bmkg.go.id/)
- **Stasiun:** Meteorologi Kemayoran, Jakarta Pusat
- **Rentang Waktu:** 01 Januari 2013 – 01 Juli 2024
- **Format:** CSV

## 🛠 Teknologi yang Digunakan

- Python, Streamlit, Pandas, NumPy
- Keras, Scikit-learn
- Matplotlib, Plotly, Altair
- RFE (Recursive Feature Elimination) untuk seleksi fitur
- LSTM dan RNN pretrained model (`lstm.h5`, `rnn.h5`)

## 📦 Cara Menjalankan Aplikasi

1. Clone repository ini:
   ```bash
   git clone https://github.com/username/nama-repo.git
   cd nama-repo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi:
   ```bash
   streamlit run streamlit_app.py
   ```

> Pastikan file model (`lstm.h5` dan `rnn.h5`) tersedia di direktori yang sama dengan script.

## 🧠 Model Deep Learning

- Model telah dilatih sebelumnya dengan data historis suhu.
- Opsi untuk memilih model: LSTM (default) atau RNN.
- Model dievaluasi menggunakan data training dan testing secara terpisah.

## 👨‍💻 Tim Pengembang

Proyek ini dikembangkan oleh mahasiswa S1 Sistem Informasi dari Universitas Bina Sarana Informatika:

- Ahmad Haitami Hatta
- Alvian Ibnu Farhan
- Dzulfiqar Ramazan

## 📄 Lisensi

Proyek ini bersifat open-source dan dilisensikan di bawah lisensi MIT.
