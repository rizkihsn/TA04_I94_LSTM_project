# TA-04: Peramalan Volume Lalu Lintas per Jam I-94 dengan LSTM

Proyek Tugas Akhir Modul 4 - Praktikum Pembelajaran Mesin: Peramalan data sekuensial (*Time Series Forecasting*) volume lalu lintas jalan tol Interstate 94 (I-94) Minneapolis-St. Paul, Minnesota menggunakan arsitektur Deep Learning **Long Short-Term Memory (LSTM)**.

---

## 👥 Informasi Proyek
- **Mata Kuliah:** Praktikum Pembelajaran Mesin (Modul 4 - TA-04)
- **Topik:** Time Series Forecasting / Data Sekuensial
- **Dataset:** Metro Interstate Traffic Volume (UCI Machine Learning Repository / Kaggle)
- **Model:** Deep Learning LSTM (2-Layer Recurrent Neural Network) + Dense Output
- **Framework Web:** Flask (Python) & Dashboard Interaktif (Chart.js + HTML5)

---

## 📁 Struktur Repositori
```text
├── TA04_LSTM_Traffic_I94.ipynb   # Jupyter Notebook: EDA, visualisasi, pelatihan, & evaluasi model
├── app.py                        # Aplikasi backend Flask REST API & Web Dashboard
├── templates/
│   └── index.html                # Tampilan Dashboard interaktif modern (Chart.js)
├── model/                        # Artefak model hasil pelatihan
│   ├── lstm_traffic.h5           # Bobot model LSTM tersimpan
│   ├── scaler.joblib             # Objek MinMaxScaler untuk normalisasi fitur
│   ├── metrics.json              # Metrik evaluasi (RMSE, MAE, R²) & konfigurasi fitur
│   ├── recent.csv                # Data pengamatan historis terakhir untuk inferensi rekursif
│   └── test_results.csv          # Hasil perbandingan data aktual vs prediksi uji
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv # Dataset historis lalu lintas I-94
├── figures/                      # Grafik visualisasi data & evaluasi hasil pelatihan
├── Procfile                      # Perintah startup Gunicorn untuk deployment cloud
├── railway.json                  # Konfigurasi deployment Railway
├── nixpacks.toml                 # Konfigurasi builder Nixpacks & Git LFS
├── requirements.txt              # Daftar pustaka Python minimal
└── .python-version               # Versi Python (3.11.9)
```

---

## 📊 Kinerja Model (Test Set)
| Model | RMSE (Kendaraan/Jam) | MAE (Kendaraan/Jam) | R² Score |
| :--- | :---: | :---: | :---: |
| **LSTM (Model Utama)** | **301.02** | **226.02** | **0.9768** |
| Seasonal Naive (Baseline 2) | 512.44 | 340.18 | 0.9328 |
| Persistence (Baseline 1) | 851.35 | 588.60 | 0.8143 |

---

## 🚀 Menjalankan Secara Lokal

1. **Clone repositori:**
   ```bash
   git clone https://github.com/rizkihsn/TA04_I94_LSTM_project.git
   cd TA04_I94_LSTM_project
   ```

2. **Buat & aktifkan virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi:**
   ```bash
   python app.py
   ```
   Akses melalui browser di: `http://localhost:5000`

---

## 🌐 Endpoint REST API
- `GET /` atau `GET /dashboard` : Antarmuka Web Dashboard interaktif.
- `GET /api/info` : Metadata arsitektur model, parameter input, dan skor evaluasi.
- `GET /api/history?hours=168` : Data volume lalu lintas historis.
- `GET /api/forecast?horizon=24` : Prediksi masa depan secara rekursif (*multi-step ahead*).
- `GET /api/evaluation?hours=168` : Perbandingan aktual vs prediksi pada data uji.
- `GET /health` : Endpoint status kesehatan aplikasi.
