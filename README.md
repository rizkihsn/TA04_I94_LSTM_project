---
title: TA04 LSTM Traffic Forecasting I-94
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# TA-04: Peramalan Volume Lalu Lintas per Jam I-94 dengan LSTM

Studi kasus: prediksi volume kendaraan per jam di jalan tol Interstate 94 (Minnesota, AS).
Dataset: Metro Interstate Traffic Volume (Kaggle / UCI).

## Struktur
```
TA04_LSTM_Traffic_I94.ipynb   # EDA, preprocessing, training, evaluasi, simpan model
app.py                        # aplikasi Flask
templates/index.html          # halaman web dashboard
data/                         # CSV dataset
model/                        # keluaran notebook: lstm_traffic.h5, scaler.joblib, recent.csv, test_results.csv, metrics.json
figures/                      # grafik keluaran notebook
requirements.txt, Procfile, Dockerfile, .python-version
```

## Menjalankan Lokal
```bash
pip install -r requirements.txt
python app.py        # buka http://localhost:5000
```

## Endpoint API
- `/` halaman dashboard
- `/api/history?hours=168`
- `/api/forecast?horizon=24`
- `/api/evaluation?hours=168`
- `/api/info`
- `/health`
