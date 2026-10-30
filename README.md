# TA-04: Peramalan Volume Lalu Lintas per Jam I-94 dengan LSTM

Studi kasus: prediksi volume kendaraan per jam di jalan tol Interstate 94 (Minnesota, AS).
Dataset: Metro Interstate Traffic Volume (Kaggle / UCI).

## Struktur
```
TA04_LSTM_Traffic_I94.ipynb   # EDA, preprocessing, training, evaluasi, simpan model
app.py                        # aplikasi Flask
templates/index.html          # halaman web (Chart.js)
data/                         # taruh CSV dataset di sini
model/                        # keluaran notebook: lstm_traffic.h5, scaler.joblib, recent.csv, test_results.csv, metrics.json
figures/                      # gambar keluaran notebook
requirements.txt, Procfile, .python-version   # untuk deploy (Render/Railway)
```

## Langkah menjalankan
1. Jalankan notebook sampai selesai (disarankan di Google Colab, pakai runtime GPU/CPU biasa). Notebook menghasilkan isi folder `model/`.
2. Salin folder `model/` ke repo ini, lalu jalankan lokal:
   ```
   pip install -r requirements.txt
   python app.py        # buka http://localhost:5000
   ```
3. Push ke GitHub (termasuk folder `model/`), lalu deploy di Render:
   - New > Web Service > pilih repo
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --workers 1 --threads 2 --timeout 180 --bind 0.0.0.0:$PORT`
   - Versi scikit-learn di `requirements.txt` harus sama dengan versi saat membuat `scaler.joblib` (cek `sklearn.__version__` di Colab).

## Endpoint
- `/` halaman utama
- `/api/history?hours=168`, `/api/forecast?horizon=24`, `/api/evaluation?hours=168`, `/api/info`, `/health`
