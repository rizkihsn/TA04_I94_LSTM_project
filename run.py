"""
Launcher untuk Hugging Face Spaces (Gradio SDK + Flask).
Menjalankan dashboard Flask dan antarmuka Gradio di port 7860 tanpa memerlukan kartu kredit.
"""
import os
import pandas as pd
import gradio as gr
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
from app import app as flask_app, recursive_forecast, load_artifacts

# Muat model dan metadata
state = load_artifacts()

def gradio_predict(horizon_hours):
    results = recursive_forecast(int(horizon_hours))
    df = pd.DataFrame(results)
    df.columns = ["Waktu", "Prediksi Volume (Kendaraan/Jam)"]
    v = df["Prediksi Volume (Kendaraan/Jam)"]
    summary = (
        f"### 📊 Ringkasan Prediksi {len(results)} Jam ke Depan:\n"
        f"- **Puncak Tertinggi:** {v.max():,} kendaraan/jam\n"
        f"- **Titik Terendah:** {v.min():,} kendaraan/jam\n"
        f"- **Rata-rata:** {int(v.mean()):,} kendaraan/jam"
    )
    return df, summary

# Buat UI Gradio Blocks
with gr.Blocks(title="Peramalan Volume Lalu Lintas I-94 (LSTM)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 Sistem Peramalan Volume Lalu Lintas Jalan Tol I-94 (LSTM)")
    gr.Markdown(
        "**Mata Kuliah:** Praktikum Pembelajaran Mesin (Modul 4 - TA-04)  \n"
        "**Dataset:** Metro Interstate Traffic Volume (UCI/Kaggle) | **Arsitektur:** Deep Learning LSTM"
    )

    with gr.Tab("📊 Web Dashboard Interaktif (Flask)"):
        gr.Markdown(
            "Dashboard lengkap berbasis **Flask, Chart.js, dan HTML5**. "
            "Menampilkan data historis, peramalan rekursif, perbandingan aktual vs prediksi, dan heatmap visual:"
        )
        gr.HTML(
            '<iframe src="/dashboard" style="width:100%; height:900px; border:none; '
            'border-radius:12px; background:#050a16; box-shadow:0 8px 32px rgba(0,0,0,0.4);"></iframe>'
        )
        gr.Markdown("👉 *Jika ingin membuka dashboard dalam layar penuh:* [**Buka Full Dashboard di Tab Baru**](/dashboard)")

    with gr.Tab("🔮 Prediksi Interaktif"):
        gr.Markdown("### Simulasi Peramalan Rekursif")
        gr.Markdown("Pilih horizon waktu di bawah untuk menghitung prediksi volume lalu lintas per jam ke depan:")
        with gr.Row():
            horizon_slider = gr.Slider(
                minimum=6, maximum=72, value=24, step=6,
                label="Horizon Prediksi (Jam ke Depan)"
            )
            predict_btn = gr.Button("🚀 Jalankan Prediksi", variant="primary")
        
        summary_out = gr.Markdown()
        table_out = gr.Dataframe(
            headers=["Waktu", "Prediksi Volume (Kendaraan/Jam)"],
            wrap=True
        )
        predict_btn.click(
            fn=gradio_predict,
            inputs=[horizon_slider],
            outputs=[table_out, summary_out]
        )

    with gr.Tab("📈 Metrik & Evaluasi"):
        gr.Markdown("### Kinerja Model pada Data Uji (Test Set)")
        m = state["meta"]["metrics"]
        eval_data = [
            {"Model": "LSTM (Model Utama)", "RMSE": m["LSTM"]["rmse"], "MAE": m["LSTM"]["mae"], "R² Score": m["LSTM"]["r2"]},
            {"Model": "Persistence (Baseline 1)", "RMSE": m["Persistence"]["rmse"], "MAE": m["Persistence"]["mae"], "R² Score": m["Persistence"]["r2"]},
            {"Model": "Seasonal Naive (Baseline 2)", "RMSE": m["Seasonal naive"]["rmse"], "MAE": m["Seasonal naive"]["mae"], "R² Score": m["Seasonal naive"]["r2"]},
        ]
        gr.Dataframe(pd.DataFrame(eval_data))
        gr.Markdown(
            "> **Analisis Hasil:**  \n"
            "> Model LSTM mencapai nilai **R² = 0.9768** dengan **RMSE = 301.02** kendaraan/jam, "
            "> membuktikan bahwa model mampu mempelajari siklus lalu lintas harian dan mingguan "
            "> secara signifikan lebih akurat dibanding model acuan (baseline)."
        )

# Inisialisasi FastAPI wrapper
fastapi_app = FastAPI()

# Mount backend Flask untuk dashboard dan endpoint API
fastapi_app.mount("/dashboard", WSGIMiddleware(flask_app))
fastapi_app.mount("/api", WSGIMiddleware(flask_app))

# Mount Gradio pada root
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
