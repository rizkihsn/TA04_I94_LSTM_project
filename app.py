"""Aplikasi Flask: Peramalan Volume Lalu Lintas per Jam I-94 dengan LSTM."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
MAX_HORIZON = 72

app = Flask(__name__)

_state = {}


def load_artifacts():
    """Muat model & data sekali saja (lazy) agar start-up cepat."""
    if _state:
        return _state
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        meta = json.load(f)

    timesteps  = int(meta["timesteps"])       # 48
    n_features = len(meta["features"])        # 5

    # Bangun ulang arsitektur identik dengan notebook
    model = Sequential([
        Input(shape=(timesteps, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ], name="LSTM_Traffic_I94")
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Load weights saja — bypass total deserialisasi config Keras 3
    model.load_weights(os.path.join(MODEL_DIR, "lstm_traffic.h5"))

    _state.update(
        model=model,
        scaler=joblib.load(os.path.join(MODEL_DIR, "scaler.joblib")),
        recent=pd.read_csv(os.path.join(MODEL_DIR, "recent.csv"), index_col=0, parse_dates=True),
        test=pd.read_csv(os.path.join(MODEL_DIR, "test_results.csv"), parse_dates=["date_time"]),
        meta=meta,
        timesteps=timesteps,
    )
    return _state


def calendar_features(idx):
    """Harus identik dengan fungsi di notebook."""
    return np.column_stack([
        np.sin(2 * np.pi * idx.hour / 24), np.cos(2 * np.pi * idx.hour / 24),
        np.sin(2 * np.pi * idx.dayofweek / 7), np.cos(2 * np.pi * idx.dayofweek / 7),
    ])


def recursive_forecast(horizon):
    s = load_artifacts()
    model, scaler, T = s["model"], s["scaler"], s["timesteps"]
    idx = s["recent"].index[-T:]
    vol = s["recent"]["traffic_volume"].values[-T:].astype(float)

    window = scaler.transform(np.column_stack([vol, calendar_features(idx)]))
    v_min, v_rng = scaler.data_min_[0], scaler.data_range_[0]

    last, out = idx[-1], []
    for _ in range(horizon):
        p = float(np.asarray(model.predict(window[-T:][np.newaxis], verbose=0)).reshape(-1)[0])
        p = min(max(p, 0.0), 1.0)
        last = last + pd.Timedelta(hours=1)
        cal = scaler.transform(np.concatenate([[0.0], calendar_features(pd.DatetimeIndex([last]))[0]])[None, :])[0, 1:]
        window = np.vstack([window, np.concatenate([[p], cal])])
        out.append({"time": last.strftime("%Y-%m-%d %H:%M"), "volume": round(p * v_rng + v_min)})
    return out


def _fmt(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


@app.route("/")
@app.route("/dashboard")
def index():
    return render_template("index.html")


@app.route("/api/info")
@app.route("/info")
def api_info():
    s = load_artifacts()
    return jsonify(s["meta"])


@app.route("/api/history")
@app.route("/history")
def api_history():
    s = load_artifacts()
    hours = min(max(request.args.get("hours", 168, type=int), 24), len(s["recent"]))
    r = s["recent"].iloc[-hours:]
    return jsonify({"time": [_fmt(t) for t in r.index], "volume": r["traffic_volume"].round().astype(int).tolist()})


@app.route("/api/forecast")
@app.route("/forecast")
def api_forecast():
    horizon = min(max(request.args.get("horizon", 24, type=int), 1), MAX_HORIZON)
    return jsonify(recursive_forecast(horizon))


@app.route("/api/evaluation")
@app.route("/evaluation")
def api_evaluation():
    s = load_artifacts()
    hours = min(max(request.args.get("hours", 168, type=int), 24), len(s["test"]))
    t = s["test"].iloc[:hours]
    return jsonify({
        "time": [_fmt(x) for x in t["date_time"]],
        "actual": t["actual"].round().astype(int).tolist(),
        "predicted": t["predicted"].round().astype(int).tolist(),
        "metrics": s["meta"]["metrics"],
    })


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
