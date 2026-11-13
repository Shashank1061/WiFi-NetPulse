"""
predictor.py — ML Speed Prediction Engine
Predicts next hour's download speed, upload speed, and latency
using a Random Forest Regressor trained on historical data.

Features used:
  - hour_of_day       (0–23)
  - day_of_week       (0=Mon … 6=Sun)
  - is_weekend        (0 or 1)
  - is_peak_morning   (7–9 AM)
  - is_peak_evening   (8–11 PM)
  - lag_1h_download   (speed 1 hour ago)
  - lag_2h_download   (speed 2 hours ago)
  - lag_3h_download   (speed 3 hours ago)
  - rolling_3h_avg    (3-hour rolling average)
  - rolling_6h_avg    (6-hour rolling average)

Dependencies:
    pip install scikit-learn pandas numpy joblib
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from database import get_connection

# ── Logging ────────────────────────────────────────────────────
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(MODEL_DIR, "rf_speed_model.pkl")
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH     = os.path.join(MODEL_DIR, "model_meta.pkl")

MIN_ROWS_REQUIRED = 20   # Minimum DB rows needed to train


# ══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def load_data_from_db() -> pd.DataFrame:
    """Load all network_metrics rows into a DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT timestamp, download_mbps, upload_mbps, latency_ms, "
        "       packet_loss_pct, jitter_ms "
        "FROM network_metrics "
        "ORDER BY timestamp ASC",
        conn
    )
    conn.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["download_mbps", "upload_mbps", "latency_ms"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based and lag features to the DataFrame."""
    df = df.copy()

    # ── Time features ──────────────────────────────────────────
    df["hour_of_day"]      = df["timestamp"].dt.hour
    df["day_of_week"]      = df["timestamp"].dt.dayofweek   # 0=Mon
    df["is_weekend"]       = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_morning"]  = df["hour_of_day"].between(7, 9).astype(int)
    df["is_peak_evening"]  = df["hour_of_day"].between(20, 23).astype(int)
    df["is_off_peak"]      = df["hour_of_day"].between(1, 5).astype(int)

    # ── Lag features (previous readings) ─────────────────────
    for lag in [1, 2, 3]:
        df[f"lag_{lag}_download"] = df["download_mbps"].shift(lag)
        df[f"lag_{lag}_upload"]   = df["upload_mbps"].shift(lag)
        df[f"lag_{lag}_latency"]  = df["latency_ms"].shift(lag)

    # ── Rolling averages ──────────────────────────────────────
    df["rolling_3_avg_dl"]  = df["download_mbps"].rolling(3,  min_periods=1).mean()
    df["rolling_6_avg_dl"]  = df["download_mbps"].rolling(6,  min_periods=1).mean()
    df["rolling_3_avg_lat"] = df["latency_ms"].rolling(3, min_periods=1).mean()

    # ── Trend: change from previous reading ───────────────────
    df["dl_trend"] = df["download_mbps"].diff().fillna(0)

    # Drop rows where lag features are NaN (first few rows)
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "hour_of_day", "day_of_week", "is_weekend",
    "is_peak_morning", "is_peak_evening", "is_off_peak",
    "lag_1_download", "lag_2_download", "lag_3_download",
    "lag_1_upload",   "lag_2_upload",
    "lag_1_latency",  "lag_3_latency",
    "rolling_3_avg_dl", "rolling_6_avg_dl",
    "rolling_3_avg_lat", "dl_trend",
]

TARGET_COLS = ["download_mbps", "upload_mbps", "latency_ms"]


# ══════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════

def train_model(force: bool = False) -> dict:
    """
    Train a Random Forest model on historical data.
    Returns a dict with training stats.
    Saves the model to disk.
    """
    log.info("Loading data from database…")
    raw = load_data_from_db()

    if len(raw) < MIN_ROWS_REQUIRED:
        msg = (f"Not enough data to train: {len(raw)} rows found, "
               f"need at least {MIN_ROWS_REQUIRED}. "
               f"Run collector.py to gather more data first.")
        log.warning(msg)
        return {"success": False, "message": msg, "rows": len(raw)}

    df = engineer_features(raw)
    log.info(f"Training on {len(df)} samples with {len(FEATURE_COLS)} features.")

    X = df[FEATURE_COLS].values
    results = {}

    models   = {}
    scalers  = {}
    metrics  = {}

    for target in TARGET_COLS:
        y = df[target].values

        # Random Forest — handles non-linearity and feature interactions
        rf = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])

        # Cross-validation score (3-fold, small dataset friendly)
        cv_scores = cross_val_score(rf, X, y, cv=min(3, len(df)//5),
                                     scoring="neg_mean_absolute_error")
        mean_mae = -cv_scores.mean()

        # Final fit on all data
        rf.fit(X, y)

        # In-sample R²
        y_pred = rf.predict(X)
        r2  = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        models[target]  = rf
        metrics[target] = {
            "cv_mae":   round(mean_mae, 3),
            "train_r2": round(r2, 3),
            "train_mae": round(mae, 3),
        }
        log.info(f"  [{target}] CV MAE={mean_mae:.2f}  R²={r2:.3f}")

    # Save to disk
    meta = {
        "trained_at":   datetime.now().isoformat(),
        "n_samples":    len(df),
        "feature_cols": FEATURE_COLS,
        "target_cols":  TARGET_COLS,
        "metrics":      metrics,
        "last_row": {
            "download_mbps": float(df["download_mbps"].iloc[-1]),
            "upload_mbps":   float(df["upload_mbps"].iloc[-1]),
            "latency_ms":    float(df["latency_ms"].iloc[-1]),
            "lag_2_download":float(df["download_mbps"].iloc[-2]) if len(df)>1 else 0,
            "lag_3_download":float(df["download_mbps"].iloc[-3]) if len(df)>2 else 0,
            "lag_2_upload":  float(df["upload_mbps"].iloc[-2])   if len(df)>1 else 0,
            "lag_2_latency": float(df["latency_ms"].iloc[-2])    if len(df)>1 else 0,
            "lag_3_latency": float(df["latency_ms"].iloc[-3])    if len(df)>2 else 0,
            "rolling_3_avg_dl":  float(df["rolling_3_avg_dl"].iloc[-1]),
            "rolling_6_avg_dl":  float(df["rolling_6_avg_dl"].iloc[-1]),
            "rolling_3_avg_lat": float(df["rolling_3_avg_lat"].iloc[-1]),
            "dl_trend":          float(df["dl_trend"].iloc[-1]),
        }
    }

    joblib.dump(models, MODEL_PATH)
    joblib.dump(meta,   META_PATH)

    log.info(f"Model saved to {MODEL_PATH}")
    return {
        "success":  True,
        "samples":  len(df),
        "metrics":  metrics,
        "trained_at": meta["trained_at"],
    }


# ══════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════

def _build_feature_row(target_dt: datetime, meta: dict) -> np.ndarray:
    """
    Build a single feature row for a future timestamp.
    Uses the last known values stored in meta as lag features.
    """
    lr = meta["last_row"]
    row = {
        "hour_of_day":      target_dt.hour,
        "day_of_week":      target_dt.weekday(),
        "is_weekend":       int(target_dt.weekday() >= 5),
        "is_peak_morning":  int(7 <= target_dt.hour <= 9),
        "is_peak_evening":  int(20 <= target_dt.hour <= 23),
        "is_off_peak":      int(1 <= target_dt.hour <= 5),
        "lag_1_download":   lr["download_mbps"],
        "lag_2_download":   lr["lag_2_download"],
        "lag_3_download":   lr["lag_3_download"],
        "lag_1_upload":     lr["upload_mbps"],
        "lag_2_upload":     lr["lag_2_upload"],
        "lag_1_latency":    lr["latency_ms"],
        "lag_3_latency":    lr["lag_3_latency"],
        "rolling_3_avg_dl": lr["rolling_3_avg_dl"],
        "rolling_6_avg_dl": lr["rolling_6_avg_dl"],
        "rolling_3_avg_lat":lr["rolling_3_avg_lat"],
        "dl_trend":         lr["dl_trend"],
    }
    return np.array([[row[f] for f in FEATURE_COLS]])


def predict_next_hours(n_hours: int = 6) -> list:
    """
    Predict network metrics for the next N hours.
    Returns a list of dicts with predictions + confidence info.

    Raises FileNotFoundError if model hasn't been trained yet.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError(
            "Model not trained yet. Call /api/ml/train first."
        )

    models = joblib.load(MODEL_PATH)
    meta   = joblib.load(META_PATH)

    now          = datetime.now()
    predictions  = []

    # Keep a rolling "last row" so each prediction feeds the next
    rolling_meta = dict(meta["last_row"])

    for h in range(1, n_hours + 1):
        target_dt = now + timedelta(hours=h)

        X_row = _build_feature_row(target_dt, {"last_row": rolling_meta})

        pred = {}
        for target in TARGET_COLS:
            val = float(models[target].predict(X_row)[0])
            # Get prediction interval using individual tree predictions
            tree_preds = np.array([
                tree.predict(
                    models[target]["scaler"].transform(X_row)
                )[0]
                for tree in models[target]["model"].estimators_
            ])
            pred[target] = {
                "value": round(max(0, val), 2),
                "lower": round(max(0, float(np.percentile(tree_preds, 10))), 2),
                "upper": round(max(0, float(np.percentile(tree_preds, 90))), 2),
            }

        # Quality label
        dl = pred["download_mbps"]["value"]
        lat = pred["latency_ms"]["value"]
        if dl >= 10 and lat < 80:
            quality = "Excellent"
            quality_color = "#00ff9d"
        elif dl >= 5 and lat < 150:
            quality = "Good"
            quality_color = "#00e5ff"
        elif dl >= 2:
            quality = "Fair"
            quality_color = "#ffab00"
        else:
            quality = "Poor"
            quality_color = "#ff3d5a"

        predictions.append({
            "hour":           h,
            "timestamp":      target_dt.strftime("%Y-%m-%d %H:%M"),
            "hour_label":     target_dt.strftime("%I %p").lstrip("0"),
            "download_mbps":  pred["download_mbps"],
            "upload_mbps":    pred["upload_mbps"],
            "latency_ms":     pred["latency_ms"],
            "quality":        quality,
            "quality_color":  quality_color,
        })

        # Update rolling lag for next iteration
        rolling_meta["lag_3_download"]  = rolling_meta["lag_2_download"]
        rolling_meta["lag_2_download"]  = rolling_meta["download_mbps"]
        rolling_meta["download_mbps"]   = pred["download_mbps"]["value"]
        rolling_meta["lag_2_upload"]    = rolling_meta["upload_mbps"]
        rolling_meta["upload_mbps"]     = pred["upload_mbps"]["value"]
        rolling_meta["lag_3_latency"]   = rolling_meta["lag_2_latency"]
        rolling_meta["lag_2_latency"]   = rolling_meta["latency_ms"]
        rolling_meta["latency_ms"]      = pred["latency_ms"]["value"]
        rolling_meta["dl_trend"]        = (pred["download_mbps"]["value"]
                                           - rolling_meta["lag_2_download"])
        # Update rolling averages
        recent = [rolling_meta["download_mbps"],
                  rolling_meta["lag_2_download"],
                  rolling_meta["lag_3_download"]]
        rolling_meta["rolling_3_avg_dl"] = round(np.mean(recent), 2)

    return predictions


def get_model_info() -> dict:
    """Return metadata about the currently saved model."""
    if not os.path.exists(META_PATH):
        return {"trained": False}
    meta = joblib.load(META_PATH)
    return {
        "trained":    True,
        "trained_at": meta.get("trained_at"),
        "n_samples":  meta.get("n_samples"),
        "metrics":    meta.get("metrics"),
    }


def get_best_usage_hours(n_hours: int = 24) -> list:
    """
    Predict the next 24 hours and rank them by expected download speed.
    Useful for showing students the best time to use the internet.
    """
    try:
        preds = predict_next_hours(n_hours)
        ranked = sorted(preds,
                        key=lambda x: x["download_mbps"]["value"],
                        reverse=True)
        return ranked[:5]  # Top 5 best hours
    except FileNotFoundError:
        return []


# ── CLI usage ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="WiFi Monitor ML Predictor")
    parser.add_argument("--train",   action="store_true", help="Train the model")
    parser.add_argument("--predict", type=int, default=6,
                        help="Predict next N hours (default: 6)")
    parser.add_argument("--info",    action="store_true",
                        help="Show model info")
    args = parser.parse_args()

    if args.train:
        result = train_model()
        print(json.dumps(result, indent=2))

    if args.info:
        info = get_model_info()
        print(json.dumps(info, indent=2))

    if not args.train and not args.info:
        try:
            preds = predict_next_hours(args.predict)
            print(f"\nPredictions for next {args.predict} hours:\n")
            for p in preds:
                print(f"  {p['hour_label']:>6}  |  "
                      f"↓ {p['download_mbps']['value']:5.1f} Mbps  |  "
                      f"↑ {p['upload_mbps']['value']:5.1f} Mbps  |  "
                      f"Ping {p['latency_ms']['value']:5.0f} ms  |  "
                      f"{p['quality']}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Run with --train first.")
