"""
app.py — Flask REST API for WiFi Monitoring Dashboard (v2 with ML)
"""

import threading
import logging
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

from database import (
    init_db, get_latest_metric, get_metrics_history,
    get_hourly_averages, get_daily_summary, get_recent_alerts, get_peak_hours,
    insert_metric, get_networks,
)

app = Flask(__name__, static_folder="static")
allowed_origin = os.getenv("FRONTEND_ORIGIN")
if allowed_origin:
    CORS(app, resources={r"/api/*": {"origins": [allowed_origin]}})
# Otherwise, no CORS policy is needed: the dashboard and API share one origin.
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Fix for Windows — always get the absolute path of the folder this file is in
BASE = os.path.dirname(os.path.abspath(__file__))

def success(data):
    return jsonify({"status": "ok", "data": data})

def error(message, code=400):
    return jsonify({"status": "error", "message": message}), code

@app.route("/")
def index():
    return send_from_directory(BASE, "dashboard.html")

@app.route("/predictor_dashboard.html")
def predictor():
    return send_from_directory(BASE, "predictor_dashboard.html")

@app.route("/api/status")
def api_status():
    return success({"server": "WiFi Monitor API", "version": "2.0.0",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

@app.route("/api/latest")
def api_latest():
    metric = get_latest_metric()
    if not metric:
        return error("No data collected yet.", 404)
    return success(metric)

@app.route("/api/history")
def api_history():
    try:
        hours = max(1, min(int(request.args.get("hours", 24)), 720))
        limit = max(10, min(int(request.args.get("limit", 200)), 1000))
    except ValueError:
        return error("Invalid query parameters.")
    return success({"rows": get_metrics_history(hours, limit)})


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    key = os.getenv("INGEST_API_KEY")
    supplied = request.headers.get("X-Ingest-Key", "")
    if not key:
        return error("INGEST_API_KEY is not configured.", 503)
    import hmac
    if not hmac.compare_digest(supplied, key):
        return error("Unauthorized local collector.", 401)
    payload = request.get_json(silent=True) or {}
    try:
        download = float(payload["download_mbps"]); upload = float(payload["upload_mbps"])
        latency = float(payload.get("latency_ms") or 0)
        loss = float(payload.get("packet_loss_pct") or 0)
        jitter = float(payload.get("jitter_ms") or 0)
    except (KeyError, TypeError, ValueError):
        return error("Valid download_mbps and upload_mbps are required.")
    vals = (download, upload, latency, loss, jitter)
    if any(v < 0 or not __import__("math").isfinite(v) for v in vals):
        return error("Measurement values must be finite and non-negative.")
    ssid = str(payload.get("ssid") or "Unknown")[:255]
    network_id = str(payload.get("network_id") or "")[:64] or None
    insert_metric(download, upload, latency, loss, jitter,
                  str(payload.get("isp") or "")[:255],
                  str(payload.get("server_name") or "")[:255],
                  str(payload.get("ip_address") or "")[:64],
                  ssid=ssid, network_id=network_id, source="local")
    return success({"message": "Local measurement saved.", "ssid": ssid,
                    "network_id": network_id, "source": "local"}), 201

@app.route("/api/networks")
def api_networks():
    return success({"networks": get_networks()})

@app.route("/api/hourly")
def api_hourly():
    try:
        days = max(1, min(int(request.args.get("days", 7)), 30))
    except ValueError:
        return error("Invalid days parameter.")
    return success({"rows": get_hourly_averages(days)})

@app.route("/api/summary")
def api_summary():
    return success(get_daily_summary())

@app.route("/api/alerts")
def api_alerts():
    try:
        limit = max(1, min(int(request.args.get("limit", 10)), 100))
    except ValueError:
        return error("Invalid limit parameter.")
    alerts = get_recent_alerts(limit)
    return success({"alerts": alerts, "count": len(alerts)})

def require_admin_key():
    """Protect operational endpoints; never expose collection/training publicly."""
    configured_key = os.getenv("ADMIN_API_KEY")
    supplied_key = request.headers.get("X-Admin-Key", "")
    if not configured_key:
        return error("This operation is disabled until ADMIN_API_KEY is configured.", 503)
    import hmac
    if not hmac.compare_digest(supplied_key, configured_key):
        return error("Unauthorized.", 401)
    return None


@app.route("/api/collect", methods=["POST"])
def api_collect():
    denied = require_admin_key()
    if denied:
        return denied
    def _run():
        try:
            from collector import collect
            collect()
        except Exception as e:
            log.error(f"Collection failed: {e}")
    threading.Thread(target=_run, daemon=True).start()
    return success({"message": "Collection started in background."})

@app.route("/api/stats/peak-hours")
def api_peak_hours():
    return success(get_peak_hours())

# ── ML Endpoints ──────────────────────────────────────────────

@app.route("/api/ml/train", methods=["POST"])
def api_ml_train():
    denied = require_admin_key()
    if denied:
        return denied
    try:
        from predictor import train_model
        result = train_model()
        return success(result) if result["success"] else error(result["message"], 422)
    except Exception as e:
        return error(str(e), 500)

@app.route("/api/ml/predict")
def api_ml_predict():
    try:
        hours = max(1, min(int(request.args.get("hours", 6)), 24))
    except ValueError:
        return error("Invalid hours parameter.")
    try:
        from predictor import predict_next_hours
        preds = predict_next_hours(hours)
        return success({"predictions": preds, "count": len(preds),
                        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    except FileNotFoundError as e:
        return error(str(e), 404)
    except Exception as e:
        return error(str(e), 500)

@app.route("/api/ml/best-hours")
def api_ml_best_hours():
    try:
        from predictor import get_best_usage_hours
        return success({"best_hours": get_best_usage_hours(24)})
    except FileNotFoundError as e:
        return error(str(e), 404)
    except Exception as e:
        return error(str(e), 500)

@app.route("/api/ml/info")
def api_ml_info():
    try:
        from predictor import get_model_info
        return success(get_model_info())
    except Exception as e:
        return error(str(e), 500)

@app.errorhandler(404)
def not_found(e): return error("Not found.", 404)

@app.errorhandler(500)
def server_error(e): return error("Server error.", 500)

# Idempotent schema creation also runs under Gunicorn, where __main__ is not executed.
init_db()

if __name__ == "__main__":
    log.info("Starting WiFi-NetPulse API on http://localhost:5000")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)