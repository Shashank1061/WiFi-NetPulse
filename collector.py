"""
collector.py — Data collection engine
Measures download/upload speed, latency, jitter, and packet loss.
Runs continuously on a schedule and stores results in the database.

Dependencies:
    pip install speedtest-cli ping3 schedule
"""

import time
import subprocess
import platform
import statistics
import schedule
import logging
import json
from datetime import datetime

import speedtest

from database import init_db, insert_metric, insert_alert, get_latest_metric

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("collector.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── Thresholds for alerts ───────────────────────────────────────────────────
THRESHOLDS = {
    "min_download_mbps": 5.0,     # Alert if download < 5 Mbps
    "min_upload_mbps":   2.0,     # Alert if upload   < 2 Mbps
    "max_latency_ms":    150.0,   # Alert if ping     > 150 ms
    "max_packet_loss":   5.0,     # Alert if loss     > 5%
}

# Hosts to ping for latency measurement
PING_HOSTS = ["8.8.8.8", "1.1.1.1", "8.8.4.4"]


# ─── Latency & Packet Loss ───────────────────────────────────────────────────

def _ping_once(host: str, count: int = 5):
    """
    Ping `host` `count` times using the OS ping command.
    Returns (avg_ms, packet_loss_pct, jitter_ms) or (None, None, None) on failure.
    """
    system = platform.system().lower()
    if system == "windows":
        cmd = ["ping", "-n", str(count), host]
    else:
        cmd = ["ping", "-c", str(count), "-W", "2", host]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        output = result.stdout

        if system == "windows":
            # Parse Windows ping output
            import re
            times = [float(x) for x in re.findall(r"time[=<](\d+)ms", output)]
            loss_match = re.search(r"(\d+)%\s+loss", output)
            loss = float(loss_match.group(1)) if loss_match else 100.0
        else:
            # Parse Linux/macOS ping output
            import re
            times = [float(x) for x in re.findall(r"time=(\d+\.?\d*)\s*ms", output)]
            loss_match = re.search(r"(\d+\.?\d*)% packet loss", output)
            loss = float(loss_match.group(1)) if loss_match else 100.0

        if not times:
            return None, 100.0, None

        avg_ms  = statistics.mean(times)
        jitter  = statistics.stdev(times) if len(times) > 1 else 0.0
        return round(avg_ms, 2), round(loss, 2), round(jitter, 2)

    except (subprocess.TimeoutExpired, Exception) as e:
        log.warning(f"Ping to {host} failed: {e}")
        return None, 100.0, None


def measure_latency():
    """
    Ping multiple hosts and return the best (lowest latency) result.
    Returns dict with latency_ms, packet_loss_pct, jitter_ms.
    """
    results = []
    for host in PING_HOSTS:
        avg, loss, jitter = _ping_once(host)
        if avg is not None:
            results.append({"host": host, "latency": avg,
                             "loss": loss, "jitter": jitter})

    if not results:
        log.error("All ping hosts unreachable.")
        return {"latency_ms": None, "packet_loss_pct": 100.0, "jitter_ms": None}

    best = min(results, key=lambda r: r["latency"])
    log.info(f"Latency  → {best['latency']} ms  |  Loss: {best['loss']}%  "
             f"|  Jitter: {best['jitter']} ms  (via {best['host']})")
    return {
        "latency_ms":      best["latency"],
        "packet_loss_pct": best["loss"],
        "jitter_ms":       best["jitter"],
    }


# ─── Speed Test ──────────────────────────────────────────────────────────────

def measure_speed():
    """
    Run a speedtest and return download/upload in Mbps plus ISP info.
    Returns dict or None on failure.
    """
    try:
        log.info("Running speedtest…  (this may take 10-30 seconds)")
        st = speedtest.Speedtest(secure=True)
        st.get_best_server()

        download_bps = st.download()
        upload_bps   = st.upload()
        results      = st.results.dict()

        download_mbps = round(download_bps / 1_000_000, 2)
        upload_mbps   = round(upload_bps   / 1_000_000, 2)
        isp           = results.get("client", {}).get("isp", "Unknown")
        server_name   = results.get("server",  {}).get("name", "")
        ip_address    = results.get("client", {}).get("ip", "")

        log.info(f"Download → {download_mbps} Mbps  |  Upload → {upload_mbps} Mbps  "
                 f"|  ISP: {isp}")

        return {
            "download_mbps": download_mbps,
            "upload_mbps":   upload_mbps,
            "isp":           isp,
            "server_name":   server_name,
            "ip_address":    ip_address,
        }

    except speedtest.SpeedtestException as e:
        log.error(f"Speedtest failed: {e}")
        return None


# ─── Alert Checker ───────────────────────────────────────────────────────────

def check_and_raise_alerts(metrics: dict):
    """Compare metrics against thresholds and log any alerts."""

    checks = [
        ("slow_speed",    "download_mbps",   metrics.get("download_mbps"),
         THRESHOLDS["min_download_mbps"],
         lambda v, t: v < t,
         "Low download speed: {v} Mbps (threshold: {t} Mbps)"),

        ("slow_speed",    "upload_mbps",     metrics.get("upload_mbps"),
         THRESHOLDS["min_upload_mbps"],
         lambda v, t: v < t,
         "Low upload speed: {v} Mbps (threshold: {t} Mbps)"),

        ("high_latency",  "latency_ms",      metrics.get("latency_ms"),
         THRESHOLDS["max_latency_ms"],
         lambda v, t: v > t,
         "High latency: {v} ms (threshold: {t} ms)"),

        ("packet_loss",   "packet_loss_pct", metrics.get("packet_loss_pct"),
         THRESHOLDS["max_packet_loss"],
         lambda v, t: v > t,
         "Packet loss detected: {v}% (threshold: {t}%)"),
    ]

    for alert_type, metric_name, value, threshold, condition, msg_tmpl in checks:
        if value is not None and condition(value, threshold):
            message = msg_tmpl.format(v=value, t=threshold)
            log.warning(f"[ALERT] {message}")
            insert_alert(alert_type, metric_name, value, threshold, message)


# ─── Main Collection Job ──────────────────────────────────────────────────────

def collect():
    """Single collection cycle: latency first (fast), then speed test."""
    log.info("=" * 55)
    log.info(f"Collection started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Latency & packet loss (fast — ~5 seconds)
    lat = measure_latency()

    # 2. Speed test (slow — ~20-30 seconds)
    spd = measure_speed()

    if spd is None:
        log.error("Skipping DB insert — speed test unavailable.")
        return

    combined = {**lat, **spd}

    # 3. Persist to database
    insert_metric(
        download    = combined["download_mbps"],
        upload      = combined["upload_mbps"],
        latency     = combined["latency_ms"] or 0,
        packet_loss = combined["packet_loss_pct"] or 0,
        jitter      = combined["jitter_ms"] or 0,
        isp         = combined["isp"],
        server      = combined["server_name"],
        ip          = combined["ip_address"],
    )
    log.info("Metrics saved to database.")

    # 4. Check thresholds
    check_and_raise_alerts(combined)

    log.info("Collection complete.")
    log.info("=" * 55)


# ─── Scheduler ───────────────────────────────────────────────────────────────

def run_scheduler(interval_minutes: int = 15):
    """
    Run the collector on a schedule.
    Default: every 15 minutes.
    """
    log.info(f"Scheduler started — collecting every {interval_minutes} min.")
    collect()  # Run immediately on start

    schedule.every(interval_minutes).minutes.do(collect)

    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    init_db()

    import argparse
    parser = argparse.ArgumentParser(description="WiFi Monitor Collector")
    parser.add_argument("--interval", type=int, default=15,
                        help="Collection interval in minutes (default: 15)")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (no scheduler)")
    args = parser.parse_args()

    if args.once:
        collect()
    else:
        run_scheduler(args.interval)
