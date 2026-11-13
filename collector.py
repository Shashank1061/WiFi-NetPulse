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
import os
import re
import hashlib
import urllib.request
import urllib.error
from datetime import datetime

import speedtest

from database import init_db, insert_metric, insert_alert, get_latest_metric

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
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



# Local Wi-Fi identity and secure cloud upload.
def detect_wifi_network():
    if platform.system().lower() != "windows":
        return "Unknown", None
    try:
        r = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True,
                           text=True, timeout=10, encoding="utf-8", errors="replace")
        ss = re.search(r"^\s*SSID\s*:\s*(.+?)\s*$", r.stdout, re.M)
        bs = re.search(r"^\s*BSSID\s*:\s*([0-9a-fA-F:-]+)", r.stdout, re.M)
        ssid = ss.group(1).strip() if ss else "Unknown"
        bssid = bs.group(1).lower() if bs else ""
        if ssid == "Unknown": return ssid, None
        # Hash SSID + BSSID; the raw access-point MAC is never uploaded.
        return ssid, hashlib.sha256(f"{ssid}|{bssid}".encode()).hexdigest()[:32]
    except Exception as exc:
        log.warning("Wi-Fi detection failed: %s", exc)
        return "Unknown", None


def send_to_cloud(metrics, remote_url, api_key):
    endpoint = remote_url.rstrip("/")
    if not endpoint.endswith("/api/ingest"): endpoint += "/api/ingest"
    req = urllib.request.Request(endpoint, data=json.dumps(metrics).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-Ingest-Key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = response.read().decode("utf-8", errors="replace")
            log.info("Cloud ingestion response: %s", result)
            return result
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Cloud ingestion failed ({exc.code}): {exc.read().decode(errors='replace')}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach cloud API: {exc}")

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

def collect(remote_url=None, api_key=None):
    """Run measurements on this machine; optionally upload them to Render."""
    log.info("Collection started at %s", datetime.now().isoformat(timespec="seconds"))
    ssid, network_id = detect_wifi_network()
    log.info("Connected Wi-Fi: %s", ssid)
    lat = measure_latency()
    spd = measure_speed()
    if spd is None:
        log.error("Speed test unavailable; no record saved.")
        return None
    data = {**lat, **spd, "ssid": ssid, "network_id": network_id}
    if remote_url:
        if not api_key: raise RuntimeError("WIFI_NETPULSE_API_KEY is required for cloud upload.")
        send_to_cloud(data, remote_url, api_key)
        log.info("Local measurement uploaded for network '%s'.", ssid)
    else:
        insert_metric(download=data["download_mbps"], upload=data["upload_mbps"],
            latency=data["latency_ms"] or 0, packet_loss=data["packet_loss_pct"] or 0,
            jitter=data["jitter_ms"] or 0, isp=data["isp"], server=data["server_name"],
            ip=data["ip_address"], ssid=ssid, network_id=network_id, source="local-development")
        log.info("Measurement saved to local development database.")
    check_and_raise_alerts(data)
    log.info("Collection complete.")
    return data


# ─── Scheduler ───────────────────────────────────────────────────────────────

def run_scheduler(interval_minutes: int = 15, remote_url=None, api_key=None):
    """
    Run the collector on a schedule.
    Default: every 15 minutes.
    """
    log.info(f"Scheduler started — collecting every {interval_minutes} min.")
    collect(remote_url=remote_url, api_key=api_key)  # Run immediately on start

    schedule.every(interval_minutes).minutes.do(lambda: collect(remote_url=remote_url, api_key=api_key))

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
    parser.add_argument("--remote-url", default=os.getenv("WIFI_NETPULSE_API_URL"),
                        help="Cloud URL, e.g. https://wifi-netpulse.onrender.com")
    parser.add_argument("--api-key", default=os.getenv("WIFI_NETPULSE_API_KEY"),
                        help="Ingestion key; prefer environment variable")
    args = parser.parse_args()
    if args.remote_url and not args.api_key:
        parser.error("Set WIFI_NETPULSE_API_KEY when using --remote-url")
    if args.once:
        collect(remote_url=args.remote_url, api_key=args.api_key)
    else:
        run_scheduler(args.interval, remote_url=args.remote_url, api_key=args.api_key)
