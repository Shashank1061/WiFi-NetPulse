"""
database.py — SQLite database setup and helper functions
WiFi Usage and Speed Monitoring System
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wifi_monitor.db")


def get_connection():
    """Return a SQLite connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Main metrics table — one row per measurement
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS network_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       DATETIME DEFAULT (datetime('now', 'localtime')),
            download_mbps   REAL,          -- Download speed in Mbps
            upload_mbps     REAL,          -- Upload speed in Mbps
            latency_ms      REAL,          -- Ping latency in milliseconds
            packet_loss_pct REAL,          -- Packet loss percentage (0-100)
            jitter_ms       REAL,          -- Latency variation in ms
            isp             TEXT,          -- ISP name from speedtest
            server_name     TEXT,          -- Speedtest server used
            ip_address      TEXT           -- External IP at time of test
        )
    """)

    # Hourly aggregates for faster dashboard queries
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hourly_summary (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            hour_bucket     DATETIME UNIQUE, -- Truncated to the hour
            avg_download    REAL,
            avg_upload      REAL,
            avg_latency     REAL,
            max_download    REAL,
            min_download    REAL,
            sample_count    INTEGER
        )
    """)

    # Alerts — triggered when thresholds are breached
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   DATETIME DEFAULT (datetime('now', 'localtime')),
            alert_type  TEXT,   -- 'slow_speed' | 'high_latency' | 'packet_loss'
            metric      TEXT,
            value       REAL,
            threshold   REAL,
            message     TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Database initialized at:", DB_PATH)


def insert_metric(download, upload, latency, packet_loss=0.0,
                  jitter=0.0, isp="", server="", ip=""):
    """Insert a single measurement row."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO network_metrics
            (download_mbps, upload_mbps, latency_ms, packet_loss_pct,
             jitter_ms, isp, server_name, ip_address)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (download, upload, latency, packet_loss, jitter, isp, server, ip))
    conn.commit()
    conn.close()


def insert_alert(alert_type, metric, value, threshold, message):
    """Log an alert when a threshold is breached."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO alerts (alert_type, metric, value, threshold, message)
        VALUES (?, ?, ?, ?, ?)
    """, (alert_type, metric, value, threshold, message))
    conn.commit()
    conn.close()


def get_latest_metric():
    """Return the most recent measurement as a dict."""
    conn = get_connection()
    row = conn.execute("""
        SELECT * FROM network_metrics ORDER BY timestamp DESC LIMIT 1
    """).fetchone()
    conn.close()
    return dict(row) if row else None


def get_metrics_history(hours=24, limit=200):
    """Return up to `limit` rows from the last `hours` hours."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM network_metrics
        WHERE timestamp >= datetime('now', 'localtime', ? || ' hours')
        ORDER BY timestamp ASC
        LIMIT ?
    """, (f"-{hours}", limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_hourly_averages(days=7):
    """Return average metrics grouped by hour for the past N days."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT
            strftime('%Y-%m-%d %H:00', timestamp) AS hour,
            ROUND(AVG(download_mbps), 2)   AS avg_download,
            ROUND(AVG(upload_mbps), 2)     AS avg_upload,
            ROUND(AVG(latency_ms), 2)      AS avg_latency,
            COUNT(*)                        AS sample_count
        FROM network_metrics
        WHERE timestamp >= datetime('now', 'localtime', ? || ' days')
        GROUP BY hour
        ORDER BY hour ASC
    """, (f"-{days}",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_daily_summary():
    """Return today's min/max/avg stats."""
    conn = get_connection()
    row = conn.execute("""
        SELECT
            ROUND(AVG(download_mbps), 2)  AS avg_download,
            ROUND(MAX(download_mbps), 2)  AS max_download,
            ROUND(MIN(download_mbps), 2)  AS min_download,
            ROUND(AVG(upload_mbps), 2)    AS avg_upload,
            ROUND(AVG(latency_ms), 2)     AS avg_latency,
            ROUND(MIN(latency_ms), 2)     AS best_latency,
            COUNT(*)                       AS total_tests
        FROM network_metrics
        WHERE date(timestamp) = date('now', 'localtime')
    """).fetchone()
    conn.close()
    return dict(row) if row else {}


def get_recent_alerts(limit=10):
    """Return the most recent alerts."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
