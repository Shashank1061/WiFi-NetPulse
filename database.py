"""Database layer for WiFi-NetPulse.

Uses SQLite for local development when DATABASE_URL is not set, and
PostgreSQL (Neon) in deployment when DATABASE_URL is configured.
All timestamps are stored as naive UTC values for consistent cross-platform
comparisons; API responses serialize them as ISO-8601 strings.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import (
    Column, DateTime, Float, Integer, MetaData, Numeric,
    String, Table, Text, create_engine, func, insert, select,
    cast, inspect, text,
)
from sqlalchemy.engine import URL

BASE_DIR = Path(__file__).resolve().parent
APP_TIMEZONE = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Kolkata"))
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if (os.getenv("RENDER") == "true" or os.getenv("GITHUB_ACTIONS") == "true") and not DATABASE_URL:
    raise RuntimeError("DATABASE_URL must be configured in the cloud environment; refusing to use ephemeral SQLite.")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={"sslmode": "require"} if "neon.tech" in DATABASE_URL else {},
    )
else:
    sqlite_path = BASE_DIR / "wifi_monitor.db"
    engine = create_engine(
        URL.create("sqlite", database=str(sqlite_path)),
        connect_args={"check_same_thread": False},
    )

metadata = MetaData()

network_metrics = Table(
    "network_metrics", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=False), nullable=False, server_default=func.now()),
    Column("download_mbps", Float),
    Column("upload_mbps", Float),
    Column("latency_ms", Float),
    Column("packet_loss_pct", Float),
    Column("jitter_ms", Float),
    Column("isp", String(255)),
    Column("server_name", String(255)),
    Column("ip_address", String(64)),
    Column("ssid", String(255)),
    Column("network_id", String(64)),
    Column("source", String(32), server_default="cloud"),
)

hourly_summary = Table(
    "hourly_summary", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("hour_bucket", DateTime(timezone=False), unique=True),
    Column("avg_download", Float),
    Column("avg_upload", Float),
    Column("avg_latency", Float),
    Column("max_download", Float),
    Column("min_download", Float),
    Column("sample_count", Integer),
)

alerts = Table(
    "alerts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=False), nullable=False, server_default=func.now()),
    Column("alert_type", String(64)),
    Column("metric", String(64)),
    Column("value", Float),
    Column("threshold", Float),
    Column("message", Text),
    Column("network_id", String(64)),
)


def get_engine():
    """Return the shared SQLAlchemy engine (used by pandas in predictor.py)."""
    return engine


def get_connection():
    """Compatibility helper returning a SQLAlchemy Connection."""
    return engine.connect()


def _serialize(row):
    item = dict(row)
    for key, value in item.items():
        if isinstance(value, datetime):
            if key == "hour":
                # Hourly buckets have already been converted to APP_TIMEZONE.
                item[key] = value.strftime("%Y-%m-%d %H:%M:%S")
            else:
                # Measurement timestamps are UTC; the Z suffix makes browser parsing unambiguous.
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                item[key] = value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return item


def _utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def init_db():
    """Create missing tables. Safe to run repeatedly during deployment."""
    metadata.create_all(engine)
    existing = {col["name"] for col in inspect(engine).get_columns("network_metrics")}
    with engine.begin() as conn:
        for name, kind in {"ssid": "VARCHAR(255)", "network_id": "VARCHAR(64)", "source": "VARCHAR(32) DEFAULT 'cloud'"}.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE network_metrics ADD COLUMN {name} {kind}"))
        alert_columns = {col["name"] for col in inspect(engine).get_columns("alerts")}
        if "network_id" not in alert_columns:
            conn.execute(text("ALTER TABLE alerts ADD COLUMN network_id VARCHAR(64)"))
    print(f"[DB] Database initialized ({engine.dialect.name}).")


def insert_metric(download, upload, latency, packet_loss=0.0,
                  jitter=0.0, isp="", server="", ip="", ssid=None, network_id=None, source="cloud"):
    values = {
        "timestamp": _utc_now(),
        "download_mbps": download,
        "upload_mbps": upload,
        "latency_ms": latency,
        "packet_loss_pct": packet_loss,
        "jitter_ms": jitter,
        "isp": isp,
        "server_name": server,
        "ip_address": ip, "ssid": ssid, "network_id": network_id, "source": source,
    }
    with engine.begin() as conn:
        conn.execute(insert(network_metrics).values(**values))


def insert_alert(alert_type, metric, value, threshold, message, network_id=None):
    with engine.begin() as conn:
        conn.execute(insert(alerts).values(
            timestamp=_utc_now(), alert_type=alert_type, metric=metric,
            value=value, threshold=threshold, message=message, network_id=network_id,
        ))


def _network_filter(stmt, network_id):
    return stmt.where(network_metrics.c.network_id == network_id) if network_id else stmt


def get_latest_metric(network_id=None):
    stmt = _network_filter(select(network_metrics), network_id).order_by(network_metrics.c.timestamp.desc()).limit(1)
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
    return _serialize(row) if row else None


def get_metrics_history(hours=24, limit=200, network_id=None):
    since = _utc_now() - timedelta(hours=hours)
    stmt = (select(network_metrics)
            .where(network_metrics.c.timestamp >= since)
            .order_by(network_metrics.c.timestamp.asc()))
    stmt = _network_filter(stmt, network_id).limit(limit)
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [_serialize(row) for row in rows]



def get_networks():
    stmt = (select(network_metrics.c.ssid, network_metrics.c.network_id,
                   func.count().label("tests"), func.max(network_metrics.c.timestamp).label("last_seen"))
            .where(network_metrics.c.network_id.is_not(None))
            .group_by(network_metrics.c.ssid, network_metrics.c.network_id)
            .order_by(func.max(network_metrics.c.timestamp).desc()))
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [_serialize(row) for row in rows]

def get_hourly_averages(days=7, network_id=None):
    since = _utc_now() - timedelta(days=days)
    if engine.dialect.name == "postgresql":
        local_timestamp = func.timezone(APP_TIMEZONE.key, network_metrics.c.timestamp)
        bucket = func.date_trunc("hour", local_timestamp)
    else:
        # SQLite has no IANA timezone database; use the configured zone's current offset.
        offset = datetime.now(APP_TIMEZONE).utcoffset() or timedelta(0)
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        total_minutes = abs(total_minutes)
        modifiers = [f"{sign}{total_minutes // 60} hours"]
        if total_minutes % 60:
            modifiers.append(f"{sign}{total_minutes % 60} minutes")
        bucket = func.strftime("%Y-%m-%d %H:00:00", network_metrics.c.timestamp, *modifiers)
    stmt = (select(
                bucket.label("hour"),
                func.round(cast(func.avg(network_metrics.c.download_mbps), Numeric), 2).label("avg_download"),
                func.round(cast(func.avg(network_metrics.c.upload_mbps), Numeric), 2).label("avg_upload"),
                func.round(cast(func.avg(network_metrics.c.latency_ms), Numeric), 2).label("avg_latency"),
                func.count().label("sample_count"),
            )
            .where(network_metrics.c.timestamp >= since)
            .group_by(bucket)
            .order_by(bucket.asc()))
    stmt = _network_filter(stmt, network_id)
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [_serialize(row) for row in rows]


def get_daily_summary(network_id=None):
    local_now = datetime.now(APP_TIMEZONE)
    local_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    local_end = local_start + timedelta(days=1)
    start = local_start.astimezone(timezone.utc).replace(tzinfo=None)
    end = local_end.astimezone(timezone.utc).replace(tzinfo=None)
    stmt = select(
        func.round(cast(func.avg(network_metrics.c.download_mbps), Numeric), 2).label("avg_download"),
        func.round(cast(func.max(network_metrics.c.download_mbps), Numeric), 2).label("max_download"),
        func.round(cast(func.min(network_metrics.c.download_mbps), Numeric), 2).label("min_download"),
        func.round(cast(func.avg(network_metrics.c.upload_mbps), Numeric), 2).label("avg_upload"),
        func.round(cast(func.avg(network_metrics.c.latency_ms), Numeric), 2).label("avg_latency"),
        func.round(cast(func.min(network_metrics.c.latency_ms), Numeric), 2).label("best_latency"),
        func.count().label("total_tests"),
    ).where(network_metrics.c.timestamp >= start, network_metrics.c.timestamp < end)
    stmt = _network_filter(stmt, network_id)
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
    return dict(row) if row else {}


def get_recent_alerts(limit=10, network_id=None):
    stmt = select(alerts)
    if network_id:
        stmt = stmt.where(alerts.c.network_id == network_id)
    stmt = stmt.order_by(alerts.c.timestamp.desc()).limit(limit)
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [_serialize(row) for row in rows]


def get_peak_hours(network_id=None):
    if engine.dialect.name == "postgresql":
        local_timestamp = func.timezone(APP_TIMEZONE.key, network_metrics.c.timestamp)
        hour = func.cast(func.extract("hour", local_timestamp), Integer)
    else:
        offset = datetime.now(APP_TIMEZONE).utcoffset() or timedelta(0)
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        total_minutes = abs(total_minutes)
        modifiers = [f"{sign}{total_minutes // 60} hours"]
        if total_minutes % 60:
            modifiers.append(f"{sign}{total_minutes % 60} minutes")
        hour = func.cast(func.strftime("%H", network_metrics.c.timestamp, *modifiers), Integer)
    stmt = (select(
                hour.label("hour_of_day"),
                func.round(cast(func.avg(network_metrics.c.download_mbps), Numeric), 2).label("avg_download"),
func.round(cast(func.avg(network_metrics.c.latency_ms), Numeric), 2).label("avg_latency"),
                func.count().label("samples"),
            )
            .group_by(hour).order_by(hour))
    stmt = _network_filter(stmt, network_id)
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    init_db()
