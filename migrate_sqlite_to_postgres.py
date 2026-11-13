"""One-time migration of an existing local SQLite database to Neon PostgreSQL.

Usage:
  PowerShell: $env:DATABASE_URL = "<Neon connection string>"
             python migrate_sqlite_to_postgres.py ./wifi_monitor.db

Run only after setting DATABASE_URL. The destination tables must be empty.
"""

import argparse
import os
import sqlite3
from datetime import datetime, timezone

from sqlalchemy import func, select, text

from database import engine, metadata, network_metrics, hourly_summary, alerts

TABLES = (network_metrics, hourly_summary, alerts)


def normalize_row(table, row):
    values = dict(row)
    for column_name in ("timestamp", "hour_bucket"):
        value = values.get(column_name)
        if isinstance(value, str) and value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            # The original SQLite application wrote local wall-clock timestamps.
            if parsed.tzinfo is None:
                parsed = parsed.astimezone()
            values[column_name] = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    allowed = set(table.c.keys())
    return {key: value for key, value in values.items() if key in allowed}


def main():
    parser = argparse.ArgumentParser(description="Copy WiFi-NetPulse SQLite data to PostgreSQL")
    parser.add_argument("sqlite_file", nargs="?", default="wifi_monitor.db", help="Path to local SQLite database")
    args = parser.parse_args()

    if engine.dialect.name != "postgresql":
        raise SystemExit("DATABASE_URL must point to PostgreSQL/Neon before running this migration.")
    if not os.path.isfile(args.sqlite_file):
        raise SystemExit(f"SQLite file not found: {args.sqlite_file}")

    metadata.create_all(engine)
    source = sqlite3.connect(args.sqlite_file)
    source.row_factory = sqlite3.Row

    try:
        with engine.begin() as destination:
            for table in TABLES:
                exists = source.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table.name,),
                ).fetchone()
                if not exists:
                    print(f"[SKIP] Source table {table.name} does not exist.")
                    continue

                destination_count = destination.execute(
                    select(func.count()).select_from(table)
                ).scalar_one()
                if destination_count:
                    raise SystemExit(
                        f"Destination table {table.name} already contains {destination_count} rows. "
                        "Migration stopped to prevent duplicate data. Use an empty Neon database."
                    )

                rows = source.execute(f'SELECT * FROM "{table.name}"').fetchall()
                payload = [normalize_row(table, row) for row in rows]
                if payload:
                    destination.execute(table.insert(), payload)
                print(f"[OK] {table.name}: copied {len(payload)} rows.")

                if engine.dialect.name == "postgresql" and "id" in table.c:
                    destination.execute(text(
                        f"SELECT setval(pg_get_serial_sequence('{table.name}', 'id'), "
                        f"COALESCE(MAX(id), 1), MAX(id) IS NOT NULL) FROM {table.name}"
                    ))
    finally:
        source.close()

    print("Migration finished. Verify row counts in Neon before deploying the web service.")


if __name__ == "__main__":
    main()
