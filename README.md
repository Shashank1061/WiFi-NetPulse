# WiFi Usage & Speed Monitoring System — Hostel Networks
**Department of Data Science | II CSD-A RTRP | A.Y. 2025–2026**

---

## Project Structure

```
wifi_monitor/
├── database.py       # SQLite schema + all DB helper functions
├── collector.py      # Data collection engine (speed, latency, packet loss)
├── app.py            # Flask REST API (backend)
├── dashboard.html    # Frontend dashboard (open in browser)
├── requirements.txt  # Python dependencies
└── wifi_monitor.db   # Auto-created SQLite database (after first run)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize the database
```bash
python database.py
```

### 3. Start the Flask API server (Terminal 1)
```bash
python app.py
```
API runs at: `http://localhost:5000`

### 4. Start the data collector (Terminal 2)
```bash
# Collect every 15 minutes (default)
python collector.py

# Collect every 5 minutes
python collector.py --interval 5

# Single run (for testing)
python collector.py --once
```

### 5. Open the Dashboard
Open `dashboard.html` in your browser, OR visit:
```
http://localhost:5000
```

---

## API Endpoints

| Method | Endpoint               | Description                        |
|--------|------------------------|------------------------------------|
| GET    | `/api/latest`          | Latest network measurement         |
| GET    | `/api/history?hours=24`| Raw history for the past N hours   |
| GET    | `/api/hourly?days=7`   | Hourly averages (for charts)       |
| GET    | `/api/summary`         | Today's min/max/avg statistics     |
| GET    | `/api/alerts?limit=10` | Recent threshold-breach alerts     |
| GET    | `/api/stats/peak-hours`| Average speed by hour of day       |
| POST   | `/api/collect`         | Trigger a manual collection run    |

---

## Alert Thresholds (configurable in collector.py)

| Metric         | Threshold          |
|----------------|--------------------|
| Download speed | < 5 Mbps → alert   |
| Upload speed   | < 2 Mbps → alert   |
| Latency        | > 150 ms → alert   |
| Packet loss    | > 5% → alert       |

---

## Tech Stack
- **Data Collection**: Python · speedtest-cli · OS ping
- **Database**: SQLite (zero setup, file-based)
- **Backend API**: Flask · Flask-CORS
- **Frontend**: HTML5 · CSS3 · Chart.js · Vanilla JS
- **Scheduler**: schedule (Python)
