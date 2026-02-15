"""Gunicorn configuration for Stock Analysis Pro."""

import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Worker processes
# Render free tier has limited memory (512MB), so keep workers low.
# gthread workers handle concurrent I/O (yfinance API calls) well.
workers = int(os.environ.get("WEB_CONCURRENCY", 2))
worker_class = "gthread"
threads = 4

# Timeout - long timeout for streaming dividend scans and heavy analysis
timeout = 300
graceful_timeout = 120
keep_alive = 5

# Restart workers periodically to reclaim memory leaked by matplotlib/pandas
max_requests = 500
max_requests_jitter = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")

# Preload the app so workers fork with the app already loaded,
# sharing memory via copy-on-write and speeding up boot time.
preload_app = True

# Server mechanics
tmp_dir = "/dev/shm"
