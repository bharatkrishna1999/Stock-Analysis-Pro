"""
STOCK ANALYSIS PRO PRODUCTION MASTER
Single file Flask app with Yahoo hardening and realistic resilience controls.

What this version fixes
1) One safe_download() used everywhere with
   - retries + exponential backoff + jitter
   - per request min_rows validation
   - stale data guard
   - bounded TTL LRU cache
   - hard concurrency limit via semaphore
   - per worker rate limiting
   - stale cache fallback when live fetch fails

2) Correct daily change
   - Intraday uses first candle of the latest trading date in Asia/Kolkata
   - Daily uses last close vs previous close

3) Correct labeling
   - pct_from_sma9 and pct_from_mean20 are separate
   - pos_in_range and pct_from_low are separate

4) Regression benchmark ladder with explicit warnings in response
   - ^NSEI -> NIFTYBEES.NS -> RELIANCE.NS
"""

from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import requests
import json
import os
import time
import random
from datetime import datetime, timedelta
from threading import Lock, Semaphore
from collections import OrderedDict

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ----------------------------
# Global knobs
# ----------------------------
CACHE_TTL = 600                # 10 minutes
CACHE_MAX_KEYS = 220           # bounded cache
MIN_INTERVAL = 0.6             # per worker min gap between outbound calls
MAX_CONCURRENT_DOWNLOADS = 1   # per worker concurrent outbound calls
DEFAULT_TIMEOUT = 12

# Stale guards, seconds
STALE_INTRADAY_MAX_AGE = 2 * 24 * 3600
STALE_DAILY_MAX_AGE = 7 * 24 * 3600

# In memory cache: key -> (df, ts)
_PRICE_CACHE = OrderedDict()
_CACHE_LOCK = Lock()

# Rate limit state per worker
_DL_LOCK = Lock()
_LAST_DOWNLOAD_TIME = 0.0

# Concurrency guard per worker
_DL_SEMAPHORE = Semaphore(MAX_CONCURRENT_DOWNLOADS)

# Armored session
custom_session = requests.Session()
custom_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Referer": "https://finance.yahoo.com",
    "Origin": "https://finance.yahoo.com",
})

# ----------------------------
# Small utilities
# ----------------------------
def _now_ts() -> float:
    return time.time()

def _log(event: str, **kw):
    payload = {"event": event, "ts": int(_now_ts())}
    payload.update(kw)
    print(json.dumps(payload, ensure_ascii=False))

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is not None and isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

def _cache_get(key):
    now = _now_ts()
    with _CACHE_LOCK:
        if key not in _PRICE_CACHE:
            return None
        df, ts = _PRICE_CACHE[key]
        if now - ts <= CACHE_TTL:
            _PRICE_CACHE.move_to_end(key)
            return df
        # expired, keep as stale fallback but mark by returning tuple
        return ("__STALE__", df, ts)

def _cache_put(key, df: pd.DataFrame):
    now = _now_ts()
    with _CACHE_LOCK:
        _PRICE_CACHE[key] = (df, now)
        _PRICE_CACHE.move_to_end(key)
        while len(_PRICE_CACHE) > CACHE_MAX_KEYS:
            _PRICE_CACHE.popitem(last=False)

def _rate_limit_sleep():
    global _LAST_DOWNLOAD_TIME
    with _DL_LOCK:
        now = _now_ts()
        elapsed = now - _LAST_DOWNLOAD_TIME
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)
        _LAST_DOWNLOAD_TIME = _now_ts()

def _to_kolkata_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    idx = pd.to_datetime(out.index, errors="coerce")
    if getattr(idx, "tz", None) is None:
        # yfinance often returns naive timestamps already in local exchange time
        # We treat them as Asia/Kolkata for day grouping consistency
        idx = idx.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="shift_forward")
    else:
        idx = idx.tz_convert("Asia/Kolkata")
    out.index = idx
    return out

def _is_intraday_interval(interval: str) -> bool:
    s = str(interval).lower()
    return any(x in s for x in ["m", "h"]) and "d" not in s and "wk" not in s and "mo" not in s

def _stale_guard(df: pd.DataFrame, interval: str) -> bool:
    if df is None or df.empty:
        return False
    try:
        last_ts = pd.to_datetime(df.index[-1], errors="coerce")
        if pd.isna(last_ts):
            return False
        if getattr(last_ts, "tzinfo", None) is None:
            # treat as local
            last_ts = last_ts.replace(tzinfo=None)
            last_epoch = last_ts.timestamp()
        else:
            last_epoch = last_ts.timestamp()
        age = _now_ts() - last_epoch
        max_age = STALE_INTRADAY_MAX_AGE if _is_intraday_interval(interval) else STALE_DAILY_MAX_AGE
        return age <= max_age
    except Exception:
        return False

# ----------------------------
# Core: resilient downloader
# ----------------------------
def safe_download(symbol: str, period: str, interval: str, min_rows: int, retries: int = 3):
    """
    Returns: (df, meta)
    meta includes: from_cache, stale_used, attempts, error
    """
    # Normalize symbol for yfinance
    ticker_str = symbol
    if not ticker_str.endswith((".NS", ".BO")) and not ticker_str.startswith("^"):
        ticker_str = f"{ticker_str}.NS"

    cache_key = (ticker_str, period, interval)
    cached = _cache_get(cache_key)
    stale_df = None
    stale_ts = None

    if isinstance(cached, pd.DataFrame):
        _log("cache_hit", ticker=ticker_str, period=period, interval=interval, rows=int(len(cached)))
        return cached, {"from_cache": True, "stale_used": False, "attempts": 0, "error": ""}

    if isinstance(cached, tuple) and cached and cached[0] == "__STALE__":
        stale_df, stale_ts = cached[1], cached[2]

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            _rate_limit_sleep()

            # Concurrency limit around the actual outbound call
            with _DL_SEMAPHORE:
                _log("download_start", ticker=ticker_str, period=period, interval=interval, attempt=attempt)
                df = yf.download(
                    ticker_str,
                    period=period,
                    interval=interval,
                    session=custom_session,
                    progress=False,
                    threads=False,
                    timeout=DEFAULT_TIMEOUT,
                )

            df = _flatten_columns(df)
            if df is None or df.empty:
                raise ValueError("empty_df")

            # basic validation
            if "Close" not in df.columns:
                raise ValueError("missing_close")

            close_non_na = df["Close"].dropna()
            if len(close_non_na) < min_rows:
                raise ValueError(f"insufficient_rows_close:{len(close_non_na)}")

            # timezone normalization for consistent logic
            df = _to_kolkata_index(df)

            # stale guard, reject data that is too old
            if not _stale_guard(df, interval):
                raise ValueError("stale_data")

            # success
            _cache_put(cache_key, df)
            _log("download_ok", ticker=ticker_str, period=period, interval=interval, rows=int(len(df)))
            return df, {"from_cache": False, "stale_used": False, "attempts": attempt, "error": ""}

        except Exception as e:
            last_error = str(e)
            wait = (2 ** (attempt - 1)) + random.random()
            _log("download_fail", ticker=ticker_str, period=period, interval=interval, attempt=attempt, error=last_error, backoff_s=round(wait, 2))
            time.sleep(wait)

    # If live failed but we have stale, use it with explicit flag
    if isinstance(stale_df, pd.DataFrame) and not stale_df.empty:
        _log("stale_fallback_used", ticker=ticker_str, period=period, interval=interval, stale_age_s=int(_now_ts() - stale_ts), rows=int(len(stale_df)))
        return stale_df, {"from_cache": False, "stale_used": True, "attempts": retries, "error": last_error}

    return None, {"from_cache": False, "stale_used": False, "attempts": retries, "error": last_error}

# ----------------------------
# Market data master list
# ----------------------------
class MarketData:
    CACHE_FILE = "stock_master_cache.json"

    def __init__(self):
        self.stocks_by_sector = {}
        self.all_tickers = []
        self.company_map = {}
        self._load()

    def _load(self):
        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, "r") as f:
                    d = json.load(f)
                self.stocks_by_sector = d.get("sectors", {}) or {}
                self.all_tickers = d.get("all_tickers", []) or []
                self.company_map = d.get("company_map", {}) or {}
                if self.all_tickers:
                    _log("master_loaded", tickers=len(self.all_tickers), sectors=len(self.stocks_by_sector))
                    return
            except Exception:
                pass
        self._fetch()

    def _fetch(self):
        _log("master_fetch_start")
        headers = {"User-Agent": custom_session.headers.get("User-Agent", "Mozilla/5.0")}
        try:
            # Sectors
            r = requests.get("https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv", headers=headers, timeout=15)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                for ind, group in df.groupby("Industry"):
                    self.stocks_by_sector[str(ind)] = group["Symbol"].astype(str).str.upper().tolist()[:20]

            # All equities
            r = requests.get("https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv", headers=headers, timeout=15)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                self.all_tickers = df["SYMBOL"].astype(str).str.upper().unique().tolist()
                # memory constrained name map
                head = df.head(1200)
                for _, row in head.iterrows():
                    name = str(row.get("NAME OF COMPANY", "")).upper().strip()
                    sym = str(row.get("SYMBOL", "")).upper().strip()
                    if name and sym:
                        self.company_map[name] = sym

            if not self.all_tickers:
                raise ValueError("master_empty")

        except Exception as e:
            _log("master_fetch_fail", error=str(e))
            self.stocks_by_sector = {"Top Stocks": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "BHARTIARTL"]}
            self.all_tickers = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "BHARTIARTL"]
            self.company_map = {}

        try:
            with open(self.CACHE_FILE, "w") as f:
                json.dump({"sectors": self.stocks_by_sector, "all_tickers": self.all_tickers, "company_map": self.company_map}, f)
            _log("master_fetch_ok", tickers=len(self.all_tickers), sectors=len(self.stocks_by_sector))
        except Exception as e:
            _log("master_cache_write_fail", error=str(e))

    def normalize_symbol(self, s: str):
        if not s:
            return None
        q = s.strip().upper()
        if q in self.all_tickers:
            return q
        # allow already suffixed
        if q.endswith(".NS") and q[:-3] in self.all_tickers:
            return q[:-3]
        return self.company_map.get(q)

market = MarketData()

# ----------------------------
# Analyzer
# ----------------------------
class Analyzer:
    def analyze(self, symbol: str):
        # Ladder: intraday then daily
        ladder = [
            ("10d", "1h", 20),
            ("1mo", "1h", 20),
            ("6mo", "1d", 60),
        ]

        df = None
        meta = {}
        used_period = ""
        used_interval = ""

        for period, interval, min_rows in ladder:
            df, meta = safe_download(symbol, period, interval, min_rows=min_rows, retries=3)
            if df is not None and not df.empty:
                used_period, used_interval = period, interval
                break

        if df is None or df.empty:
            return None, {"error": meta.get("error", "download_failed")}

        # ensure numeric series
        try:
            df = _flatten_columns(df)
            df = _to_kolkata_index(df)

            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            high = pd.to_numeric(df["High"], errors="coerce").dropna()
            low = pd.to_numeric(df["Low"], errors="coerce").dropna()
            open_ = pd.to_numeric(df["Open"], errors="coerce").dropna()

            if len(close) < 20:
                return None, {"error": "insufficient_rows_after_clean"}

            curr = float(close.iloc[-1])

            # SMA and mean
            sma9 = float(close.rolling(9).mean().iloc[-1])
            mean20 = float(close.iloc[-20:].mean())
            std20 = float(close.iloc[-20:].std()) if float(close.iloc[-20:].std()) > 0 else 0.0

            # RSI 14
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
            if not np.isfinite(rsi):
                rsi = 50.0

            # MACD basic
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_bullish = bool(macd_line.iloc[-1] > signal_line.iloc[-1])

            # Range metrics over last 20
            low_20 = float(low.iloc[-20:].min())
            high_20 = float(high.iloc[-20:].max())
            if high_20 == low_20:
                pos_in_range = 50.0
            else:
                pos_in_range = float((curr - low_20) / (high_20 - low_20) * 100)

            pct_from_low = float((curr - low_20) / low_20 * 100) if low_20 > 0 else 0.0

            # Z score vs mean20
            zscore = float((curr - mean20) / std20) if std20 > 0 else 0.0
            pct_from_mean20 = float((curr - mean20) / mean20 * 100) if mean20 > 0 else 0.0
            pct_from_sma9 = float((curr - sma9) / sma9 * 100) if sma9 > 0 else 0.0

            # Volatility (daily return std on available frequency)
            returns = close.pct_change().dropna()
            volatility = float(returns.std() * 100) if len(returns) else 0.0

            # Daily change logic
            daily_change = 0.0
            latest_dt = df.index[-1]
            latest_date = latest_dt.date()

            if _is_intraday_interval(used_interval):
                # first candle open of latest_date in Asia/Kolkata
                day_mask = df.index.date == latest_date
                day_df = df.loc[day_mask]
                if not day_df.empty:
                    day_open = float(pd.to_numeric(day_df["Open"], errors="coerce").dropna().iloc[0])
                    if day_open > 0:
                        daily_change = float((curr - day_open) / day_open * 100)
            else:
                # daily close vs previous close
                if len(close) >= 2 and float(close.iloc[-2]) != 0:
                    daily_change = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)

            # Signal logic, avoids contradictions
            uptrend = curr > sma9
            sig = "HOLD"
            if uptrend and macd_bullish and rsi < 70:
                sig = "BUY"
            elif (not uptrend) and (not macd_bullish) and rsi > 30:
                sig = "SELL"

            # Targets
            entry = min(sma9, mean20) if sig == "BUY" else max(sma9, mean20)
            stop = low_20 * 0.98 if sig != "SELL" else high_20 * 1.02
            target = high_20 * 1.05 if sig == "BUY" else mean20 if sig == "SELL" else high_20

            # Confidence simple and interpretable
            alignment = int((uptrend == macd_bullish) * 1)
            confidence = 50 + 20 * alignment
            if rsi < 30 or rsi > 70:
                confidence += 10
            if abs(zscore) > 1:
                confidence += 10
            if volatility > 6:
                confidence -= 10
            confidence = int(max(0, min(100, confidence)))

            return {
                "signal": {
                    "signal": sig,
                    "action": f"{sig} SETUP",
                    "rec": "Trend, momentum, and mean reversion are aligned." if confidence >= 70 else "Mixed signals. Treat as low conviction.",
                    "entry": f"₹{entry:.2f}",
                    "stop": f"₹{stop:.2f}",
                    "target": f"₹{target:.2f}",
                    "confidence": confidence,
                    "days_to_target": 7,
                    "entry_explain": "Entry near trend mean (SMA9 or Mean20).",
                    "exit_explain": "Target near recent resistance or mean reversion point.",
                    "confidence_explain": "Based on trend vs SMA9, MACD alignment, RSI extremes, Z score, and volatility.",
                    "time_explain": "Heuristic estimate, not a guarantee.",
                    "trend_explain": f"Price ₹{curr:.2f} vs SMA9 ₹{sma9:.2f} ({'above' if uptrend else 'below'}).",
                    "momentum_explain": "MACD bullish." if macd_bullish else "MACD bearish.",
                    "rsi_explain": f"RSI(14): {rsi:.1f}.",
                    "position_explain": f"Position in 20 bar range: {pos_in_range:.0f}%. Pct from 20 bar low: {pct_from_low:.1f}%.",
                    "zscore_explain": f"Z score vs Mean20: {zscore:.2f}. Pct from Mean20: {pct_from_mean20:+.2f}%.",
                    "bb_explain": "Not shown in this build.",
                    "macd_text": "BULLISH" if macd_bullish else "BEARISH",
                    "data_meta": {
                        "period": used_period,
                        "interval": used_interval,
                        "from_cache": bool(meta.get("from_cache")),
                        "stale_used": bool(meta.get("stale_used")),
                    }
                },
                "details": {
                    "price": f"₹{curr:.2f}",
                    "daily": f"{daily_change:+.2f}%",
                    "rsi": f"{rsi:.1f}",
                    "zscore": f"{zscore:.2f}",
                    "pct_from_mean20": f"{pct_from_mean20:+.2f}%",
                    "pct_from_sma9": f"{pct_from_sma9:+.2f}%",
                    "mean20": f"₹{mean20:.2f}",
                    "sma9": f"₹{sma9:.2f}",
                    "high20": f"₹{high_20:.2f}",
                    "low20": f"₹{low_20:.2f}",
                    "volatility": f"{volatility:.2f}%",
                    "macd": "BULLISH" if macd_bullish else "BEARISH",
                }
            }, {"error": ""}

        except Exception as e:
            _log("analyze_logic_fail", symbol=symbol, error=str(e))
            return None, {"error": str(e)}

    def regression_analysis(self, symbol: str):
        benchmarks = [
            ("^NSEI", "Nifty 50 Index"),
            ("NIFTYBEES.NS", "NIFTYBEES ETF"),
            ("RELIANCE.NS", "RELIANCE proxy"),
        ]

        # Stock returns
        s_df, s_meta = safe_download(symbol, "1y", "1d", min_rows=120, retries=3)
        if s_df is None or s_df.empty:
            return None

        s_df = _flatten_columns(s_df)
        s_df = _to_kolkata_index(s_df)
        s_close = pd.to_numeric(s_df["Close"], errors="coerce").dropna()
        s_ret = s_close.pct_change().dropna()

        m_df = None
        bench_name = ""
        bench_warning = ""

        for b_sym, b_name in benchmarks:
            m_df, m_meta = safe_download(b_sym, "1y", "1d", min_rows=120, retries=3)
            if m_df is not None and not m_df.empty:
                bench_name = b_name
                if b_sym != "^NSEI":
                    bench_warning = f"Benchmark fallback in use: {b_name}."
                break

        if m_df is None or m_df.empty:
            return None

        m_df = _flatten_columns(m_df)
        m_df = _to_kolkata_index(m_df)
        m_close = pd.to_numeric(m_df["Close"], errors="coerce").dropna()
        m_ret = m_close.pct_change().dropna()

        rets = pd.concat([s_ret.rename("stock"), m_ret.rename("market")], axis=1, join="inner").dropna()
        if len(rets) < 60:
            return None

        X = rets["market"].to_numpy()
        y = rets["stock"].to_numpy()
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        r_squared = r_value ** 2

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X * 100, y * 100, alpha=0.35)
        x_range = np.linspace(X.min(), X.max(), 100)
        y_pred = slope * x_range + intercept
        ax.plot(x_range * 100, y_pred * 100, linewidth=2)
        ax.set_title(f"{symbol} vs {bench_name}")
        ax.set_xlabel(f"{bench_name} returns (%)")
        ax.set_ylabel(f"{symbol} returns (%)")
        ax.grid(True, alpha=0.25)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        plt.close(fig)
        buf.seek(0)

        residuals = y - (slope * X + intercept)
        residual_std = float(np.std(residuals))
        correlation = float(np.corrcoef(X, y)[0, 1])

        return {
            "beta": float(slope),
            "alpha": float(intercept),
            "r_squared": float(r_squared),
            "correlation": float(correlation),
            "p_value": float(p_value),
            "std_error": float(std_err),
            "residual_std": float(residual_std),
            "data_points": int(len(rets)),
            "market_source": bench_name,
            "benchmark_warning": bench_warning,
            "plot_url": base64.b64encode(buf.getvalue()).decode(),
        }

analyzer = Analyzer()

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    # Keep it minimal but functional
    sectors_json = json.dumps(market.stocks_by_sector)
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Analysis Pro</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ border: 1px solid #ddd; padding: 14px; border-radius: 10px; }}
    input {{ width: 100%; padding: 10px; font-size: 16px; }}
    button {{ padding: 8px 12px; margin: 4px; cursor: pointer; }}
    .pill {{ display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid #ccc; }}
    .error {{ color: #b00020; }}
    .muted {{ color: #666; }}
    pre {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h2>Stock Analysis Pro</h2>
  <div class="row">
    <div class="card">
      <h3>Technical Analysis</h3>
      <input id="sym" placeholder="RELIANCE, TCS, INFY..." />
      <div id="sug" class="muted"></div>
      <button onclick="run()">Analyze</button>
      <div id="out"></div>
    </div>

    <div class="card">
      <h3>Regression</h3>
      <input id="regsym" placeholder="RELIANCE, TCS, INFY..." />
      <button onclick="reg()">Run Regression</button>
      <div id="regout"></div>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3>Sectors</h3>
    <div id="sectors"></div>
  </div>

<script>
const sectors = {sectors_json};

function renderSectors() {{
  const el = document.getElementById("sectors");
  let html = "";
  Object.entries(sectors).slice(0, 20).forEach(([name, list]) => {{
    html += `<div><strong>${{name}}</strong><div>`;
    list.slice(0, 20).forEach(s => {{
      html += `<button onclick="pick('${{s}}')">${{s}}</button>`;
    }});
    html += `</div></div><hr/>`;
  }});
  el.innerHTML = html;
}}

function pick(s) {{
  document.getElementById("sym").value = s;
  run();
}}

let t = null;
document.getElementById("sym").addEventListener("input", (e) => {{
  clearTimeout(t);
  const q = e.target.value.trim();
  if (q.length < 2) return;
  t = setTimeout(async () => {{
    const r = await fetch(`/search?q=${{encodeURIComponent(q)}}`);
    const d = await r.json();
    document.getElementById("sug").innerHTML = d.results.map(x => `<span class="pill" onclick="pick('${{x}}')">${{x}}</span>`).join(" ");
  }}, 250);
}});

async function run() {{
  const s = document.getElementById("sym").value.trim();
  document.getElementById("out").innerHTML = "Loading...";
  const r = await fetch(`/analyze?symbol=${{encodeURIComponent(s)}}`);
  const d = await r.json();
  if (d.error) {{
    document.getElementById("out").innerHTML = `<div class="error">${{d.error}}</div>`;
    return;
  }}
  const sig = d.signal;
  const det = d.details;
  document.getElementById("out").innerHTML = `
    <div><strong>${{sig.signal}}</strong> <span class="pill">${{sig.confidence}}% confidence</span></div>
    <div class="muted">Data: ${{sig.data_meta.period}} / ${{sig.data_meta.interval}} | cache=${{sig.data_meta.from_cache}} | stale=${{sig.data_meta.stale_used}}</div>
    <pre>${{JSON.stringify({{signal:sig, details:det}}, null, 2)}}</pre>
  `;
}}

async function reg() {{
  const s = document.getElementById("regsym").value.trim();
  document.getElementById("regout").innerHTML = "Loading...";
  const r = await fetch(`/regression?symbol=${{encodeURIComponent(s)}}`);
  const d = await r.json();
  if (d.error) {{
    document.getElementById("regout").innerHTML = `<div class="error">${{d.error}}</div>`;
    return;
  }}
  const warn = d.benchmark_warning ? `<div class="error">${{d.benchmark_warning}}</div>` : "";
  document.getElementById("regout").innerHTML = `
    ${{warn}}
    <div class="muted">Benchmark: ${{d.market_source}} | points: ${{d.data_points}}</div>
    <div>Beta: ${{d.beta.toFixed(3)}} | R²: ${{(d.r_squared*100).toFixed(1)}}% | Alpha: ${{(d.alpha*100).toFixed(3)}}%</div>
    <img style="max-width:100%; margin-top:10px;" src="data:image/png;base64,${{d.plot_url}}" />
  `;
}}

renderSectors();
</script>
</body>
</html>
"""

@app.route("/search")
def search_route():
    q = request.args.get("q", "").strip().upper()
    if len(q) < 2:
        return jsonify({"results": []})
    # fast substring search on tickers
    matches = [t for t in market.all_tickers if q in t]
    matches = sorted(matches)[:15]
    return jsonify({"results": matches})

@app.route("/analyze")
def analyze_route():
    symbol = request.args.get("symbol", "").strip()
    norm = market.normalize_symbol(symbol) or symbol.strip().upper()
    # allow direct .NS
    if norm.endswith(".NS"):
        norm = norm[:-3]

    # prevent nonsense
    if not norm:
        return jsonify({"error": "Missing symbol"}), 400

    res, meta = analyzer.analyze(norm)
    if res is None:
        # This is the honest message: upstream data blocked or insufficient
        return jsonify({"error": f"Data fetch failed or throttled. Try again later. Details: {meta.get('error','')}".strip()}), 503
    return jsonify(res)

@app.route("/regression")
def regression_route():
    symbol = request.args.get("symbol", "").strip()
    norm = market.normalize_symbol(symbol) or symbol.strip().upper()
    if norm.endswith(".NS"):
        norm = norm[:-3]
    if not norm:
        return jsonify({"error": "Missing symbol"}), 400

    res = analyzer.regression_analysis(norm)
    if not res:
        return jsonify({"error": "Regression failed due to missing overlap or upstream throttling."}), 503
    return jsonify(res)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
