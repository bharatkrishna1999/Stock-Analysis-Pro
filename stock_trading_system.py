"""
Enhanced Large Cap Stocks Trading Dashboard - Flask Version (ALL NSE STOCKS)
Features:
- ALL NSE-listed stocks (500+ companies)
- Z-score with percentage deviation
- HSIC non-linear dependency analysis vs Nifty 50
- VISUAL Scatter Plots with Dependency Scores
- Clear entry/exit explanations with confidence levels
- Time-to-target predictions
- Autocomplete for Search
"""

from flask import Flask, jsonify, request, Response, stream_with_context, make_response
from flask_compress import Compress
import hashlib
import json
import os
import pickle
import random
import re
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import warnings
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import requests
import logging
import gc
from scipy.optimize import minimize as scipy_minimize
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from stock_alerts import StockAlertMonitor

# Set non-interactive backend for Render server
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("yfinance").propagate = False

app = Flask(__name__)
Compress(app)
app.config['COMPRESS_MIMETYPES'] = ['text/html', 'text/css', 'application/javascript', 'application/json']
app.config['COMPRESS_MIN_SIZE'] = 500

DIVIDEND_CACHE_TTL = timedelta(hours=6)
DIVIDEND_CACHE = {}
DIVIDEND_MAX_RESULTS = 150

DEFAULT_ANALYSIS_PERIOD = '6mo'
DEFAULT_ANALYSIS_INTERVAL = '1d'
MAX_HISTORY_POINTS = 160
REGRESSION_WAIT_TIMEOUT_SECONDS = 2.0
REGRESSION_CACHE_TTL = timedelta(hours=8)
ANALYZE_CACHE_TTL = timedelta(minutes=20)
PRICE_HISTORY_CACHE_TTL = timedelta(minutes=30)
REGIME_CACHE_TTL = timedelta(minutes=30)

YAHOO_TICKER_ALIASES = {
    "ETERNAL": ["ZOMATO"],
}

# ===== EXPANDED STOCK LIST - ALL NSE STOCKS =====
# Organized by sector for better UX, but includes 500+ stocks

STOCKS = {
    'IT Sector': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM', 'LTIM', 'COFORGE', 'MPHASIS', 'PERSISTENT',
                  'LTTS', 'SONATSOFTW', 'TATAELXSI', 'CYIENT', 'KPITTECH',
                  'INTELLECT', 'MASTEK', 'ZENSAR'],
    
    'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'BANKBARODA', 
                'PNB', 'FEDERALBNK', 'AUBANK', 'BANDHANBNK', 'IDFCFIRSTB', 'RBLBANK', 'CANBK', 
                'UNIONBANK', 'INDIANB', 'CENTRALBK', 'MAHABANK', 'JKBANK', 'KARNATBANK', 'DCBBANK'],
    
    'Financial Services': ['BAJFINANCE', 'BAJAJFINSV', 'SBILIFE', 'HDFCLIFE', 'ICICIGI', 'ICICIPRULI', 
                           'CHOLAFIN', 'PFC', 'RECLTD', 'MUTHOOTFIN', 'HDFCAMC', 'CDSL', 'CAMS', 
                           'LICHSGFIN', 'M&MFIN', 'SHRIRAMFIN', 'PNBHOUSING', 'IIFL', 'CREDITACC'],
    
    'Auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT', 'HEROMOTOCO', 'TVSMOTOR', 
             'ASHOKLEY', 'ESCORTS', 'FORCEMOT', 'MAHINDCIE', 'SONACOMS', 'TIINDIA'],
    
    'Auto Components': ['BOSCHLTD', 'MOTHERSON', 'BALKRISIND', 'MRF', 'APOLLOTYRE', 'EXIDEIND',
                        'ARE&M', 'BHARATFORG', 'CEATLTD', 'SCHAEFFLER', 'SUPRAJIT', 'ENDURANCE'],
    
    'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'LUPIN', 'BIOCON', 'AUROPHARMA',
               'TORNTPHARM', 'ALKEM', 'ZYDUSLIFE', 'IPCALAB', 'GRANULES', 'GLENMARK', 'NATCOPHARMA',
               'JBCHEPHARM', 'LAURUSLABS', 'PFIZER', 'ABBOTINDIA', 'GLAXO', 'SANOFI'],
    
    'Healthcare': ['APOLLOHOSP', 'MAXHEALTH', 'FORTIS', 'LALPATHLAB', 'METROPOLIS', 'DRREDDY',
                   'THYROCARE', 'ASTER', 'RAINBOW'],
    
    'Consumer Goods': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO', 'GODREJCP',
                       'COLPAL', 'TATACONSUM', 'EMAMILTD', 'VBL', 'RADICO', 'UBL', 'MCDOWELL-N',
                       'PGHH', 'GILLETTE', 'JYOTHYLAB', 'BAJAJCON'],
    
    'Retail': ['DMART', 'TRENT', 'TITAN', 'ABFRL', 'SHOPERSTOP', 'JUBLFOOD', 'WESTLIFE',
               'DEVYANI', 'SPENCERS', 'VMART', 'BATA', 'KALYANKJIL'],
    
    'Energy - Oil & Gas': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL', 'HINDPETRO', 'PETRONET', 
                           'OIL', 'MGL', 'IGL', 'GUJGASLTD', 'ATGL'],
    
    'Power': ['NTPC', 'POWERGRID', 'ADANIPOWER', 'TATAPOWER', 'TORNTPOWER', 'ADANIGREEN',
              'NHPC', 'SJVN', 'JSWENERGY', 'CESC', 'PFC', 'RECLTD'],
    
    'Metals & Mining': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'COALINDIA', 'VEDL', 'NMDC', 'SAIL',
                        'NATIONALUM', 'JINDALSTEL', 'HINDZINC', 'RATNAMANI', 'WELCORP',
                        'MOIL'],
    
    'Cement': ['ULTRACEMCO', 'GRASIM', 'SHREECEM', 'AMBUJACEM', 'ACC', 'DALMIACEM', 'JKCEMENT',
               'RAMCOCEM', 'HEIDELBERG', 'ORIENTCEM', 'JKLAKSHMI', 'STARCEMENT'],
    
    'Real Estate': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'BRIGADE', 'PHOENIXLTD', 
                    'SOBHA', 'LODHA', 'MAHLIFE', 'SUNTECK'],
    
    'Infrastructure': ['LT', 'ADANIENT', 'ADANIPORTS', 'SIEMENS', 'ABB', 'CUMMINSIND', 'VOLTAS',
                       'NCC', 'PNC', 'KNR', 'IRCTC', 'CONCOR', 'IRFC', 'GMRINFRA', 'AIAENG'],
    
    'Telecom': ['BHARTIARTL', 'IDEA', 'TATACOMM'],
    
    'Media': ['ZEEL', 'SUNTV', 'PVRINOX', 'SAREGAMA', 'TIPS', 'NAZARA', 'NETWORK18'],
    
    'Chemicals': ['UPL', 'PIDILITIND', 'AARTIIND', 'SRF', 'DEEPAKNTR', 'GNFC', 'CHAMBLFERT',
                  'TATACHEM', 'ALKYLAMINE', 'CLEAN', 'NOCIL', 'VINATIORGA',
                  'ATUL', 'FINEORG', 'NAVINFLUOR'],
    
    'Paints': ['ASIANPAINT', 'BERGEPAINT', 'KANSAINER'],
    
    'Textiles': ['GRASIM', 'RAYMOND', 'ARVIND', 'WELSPUNIND', 'TRIDENT', 'KPR'],
    
    'Logistics': ['CONCOR', 'VRL', 'MAHLOG', 'BLUEDART', 'TCI', 'AEGISCHEM', 'GATI'],
    
    'Aviation': ['INDIGO', 'SPICEJET'],
    
    'Hospitality': ['INDHOTEL', 'LEMONTREE', 'EIH', 'CHALET', 'ITCHOTELS'],
    
    'Construction': ['LT', 'NCC', 'PNC', 'KNR', 'ASHOKA', 'SADBHAV', 'HGINFRA'],
    
    'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO', 'GODREJCP',
             'TATACONSUM', 'EMAMILTD', 'JYOTHYLAB', 'BAJAJCON', 'VBL', 'BALRAMCHIN'],
    
    'Electronics': ['DIXON', 'AMBER', 'POLYCAB', 'HAVELLS', 'CROMPTON', 'VGUARD',
                    'KEI', 'FINOLEX'],
    
    'Conglomerate': ['RELIANCE', 'LT', 'ITC', 'ADANIENT', 'TATASTEEL', 'M&M', 'SIEMENS'],
    
    'Nifty 50': ['ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO',
                 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA',
                 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE',
                 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'ITC', 'INDUSINDBK', 'INFY',
                 'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC',
                 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SUNPHARMA', 'TCS', 'TATACONSUM',
                 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'ULTRACEMCO', 'UPL', 'WIPRO'],
    
    'Nifty Next 50': ['ACC', 'AMBUJACEM', 'BANDHANBNK', 'BERGEPAINT', 'BOSCHLTD', 'CHOLAFIN',
                      'COLPAL', 'DABUR', 'DLF', 'GODREJCP', 'GAIL', 'HAVELLS', 'HDFCAMC', 'HINDPETRO',
                      'ICICIGI', 'ICICIPRULI', 'IDEA', 'INDIGO', 'LUPIN', 'MCDOWELL-N', 'MARICO',
                      'MOTHERSON', 'MUTHOOTFIN', 'NMDC', 'NYKAA', 'OFSS', 'OIL', 'PAGEIND', 'PIDILITIND',
                      'PNB', 'PEL', 'PETRONET', 'PFIZER', 'SIEMENS', 'SRF', 'SBICARD', 'SHREECEM',
                      'TATACOMM', 'TORNTPHARM', 'TRENT', 'TVSMOTOR', 'VEDL', 'VOLTAS', 'ZEEL', 'ETERNAL'],
    
    'Others': ['ETERNAL', 'PAYTM', 'NYKAA', 'POLICYBZR', 'DELHIVERY', 'CARTRADE', 'EASEMYTRIP',
               'ROUTE', 'LATENTVIEW', 'APTUS', 'RAINBOW', 'LAXMIMACH', 'SYNGENE', 'METROPOLIS']
}

UNIVERSE_SECTOR_NAME = "All NSE"
UNIVERSE_SOURCE = "Static list"

# Persistent cache file stored next to this script so a known-good universe
# survives process restarts (on Render this persists within a single deploy).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_CACHE_FILE = os.path.join(_SCRIPT_DIR, ".nse_universe_cache.json")
CACHE_DIR = os.path.join(_SCRIPT_DIR, '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)


class HybridTTLCache:
    def __init__(self, name, ttl, max_entries=96):
        self.name = name
        self.ttl = ttl
        self.max_entries = max_entries
        self.memory = {}
        self.lock = Lock()
        self.disk_file = os.path.join(CACHE_DIR, f"{name}.pkl")
        self._load_disk()

    def _load_disk(self):
        if not os.path.exists(self.disk_file):
            return
        try:
            with open(self.disk_file, 'rb') as f:
                self.memory = pickle.load(f)
        except Exception:
            self.memory = {}

    def _persist_disk(self):
        try:
            with open(self.disk_file, 'wb') as f:
                pickle.dump(self.memory, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def _prune(self):
        now = datetime.utcnow()
        keys = [k for k, v in self.memory.items() if (now - v['timestamp']) > self.ttl]
        for k in keys:
            self.memory.pop(k, None)
        if len(self.memory) > self.max_entries:
            drop = sorted(self.memory.items(), key=lambda kv: kv[1]['timestamp'])[:len(self.memory)-self.max_entries]
            for k, _ in drop:
                self.memory.pop(k, None)

    def get(self, key):
        with self.lock:
            item = self.memory.get(key)
            if not item:
                return None
            if (datetime.utcnow() - item['timestamp']) > self.ttl:
                self.memory.pop(key, None)
                return None
            return item['value']

    def set(self, key, value):
        with self.lock:
            self.memory[key] = {'timestamp': datetime.utcnow(), 'value': value}
            self._prune()
            self._persist_disk()

    def clear(self):
        """Wipe all entries (memory + disk)."""
        with self.lock:
            self.memory = {}
            self._persist_disk()


PRICE_HISTORY_CACHE = HybridTTLCache('price_history', PRICE_HISTORY_CACHE_TTL, max_entries=180)
ANALYSIS_CACHE = HybridTTLCache('analysis', ANALYZE_CACHE_TTL, max_entries=180)
REGRESSION_CACHE = HybridTTLCache('regression', REGRESSION_CACHE_TTL, max_entries=96)
REGIME_CACHE = HybridTTLCache('regime', REGIME_CACHE_TTL, max_entries=32)
REGRESSION_JOB_CACHE = {}
REGRESSION_EXECUTOR = ThreadPoolExecutor(max_workers=1)


@app.before_request
def _begin_request_metrics():
    request._start_time = time.perf_counter()


@app.after_request
def _end_request_metrics(response):
    start = getattr(request, '_start_time', None)
    if start is not None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers['X-Endpoint-Time-Ms'] = f"{elapsed_ms:.1f}"
        app.logger.info("endpoint=%s method=%s status=%s ttfb_ms=%.1f", request.path, request.method, response.status_code, elapsed_ms)
    return response


_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _extract_symbols_from_csv(text):
    """Parse EQUITY_L.csv text and return a sorted list of symbols."""
    df = pd.read_csv(io.StringIO(text))
    symbols = (
        df.get("SYMBOL", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    return sorted({s for s in symbols if s})


def _fetch_nsearchives():
    """Source 1: New nsearchives subdomain (most reliable for CSV)."""
    resp = requests.get(
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        headers={"User-Agent": _BROWSER_UA},
        timeout=15,
    )
    resp.raise_for_status()
    return _extract_symbols_from_csv(resp.text)


def _fetch_old_archives():
    """Source 2: Legacy archives subdomain."""
    resp = requests.get(
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        headers={"User-Agent": _BROWSER_UA},
        timeout=15,
    )
    resp.raise_for_status()
    return _extract_symbols_from_csv(resp.text)


def _fetch_nse_api_indices():
    """Source 3: NSE website JSON API (needs session cookies).

    Fetches NIFTY TOTAL MARKET + NIFTY 500 and merges them for broad
    coverage (~750 stocks).  Not the full universe but a good fallback.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": _BROWSER_UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    # Establish session cookies
    session.get("https://www.nseindia.com/", timeout=10)

    symbols = set()
    for index_name in ["NIFTY%20TOTAL%20MARKET", "NIFTY%20500"]:
        try:
            resp = session.get(
                f"https://www.nseindia.com/api/equity-stockIndices?index={index_name}",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("data", []):
                sym = item.get("symbol", "").strip()
                if sym and sym != "NIFTY TOTAL MARKET" and sym != "NIFTY 500":
                    symbols.add(sym)
        except Exception:
            continue
    if not symbols:
        raise RuntimeError("NSE API returned no symbols")
    return sorted(symbols)


def _load_universe_cache():
    """Load cached universe from disk."""
    try:
        if os.path.exists(UNIVERSE_CACHE_FILE):
            with open(UNIVERSE_CACHE_FILE, "r") as f:
                data = json.load(f)
            symbols = data.get("symbols", [])
            source = data.get("source", "disk cache")
            if symbols:
                print(f"  [cache] Loaded {len(symbols)} symbols from disk (source: {source})")
                return symbols
    except Exception as e:
        print(f"  [cache] Failed to load cache: {e}")
    return []


def _save_universe_cache(symbols, source):
    """Persist a known-good universe to disk."""
    try:
        with open(UNIVERSE_CACHE_FILE, "w") as f:
            json.dump({"symbols": symbols, "source": source,
                        "saved_at": datetime.utcnow().isoformat()}, f)
        print(f"  [cache] Saved {len(symbols)} symbols to disk (source: {source})")
    except Exception as e:
        print(f"  [cache] Failed to save cache: {e}")


def _fy_dates(n_years_back=0):
    """Return (start_str, end_str) for a completed Indian financial year.

    Indian FY: April 1 → March 31.
    n_years_back=0 → most recently completed FY
                      (e.g. FY2024-25 when today is Feb 2026)
    n_years_back=1 → the FY before that, etc.
    Returns strings in 'YYYY-MM-DD' format.
    """
    today = date.today()
    # fy_end_year is the calendar year in which the FY ends (March)
    fy_end_year = today.year if today.month >= 4 else today.year - 1
    fy_end_year -= n_years_back
    fy_start = date(fy_end_year - 1, 4, 1)
    fy_end   = date(fy_end_year,     3, 31)
    return fy_start.strftime('%Y-%m-%d'), fy_end.strftime('%Y-%m-%d')


def _fy_label(n_years_back=0):
    """Return a human-readable label, e.g. 'FY2024-25'."""
    today = date.today()
    fy_end_year = today.year if today.month >= 4 else today.year - 1
    fy_end_year -= n_years_back
    return f"FY{fy_end_year - 1}-{str(fy_end_year)[2:]}"


def _compute_sustainable_dividend(dividends, current_price, max_single_pct=0.08):
    """Compute sustainable annual dividend by comparing last 2 completed Indian FYs.

    If the most recent FY's dividend is ≥ 2× the previous FY's dividend (and
    the previous FY had meaningful dividends), the inflated year is likely a
    one-time special dividend.  In that case use the *lower* FY figure so the
    yield reflects what an investor can realistically expect going forward.

    Returns (sustainable_dividend, latest_fy_div, prev_fy_div, was_capped, fy_count).
    fy_count = number of the last 2 FYs that had positive dividends (0, 1, or 2).
    """
    if dividends is None or dividends.empty:
        return 0.0, 0.0, 0.0, False, 0

    def _fy_sum(n_back):
        s_str, e_str = _fy_dates(n_back)
        s_ts, e_ts = pd.Timestamp(s_str), pd.Timestamp(e_str)
        try:
            if dividends.index.tz is not None:
                s_ts = s_ts.tz_localize('UTC')
                e_ts = e_ts.tz_localize('UTC')
        except Exception:
            pass
        fy_divs = dividends[(dividends.index >= s_ts) & (dividends.index <= e_ts)]
        # Strip corporate-action outliers (same 8% rule)
        if not fy_divs.empty:
            fy_divs = fy_divs[fy_divs <= current_price * max_single_pct]
        return float(fy_divs.sum()) if not fy_divs.empty else 0.0

    latest_fy = _fy_sum(0)
    prev_fy   = _fy_sum(1)

    # Count how many of the last 2 FYs had dividends
    fy_count = (1 if latest_fy > 0 else 0) + (1 if prev_fy > 0 else 0)

    if latest_fy <= 0:
        return 0.0, latest_fy, prev_fy, False, fy_count

    # If no meaningful previous-year data, trust the latest FY as-is
    if prev_fy <= 0:
        return latest_fy, latest_fy, prev_fy, False, fy_count

    # Flag as unsustainable if latest is ≥ 2× previous FY
    if latest_fy >= 2.0 * prev_fy:
        return prev_fy, latest_fy, prev_fy, True, fy_count

    # Also flag if previous FY was the spike (user scrolled back 2 years)
    # — use the lower of the two for a conservative estimate
    if prev_fy >= 2.0 * latest_fy:
        return latest_fy, latest_fy, prev_fy, False, fy_count

    # Both FYs are comparable — use the latest
    return latest_fy, latest_fy, prev_fy, False, fy_count


def fetch_nse_universe():
    """Fetch the full NSE equity universe with multiple redundant sources.

    Tries four sources in order, then merges with the last known-good cache
    so that the list never shrinks (stocks never silently disappear).

    Sources tried:
      1. nsearchives.nseindia.com  (CSV, new subdomain)
      2. archives.nseindia.com     (CSV, legacy subdomain)
      3. NSE website JSON API      (session-cookie gated)
      4. Disk cache                (last successful fetch)
    """
    sources = [
        ("NSE Archives (new)", _fetch_nsearchives),
        ("NSE Archives (legacy)", _fetch_old_archives),
        ("NSE Website API", _fetch_nse_api_indices),
    ]

    fetched_symbols = []
    fetched_source = None

    for name, fetcher in sources:
        try:
            print(f"  [universe] Trying {name}...")
            result = fetcher()
            if result and len(result) >= 100:  # sanity: expect at least 100 stocks
                fetched_symbols = result
                fetched_source = name
                print(f"  [universe] {name} returned {len(result)} symbols ✓")
                break
            else:
                print(f"  [universe] {name} returned only {len(result)} symbols, skipping")
        except Exception as e:
            print(f"  [universe] {name} failed: {e}")

    # Load previously cached list
    cached_symbols = _load_universe_cache()

    # Merge: take the union so the list never shrinks
    if fetched_symbols:
        merged = sorted(set(fetched_symbols) | set(cached_symbols))
        _save_universe_cache(merged, fetched_source)
        return merged, fetched_source
    elif cached_symbols:
        return cached_symbols, "Disk cache (all APIs unavailable)"
    else:
        return [], None


def add_universe_sector(stocks_dict):
    """Attach full NSE universe (API-driven with fallbacks) to stock sectors."""
    global UNIVERSE_SOURCE
    static_fallback = sorted({t for sector in stocks_dict.values() for t in sector})

    api_symbols, source = fetch_nse_universe()

    if api_symbols:
        # Also merge with the static list so hand-curated stocks are never lost
        merged = sorted(set(api_symbols) | set(static_fallback))
        UNIVERSE_SOURCE = f"{source} ({len(api_symbols)} API + {len(static_fallback)} static = {len(merged)} merged)"
        stocks_dict[UNIVERSE_SECTOR_NAME] = merged
    else:
        UNIVERSE_SOURCE = "Static list (all APIs unavailable)"
        stocks_dict[UNIVERSE_SECTOR_NAME] = static_fallback


add_universe_sector(STOCKS)

# ===== DYNAMIC SECTOR CLASSIFICATION FOR UNASSIGNED STOCKS =====
# Maps yfinance sector/industry strings to our sector names in STOCKS dict.

SECTOR_CACHE_FILE = os.path.join(_SCRIPT_DIR, ".sector_classification_cache.json")

# yfinance sector field -> our sector name
YF_SECTOR_MAP = {
    'Technology': 'IT Sector',
    'Financial Services': 'Financial Services',
    'Healthcare': 'Healthcare',
    'Consumer Cyclical': 'Consumer Goods',
    'Consumer Defensive': 'FMCG',
    'Basic Materials': 'Chemicals',
    'Energy': 'Energy - Oil & Gas',
    'Industrials': 'Infrastructure',
    'Real Estate': 'Real Estate',
    'Communication Services': 'Telecom',
    'Utilities': 'Power',
}

# yfinance industry field -> our sector name (overrides YF_SECTOR_MAP when matched)
YF_INDUSTRY_MAP = {
    # Banking
    'Banks—Regional': 'Banking',
    'Banks—Diversified': 'Banking',
    'Banks - Regional': 'Banking',
    'Banks - Diversified': 'Banking',
    # Pharma & Healthcare
    'Drug Manufacturers—General': 'Pharma',
    'Drug Manufacturers—Specialty & Generic': 'Pharma',
    'Drug Manufacturers - General': 'Pharma',
    'Drug Manufacturers - Specialty & Generic': 'Pharma',
    'Pharmaceutical Retailers': 'Pharma',
    'Biotechnology': 'Pharma',
    'Diagnostics & Research': 'Healthcare',
    'Medical Instruments & Supplies': 'Healthcare',
    'Medical Devices': 'Healthcare',
    'Medical Care Facilities': 'Healthcare',
    'Medical Distribution': 'Healthcare',
    'Health Information Services': 'Healthcare',
    'Healthcare Plans': 'Healthcare',
    # Auto
    'Auto Manufacturers': 'Auto',
    'Auto - Manufacturers': 'Auto',
    'Auto Parts': 'Auto Components',
    'Auto - Parts': 'Auto Components',
    'Auto & Truck Dealerships': 'Auto',
    'Farm & Heavy Construction Machinery': 'Auto Components',
    'Recreational Vehicles': 'Auto',
    # Metals & Mining
    'Steel': 'Metals & Mining',
    'Aluminum': 'Metals & Mining',
    'Copper': 'Metals & Mining',
    'Other Industrial Metals & Mining': 'Metals & Mining',
    'Gold': 'Metals & Mining',
    'Silver': 'Metals & Mining',
    'Coking Coal': 'Metals & Mining',
    'Thermal Coal': 'Metals & Mining',
    'Industrial Metals & Minerals': 'Metals & Mining',
    # Cement & Building Materials
    'Building Materials': 'Cement',
    'Building Products & Equipment': 'Cement',
    # Media & Entertainment
    'Entertainment': 'Media',
    'Broadcasting': 'Media',
    'Electronic Gaming & Multimedia': 'Media',
    'Publishing': 'Media',
    'Advertising Agencies': 'Media',
    'Internet Content & Information': 'Media',
    # Aviation
    'Airlines': 'Aviation',
    'Airports & Air Services': 'Aviation',
    # Hospitality
    'Lodging': 'Hospitality',
    'Resorts & Casinos': 'Hospitality',
    'Restaurants': 'Hospitality',
    'Travel Services': 'Hospitality',
    # Textiles & Apparel
    'Textile Manufacturing': 'Textiles',
    'Apparel Manufacturing': 'Textiles',
    'Apparel Retail': 'Textiles',
    'Footwear & Accessories': 'Retail',
    # Logistics
    'Integrated Freight & Logistics': 'Logistics',
    'Marine Shipping': 'Logistics',
    'Trucking': 'Logistics',
    'Railroads': 'Logistics',
    # Retail
    'Specialty Retail': 'Retail',
    'Department Stores': 'Retail',
    'Luxury Goods': 'Retail',
    'Grocery Stores': 'Retail',
    'Home Improvement Retail': 'Retail',
    'Internet Retail': 'Retail',
    'Discount Stores': 'Retail',
    # Insurance & Financial Services
    'Insurance—Life': 'Financial Services',
    'Insurance—Diversified': 'Financial Services',
    'Insurance—Property & Casualty': 'Financial Services',
    'Insurance—Reinsurance': 'Financial Services',
    'Insurance—Specialty': 'Financial Services',
    'Insurance - Life': 'Financial Services',
    'Insurance - Diversified': 'Financial Services',
    'Insurance - Property & Casualty': 'Financial Services',
    'Insurance - Reinsurance': 'Financial Services',
    'Insurance - Specialty': 'Financial Services',
    'Insurance Brokers': 'Financial Services',
    'Capital Markets': 'Financial Services',
    'Asset Management': 'Financial Services',
    'Financial Data & Stock Exchanges': 'Financial Services',
    'Credit Services': 'Financial Services',
    'Mortgage Finance': 'Financial Services',
    'Financial Conglomerates': 'Financial Services',
    'Shell Companies': 'Financial Services',
    # Chemicals
    'Specialty Chemicals': 'Chemicals',
    'Chemicals': 'Chemicals',
    'Agricultural Inputs': 'Chemicals',
    'Chemicals - Major Diversified': 'Chemicals',
    # FMCG
    'Packaged Foods': 'FMCG',
    'Beverages—Non-Alcoholic': 'FMCG',
    'Beverages—Brewers': 'FMCG',
    'Beverages—Wineries & Distilleries': 'FMCG',
    'Beverages - Non-Alcoholic': 'FMCG',
    'Beverages - Brewers': 'FMCG',
    'Beverages - Wineries & Distilleries': 'FMCG',
    'Tobacco': 'FMCG',
    'Household & Personal Products': 'FMCG',
    'Personal Products & Services': 'FMCG',
    'Confectioners': 'FMCG',
    'Meat Products': 'FMCG',
    'Farm Products': 'FMCG',
    # Consumer Goods
    'Furnishings, Fixtures & Appliances': 'Consumer Goods',
    'Home Furnishings & Fixtures': 'Consumer Goods',
    'Packaging & Containers': 'Consumer Goods',
    'Paper & Paper Products': 'Consumer Goods',
    'Rubber & Plastics': 'Consumer Goods',
    'Leisure': 'Consumer Goods',
    # Paints
    'Paints': 'Paints',
    # Power / Utilities
    'Utilities—Regulated Electric': 'Power',
    'Utilities—Renewable': 'Power',
    'Utilities—Diversified': 'Power',
    'Utilities—Independent Power Producers': 'Power',
    'Utilities—Regulated Gas': 'Power',
    'Utilities—Regulated Water': 'Power',
    'Utilities - Regulated Electric': 'Power',
    'Utilities - Renewable': 'Power',
    'Utilities - Diversified': 'Power',
    'Utilities - Independent Power Producers': 'Power',
    'Utilities - Regulated Gas': 'Power',
    'Utilities - Regulated Water': 'Power',
    'Independent Power Producers': 'Power',
    'Solar': 'Power',
    # Oil & Gas specifics
    'Oil & Gas E&P': 'Energy - Oil & Gas',
    'Oil & Gas Integrated': 'Energy - Oil & Gas',
    'Oil & Gas Refining & Marketing': 'Energy - Oil & Gas',
    'Oil & Gas Equipment & Services': 'Energy - Oil & Gas',
    'Oil & Gas Drilling': 'Energy - Oil & Gas',
    'Oil & Gas Midstream': 'Energy - Oil & Gas',
    # Electronics & Electrical
    'Consumer Electronics': 'Electronics',
    'Electronic Components': 'Electronics',
    'Electrical Equipment & Parts': 'Electronics',
    'Scientific & Technical Instruments': 'Electronics',
    'Semiconductor Equipment & Materials': 'Electronics',
    'Semiconductors': 'Electronics',
    # Construction & Infrastructure
    'Engineering & Construction': 'Construction',
    'Infrastructure Operations': 'Infrastructure',
    'Rental & Leasing Services': 'Infrastructure',
    'Waste Management': 'Infrastructure',
    'Conglomerates': 'Infrastructure',
    'Industrial Distribution': 'Infrastructure',
    'Diversified Industrials': 'Infrastructure',
    'Pollution & Treatment Controls': 'Infrastructure',
    'Security & Protection Services': 'Infrastructure',
    'Staffing & Employment Services': 'Infrastructure',
    'Consulting Services': 'Infrastructure',
    'Business Equipment & Supplies': 'Infrastructure',
    'Specialty Business Services': 'Infrastructure',
    # IT Sector
    'Software—Application': 'IT Sector',
    'Software—Infrastructure': 'IT Sector',
    'Software - Application': 'IT Sector',
    'Software - Infrastructure': 'IT Sector',
    'Information Technology Services': 'IT Sector',
    'Computer Hardware': 'IT Sector',
    'Communication Equipment': 'IT Sector',
    'Data Storage': 'IT Sector',
    'IT Consulting & Other Services': 'IT Sector',
    # Telecom
    'Telecom Services': 'Telecom',
    'Telecommunication Services': 'Telecom',
    # Real Estate
    'Real Estate—Development': 'Real Estate',
    'Real Estate—Diversified': 'Real Estate',
    'Real Estate - Development': 'Real Estate',
    'Real Estate - Diversified': 'Real Estate',
    'Real Estate Services': 'Real Estate',
    'REIT—Diversified': 'Real Estate',
    'REIT—Specialty': 'Real Estate',
    'REIT - Diversified': 'Real Estate',
    'REIT - Specialty': 'Real Estate',
    # Education (-> Others)
    'Education & Training Services': 'Others',
}


def _load_sector_cache():
    """Load cached sector classifications from disk."""
    try:
        if os.path.exists(SECTOR_CACHE_FILE):
            with open(SECTOR_CACHE_FILE, "r") as f:
                data = json.load(f)
            mapping = data.get("mapping", {})
            if mapping:
                print(f"  [sectors] Loaded {len(mapping)} cached sector classifications")
                return mapping
    except Exception as e:
        print(f"  [sectors] Failed to load sector cache: {e}")
    return {}


def _save_sector_cache(mapping):
    """Persist sector classifications to disk."""
    try:
        with open(SECTOR_CACHE_FILE, "w") as f:
            json.dump({
                "mapping": mapping,
                "saved_at": datetime.utcnow().isoformat(),
                "count": len(mapping),
            }, f)
        print(f"  [sectors] Saved {len(mapping)} sector classifications to cache")
    except Exception as e:
        print(f"  [sectors] Failed to save sector cache: {e}")


def _resolve_sector(yf_sector, yf_industry):
    """Map yfinance sector/industry to our sector name. Industry takes priority.

    Falls back to 'Others' if yfinance provides a sector but we have no mapping,
    so every stock with yfinance data gets classified.
    """
    if yf_industry and yf_industry in YF_INDUSTRY_MAP:
        return YF_INDUSTRY_MAP[yf_industry]
    if yf_sector and yf_sector in YF_SECTOR_MAP:
        return YF_SECTOR_MAP[yf_sector]
    # If yfinance returned *some* sector but we don't have a mapping, use Others
    if yf_sector:
        return 'Others'
    return None


def _fetch_sector_for_symbol(symbol):
    """Fetch sector classification for a single symbol from yfinance."""
    try:
        ns_sym = symbol if symbol.endswith('.NS') else symbol + '.NS'
        info = yf.Ticker(ns_sym).info
        yf_sector = info.get('sector', '')
        yf_industry = info.get('industry', '')
        resolved = _resolve_sector(yf_sector, yf_industry)
        return resolved, yf_sector, yf_industry
    except Exception:
        return None, '', ''


def classify_unassigned_stocks(stocks_dict, max_fetch=0, workers=16):
    """Classify stocks that are only in 'All NSE' into proper sectors.

    Uses a disk cache so that yfinance is only called for new/unknown symbols.
    Set max_fetch=0 (default) to fetch ALL uncached symbols. First run may
    take a few minutes but subsequent startups use the cache instantly.
    """
    universe_key = UNIVERSE_SECTOR_NAME
    if universe_key not in stocks_dict:
        return

    # Collect symbols already assigned to a real sector
    skip_sectors = {universe_key, 'Nifty 50', 'Nifty Next 50', 'Conglomerate', 'Others'}
    already_assigned = set()
    for sector_name, tickers in stocks_dict.items():
        if sector_name not in skip_sectors:
            already_assigned.update(tickers)

    # Symbols that need classification
    universe_symbols = set(stocks_dict.get(universe_key, []))
    unassigned = sorted(universe_symbols - already_assigned)

    if not unassigned:
        print("  [sectors] All stocks already have sector assignments")
        return

    print(f"  [sectors] {len(unassigned)} stocks need sector classification")

    # Load cache
    cache = _load_sector_cache()

    # Split into cached vs needs-fetch
    # Also re-fetch stocks cached as "Others" — expanded mappings may classify them now
    to_fetch = []
    for sym in unassigned:
        if sym not in cache:
            to_fetch.append(sym)
        elif cache[sym] in ('Others', ''):
            to_fetch.append(sym)
            del cache[sym]  # remove so they get re-fetched and re-applied

    # Apply cached classifications first
    assigned_count = 0
    for sym in unassigned:
        if sym in cache:
            sector = cache[sym]
            if not sector:
                sector = 'Others'           # re-map old empty cache entries
                cache[sym] = sector
            if sector in stocks_dict:
                stocks_dict[sector].append(sym)
            else:
                stocks_dict[sector] = [sym]
            assigned_count += 1

    if assigned_count:
        print(f"  [sectors] Applied {assigned_count} cached sector assignments")

    # Fetch new classifications (no cap by default — all uncached get fetched)
    fetch_batch = to_fetch[:max_fetch] if max_fetch > 0 else to_fetch
    if fetch_batch:
        print(f"  [sectors] Fetching sectors for {len(fetch_batch)} new symbols from yfinance...")

        def _fetch_one(sym):
            resolved, yf_sec, yf_ind = _fetch_sector_for_symbol(sym)
            return sym, resolved, yf_sec, yf_ind

        new_assigned = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_fetch_one, fetch_batch))

        for sym, resolved, yf_sec, yf_ind in results:
            # If yfinance returned nothing at all, assign to Others so no stock is left orphaned
            if not resolved:
                resolved = 'Others'
            cache[sym] = resolved
            if resolved in stocks_dict:
                stocks_dict[resolved].append(sym)
            else:
                stocks_dict[resolved] = [sym]
            new_assigned += 1

        print(f"  [sectors] Newly classified {new_assigned}/{len(fetch_batch)} stocks")

        _save_sector_cache(cache)


# Run classification before deduplication
classify_unassigned_stocks(STOCKS)

# Enhanced company name mapping with MANY more variations
COMPANY_TO_TICKER = {
    # IT Sector
    'VEDANTA': 'VEDL', 'TATA CONSULTANCY': 'TCS', 'TATA CONSULTANCY SERVICES': 'TCS', 'INFOSYS': 'INFY',
    'HCL TECH': 'HCLTECH', 'HCL TECHNOLOGIES': 'HCLTECH', 'TECH MAHINDRA': 'TECHM', 'L&T INFOTECH': 'LTIM',
    'LTI': 'LTIM', 'MINDTREE': 'LTIM', 'L&TTS': 'LTTS', 'COFORGE': 'COFORGE', 'MPHASIS': 'MPHASIS',
    'PERSISTENT': 'PERSISTENT', 'TATA ELXSI': 'TATAELXSI', 'CYIENT': 'CYIENT',
    
    # Banking
    'HDFC BANK': 'HDFCBANK', 'HDFC': 'HDFCBANK', 'ICICI BANK': 'ICICIBANK', 'ICICI': 'ICICIBANK',
    'STATE BANK': 'SBIN', 'STATE BANK OF INDIA': 'SBIN', 'SBI': 'SBIN', 'KOTAK': 'KOTAKBANK',
    'KOTAK MAHINDRA': 'KOTAKBANK', 'AXIS BANK': 'AXISBANK', 'AXIS': 'AXISBANK',
    'INDUSIND': 'INDUSINDBK', 'INDUSIND BANK': 'INDUSINDBK', 'BANK OF BARODA': 'BANKBARODA',
    'PUNJAB NATIONAL BANK': 'PNB', 'PUNJAB NATIONAL': 'PNB', 'FEDERAL BANK': 'FEDERALBNK',
    'FEDERAL': 'FEDERALBNK', 'AU BANK': 'AUBANK', 'AU SMALL FINANCE': 'AUBANK',
    'BANDHAN BANK': 'BANDHANBNK', 'BANDHAN': 'BANDHANBNK', 'IDFC FIRST BANK': 'IDFCFIRSTB',
    'IDFC': 'IDFCFIRSTB', 'RBL BANK': 'RBLBANK', 'RBL': 'RBLBANK',
    
    # Financial Services
    'BAJAJ FINANCE': 'BAJFINANCE', 'BAJAJ FIN': 'BAJFINANCE', 'BAJAJ FINSERV': 'BAJAJFINSV',
    'SBI LIFE': 'SBILIFE', 'HDFC LIFE': 'HDFCLIFE', 'ICICI LOMBARD': 'ICICIGI',
    'ICICI PRUDENTIAL': 'ICICIPRULI', 'CHOLAMANDALAM': 'CHOLAFIN', 'CHOLA': 'CHOLAFIN',
    'PFC': 'PFC', 'POWER FINANCE': 'PFC', 'REC': 'RECLTD', 'MUTHOOT': 'MUTHOOTFIN',
    'MUTHOOT FINANCE': 'MUTHOOTFIN', 'HDFC AMC': 'HDFCAMC', 'CDSL': 'CDSL', 'CAMS': 'CAMS',
    'SHRIRAM': 'SHRIRAMFIN', 'SHRIRAM FINANCE': 'SHRIRAMFIN',
    
    # Auto
    'MARUTI': 'MARUTI', 'MARUTI SUZUKI': 'MARUTI', 'TATA MOTORS': 'TATAMOTORS', 'TATA MOTOR': 'TATAMOTORS',
    'MAHINDRA': 'M&M', 'M AND M': 'M&M', 'BAJAJ AUTO': 'BAJAJ-AUTO', 'EICHER': 'EICHERMOT',
    'EICHER MOTORS': 'EICHERMOT', 'HERO MOTOCORP': 'HEROMOTOCO', 'HERO': 'HEROMOTOCO',
    'TVS': 'TVSMOTOR', 'TVS MOTOR': 'TVSMOTOR', 'ASHOK LEYLAND': 'ASHOKLEY', 'ASHOK': 'ASHOKLEY',
    
    # Auto Components
    'BOSCH': 'BOSCHLTD', 'MRF': 'MRF', 'APOLLO TYRES': 'APOLLOTYRE', 'APOLLO': 'APOLLOTYRE',
    'EXIDE': 'EXIDEIND', 'EXIDE INDUSTRIES': 'EXIDEIND', 'AMARA RAJA': 'ARE&M',
    'AMARAJABAT': 'ARE&M', 'AMARA RAJA ENERGY': 'ARE&M',
    
    # Pharma
    'SUN PHARMA': 'SUNPHARMA', 'SUN PHARMACEUTICAL': 'SUNPHARMA', 'DR REDDY': 'DRREDDY',
    'DR REDDYS': 'DRREDDY', 'CIPLA': 'CIPLA', 'DIVIS': 'DIVISLAB', 'DIVIS LAB': 'DIVISLAB',
    'LUPIN': 'LUPIN', 'BIOCON': 'BIOCON', 'AUROBINDO': 'AUROPHARMA', 'TORRENT PHARMA': 'TORNTPHARM',
    'ALKEM': 'ALKEM', 'CADILA': 'ZYDUSLIFE', 'CADILAHC': 'ZYDUSLIFE', 'ZYDUS': 'ZYDUSLIFE', 'IPCA': 'IPCALAB',
    'GRANULES': 'GRANULES', 'GLENMARK': 'GLENMARK', 'NATCO': 'NATCOPHARMA',
    
    # Healthcare
    'APOLLO HOSPITAL': 'APOLLOHOSP', 'APOLLO HOSPITALS': 'APOLLOHOSP', 'MAX HEALTH': 'MAXHEALTH',
    'MAX HEALTHCARE': 'MAXHEALTH', 'FORTIS': 'FORTIS', 'LALPATHLAB': 'LALPATHLAB',
    'DR LALPATHLAB': 'LALPATHLAB', 'METROPOLIS': 'METROPOLIS',
    
    # Consumer Goods
    'HUL': 'HINDUNILVR', 'HINDUSTAN UNILEVER': 'HINDUNILVR', 'ITC': 'ITC', 'NESTLE': 'NESTLEIND',
    'BRITANNIA': 'BRITANNIA', 'DABUR': 'DABUR', 'MARICO': 'MARICO', 'GODREJ': 'GODREJCP',
    'GODREJ CONSUMER': 'GODREJCP', 'COLGATE': 'COLPAL', 'TATA CONSUMER': 'TATACONSUM',
    'EMAMI': 'EMAMILTD', 'VARUN': 'VBL', 'VARUN BEVERAGES': 'VBL',
    
    # Retail
    'DMART': 'DMART', 'AVENUE SUPERMARTS': 'DMART', 'TRENT': 'TRENT', 'TITAN': 'TITAN',
    'ADITYA BIRLA': 'ABFRL', 'SHOPPERS STOP': 'SHOPERSTOP', 'JUBILANT': 'JUBLFOOD',
    'JUBILANT FOODWORKS': 'JUBLFOOD',
    
    # Energy
    'RELIANCE': 'RELIANCE', 'RIL': 'RELIANCE', 'ONGC': 'ONGC', 'OIL AND NATURAL GAS': 'ONGC',
    'BPCL': 'BPCL', 'BHARAT PETROLEUM': 'BPCL', 'IOC': 'IOC', 'INDIAN OIL': 'IOC',
    'GAIL': 'GAIL', 'HPCL': 'HINDPETRO', 'HINDUSTAN PETROLEUM': 'HINDPETRO',
    'PETRONET': 'PETRONET', 'PETRONET LNG': 'PETRONET', 'MGL': 'MGL', 'MAHANAGAR GAS': 'MGL',
    'IGL': 'IGL', 'INDRAPRASTHA GAS': 'IGL',
    
    # Power
    'NTPC': 'NTPC', 'POWER GRID': 'POWERGRID', 'POWERGRID': 'POWERGRID', 'ADANI POWER': 'ADANIPOWER',
    'TATA POWER': 'TATAPOWER', 'TORRENT POWER': 'TORNTPOWER', 'ADANI GREEN': 'ADANIGREEN',
    
    # Metals
    'TATA STEEL': 'TATASTEEL', 'HINDALCO': 'HINDALCO', 'JSW': 'JSWSTEEL', 'JSW STEEL': 'JSWSTEEL',
    'COAL INDIA': 'COALINDIA', 'VEDL': 'VEDL', 'NMDC': 'NMDC', 'SAIL': 'SAIL',
    'NATIONAL ALUMINIUM': 'NATIONALUM', 'NALCO': 'NATIONALUM', 'JINDAL STEEL': 'JINDALSTEL',
    'HINDZINC': 'HINDZINC', 'HINDUSTAN ZINC': 'HINDZINC',
    
    # Cement
    'ULTRATECH': 'ULTRACEMCO', 'ULTRATECH CEMENT': 'ULTRACEMCO', 'GRASIM': 'GRASIM',
    'SHREE CEMENT': 'SHREECEM', 'AMBUJA': 'AMBUJACEM', 'AMBUJA CEMENT': 'AMBUJACEM',
    'ACC': 'ACC', 'DALMIA': 'DALMIACEM', 'DALMIA BHARAT': 'DALMIACEM', 'JK CEMENT': 'JKCEMENT',
    
    # Real Estate
    'DLF': 'DLF', 'GODREJ PROPERTIES': 'GODREJPROP', 'GODREJ PROP': 'GODREJPROP',
    'OBEROI': 'OBEROIRLTY', 'OBEROI REALTY': 'OBEROIRLTY', 'PRESTIGE': 'PRESTIGE',
    'BRIGADE': 'BRIGADE', 'PHOENIX': 'PHOENIXLTD',
    
    # Infrastructure
    'LT': 'LT', 'L&T': 'LT', 'LARSEN': 'LT', 'LARSEN AND TOUBRO': 'LT', 'ADANI': 'ADANIENT',
    'ADANI ENTERPRISES': 'ADANIENT', 'ADANI PORTS': 'ADANIPORTS', 'SIEMENS': 'SIEMENS',
    'ABB': 'ABB', 'CUMMINS': 'CUMMINSIND', 'VOLTAS': 'VOLTAS', 'IRCTC': 'IRCTC',
    
    # Telecom
    'BHARTI': 'BHARTIARTL', 'BHARTI AIRTEL': 'BHARTIARTL', 'AIRTEL': 'BHARTIARTL',
    'IDEA': 'IDEA', 'VODAFONE': 'IDEA', 'VI': 'IDEA',
    
    # Media
    'ZEE': 'ZEEL', 'ZEE ENTERTAINMENT': 'ZEEL', 'SUN TV': 'SUNTV', 'PVR': 'PVRINOX',
    'PVR INOX': 'PVRINOX',
    
    # Others
    'ZOMATO': 'ETERNAL', 'ETERNAL': 'ETERNAL', 'PAYTM': 'PAYTM', 'NYKAA': 'NYKAA', 'POLICYBAZAAR': 'POLICYBZR',
    'DELHIVERY': 'DELHIVERY', 'DIXON': 'DIXON', 'POLYCAB': 'POLYCAB', 'HAVELLS': 'HAVELLS',
    'AARTI': 'AARTIIND', 'AARTI INDUSTRIES': 'AARTIIND',
}

# Build reverse mapping: ticker -> best company name (longest/most descriptive)
TICKER_TO_NAME = {}
for _name, _ticker in COMPANY_TO_TICKER.items():
    _title = _name.title()
    if _ticker not in TICKER_TO_NAME or len(_title) > len(TICKER_TO_NAME[_ticker]):
        TICKER_TO_NAME[_ticker] = _title

def deduplicate_stocks(stocks_dict, universe_sector=UNIVERSE_SECTOR_NAME):
    """
    Remove duplicate stock entries across sectors.

    Strategy:
    - Each stock is assigned to exactly ONE primary sector (the first non-index
      sector it appears in).
    - Index/collection sectors ('Nifty 50', 'Nifty Next 50', 'Conglomerate',
      'Others') keep only stocks not already placed elsewhere.
    - The universe sector keeps the full list for "all stocks" scans.
    - Within each sector list, duplicates are removed while preserving order.
    """
    index_sectors = {'Nifty 50', 'Nifty Next 50', 'Conglomerate', 'Others'}

    seen_globally = set()
    cleaned = {}

    if universe_sector in stocks_dict:
        cleaned[universe_sector] = sorted(set(stocks_dict[universe_sector]))

    # Pass 1: Process non-index sectors first (primary assignment)
    for sector, tickers in stocks_dict.items():
        if sector in index_sectors or sector == universe_sector:
            continue
        unique_in_sector = []
        seen_in_sector = set()
        for t in tickers:
            if t not in seen_globally and t not in seen_in_sector:
                unique_in_sector.append(t)
                seen_in_sector.add(t)
                seen_globally.add(t)
        if unique_in_sector:
            cleaned[sector] = unique_in_sector

    # Pass 2: Process index/collection sectors (keep only unassigned stocks)
    for sector in index_sectors:
        if sector not in stocks_dict:
            continue
        tickers = stocks_dict[sector]
        unique_in_sector = []
        seen_in_sector = set()
        for t in tickers:
            if t not in seen_globally and t not in seen_in_sector:
                unique_in_sector.append(t)
                seen_in_sector.add(t)
                seen_globally.add(t)
        if unique_in_sector:
            cleaned[sector] = unique_in_sector

    return cleaned

# Preserve original Nifty 50 list before deduplication removes them
NIFTY_50_STOCKS = list(STOCKS.get('Nifty 50', []))

# Deduplicate and build ticker set
STOCKS = deduplicate_stocks(STOCKS)
ALL_VALID_TICKERS = set()
for sector_stocks in STOCKS.values():
    ALL_VALID_TICKERS.update(sector_stocks)

print(
    f"✅ Loaded {len(ALL_VALID_TICKERS)} unique stocks across {len(STOCKS)} sectors "
    f"(duplicates removed). Universe source: {UNIVERSE_SOURCE}"
)

# ===== TICKER-TO-SECTOR REVERSE MAPPING =====
TICKER_TO_SECTOR = {}
_SKIP_SECTORS_FOR_MAP = {'All NSE', 'Nifty 50', 'Nifty Next 50', 'Conglomerate'}
for _sector_name, _sector_tickers in STOCKS.items():
    if _sector_name in _SKIP_SECTORS_FOR_MAP:
        continue
    for _t in _sector_tickers:
        if _t not in TICKER_TO_SECTOR:
            TICKER_TO_SECTOR[_t] = _sector_name

# Clear stale analysis & regime disk caches so old "All NSE" entries don't persist
ANALYSIS_CACHE.clear()
REGIME_CACHE.clear()
print("  [cache] Cleared analysis + regime caches (sector mapping updated)")

# ===== SECTOR INDEX MAP (Yahoo Finance tickers for NSE sectoral indices) =====
SECTOR_INDEX_MAP = {
    'IT Sector': '^CNXIT',
    'Banking': '^NSEBANK',
    'Financial Services': '^CNXFIN',
    'Auto': '^CNXAUTO',
    'Auto Components': '^CNXAUTO',
    'Pharma': '^CNXPHARMA',
    'Healthcare': '^CNXPHARMA',
    'Consumer Goods': '^CNXFMCG',
    'FMCG': '^CNXFMCG',
    'Retail': '^CNXFMCG',
    'Energy - Oil & Gas': '^CNXENERGY',
    'Power': '^CNXENERGY',
    'Metals & Mining': '^CNXMETAL',
    'Cement': '^CNXINFRA',
    'Real Estate': '^CNXREALTY',
    'Infrastructure': '^CNXINFRA',
    'Construction': '^CNXINFRA',
    'Media': '^CNXMEDIA',
    'Telecom': '^CNXMEDIA',
    'Electronics': '^CNXIT',
    'Paints': '^CNXFMCG',
    'Textiles': '^CNXFMCG',
    'Chemicals': '^CNXPHARMA',
    'Logistics': '^CNXINFRA',
    'Aviation': '^CNXINFRA',
    'Hospitality': '^CNXFMCG',
}

# ===== REST OF CODE REMAINS IDENTICAL =====

class Analyzer:
    @staticmethod
    def normalize_symbol(symbol):
        if not symbol:
            return None, ""
        original = symbol.strip()
        symbol_upper = original.upper()
        if symbol_upper in ALL_VALID_TICKERS:
            return symbol_upper, original
        if symbol_upper in COMPANY_TO_TICKER:
            return COMPANY_TO_TICKER[symbol_upper], original
        matches = [ticker for ticker in ALL_VALID_TICKERS if symbol_upper in ticker]
        if len(matches) == 1:
            return matches[0], original
        reverse_matches = [ticker for ticker in ALL_VALID_TICKERS if ticker in symbol_upper]
        if len(reverse_matches) == 1:
            return reverse_matches[0], original
        for company_name, ticker in COMPANY_TO_TICKER.items():
            if symbol_upper in company_name or company_name in symbol_upper:
                return ticker, original
        return None, original

    def get_data(self, symbol, period=DEFAULT_ANALYSIS_PERIOD, interval=DEFAULT_ANALYSIS_INTERVAL):
        cache_key = f"{symbol}:{period}:{interval}"
        cached = PRICE_HISTORY_CACHE.get(cache_key)
        if cached is not None and not cached.empty:
            return cached.copy(deep=False)

        def _download(ticker_symbol, period_value, interval_value):
            try:
                return yf.download(
                    ticker_symbol,
                    period=period_value,
                    interval=interval_value,
                    progress=False,
                    threads=False,
                    timeout=8,
                )
            except Exception:
                return None

        def _history_fallback(ticker_symbol):
            try:
                return yf.Ticker(ticker_symbol).history(period="1y", interval="1d", timeout=8)
            except Exception:
                return None

        def _has_enough_data(df, minimum_rows):
            if df is None or df.empty:
                return False
            close_series = df.get("Close")
            if close_series is None:
                return False
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
            return len(close_series.dropna()) >= minimum_rows

        def _downsample(df):
            if df is None or df.empty:
                return df
            if len(df) <= MAX_HISTORY_POINTS:
                return df
            step = max(1, len(df) // MAX_HISTORY_POINTS)
            return df.iloc[::step].tail(MAX_HISTORY_POINTS)

        try:
            base_symbols = [symbol] + YAHOO_TICKER_ALIASES.get(symbol, [])
            tickers = [f"{base}.NS" for base in base_symbols] + [f"{base}.BO" for base in base_symbols]
            attempts = [
                (period, interval, 8),
                ("1y", "1d", 8),
                ("6mo", "1d", 6),
                ("3mo", "1d", 5),
                ("1mo", "1d", 5),
                ("5d", "1h", 5),
            ]
            for ticker in tickers:
                for try_period, try_interval, min_rows in attempts:
                    data = _download(ticker, try_period, try_interval)
                    if _has_enough_data(data, min_rows):
                        data = _downsample(data)
                        PRICE_HISTORY_CACHE.set(cache_key, data)
                        return data.copy(deep=False)
                fallback = _history_fallback(ticker)
                if _has_enough_data(fallback, 5):
                    fallback = _downsample(fallback)
                    PRICE_HISTORY_CACHE.set(cache_key, fallback)
                    return fallback.copy(deep=False)
            return None
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None


    def calc_indicators(self, data):
        if data is None or len(data) < 5:
            return None
        try:
            close = data['Close'].dropna()
            high = data['High'].dropna()
            low = data['Low'].dropna()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if isinstance(high, pd.DataFrame):
                high = high.iloc[:, 0]
            if isinstance(low, pd.DataFrame):
                low = low.iloc[:, 0]
            if len(close) < 5:
                return None
            rsi_window = min(14, len(close))
            sma9_window = min(9, len(close))
            sma5_window = min(5, len(close))
            curr = float(close.iloc[-1])
            sma9 = float(close.rolling(sma9_window).mean().iloc[-1])
            sma5 = float(close.rolling(sma5_window).mean().iloc[-1])
            prev_close = float(close.iloc[-2] if len(close) > 1 else curr)
            daily_ret = ((curr - prev_close) / prev_close) * 100 if prev_close > 0 else 0
            hourly_ret = ((curr - prev_close) / prev_close) * 100 if prev_close > 0 else 0
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(rsi_window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(rsi_window).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi = float(rsi_series.iloc[-1])
            if pd.isna(rsi) or np.isinf(rsi):
                rsi = 50.0
            ema5 = close.ewm(5).mean()
            ema13 = close.ewm(13).mean()
            macd = ema5 - ema13
            signal = macd.ewm(5).mean()
            macd_hist = macd - signal
            macd_val = float(macd_hist.iloc[-1])
            macd_bullish = macd_val > 0 if not pd.isna(macd_val) else True
            h = float(high.iloc[-18:].max() if len(high) > 18 else high.iloc[-1])
            l = float(low.iloc[-18:].min() if len(low) > 18 else low.iloc[-1])
            pct_from_low = ((curr - l) / l) * 100 if l > 0 else 50
            lookback = 20 if len(close) >= 20 else len(close)
            mean_price = float(close.iloc[-lookback:].mean())
            std_price = float(close.iloc[-lookback:].std())
            if std_price > 0:
                zscore = (curr - mean_price) / std_price
                pct_deviation = ((curr - mean_price) / mean_price) * 100
            else:
                zscore = 0.0
                pct_deviation = 0.0
            bb_upper = mean_price + (2 * std_price)
            bb_lower = mean_price - (2 * std_price)
            bb_position = ((curr - bb_lower) / (bb_upper - bb_lower)) * 100 if (bb_upper - bb_lower) > 0 else 50
            returns = close.pct_change().dropna()
            volatility = float(returns.std() * 100)
            sma20_window = min(20, len(close))
            sma50_window = min(50, len(close))
            sma20 = float(close.rolling(sma20_window).mean().iloc[-1])
            sma50 = float(close.rolling(sma50_window).mean().iloc[-1])

            # ── Williams %R ──
            wr_period = min(14, len(close))
            highest_high = high.rolling(wr_period).max()
            lowest_low = low.rolling(wr_period).min()
            wr_denom = highest_high - lowest_low
            williams_r_series = pd.Series(np.where(wr_denom > 0, (highest_high - close) / wr_denom * -100, -50.0), index=close.index)
            williams_r = float(williams_r_series.iloc[-1])
            if pd.isna(williams_r):
                williams_r = -50.0
            # Signal: check for cross-back from extremes
            williams_r_prev = float(williams_r_series.iloc[-2]) if len(williams_r_series) > 1 else williams_r
            if williams_r_prev < -80 and williams_r > -80:
                williams_r_signal = 'POTENTIAL BOTTOM'
            elif williams_r_prev > -20 and williams_r < -20:
                williams_r_signal = 'POTENTIAL TOP'
            else:
                williams_r_signal = 'NEUTRAL'

            # ── Bollinger %B ──
            if (bb_upper - bb_lower) > 0:
                percent_b = (curr - bb_lower) / (bb_upper - bb_lower)
            else:
                percent_b = 0.5
            if percent_b < 0:
                percent_b_signal = 'OVERSOLD'
            elif percent_b > 1:
                percent_b_signal = 'OVERBOUGHT'
            elif percent_b < 0.5:
                percent_b_signal = 'LOWER HALF'
            else:
                percent_b_signal = 'UPPER HALF'

            # ── RSI Divergence Detection ──
            rsi_divergence = 'NONE'
            rsi_divergence_detail = ''
            try:
                lookback_div = min(20, len(close) - 1)
                if lookback_div >= 5:
                    price_window = close.iloc[-lookback_div:].values.astype(float)
                    rsi_window_vals = rsi_series.iloc[-lookback_div:].values.astype(float)
                    # Find local minima and maxima using simple comparison
                    price_lows = []
                    price_highs = []
                    for idx in range(1, len(price_window) - 1):
                        if price_window[idx] < price_window[idx - 1] and price_window[idx] < price_window[idx + 1]:
                            price_lows.append(idx)
                        if price_window[idx] > price_window[idx - 1] and price_window[idx] > price_window[idx + 1]:
                            price_highs.append(idx)
                    # Bullish divergence: price lower low + RSI higher low
                    if len(price_lows) >= 2:
                        i1, i2 = price_lows[-2], price_lows[-1]
                        if price_window[i2] < price_window[i1] and rsi_window_vals[i2] > rsi_window_vals[i1]:
                            rsi_divergence = 'BULLISH DIVERGENCE'
                            rsi_divergence_detail = f'Price made lower low, RSI made higher low over last {lookback_div} candles. Selling momentum is exhausting.'
                    # Bearish divergence: price higher high + RSI lower high
                    if rsi_divergence == 'NONE' and len(price_highs) >= 2:
                        i1, i2 = price_highs[-2], price_highs[-1]
                        if price_window[i2] > price_window[i1] and rsi_window_vals[i2] < rsi_window_vals[i1]:
                            rsi_divergence = 'BEARISH DIVERGENCE'
                            rsi_divergence_detail = f'Price made higher high, RSI made lower high over last {lookback_div} candles. Buying momentum is fading.'
            except Exception:
                pass
            if not rsi_divergence_detail:
                rsi_divergence_detail = 'No divergence detected in recent price action.'

            # ── Volume exhaustion (for confidence score) ──
            volume_exhaustion = False
            try:
                vol = data.get('Volume')
                if vol is not None:
                    if isinstance(vol, pd.DataFrame):
                        vol = vol.iloc[:, 0]
                    vol = vol.dropna()
                    if len(vol) >= 20:
                        vol_sma20 = float(vol.iloc[-20:].mean())
                        curr_vol = float(vol.iloc[-1])
                        volume_exhaustion = curr_vol < vol_sma20
            except Exception:
                pass

            # ── Local Minima/Maxima Confidence Score (out of 5) ──
            bottom_points = 0
            top_points = 0
            confidence_checks = {}
            # 1. RSI extreme
            if rsi < 30:
                bottom_points += 1
                confidence_checks['rsi'] = {'met': True, 'label': f'RSI Oversold ({rsi:.1f})'}
            elif rsi > 70:
                top_points += 1
                confidence_checks['rsi'] = {'met': True, 'label': f'RSI Overbought ({rsi:.1f})'}
            else:
                confidence_checks['rsi'] = {'met': False, 'label': f'RSI Neutral ({rsi:.1f})'}
            # 2. Williams %R reversing from extreme
            if williams_r_signal == 'POTENTIAL BOTTOM':
                bottom_points += 1
                confidence_checks['williams'] = {'met': True, 'label': 'Williams %R reversing from oversold'}
            elif williams_r_signal == 'POTENTIAL TOP':
                top_points += 1
                confidence_checks['williams'] = {'met': True, 'label': 'Williams %R reversing from overbought'}
            else:
                confidence_checks['williams'] = {'met': False, 'label': 'Williams %R not at extreme'}
            # 3. Bollinger %B outside 0-1
            if percent_b < 0:
                bottom_points += 1
                confidence_checks['bb'] = {'met': True, 'label': 'Price below Bollinger Lower Band'}
            elif percent_b > 1:
                top_points += 1
                confidence_checks['bb'] = {'met': True, 'label': 'Price above Bollinger Upper Band'}
            else:
                confidence_checks['bb'] = {'met': False, 'label': 'Price within Bollinger Bands'}
            # 4. Volume exhaustion
            if volume_exhaustion:
                bottom_points += 1
                top_points += 1
                confidence_checks['volume'] = {'met': True, 'label': 'Volume declining (exhaustion)'}
            else:
                confidence_checks['volume'] = {'met': False, 'label': 'Volume not showing exhaustion'}
            # 5. RSI Divergence
            if rsi_divergence == 'BULLISH DIVERGENCE':
                bottom_points += 1
                confidence_checks['divergence'] = {'met': True, 'label': 'Bullish RSI divergence detected'}
            elif rsi_divergence == 'BEARISH DIVERGENCE':
                top_points += 1
                confidence_checks['divergence'] = {'met': True, 'label': 'Bearish RSI divergence detected'}
            else:
                confidence_checks['divergence'] = {'met': False, 'label': 'No RSI divergence confirmed'}

            if bottom_points >= 3 and bottom_points >= top_points:
                minmax_type = 'BOTTOM'
                minmax_score = bottom_points
            elif top_points >= 3 and top_points > bottom_points:
                minmax_type = 'TOP'
                minmax_score = top_points
            else:
                minmax_type = 'MIXED'
                minmax_score = max(bottom_points, top_points)

            result = {
                'price': curr, 'sma9': sma9, 'sma5': sma5, 'sma20': sma20, 'sma50': sma50,
                'daily': daily_ret, 'hourly': hourly_ret,
                'rsi': rsi, 'macd_bullish': bool(macd_bullish), 'high': h, 'low': l,
                'pct_from_low': pct_from_low, 'zscore': zscore, 'pct_deviation': pct_deviation,
                'mean_price': mean_price, 'std_price': std_price, 'bb_upper': bb_upper,
                'bb_lower': bb_lower, 'bb_position': bb_position, 'volatility': volatility,
                'williams_r': williams_r, 'williams_r_signal': williams_r_signal,
                'percent_b': round(percent_b, 4), 'percent_b_signal': percent_b_signal,
                'rsi_divergence': rsi_divergence, 'rsi_divergence_detail': rsi_divergence_detail,
                'volume_exhaustion': volume_exhaustion,
                'minmax_type': minmax_type, 'minmax_score': minmax_score,
                'confidence_checks': confidence_checks,
            }
            return result
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return None

    def calculate_confidence(self, ind):
        confidence_score = 50
        uptrend = ind['price'] > ind['sma9']
        if uptrend and ind['macd_bullish']:
            confidence_score += 15
        elif not uptrend and not ind['macd_bullish']:
            confidence_score += 15
        if ind['rsi'] > 70 or ind['rsi'] < 30:
            confidence_score += 10
        if abs(ind['zscore']) > 2:
            confidence_score += 15
        elif abs(ind['zscore']) > 1:
            confidence_score += 5
        if ind['volatility'] < 2:
            confidence_score += 10
        elif ind['volatility'] > 5:
            confidence_score -= 10
        if ind['pct_from_low'] < 20 or ind['pct_from_low'] > 80:
            confidence_score += 5
        return min(max(confidence_score, 0), 100)

    def estimate_days_to_target(self, current, target, ind):
        try:
            distance = abs(target - current)
            pct_move_needed = (distance / current) * 100
            daily_vol = ind['volatility']
            if daily_vol > 0:
                days = (pct_move_needed / daily_vol) * 1.5
                return max(1, min(int(days), 30))
            else:
                return 7
        except:
            return 7

    def signal(self, ind):
        if not ind:
            return None
        i = ind
        uptrend = i['price'] > i['sma9']
        strong_move = abs(i['daily']) > 2
        not_extreme = 20 < i['pct_from_low'] < 80
        trend_explain = f"Price ₹{i['price']:.2f} vs SMA9 ₹{i['sma9']:.2f}"
        if uptrend:
            trend_explain += " → Price ABOVE average = UPTREND. Buyers in control."
        else:
            trend_explain += " → Price BELOW average = DOWNTREND. Sellers in control."
        momentum_explain = f"Daily move: {i['daily']:+.2f}%"
        if abs(i['daily']) > 2:
            momentum_explain += f" → STRONG move. Hourly: {i['hourly']:+.2f}%."
        else:
            momentum_explain += f" → Weak move. Hourly: {i['hourly']:+.2f}%."
        rsi_explain = f"RSI: {i['rsi']:.1f}"
        if i['rsi'] > 70:
            if uptrend:
                rsi_explain += " → OVERBOUGHT in UPTREND = Strength, not weakness."
            else:
                rsi_explain += " → OVERBOUGHT in downtrend. Reversal risk."
        elif i['rsi'] < 30:
            if uptrend:
                rsi_explain += " → OVERSOLD in uptrend. Good dip to buy."
            else:
                rsi_explain += " → OVERSOLD in downtrend. Catch falling knife risk."
        else:
            rsi_explain += " → NORMAL range. No extreme signal."
        position_explain = f"Price at {i['pct_from_low']:.0f}% from low"
        if i['pct_from_low'] > 80:
            position_explain += " → NEAR TOP of recent range."
        elif i['pct_from_low'] < 20:
            position_explain += " → NEAR BOTTOM of recent range."
        else:
            position_explain += " → MIDDLE of range."
        deviation_direction = "above" if i['pct_deviation'] > 0 else "below"
        zscore_explain = f"Z-Score: {i['zscore']:.2f} | Price is {abs(i['pct_deviation']):.2f}% {deviation_direction} mean (₹{i['mean_price']:.2f})"
        if i['zscore'] > 2:
            zscore_explain += f" → EXTREME OVEREXTENSION (+2σ). Price {abs(i['pct_deviation']):.1f}% above average, with HIGH probability mean reversion DOWN expected."
        elif i['zscore'] > 1:
            zscore_explain += f" → MODERATELY OVERBOUGHT. Price {abs(i['pct_deviation']):.1f}% above mean. Potential pullback zone."
        elif i['zscore'] < -2:
            zscore_explain += f" → EXTREME OVERSOLD (-2σ). Price {abs(i['pct_deviation']):.1f}% below average, with HIGH probability bounce to mean."
        elif i['zscore'] < -1:
            zscore_explain += f" → MODERATELY OVERSOLD. Price {abs(i['pct_deviation']):.1f}% below mean. Bounce opportunity."
        else:
            zscore_explain += f" → NEAR MEAN (within ±1σ). Price at fair value."
        bb_explain = f"Bollinger Band: {i['bb_position']:.0f}% position"
        if i['bb_position'] > 95:
            bb_explain += f" → TOUCHING UPPER BAND (₹{i['bb_upper']:.2f}). Overextended."
        elif i['bb_position'] < 5:
            bb_explain += f" → TOUCHING LOWER BAND (₹{i['bb_lower']:.2f}). Oversold."
        elif i['bb_position'] > 70:
            bb_explain += " → UPPER half. Bullish zone."
        elif i['bb_position'] < 30:
            bb_explain += " → LOWER half. Support zone."
        else:
            bb_explain += " → MIDDLE of bands. Neutral."
        rsi_overbought = i['rsi'] > 70
        rsi_oversold = i['rsi'] < 30
        near_top = i['pct_from_low'] > 80
        near_bottom = i['pct_from_low'] < 20
        extreme_high_zscore = i['zscore'] > 2
        high_zscore = i['zscore'] > 1
        extreme_low_zscore = i['zscore'] < -2
        low_zscore = i['zscore'] < -1
        sig = ""
        action = ""
        rec = ""
        entry_price = 0
        stop_price = 0
        target_price = 0
        if uptrend and i['macd_bullish'] and not near_top:
            sig = "BUY"
            action = "BULLISH SETUP"
            entry_price = min(i['sma9'], i['mean_price'])
            stop_price = i['low'] * 0.98
            target_price = i['high'] * 1.03
            if rsi_oversold and low_zscore:
                rec = "TRIPLE CONFLUENCE: Uptrend + oversold RSI + negative Z-score. Excellent dip-buying opportunity."
            elif extreme_low_zscore:
                rec = "Strong uptrend + extreme mean reversion setup. High-probability bounce."
            elif rsi_oversold:
                rec = "Strong uptrend + bullish MACD + RSI oversold. Excellent dip."
            elif strong_move:
                rec = "Momentum breakout in uptrend. Follow the trend."
            else:
                rec = "Confirmed uptrend with positive momentum. Good swing trade."
        elif not uptrend and not i['macd_bullish']:
            sig = "SELL"
            action = "BEARISH SETUP"
            entry_price = max(i['sma9'], i['mean_price'])
            stop_price = i['high'] * 1.02
            target_price = min(i['mean_price'], i['low'] * 0.97)
            if rsi_overbought and high_zscore:
                rec = "BEARISH CONFLUENCE: Downtrend + overbought + high Z-score. Mean reversion DOWN likely."
            elif extreme_high_zscore:
                rec = "Downtrend + extreme overextension. Pullback highly probable."
            elif rsi_overbought:
                rec = "Downtrend + bearish MACD + overbought RSI. High-probability short."
            elif strong_move:
                rec = "Breakdown with momentum. Exit longs, avoid catching knife."
            else:
                rec = "Confirmed downtrend. Stay out or short with tight stops."
        elif uptrend and not i['macd_bullish']:
            sig = "HOLD"
            action = "UPTREND WEAKENING"
            entry_price = min(i['sma9'], i['mean_price'])
            stop_price = i['sma9'] * 0.98
            target_price = i['high']
            if extreme_high_zscore:
                rec = "Uptrend intact but EXTREME overextension. Mean reversion pullback probable."
            elif near_top and rsi_overbought:
                rec = "Uptrend overextended. Hold longs, wait for pullback to add."
            else:
                rec = "Trend up but momentum fading. Hold with trailing stops."
        elif not uptrend and i['macd_bullish']:
            sig = "HOLD"
            action = "DOWNTREND STABILIZING"
            entry_price = i['sma9']
            stop_price = i['low'] * 0.98
            target_price = i['mean_price']
            if extreme_low_zscore and near_bottom:
                rec = "REVERSAL WATCH: Extreme oversold + early reversal signs. Monitor closely."
            elif near_bottom and rsi_oversold:
                rec = "Downtrend but oversold near support. Watch for reversal."
            else:
                rec = "Downtrend but momentum turning. Too early to buy."
        else:
            sig = "HOLD"
            action = "RANGE-BOUND"
            entry_price = (i['high'] + i['low']) / 2
            stop_price = i['low'] * 0.98
            target_price = i['high'] * 1.02
            rec = "Stock consolidating. Wait for breakout direction."
        confidence = self.calculate_confidence(i)
        days_to_target = self.estimate_days_to_target(i['price'], target_price, i)
        # Calculate risk-reward metrics
        # expected_move_pct: always positive magnitude (used for R-multiple math)
        # expected_move_signed: signed for display (negative for SELL = price expected to drop)
        if sig == "BUY":
            expected_move_pct = ((target_price - i['price']) / i['price']) * 100
            max_risk_pct = ((i['price'] - stop_price) / i['price']) * 100
            expected_move_signed = expected_move_pct
        elif sig == "SELL":
            expected_move_pct = ((i['price'] - target_price) / i['price']) * 100
            max_risk_pct = ((stop_price - i['price']) / i['price']) * 100
            expected_move_signed = -expected_move_pct  # negative: price expected to fall
        else:
            expected_move_pct = abs((target_price - i['price']) / i['price']) * 100
            max_risk_pct = abs((i['price'] - stop_price) / i['price']) * 100
            expected_move_signed = expected_move_pct if target_price > i['price'] else -expected_move_pct
        risk_reward = round(expected_move_pct / max_risk_pct, 2) if max_risk_pct > 0 else 0
        risk_per_share = abs(i['price'] - stop_price)
        # ── Auto-compute recommended risk % ──
        # Base risk: 1% of capital (standard retail default)
        # Adjustments:
        #   +0.5 if RR >= 2 (favourable payoff skew)
        #   +0.5 if confidence >= 70 (high-conviction setup)
        #   -0.5 if volatility > 3% daily (wider expected swings)
        #   -0.5 if confidence < 50 (low-conviction)
        #   Clamped to [0.25, 3.0]
        rec_risk_pct = 1.0
        risk_reasons = []
        if risk_reward >= 2:
            rec_risk_pct += 0.5
            risk_reasons.append(f"R-multiple is strong ({risk_reward}x), so the algorithm added +0.5% to the base risk.")
        elif risk_reward < 1:
            rec_risk_pct -= 0.25
            risk_reasons.append(f"R-multiple is below 1.0x ({risk_reward}x), meaning risk exceeds reward, so the algorithm reduced risk by 0.25%.")
        if confidence >= 70:
            rec_risk_pct += 0.5
            risk_reasons.append(f"Confidence is high ({confidence}%), indicating most indicators agree, so +0.5% was added.")
        elif confidence < 50:
            rec_risk_pct -= 0.5
            risk_reasons.append(f"Confidence is low ({confidence}%), meaning indicators disagree, so the algorithm cut risk by 0.5%.")
        if i['volatility'] > 3:
            rec_risk_pct -= 0.5
            risk_reasons.append(f"Daily volatility is elevated ({i['volatility']:.1f}%), meaning bigger daily swings, so risk was reduced by 0.5% to keep losses manageable.")
        elif i['volatility'] < 1.5:
            rec_risk_pct += 0.25
            risk_reasons.append(f"Volatility is low ({i['volatility']:.1f}%), making the stop loss tighter, so +0.25% was added.")
        rec_risk_pct = round(max(0.25, min(rec_risk_pct, 3.0)), 2)
        if not risk_reasons:
            risk_reasons.append("All factors are near baseline levels, so the standard 1% risk-per-trade is used.")
        risk_reason_text = f"Recommended risk: {rec_risk_pct}% of capital. Starting from a 1% base: " + " ".join(risk_reasons) + f" Final value clamped to [{0.25}%-{3.0}%]. To increase this, a higher confidence score (currently {confidence}%), a better R-multiple (currently {risk_reward}x), or lower volatility (currently {i['volatility']:.1f}%) would be needed."
        # ── Formula tooltip texts (used verbatim in the frontend) ──
        expected_move_tooltip = (
            f"Definition: The percentage distance from the current price to the target exit price. "
            f"Inputs: Current price ({i['price']:.2f}), Target price ({target_price:.2f}). "
            f"Formula: ((Target - Current) / Current) x 100 = "
            f"(({target_price:.2f} - {i['price']:.2f}) / {i['price']:.2f}) x 100 = {expected_move_pct:+.1f}%. "
            f"The target is set at the 18-day high + 3% for BUY setups (recent resistance breakout), "
            f"or the lower of the 20-day mean and the 18-day low - 3% for SELL setups."
        ) if sig == "BUY" else (
            f"Definition: The percentage distance from the current price to the target exit price. "
            f"Inputs: Current price ({i['price']:.2f}), Target price ({target_price:.2f}). "
            f"Formula: ((Current - Target) / Current) x 100 = "
            f"(({i['price']:.2f} - {target_price:.2f}) / {i['price']:.2f}) x 100 = {expected_move_pct:+.1f}%. "
            f"The target is set at the lower of the 20-day mean price and the 18-day low - 3% for SELL setups."
        ) if sig == "SELL" else (
            f"Definition: The percentage distance from the current price to the target exit price. "
            f"Inputs: Current price ({i['price']:.2f}), Target price ({target_price:.2f}). "
            f"Formula: |Target - Current| / Current x 100 = {expected_move_pct:.1f}%. "
            f"For HOLD setups the target is the 18-day high + 2% (breakout level)."
        )
        max_risk_tooltip = (
            f"Definition: The percentage you could lose if the trade hits your stop loss, which is the invalidation level "
            f"below which the trade thesis no longer holds. "
            f"Inputs: Current price ({i['price']:.2f}), Stop loss ({stop_price:.2f}). "
            f"Formula: ((Current - Stop) / Current) x 100 = "
            f"(({i['price']:.2f} - {stop_price:.2f}) / {i['price']:.2f}) x 100 = {max_risk_pct:.1f}%. "
            f"The stop loss is placed 2% below the 18-day low (recent support with buffer), acting as the "
            f"invalidation level. If price breaks this, the original setup is no longer valid."
        ) if sig == "BUY" else (
            f"Definition: The percentage you could lose if the trade hits your stop loss. "
            f"Inputs: Current price ({i['price']:.2f}), Stop loss ({stop_price:.2f}). "
            f"Formula: ((Stop - Current) / Current) x 100 = "
            f"(({stop_price:.2f} - {i['price']:.2f}) / {i['price']:.2f}) x 100 = {max_risk_pct:.1f}%. "
            f"The stop loss is placed 2% above the 18-day high (resistance + buffer)."
        ) if sig == "SELL" else (
            f"Definition: The percentage you could lose if the trade hits your stop loss. "
            f"Inputs: Current price ({i['price']:.2f}), Stop loss ({stop_price:.2f}). "
            f"Formula: |Current - Stop| / Current x 100 = {max_risk_pct:.1f}%. "
            f"The stop is placed 2% below a key support/average level."
        )
        if sig == "SELL":
            risk_reward_tooltip = (
                f"Definition: How many units of potential gain you get for every 1 unit of risk (the R-multiple). "
                f"For a SHORT/SELL setup, 'gain' means the price dropping to your target. "
                f"Inputs: Expected downward move ({expected_move_pct:.1f}%), Max Risk if price rises ({max_risk_pct:.1f}%). "
                f"Formula: Expected Move / Max Risk = {expected_move_pct:.1f} / {max_risk_pct:.1f} = {risk_reward:.2f}x. "
                f"Example: at {risk_reward:.2f}x, if you risk ₹100, you stand to gain ₹{risk_reward * 100:.0f} if the price falls to target. "
                f"Above 1.5x is favourable. Below 1.0x means the potential loss (price rising to stop) exceeds the potential gain."
            )
        else:
            risk_reward_tooltip = (
                f"Definition: How many units of potential gain you get for every 1 unit of risk (the R-multiple). "
                f"Inputs: Expected Move ({expected_move_pct:.1f}%), Max Risk ({max_risk_pct:.1f}%). "
                f"Formula: Expected Move / Max Risk = {expected_move_pct:.1f} / {max_risk_pct:.1f} = {risk_reward:.2f}x. "
                f"Example: at {risk_reward:.2f}x, if you risk ₹100, you stand to gain ₹{risk_reward * 100:.0f}. "
                f"Above 1.5x is favourable. Below 1.0x means the potential loss exceeds the potential gain."
            )
        # Determine setup duration label
        if days_to_target <= 7:
            setup_duration = "Short Term Setup"
        elif days_to_target <= 15:
            setup_duration = "Medium Term Setup"
        else:
            setup_duration = "Swing Trade Setup"
        # Build "Why This Makes Sense" plain-English summary
        trend_word = "strong uptrend" if uptrend and i['macd_bullish'] else "uptrend" if uptrend else "downtrend" if not uptrend and not i['macd_bullish'] else "sideways"
        buyer_seller = "buyers remain dominant" if uptrend else "sellers are in control" if not uptrend and not i['macd_bullish'] else "neither buyers nor sellers have clear control"
        momentum_word = "Momentum supports continuation despite extended conditions." if (uptrend and i['rsi'] > 60) or (not uptrend and i['rsi'] < 40) else "Momentum is building in favor of the current direction." if abs(i['daily']) > 1 else "Momentum is neutral, suggesting a potential shift ahead."
        why_makes_sense = f"Price is in a {trend_word} and {buyer_seller}. {momentum_word}"
        # Confidence one-liner
        if confidence >= 70:
            confidence_oneliner = "Past similar setups moved in this direction most of the time."
        elif confidence >= 55:
            confidence_oneliner = "Past similar setups moved upward more often than not." if sig == "BUY" else "Past similar setups moved downward more often than not." if sig == "SELL" else "Mixed signals. Waiting for clearer direction is advisable."
        else:
            confidence_oneliner = "Setup shows potential but has mixed signals. Use tighter risk controls."
        # SMA status for technical details
        above_sma20 = i['price'] > i.get('sma20', i['sma9'])
        above_sma50 = i['price'] > i.get('sma50', i['sma9'])
        # BB position label
        if i['bb_position'] > 90:
            bb_label = "Near Upper Bollinger"
        elif i['bb_position'] < 10:
            bb_label = "Near Lower Bollinger"
        elif i['bb_position'] > 60:
            bb_label = "Upper Half Bollinger"
        elif i['bb_position'] < 40:
            bb_label = "Lower Half Bollinger"
        else:
            bb_label = "Middle Bollinger"
        if sig == "BUY":
            entry_explain = f"Enter when price dips to ₹{entry_price:.2f}. This is a good entry because it's near the average price and provides a better risk-reward ratio."
            exit_explain = f"Exit (sell) at ₹{target_price:.2f}. This target is {expected_move_pct:.1f}% above current price."
            confidence_explain = f"{confidence}% confidence based on: trend strength, RSI level, mean reversion signal, and market volatility. Higher confidence = more reliable setup."
            time_explain = f"Expected to reach target in approximately {days_to_target} trading days based on historical price movement patterns and current momentum."
        elif sig == "SELL":
            entry_explain = f"Exit long positions or enter short at ₹{entry_price:.2f}. Price is likely to fall towards mean."
            exit_explain = f"Cover shorts or re-enter longs at ₹{target_price:.2f}. This is {expected_move_pct:.1f}% below current price."
            confidence_explain = f"{confidence}% confidence based on: downtrend confirmation, overbought conditions, and mean reversion probability."
            time_explain = f"Expected downward move in approximately {days_to_target} trading days."
        else:
            entry_explain = f"Wait for clearer signals. Consider entry only if price moves decisively above ₹{entry_price:.2f}."
            exit_explain = f"If already holding, consider taking profits at ₹{target_price:.2f}."
            confidence_explain = f"{confidence}% confidence. Moderate confidence suggests waiting for better setup."
            time_explain = f"Market consolidating. Wait for breakout confirmation."
        # ── Williams %R explain ──
        wr_val = i.get('williams_r', -50)
        wr_signal = i.get('williams_r_signal', 'NEUTRAL')
        williams_r_explain = f"Williams %R: {wr_val:.1f}"
        if wr_signal == 'POTENTIAL BOTTOM':
            williams_r_explain += " → Reversing from oversold zone. Potential bottom forming."
        elif wr_signal == 'POTENTIAL TOP':
            williams_r_explain += " → Reversing from overbought zone. Potential top forming."
        elif wr_val < -80:
            williams_r_explain += " → Deep oversold. Watching for reversal signal."
        elif wr_val > -20:
            williams_r_explain += " → Overbought territory. Watching for pullback."
        else:
            williams_r_explain += " → Neutral zone. No extreme reading."

        # ── Bollinger %B explain ──
        pb_val = i.get('percent_b', 0.5)
        pb_signal = i.get('percent_b_signal', 'LOWER HALF')
        percent_b_explain = f"%B: {pb_val:.2f}"
        if pb_val < 0:
            percent_b_explain += " → Price is outside the lower band. Statistically extreme. Mean reversion likely."
        elif pb_val > 1:
            percent_b_explain += " → Price is outside the upper band. Statistically extreme. Pullback likely."
        elif pb_val < 0.5:
            percent_b_explain += " → Price in lower half of bands. Closer to support."
        else:
            percent_b_explain += " → Price in upper half of bands. Closer to resistance."

        # ── RSI Divergence explain ──
        rsi_div = i.get('rsi_divergence', 'NONE')
        rsi_div_detail = i.get('rsi_divergence_detail', 'No divergence detected.')
        rsi_divergence_explain = f"RSI Divergence: {rsi_div_detail}"

        # ── Min/Max Confidence Score explain ──
        mm_type = i.get('minmax_type', 'MIXED')
        mm_score = i.get('minmax_score', 0)
        checks = i.get('confidence_checks', {})

        # ── Confidence score note for regime integration ──
        minmax_regime_note = ''
        if mm_type == 'BOTTOM' and mm_score >= 4:
            minmax_regime_note = f"High bottom confidence score ({mm_score}/5) detected. Consider scaling in cautiously despite bearish regime."
        elif mm_type == 'TOP' and mm_score >= 4:
            minmax_regime_note = f"High top confidence score ({mm_score}/5) detected. This reinforces caution on long positions."

        return {
            'signal': {
                'signal': sig, 'action': action, 'rec': rec,
                'entry': f"₹{entry_price:.2f}", 'stop': f"₹{stop_price:.2f}", 'target': f"₹{target_price:.2f}",
                'entry_raw': round(entry_price, 2), 'stop_raw': round(stop_price, 2), 'target_raw': round(target_price, 2),
                'confidence': confidence, 'days_to_target': days_to_target,
                'expected_move_pct': round(expected_move_pct, 1),
                'expected_move_signed': round(expected_move_signed, 1),
                'max_risk_pct': round(max_risk_pct, 1),
                'risk_reward': risk_reward,
                'risk_per_share': round(risk_per_share, 2),
                'rec_risk_pct': rec_risk_pct,
                'risk_reason_text': risk_reason_text,
                'expected_move_tooltip': expected_move_tooltip,
                'max_risk_tooltip': max_risk_tooltip,
                'risk_reward_tooltip': risk_reward_tooltip,
                'setup_duration': setup_duration,
                'why_makes_sense': why_makes_sense,
                'confidence_oneliner': confidence_oneliner,
                'entry_explain': entry_explain, 'exit_explain': exit_explain, 'confidence_explain': confidence_explain,
                'time_explain': time_explain, 'trend_explain': trend_explain, 'momentum_explain': momentum_explain,
                'rsi_explain': rsi_explain, 'position_explain': position_explain, 'zscore_explain': zscore_explain,
                'bb_explain': bb_explain, 'macd_text': "BULLISH: momentum favors buyers" if i['macd_bullish'] else "BEARISH: momentum favors sellers",
                'williams_r_explain': williams_r_explain,
                'percent_b_explain': percent_b_explain,
                'rsi_divergence_explain': rsi_divergence_explain,
                'minmax_regime_note': minmax_regime_note,
            },
            'details': {
                'price': f"₹{i['price']:.2f}", 'price_raw': round(i['price'], 2),
                'daily': f"{i['daily']:+.2f}%", 'daily_raw': round(i['daily'], 2),
                'hourly': f"{i['hourly']:+.2f}%",
                'rsi': f"{i['rsi']:.1f}", 'rsi_raw': round(i['rsi'], 1),
                'zscore': f"{i['zscore']:.2f}", 'zscore_raw': round(i['zscore'], 2),
                'pct_deviation': f"{i['pct_deviation']:+.2f}%",
                'mean': f"₹{i['mean_price']:.2f}", 'sma9': f"₹{i['sma9']:.2f}",
                'sma20': f"₹{i.get('sma20', i['sma9']):.2f}", 'sma50': f"₹{i.get('sma50', i['sma9']):.2f}",
                'above_sma20': above_sma20, 'above_sma50': above_sma50,
                'high': f"₹{i['high']:.2f}", 'low': f"₹{i['low']:.2f}",
                'bb_upper': f"₹{i['bb_upper']:.2f}", 'bb_lower': f"₹{i['bb_lower']:.2f}",
                'bb_position': round(i['bb_position'], 0), 'bb_label': bb_label,
                'volatility': f"{i['volatility']:.2f}%", 'macd': "BULLISH" if i['macd_bullish'] else "BEARISH",
                'macd_bullish': i['macd_bullish'],
                'williams_r': round(i.get('williams_r', -50), 1),
                'williams_r_signal': i.get('williams_r_signal', 'NEUTRAL'),
                'percent_b': round(i.get('percent_b', 0.5), 4),
                'percent_b_signal': i.get('percent_b_signal', 'NEUTRAL'),
                'rsi_divergence': i.get('rsi_divergence', 'NONE'),
                'rsi_divergence_detail': i.get('rsi_divergence_detail', ''),
                'minmax_type': i.get('minmax_type', 'MIXED'),
                'minmax_score': i.get('minmax_score', 0),
                'confidence_checks': i.get('confidence_checks', {}),
            }
        }

    def generate_projection_chart(self, data, signal_result):
        """Generate a price projection chart with historical data and forecast zone."""
        try:
            s = signal_result.get('signal', {})
            sig = s.get('signal', 'HOLD')
            target_raw = s.get('target_raw', 0)
            stop_raw = s.get('stop_raw', 0)
            days_to_target = s.get('days_to_target', 7)
            price_raw = signal_result.get('details', {}).get('price_raw', 0)
            if not price_raw or not target_raw or not stop_raw:
                return None

            close = data['Close'].dropna()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            # Take last 60 data points for history
            hist_len = min(60, len(close))
            hist_prices = close.iloc[-hist_len:].values.astype(float)
            hist_x = list(range(hist_len))

            # Forecast zone
            forecast_days = max(days_to_target, 3)
            forecast_x = list(range(hist_len - 1, hist_len + forecast_days))

            # Central projected path: linear interpolation from current price to target
            current = hist_prices[-1]
            central_path = np.linspace(current, target_raw, len(forecast_x))

            # Probability band: expands with time, bounded by stop and target
            daily_vol = float(close.pct_change().dropna().std()) if len(close) > 2 else 0.02
            band_widths = [daily_vol * current * np.sqrt(i + 1) * 1.5 for i in range(len(forecast_x))]
            upper_band = [central_path[i] + band_widths[i] for i in range(len(forecast_x))]
            lower_band = [central_path[i] - band_widths[i] for i in range(len(forecast_x))]

            # Colour scheme based on signal
            if sig == "BUY":
                action_color = '#10b981'
                action_color_light = '#10b98133'
            elif sig == "SELL":
                action_color = '#ef4444'
                action_color_light = '#ef444433'
            else:
                action_color = '#f59e0b'
                action_color_light = '#f59e0b33'

            fig, ax = plt.subplots(figsize=(8, 3.5), dpi=100)
            fig.patch.set_facecolor('#0a0c12')
            ax.set_facecolor('#080a10')

            # Historical price line
            ax.plot(hist_x, hist_prices, color='#a0aec0', linewidth=1.5, label='Historical', zorder=3)

            # Forecast zone
            ax.fill_between(forecast_x, lower_band, upper_band,
                            color=action_color_light, zorder=1, label='Probability band')
            ax.plot(forecast_x, central_path, color=action_color,
                    linewidth=2, linestyle='--', zorder=4, label='Projected path')

            # Current price marker
            ax.scatter([hist_len - 1], [current], color='#ffffff', s=50, zorder=6, edgecolors=action_color, linewidths=2)
            ax.annotate(f'CMP ₹{current:.2f}', xy=(hist_len - 1, current),
                        xytext=(hist_len - 1 - 8, current + (max(hist_prices) - min(hist_prices)) * 0.12),
                        color='#ffffff', fontsize=8, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#ffffff', lw=0.8),
                        zorder=7)

            # Horizontal target line
            total_x = hist_len + forecast_days
            ax.axhline(y=target_raw, color=action_color, linewidth=1, linestyle=':',
                        alpha=0.7, zorder=2)
            ax.text(total_x - 1, target_raw, f' Target ₹{target_raw:.2f}',
                    color=action_color, fontsize=7.5, va='bottom' if target_raw > current else 'top',
                    fontweight='bold', zorder=7)

            # Horizontal stop loss line
            stop_color = '#ef4444' if sig == "BUY" else '#10b981'
            ax.axhline(y=stop_raw, color=stop_color, linewidth=1, linestyle=':',
                        alpha=0.7, zorder=2)
            ax.text(total_x - 1, stop_raw, f' Stop ₹{stop_raw:.2f}',
                    color=stop_color, fontsize=7.5, va='top' if stop_raw < current else 'bottom',
                    fontweight='bold', zorder=7)

            # Vertical line separating history from forecast
            ax.axvline(x=hist_len - 1, color='#1a2030', linewidth=1, linestyle='-', alpha=0.5, zorder=1)
            mid_y = (max(hist_prices) + min(hist_prices)) / 2
            ax.text(hist_len + 1, ax.get_ylim()[1] * 0.99, 'FORECAST',
                    color='#718096', fontsize=7, fontstyle='italic', va='top', zorder=7)

            # Styling
            ax.tick_params(colors='#718096', labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#1a2030')
            ax.spines['left'].set_color('#1a2030')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            # Rupee labels on y-axis
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))

            plt.tight_layout(pad=0.5)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.15)
            plt.close(fig)
            buf.seek(0)
            chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            gc.collect()
            return chart_b64
        except Exception as e:
            print(f"Error generating projection chart: {e}")
            return None

    # ===== REGIME DETECTION LAYER =====

    def _compute_regime(self, close_series):
        """Compute market/sector regime from a close price series.

        Rules (price vs SMA20/SMA50 + RSI):
          bullish (1.0): price > SMA20 AND price > SMA50 AND RSI > 40
          bearish (0.0): price < SMA20 AND price < SMA50 AND RSI < 60
          neutral (0.5): everything else

        RSI guards prevent false classification during extreme mean-reversion:
          - RSI > 40 filter: avoids calling a regime "bullish" if a crash just
            pushed RSI into deep oversold territory despite price still above MAs.
          - RSI < 60 filter: avoids calling a regime "bearish" during a strong
            bounce that hasn't yet lifted price above the MAs.
        """
        if close_series is None or len(close_series) < 20:
            return 'neutral', 0.5, {'reason': 'Insufficient data for regime detection'}

        price = float(close_series.iloc[-1])
        sma20 = float(close_series.rolling(20).mean().iloc[-1])
        sma50_window = min(50, len(close_series))
        sma50 = float(close_series.rolling(sma50_window).mean().iloc[-1])

        # RSI (14-period)
        rsi_window = min(14, len(close_series))
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_window).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        current_rsi = float(rsi_series.iloc[-1])
        if pd.isna(current_rsi) or np.isinf(current_rsi):
            current_rsi = 50.0

        above_sma20 = price > sma20
        above_sma50 = price > sma50

        if above_sma20 and above_sma50 and current_rsi > 40:
            regime = 'bullish'
            score = 1.0
        elif not above_sma20 and not above_sma50 and current_rsi < 60:
            regime = 'bearish'
            score = 0.0
        else:
            regime = 'neutral'
            score = 0.5

        details = {
            'price': round(price, 2),
            'sma20': round(sma20, 2),
            'sma50': round(sma50, 2),
            'rsi': round(current_rsi, 1),
            'above_sma20': above_sma20,
            'above_sma50': above_sma50,
        }
        return regime, score, details

    def _fetch_regime(self, yahoo_ticker, cache_key):
        """Fetch price data for a ticker, compute its regime, and cache."""
        cached = REGIME_CACHE.get(cache_key)
        if cached is not None:
            return cached['regime'], cached['score'], cached['details']
        try:
            data = yf.download(yahoo_ticker, period='3mo', interval='1d',
                               progress=False, threads=False, timeout=8)
            if data is None or data.empty:
                result = {'regime': 'neutral', 'score': 0.5,
                          'details': {'reason': f'No data for {yahoo_ticker}', 'source': yahoo_ticker}}
                REGIME_CACHE.set(cache_key, result)
                return result['regime'], result['score'], result['details']

            close = data.get('Close')
            if close is None:
                result = {'regime': 'neutral', 'score': 0.5,
                          'details': {'reason': f'No close data for {yahoo_ticker}', 'source': yahoo_ticker}}
                REGIME_CACHE.set(cache_key, result)
                return result['regime'], result['score'], result['details']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()

            regime, score, details = self._compute_regime(close)
            details['source'] = yahoo_ticker
            result = {'regime': regime, 'score': score, 'details': details}
            REGIME_CACHE.set(cache_key, result)
            return regime, score, details
        except Exception as e:
            result = {'regime': 'neutral', 'score': 0.5,
                      'details': {'reason': f'Error: {str(e)}', 'source': yahoo_ticker}}
            REGIME_CACHE.set(cache_key, result)
            return result['regime'], result['score'], result['details']

    def _get_market_regime(self):
        """Get Nifty 50 market regime. Falls back to NIFTYBEES ETF if index unavailable."""
        regime, score, details = self._fetch_regime('^NSEI', 'regime:market:NSEI')
        if details.get('reason') and 'No data' in details.get('reason', ''):
            regime, score, details = self._fetch_regime('NIFTYBEES.NS', 'regime:market:NIFTYBEES')
        return regime, score, details

    def _get_sector_regime(self, symbol):
        """Get sector regime for a stock. Falls back to neutral if sector data unavailable.

        Returns (regime, score, details) where details always includes 'sector_name'.
        """
        # Look up real sector, skipping meta-sectors
        sector = TICKER_TO_SECTOR.get(symbol, '')
        meta = {'All NSE', 'Nifty 50', 'Nifty Next 50', 'Conglomerate'}
        if not sector or sector in meta:
            # Fallback: scan STOCKS for first real sector containing this symbol
            for sn, st in STOCKS.items():
                if sn in meta:
                    continue
                if symbol in st:
                    sector = sn
                    break
        if not sector or sector in meta:
            return 'neutral', 0.5, {'reason': f'No sector mapping for {symbol}', 'sector_name': 'Unknown'}
        sector_ticker = SECTOR_INDEX_MAP.get(sector)
        if not sector_ticker:
            return 'neutral', 0.5, {'reason': f'No index for sector "{sector}"', 'sector_name': sector}
        regime, score, details = self._fetch_regime(sector_ticker, f'regime:sector:{sector}')
        # Copy so we don't mutate the cached dict in _fetch_regime
        details = dict(details)
        details['sector_name'] = sector
        return regime, score, details

    def _build_verdict(self, signal_result, original_signal, gated_signal,
                       gate_reason, market_regime, sector_regime, sector_name,
                       regime_factor, factor_label, final_score):
        """Build a plain-English verdict that tells the complete story.

        Weaves together trend, momentum, valuation, regime context, and
        risk-reward into a narrative a beginner can follow.
        """
        s = signal_result.get('signal', {})
        d = signal_result.get('details', {})
        price = d.get('price', '?')
        rsi_raw = d.get('rsi_raw', 50)
        zscore_raw = d.get('zscore_raw', 0)
        volatility = d.get('volatility', '?')
        macd_bullish = d.get('macd_bullish', True)
        above_sma20 = d.get('above_sma20', False)
        above_sma50 = d.get('above_sma50', False)
        confidence = s.get('confidence', 50)
        risk_reward = s.get('risk_reward', 0)
        expected_move = s.get('expected_move_signed', 0)
        max_risk = s.get('max_risk_pct', 0)
        target = s.get('target', '?')
        stop = s.get('stop', '?')
        days = s.get('days_to_target', 7)
        rec_risk = s.get('rec_risk_pct', 1.0)
        final_signal = gated_signal

        parts = []

        # --- Sentence 1: The headline verdict ---
        # R/R quality dominates the tone. A "strong buy" with 0.5x R/R is
        # misleading - the headline must be honest about the trade quality.
        if final_signal == 'BUY':
            if risk_reward < 1.0:
                parts.append(
                    f"The trend is bullish, but the risk-reward setup is poor "
                    f"({risk_reward}x, meaning you'd risk more than you stand to gain). "
                    f"Consider waiting for a better entry point."
                )
            elif risk_reward < 1.5 and confidence < 70:
                parts.append(f"This stock shows a mildly favorable setup for buying, but the reward barely outweighs the risk. Proceed with caution.")
            elif confidence >= 75 and risk_reward >= 1.5:
                parts.append(f"This stock looks like a strong buying opportunity. The trend, momentum, and risk-reward are all lining up.")
            elif confidence >= 60:
                parts.append(f"This stock shows a decent setup for buying, though not without some caution.")
            else:
                parts.append(f"There's a mild case for buying this stock, but the setup isn't particularly strong.")
        elif final_signal == 'SELL':
            if risk_reward < 1.0:
                parts.append(
                    f"The stock is weakening, but the risk-reward on the short side "
                    f"is poor ({risk_reward}x). The potential loss exceeds the potential gain."
                )
            elif confidence >= 75 and risk_reward >= 1.5:
                parts.append(f"This stock is showing strong signs of weakness, and it's time to exit or consider shorting.")
            elif confidence >= 60:
                parts.append(f"The stock is leaning bearish. Consider reducing your position or staying out.")
            else:
                parts.append(f"There are some bearish signals here, but conviction is low.")
        else:
            if factor_label in ('rr_reject', 'rr_conflict'):
                parts.append(
                    f"The stock's trend says {original_signal}, but the risk-reward "
                    f"is only {risk_reward}x. The potential downside far exceeds "
                    f"the upside. The system has downgraded this to HOLD until "
                    f"the setup improves."
                )
            elif original_signal and original_signal != 'HOLD':
                parts.append(f"The stock's own indicators say {original_signal}, but the bigger picture is conflicting, so the recommendation is to HOLD and wait for clarity.")
            else:
                parts.append(f"This stock is in a wait-and-watch zone. There's no clear edge for buying or selling right now.")

        # --- Sentence 2: Trend & momentum story ---
        trend_parts = []
        if above_sma20 and above_sma50:
            trend_parts.append("the price is trading above both its 20-day and 50-day moving averages, which is a healthy uptrend")
        elif above_sma20 and not above_sma50:
            trend_parts.append("the price is above the 20-day average but still below the 50-day, suggesting a short-term recovery within a longer-term weakness")
        elif not above_sma20 and above_sma50:
            trend_parts.append("the price has dipped below the 20-day average but is still above the 50-day, hinting at a short-term pullback in an otherwise intact trend")
        else:
            trend_parts.append("the price is below both the 20-day and 50-day averages, which means the trend is pointing down")

        if macd_bullish:
            trend_parts.append("MACD momentum is bullish (buying pressure is building)")
        else:
            trend_parts.append("MACD momentum is bearish (selling pressure is dominant)")

        parts.append(f"On the technical side, {', and '.join(trend_parts)}.")

        # --- Sentence 3: RSI & valuation ---
        rsi_text = ""
        if rsi_raw > 70:
            rsi_text = f"RSI is at {rsi_raw:.0f} (overbought, meaning the stock may be stretched too far up and due for a pullback)"
        elif rsi_raw < 30:
            rsi_text = f"RSI is at {rsi_raw:.0f} (oversold, meaning the stock has been beaten down and could bounce)"
        elif rsi_raw > 60:
            rsi_text = f"RSI is at {rsi_raw:.0f} (strong momentum, but not yet overbought)"
        elif rsi_raw < 40:
            rsi_text = f"RSI is at {rsi_raw:.0f} (weak momentum, but not yet oversold)"
        else:
            rsi_text = f"RSI is at {rsi_raw:.0f} (neutral range, no extreme reading)"

        zscore_text = ""
        if zscore_raw > 2:
            zscore_text = "the price is stretched far above its recent average (high chance of a pullback)"
        elif zscore_raw > 1:
            zscore_text = "the price is moderately above its recent average"
        elif zscore_raw < -2:
            zscore_text = "the price is well below its recent average (potential bounce zone)"
        elif zscore_raw < -1:
            zscore_text = "the price is moderately below its recent average"
        else:
            zscore_text = "the price is near its recent average (fair value zone)"

        parts.append(f"{rsi_text}, and {zscore_text}.")

        # --- Sentence 4: Market & sector context ---
        market_word = {'bullish': 'supportive (bullish)', 'bearish': 'hostile (bearish)', 'neutral': 'mixed (neutral)'}
        sector_word = {'bullish': 'in favor (bullish)', 'bearish': 'working against it (bearish)', 'neutral': 'not giving a clear signal (neutral)'}
        parts.append(
            f"Looking at the bigger picture, the overall market (Nifty 50) is {market_word.get(market_regime, 'unclear')}, "
            f"and the {sector_name} sector is {sector_word.get(sector_regime, 'unclear')}."
        )

        # --- Sentence 5: What the regime + R/R layer did ---
        if factor_label == 'rr_reject':
            parts.append(
                f"The system has downgraded this to HOLD primarily because "
                f"the risk-reward ratio ({risk_reward}x) is too unfavorable. "
                f"The stop loss is far from the entry while the target is close, "
                f"meaning the downside exposure far outweighs the upside potential."
            )
        elif factor_label == 'rr_conflict':
            parts.append(
                f"The combination of a weak risk-reward ({risk_reward}x) and "
                f"opposing regime conditions makes this trade too risky. "
                f"The system has downgraded to HOLD."
            )
        elif factor_label == 'full_alignment':
            parts.append("Since the stock, sector, and market all agree on direction, this is a high-conviction setup and the system has increased the recommended position size by 20%.")
        elif factor_label == 'hard_conflict':
            if gated_signal == original_signal:
                parts.append(f"Both the market and sector are moving against this stock's signal, but the stock's own bearish conviction is strong enough to keep the {original_signal} call. Position size has been sharply reduced as a safety measure.")
            else:
                parts.append(f"Because both the market and sector are moving against this stock's signal, the system has overridden the {original_signal} to HOLD and sharply reduced the position size as a safety measure.")
        elif factor_label == 'conflict':
            parts.append("The stock says one thing but part of the broader environment disagrees. The system has reduced risk by 30% to account for this headwind.")
        else:
            parts.append("The broader environment is mixed, so position sizing stays at the default level.")

        # --- Sentence 6: Risk-reward & what to do ---
        rr_quality = "excellent" if risk_reward >= 2.5 else "strong" if risk_reward >= 2 else "good" if risk_reward >= 1.5 else "acceptable" if risk_reward >= 1 else "unfavorable (risk exceeds reward)"
        if final_signal == 'BUY':
            parts.append(
                f"If you buy at the current price of {price}, the target is {target} "
                f"(+{abs(expected_move):.1f}%) with a stop loss at {stop} (-{max_risk:.1f}%). "
                f"The risk-reward ratio is {risk_reward}x which is {rr_quality}. "
                f"The system recommends risking {rec_risk}% of your capital on this trade."
            )
        elif final_signal == 'SELL':
            parts.append(
                f"The target on the downside is {target} ({expected_move:+.1f}%) "
                f"with a stop loss at {stop}. Risk-reward is {risk_reward}x ({rr_quality}). "
                f"Risk no more than {rec_risk}% of your capital."
            )
        else:
            if risk_reward > 0 and risk_reward < 1.0:
                parts.append(
                    f"The numbers: target {target}, stop {stop}, risk-reward {risk_reward}x ({rr_quality}). "
                    f"Don't take a new position until the risk-reward improves. "
                    f"Either wait for a pullback to a better entry, or for the stop/target levels to shift."
                )
            else:
                parts.append(
                    f"For now, don't take a new position. Watch for the market or sector conditions to improve "
                    f"before acting on this stock's signal."
                )

        # --- Sentence 7: Confidence summary ---
        parts.append(
            f"Overall confidence across all factors is {confidence}% (regime-adjusted score: {final_score * 100:.0f}%)."
        )

        return "<ul>" + "".join(f"<li>{p}</li>" for p in parts) + "</ul>"

    def _apply_regime_layer(self, signal_result, symbol):
        """Apply market + sector regime layer on top of stock-level signal.

        This method:
        1. Fetches market regime (Nifty 50) and sector regime (sectoral index).
        2. Computes a directionally-aware blended score.
        3. Applies signal gating (BUY/SELL -> HOLD when regime conflicts).
        4. Applies risk factor to rec_risk_pct.
        5. Adds explainability fields to the response payload.

        Directional awareness in the blended score:
          For BUY: bullish regime = favourable (score 1.0 as-is).
          For SELL: bearish regime = favourable (score inverted: 1.0 - score).
          For HOLD: regime scores fixed at 0.5 (no directional bias).
        This ensures the blended score always represents "how favourable is
        the overall environment for the *direction* of the current signal".
        """
        if not signal_result or 'signal' not in signal_result:
            return signal_result

        sig_data = signal_result['signal']
        original_signal = sig_data['signal']
        original_confidence = sig_data['confidence']
        stock_score = original_confidence / 100.0

        # --- Fetch regimes (failsafe: defaults to neutral on error) ---
        try:
            market_regime, market_score, market_details = self._get_market_regime()
        except Exception:
            market_regime, market_score, market_details = 'neutral', 0.5, {'reason': 'Error fetching market data'}
        try:
            sector_regime, sector_score, sector_details = self._get_sector_regime(symbol)
        except Exception:
            sector_regime, sector_score, sector_details = 'neutral', 0.5, {'reason': 'Error fetching sector data'}

        # --- Directionally-aware blended score ---
        # Raw regime scores: bullish=1.0, neutral=0.5, bearish=0.0
        # For SELL signals we invert so bearish=1.0 (favourable) and bullish=0.0.
        if original_signal == 'BUY':
            market_score_adj = market_score
            sector_score_adj = sector_score
        elif original_signal == 'SELL':
            market_score_adj = 1.0 - market_score
            sector_score_adj = 1.0 - sector_score
        else:
            market_score_adj = 0.5
            sector_score_adj = 0.5

        final_score = 0.5 * stock_score + 0.3 * sector_score_adj + 0.2 * market_score_adj

        # --- Signal gating ---
        gated_signal = original_signal
        gate_reason = None

        if original_signal == 'BUY' and market_regime == 'bearish' and sector_regime == 'bearish':
            if original_confidence < 65:
                gated_signal = 'HOLD'
                gate_reason = (
                    "BUY downgraded to HOLD: both market and sector are in bearish regime "
                    "and stock conviction is moderate. "
                    "Broad weakness across the market and sector makes initiating long positions risky."
                )
            else:
                gate_reason = (
                    "Both market and sector are bearish, which is a headwind for this BUY. "
                    "Proceed with caution and a smaller position."
                )
        elif original_signal == 'SELL' and market_regime == 'bullish' and sector_regime == 'bullish':
            if original_confidence < 65:
                gated_signal = 'HOLD'
                gate_reason = (
                    "SELL downgraded to HOLD: both market and sector are in bullish regime "
                    "and stock conviction is moderate. "
                    "Broad strength makes shorting against the trend risky."
                )
            else:
                gate_reason = (
                    "Both market and sector are bullish, which is a headwind for this SELL. "
                    "The stock's own bearish signals are strong enough to maintain the call, "
                    "but use tighter risk management."
                )

        # --- Regime factor for risk ---
        market_aligned = False
        sector_aligned = False
        market_conflict = False
        sector_conflict = False

        if original_signal in ('BUY', 'SELL'):
            if original_signal == 'BUY':
                market_aligned = market_regime == 'bullish'
                sector_aligned = sector_regime == 'bullish'
                market_conflict = market_regime == 'bearish'
                sector_conflict = sector_regime == 'bearish'
            else:
                market_aligned = market_regime == 'bearish'
                sector_aligned = sector_regime == 'bearish'
                market_conflict = market_regime == 'bullish'
                sector_conflict = sector_regime == 'bullish'

            if market_aligned and sector_aligned:
                regime_factor = 1.2
                factor_label = 'full_alignment'
            elif market_conflict and sector_conflict:
                regime_factor = 0.4
                factor_label = 'hard_conflict'
                if original_confidence < 65:
                    gated_signal = 'HOLD'
                    if gate_reason is None:
                        gate_reason = (
                            f"Signal downgraded to HOLD due to hard conflict. Both market ({market_regime}) "
                            f"and sector ({sector_regime}) regimes oppose the {original_signal} signal."
                        )
                else:
                    regime_factor = 0.5
                    if gate_reason is None:
                        gate_reason = (
                            f"Both market ({market_regime}) and sector ({sector_regime}) regimes oppose "
                            f"the {original_signal} signal, but stock conviction is high enough to "
                            f"maintain the call. Position size reduced as a safety measure."
                        )
            elif market_conflict or sector_conflict:
                regime_factor = 0.7
                factor_label = 'conflict'
            else:
                regime_factor = 1.0
                factor_label = 'mixed'
        else:
            regime_factor = 1.0
            factor_label = 'neutral'

        # --- R/R quality gating ---
        # Risk-reward is computed from entry/stop/target levels. If risk
        # exceeds reward, the trade setup is structurally unfavorable
        # regardless of how strong the trend looks.
        #   R/R < 0.5  → always downgrade to HOLD (risking 2x the reward)
        #   R/R < 1.0 + any regime conflict → downgrade to HOLD
        #   R/R < 1.0 without conflict → keep signal but reduce risk further
        risk_reward = sig_data.get('risk_reward', 0)
        rr_gated = False

        if gated_signal in ('BUY', 'SELL') and 0 < risk_reward < 1.0:
            if risk_reward < 0.5:
                # Extremely poor R/R - mathematically doesn't make sense
                gated_signal = 'HOLD'
                regime_factor = min(regime_factor, 0.5)
                factor_label = 'rr_reject'
                rr_gate_text = (
                    f"Risk-reward is only {risk_reward}x, meaning you'd risk "
                    f"roughly {1/risk_reward:.1f}x more than you stand to gain. "
                    f"The trade setup doesn't justify the risk at current levels."
                )
                gate_reason = rr_gate_text + (" " + gate_reason if gate_reason else "")
                rr_gated = True
            elif market_conflict or sector_conflict:
                # Poor R/R AND regime headwind - too many factors against
                gated_signal = 'HOLD'
                regime_factor = min(regime_factor, 0.6)
                factor_label = 'rr_conflict'
                rr_gate_text = (
                    f"Weak risk-reward ({risk_reward}x) combined with regime "
                    f"headwinds. Risk exceeds reward while broader conditions "
                    f"are also unfavorable."
                )
                gate_reason = rr_gate_text + (" " + gate_reason if gate_reason else "")
                rr_gated = True
            else:
                # Poor R/R but no regime conflict - warn but don't override
                regime_factor = min(regime_factor, 0.8)
                if factor_label == 'full_alignment':
                    factor_label = 'mixed'  # can't call it full alignment with poor R/R

        sig_data['rr_gated'] = rr_gated

        # --- Apply regime factor to risk ---
        original_risk = sig_data['rec_risk_pct']
        adjusted_risk = round(max(0.25, min(original_risk * regime_factor, 3.0)), 2)

        # --- Build regime reason text ---
        reason_parts = []
        reason_parts.append(f"Market regime: {market_regime.upper()} (Nifty 50).")
        sector_name = sector_details.get('sector_name') or TICKER_TO_SECTOR.get(symbol, 'Unknown')
        reason_parts.append(f"Sector regime: {sector_regime.upper()} ({sector_name}).")

        alignment_labels = {
            'full_alignment': 'Full alignment: market, sector, and stock signal all agree.',
            'mixed': 'Mixed: partial alignment between regime and signal.',
            'conflict': 'Conflict: one of market or sector regime opposes the signal.',
            'hard_conflict': 'Hard conflict: both market and sector regimes oppose the signal.',
            'neutral': 'Neutral: HOLD signal, no directional alignment applicable.',
            'rr_reject': f'Risk-reward too low ({risk_reward}x), trade rejected regardless of regime.',
            'rr_conflict': f'Weak risk-reward ({risk_reward}x) compounded by regime conflict.',
        }
        reason_parts.append(alignment_labels.get(factor_label, ''))

        if gate_reason:
            reason_parts.append(gate_reason)
        if regime_factor != 1.0:
            reason_parts.append(
                f"Risk adjusted from {original_risk}% to {adjusted_risk}% "
                f"(regime factor x{regime_factor})."
            )

        # --- Integrate minmax confidence note ---
        minmax_note = sig_data.get('minmax_regime_note', '')
        if minmax_note:
            reason_parts.append(minmax_note)

        regime_reason_text = " ".join(reason_parts)

        # --- Update signal data ---
        if gated_signal != original_signal:
            sig_data['signal'] = gated_signal
            sig_data['original_signal'] = original_signal
            sig_data['action'] = f"REGIME OVERRIDE ({original_signal}\u2192HOLD)"
        sig_data['rec_risk_pct'] = adjusted_risk

        # Update risk_reason_text to include regime info
        sig_data['risk_reason_text'] = (
            sig_data.get('risk_reason_text', '') +
            f" [Regime layer: {factor_label} (x{regime_factor}). "
            f"Risk adjusted to {adjusted_risk}%.]"
        )

        # --- Add regime fields to signal payload ---
        regime_confidence = max(0, min(round(final_score * 100), 100))
        sig_data['market_regime'] = market_regime
        sig_data['sector_regime'] = sector_regime
        sig_data['regime_score'] = round(final_score, 3)
        sig_data['regime_factor'] = regime_factor
        sig_data['regime_reason_text'] = regime_reason_text

        # --- Build plain-English verdict narrative ---
        sig_data['verdict_text'] = self._build_verdict(
            signal_result, original_signal, gated_signal, gate_reason,
            market_regime, sector_regime, sector_name,
            regime_factor, factor_label, final_score,
        )

        # --- Add detailed regime block for transparency ---
        signal_result['regime'] = {
            'market_regime': market_regime,
            'market_score': market_score,
            'market_details': market_details,
            'sector_regime': sector_regime,
            'sector_score': sector_score,
            'sector_name': sector_name,
            'sector_details': sector_details,
            'stock_score': round(stock_score, 3),
            'final_score': round(final_score, 3),
            'regime_confidence': regime_confidence,
            'regime_factor': regime_factor,
            'factor_label': factor_label,
            'original_signal': original_signal,
            'gated_signal': gated_signal,
            'gate_reason': gate_reason,
            'original_risk_pct': original_risk,
            'adjusted_risk_pct': adjusted_risk,
        }

        return signal_result

    def analyze(self, symbol):
        """Main analysis method"""
        cached = ANALYSIS_CACHE.get(symbol)
        if cached:
            return cached
        data = self.get_data(symbol)
        if data is None:
            return None
        ind = self.calc_indicators(data)
        if not ind:
            return None
        result = self.signal(ind)
        if result:
            # Apply regime layer (failsafe: if it errors, stock signal is unchanged)
            try:
                self._apply_regime_layer(result, symbol)
            except Exception as e:
                print(f"[WARN] Regime layer failed for {symbol}: {e}")
            try:
                chart_key = f"projection:{symbol}"
                chart_b64 = ANALYSIS_CACHE.get(chart_key)
                if not chart_b64:
                    chart_b64 = self.generate_projection_chart(data, result)
                    if chart_b64:
                        ANALYSIS_CACHE.set(chart_key, chart_b64)
                if chart_b64:
                    result['projection_chart'] = chart_b64
            except Exception:
                pass
            ANALYSIS_CACHE.set(symbol, result)
        del data
        gc.collect()
        return result

    def regression_analysis(self, stock_symbol):
        """Perform HSIC (Hilbert-Schmidt Independence Criterion) non-linear dependency analysis of stock vs Nifty 50.

        Uses an RBF kernel with median-heuristic bandwidth to detect arbitrary
        (including non-linear) statistical dependencies between daily returns.
        Capped to the last 90 trading days and computed in float32 to stay
        within the 512 MB Free-Tier RAM limit.
        """
        cached = REGRESSION_CACHE.get(stock_symbol)
        if cached:
            return cached
        def _clean_close(px_df):
            c = px_df.get("Close", None)
            if c is None: return pd.Series(dtype=float)
            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            s = c.copy()
            s.index = pd.to_datetime(s.index, errors="coerce")
            try:
                if getattr(s.index, "tz", None) is not None: s.index = s.index.tz_convert(None)
            except Exception: pass
            s = s[~s.index.isna()]
            s.index = s.index.normalize()
            s = s.groupby(s.index).last()
            s = pd.to_numeric(s, errors="coerce").dropna()
            return s

        def _rbf_kernel_f32(x, sigma):
            """Compute RBF (Gaussian) kernel matrix in float32.
            K_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
            """
            x = x.astype(np.float32).reshape(-1, 1)
            sq_dists = (x - x.T) ** 2
            return np.exp(-sq_dists / (np.float32(2.0) * np.float32(sigma) ** 2))

        def _median_heuristic(x):
            """Median heuristic for RBF bandwidth: sigma = median(|x_i - x_j|) for i != j."""
            x = x.astype(np.float32).reshape(-1, 1)
            dists = np.abs(x - x.T)
            # Extract upper triangle (exclude diagonal zeros)
            upper = dists[np.triu_indices_from(dists, k=1)]
            med = float(np.median(upper))
            return med if med > 1e-8 else 1e-4  # guard against zero bandwidth

        def _hsic_score(x, y, max_samples=252):
            """Compute normalised HSIC dependency score in [0, 1].

            Steps:
            1. Cap to last `max_samples` observations (memory-safe for 512 MB).
               252 days ≈ 1 trading year. 252×252 float32 matrix ≈ 254 KB.
            2. Compute RBF kernels K (for x) and L (for y) using median heuristic.
            3. Center both kernels:  K_c = H K H,  L_c = H L H  where H = I - 1/n 11^T.
            4. HSIC = tr(K_c L_c) / n^2.
            5. Normalise: score = HSIC / sqrt(HSIC_xx * HSIC_yy) clamped to [0, 1].
            """
            # --- cap to last N trading days --------------------------------
            if len(x) > max_samples:
                x = x[-max_samples:]
                y = y[-max_samples:]
            n = len(x)

            # --- float32 throughout ----------------------------------------
            x = x.astype(np.float32)
            y = y.astype(np.float32)

            sigma_x = _median_heuristic(x)
            sigma_y = _median_heuristic(y)

            K = _rbf_kernel_f32(x, sigma_x)
            L = _rbf_kernel_f32(y, sigma_y)

            # Centering matrix H = I - (1/n) * 11^T
            H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)

            Kc = H @ K @ H
            Lc = H @ L @ H

            hsic_xy = float(np.trace(Kc @ Lc)) / (n * n)
            hsic_xx = float(np.trace(Kc @ Kc)) / (n * n)
            hsic_yy = float(np.trace(Lc @ Lc)) / (n * n)

            denom = np.sqrt(max(hsic_xx, 0) * max(hsic_yy, 0))
            if denom < 1e-12:
                return 0.0, hsic_xy, n
            score = float(np.clip(hsic_xy / denom, 0.0, 1.0))
            return score, hsic_xy, n

        try:
            periods_to_try = ['1y', '6mo', '3mo']
            stock_data, nifty_data = None, None
            nifty_source = None

            for period in periods_to_try:
                try:
                    stock_data = yf.download(f"{stock_symbol}.NS", period=period, interval='1d', progress=False, threads=False)
                    if stock_data is None or stock_data.empty: continue
                    nifty_data = yf.download("^NSEI", period=period, interval='1d', progress=False, threads=False)
                    if nifty_data is None or nifty_data.empty:
                        nifty_data = yf.download("NIFTYBEES.NS", period=period, interval='1d', progress=False, threads=False)
                        if nifty_data is None or nifty_data.empty:
                            nifty_data = yf.download("RELIANCE.NS", period=period, interval='1d', progress=False, threads=False)
                            nifty_source = "RELIANCE (proxy)"
                        else: nifty_source = "NIFTYBEES ETF"
                    else: nifty_source = "Nifty 50 Index"

                    if nifty_data is not None and not nifty_data.empty: break
                except Exception: continue

            if stock_data is None or stock_data.empty or nifty_data is None or nifty_data.empty:
                return None

            stock_close = _clean_close(stock_data)
            market_close = _clean_close(nifty_data)
            stock_ret = stock_close.pct_change()
            market_ret = market_close.pct_change()
            rets = pd.concat([stock_ret.rename("stock"), market_ret.rename("market")], axis=1, join="inner").dropna()

            if len(rets) < 20: return None

            X = rets["market"].to_numpy()
            y = rets["stock"].to_numpy()

            # --- HSIC computation (252 trading-day lookback, float32) ------
            # 252 days ≈ 1 trading year, sufficient for structural claims.
            # 252×252 float32 matrix ≈ 254 KB; well within 512 MB limit.
            dep_score, raw_hsic, n_used = _hsic_score(X, y, max_samples=252)

            # Slice arrays to the window actually used
            X_win = X[-n_used:]
            y_win = y[-n_used:]

            # --- Beta & downside beta (OLS, contextualises crash behaviour) -
            X32 = X_win.astype(np.float32)
            y32 = y_win.astype(np.float32)
            cov_xy = float(np.mean((X32 - X32.mean()) * (y32 - y32.mean())))
            var_x = float(np.var(X32, ddof=0))
            beta = round(cov_xy / var_x, 4) if var_x > 1e-12 else 0.0

            # Downside beta: beta computed only on days when market fell
            down_mask = X32 < 0
            if down_mask.sum() >= 10:
                Xd = X32[down_mask]
                yd = y32[down_mask]
                cov_d = float(np.mean((Xd - Xd.mean()) * (yd - yd.mean())))
                var_d = float(np.var(Xd, ddof=0))
                downside_beta = round(cov_d / var_d, 4) if var_d > 1e-12 else beta
            else:
                downside_beta = beta  # too few down days to estimate separately

            # --- Pearson correlation ----------------------------------------
            correlation = round(float(np.corrcoef(X_win, y_win)[0, 1]), 4)
            abs_corr = abs(correlation)

            # --- Composite connection score (accounts for BOTH measures) ----
            # Neither HSIC nor correlation alone tells the full story.
            # Composite = weighted blend so that high correlation can never
            # be overridden by a low HSIC into a "low connection" verdict.
            composite = round(0.5 * dep_score + 0.3 * abs_corr + 0.2 * min(abs(beta), 2.0) / 2.0, 4)
            composite = min(composite, 1.0)

            # --- Classification (recalibrated thresholds) -------------------
            # The composite ensures a stock with corr=0.64 never falls below
            # "Moderate".  Thresholds: High ≥0.55, Moderate ≥0.35, Low ≥0.18.
            if composite >= 0.55:
                dep_label = "High Dependence"
                dep_color = "#ff6b6b"
                badge_text = "High Connection"
                badge_bg = "#ff6b6b"
                magnetism_plain = (
                    "This stock is closely tied to the Nifty 50. Both visible day-to-day co-movement "
                    "and deeper structural links are present. It tends to amplify market swings."
                )
                diversification_note = (
                    "Adding this to a Nifty-heavy portfolio provides limited extra protection. "
                    "In a broad market sell-off, this stock is likely to decline alongside the index."
                )
            elif composite >= 0.35:
                dep_label = "Moderate Dependence"
                dep_color = "#ffa94d"
                badge_text = "Moderate Connection"
                badge_bg = "#ffa94d"
                magnetism_plain = (
                    "This stock has a meaningful connection to the Nifty 50. It doesn't mirror every move, "
                    "but during significant market swings they often move in the same direction."
                )
                diversification_note = (
                    "Provides partial diversification, but may still follow the market during sharp corrections. "
                    "Consider pairing with genuinely uncorrelated assets for better risk spread."
                )
            elif composite >= 0.18:
                dep_label = "Low Dependence"
                dep_color = "#69db7c"
                badge_text = "Low Connection"
                badge_bg = "#69db7c"
                magnetism_plain = (
                    "This stock shows limited connection to the Nifty 50. It appears to be driven "
                    "more by company-specific factors than broad market sentiment."
                )
                diversification_note = (
                    "Can contribute to portfolio diversification. Historical data suggests limited "
                    "co-movement with the index, though this is not a guarantee of future behaviour."
                )
            else:
                dep_label = "Near Independence"
                dep_color = "#a0aec0"
                badge_text = "Very Low Connection"
                badge_bg = "#10b981"
                magnetism_plain = (
                    "Very little statistical connection to the Nifty 50 was detected over the lookback window. "
                    "This stock's returns appear largely independent of market-wide movements."
                )
                diversification_note = (
                    "Among the better diversifiers against Nifty 50 exposure. Historical independence "
                    "from the index suggests it may behave differently during market stress, though "
                    "correlations can spike in extreme sell-offs."
                )

            # --- Mirror Test (correlation vs HSIC divergence) ---------------
            if dep_score >= 0.40 and abs_corr < 0.35:
                hidden_sync = True
                mirror_verdict = "Hidden Sync Detected"
                mirror_color = "#ff6b6b"
                mirror_explain = (
                    f"The Mirror Test shows a low surface-level correlation ({correlation:+.2f}), "
                    f"but the Magnetism Score is elevated ({dep_score:.0%}). This suggests non-linear "
                    "links: for example, the stock and index may diverge on normal days but move "
                    "together during tail events. Exercise caution when relying on this stock as a hedge."
                )
            elif abs_corr >= 0.5 and dep_score >= 0.30:
                hidden_sync = False
                mirror_verdict = "Confirmed Co-Movement"
                mirror_color = "#ffa94d"
                mirror_explain = (
                    f"Both linear correlation ({correlation:+.2f}) and the Magnetism Score ({dep_score:.0%}) "
                    "indicate meaningful co-movement with the market. The connection is straightforward "
                    "and visible in daily returns. No hidden surprises, but limited hedging value."
                )
            elif abs_corr >= 0.5 and dep_score < 0.30:
                hidden_sync = False
                mirror_verdict = "Linear Co-Movement"
                mirror_color = "#ffa94d"
                mirror_explain = (
                    f"Daily returns show noticeable correlation ({correlation:+.2f}), though the deeper "
                    f"kernel analysis ({dep_score:.0%}) is lower. The relationship appears mostly linear "
                    ": the stock tracks the market's direction but without complex non-linear coupling."
                )
            elif abs_corr < 0.3 and dep_score < 0.25:
                hidden_sync = False
                mirror_verdict = "Independent"
                mirror_color = "#10b981"
                mirror_explain = (
                    f"Both correlation ({correlation:+.2f}) and magnetism ({dep_score:.0%}) are low. "
                    "This stock appears largely independent of Nifty 50 over the analysis window. "
                    "Note that independence can break down during extreme market events."
                )
            else:
                hidden_sync = False
                mirror_verdict = "Mixed Signals"
                mirror_color = "#a0aec0"
                mirror_explain = (
                    f"Correlation ({correlation:+.2f}) and magnetism ({dep_score:.0%}) tell slightly "
                    "different stories. The relationship may be context-dependent, potentially stronger during "
                    "certain market regimes. Treat diversification claims with caution."
                )

            # --- Trading insight (hedged language, no false certainty) ------
            if composite >= 0.55:
                if downside_beta > 1.2:
                    trading_insight = f"HIGH MARKET EXPOSURE: Composite score {composite:.0%} with downside beta {downside_beta:.2f}. This stock has historically amplified market losses."
                else:
                    trading_insight = f"SIGNIFICANT MARKET LINK: Composite score {composite:.0%}. This stock tends to track Nifty 50 closely in both up and down markets."
            elif composite >= 0.35:
                if hidden_sync:
                    trading_insight = f"CAUTION, HIDDEN LINK: Despite moderate surface correlation, non-linear analysis ({dep_score:.0%}) reveals deeper coupling. Hedging effectiveness may be limited."
                else:
                    trading_insight = f"MODERATE MARKET LINK: Composite score {composite:.0%}. Partial co-movement with Nifty 50. Some diversification value, but not a reliable hedge."
            elif composite >= 0.18:
                trading_insight = f"LIMITED MARKET LINK: Composite score {composite:.0%}. Historically shows some independence from Nifty 50, though this may not hold in all market conditions."
            else:
                trading_insight = f"HISTORICALLY INDEPENDENT: Composite score {composite:.0%}. Low connection to Nifty 50 over the lookback window. Past independence does not guarantee future behaviour."

            # --- Plot generation (scatter only, no trend line) -------------
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 6))

            bg_color = '#0a0c12'
            grid_color = '#1a2030'

            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)

            ax.scatter(X_win * 100, y_win * 100, alpha=0.6, c='#C9A84C',
                       edgecolors='none', s=50, label='Daily Returns', zorder=1)

            stats_text = (
                f"$\\bf{{HSIC\\ Analysis}}$\n"
                f"• Magnetism: {dep_score:.4f}\n"
                f"• Correlation: {correlation:+.4f}\n"
                f"• Beta: {beta:.3f}  |  Down-Beta: {downside_beta:.3f}\n"
                f"• Composite: {composite:.4f}  |  Days: {n_used}"
            )

            ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=10, color='white',
                    verticalalignment='top', bbox=dict(facecolor=bg_color, alpha=0.95, edgecolor=grid_color, boxstyle='round,pad=0.5'), zorder=3)

            ax.set_title(f'{stock_symbol} vs {nifty_source} | HSIC Dependency Analysis', fontsize=14, color='white', pad=15)
            ax.set_xlabel(f'{nifty_source} Returns (%)', fontsize=12, color='#a0aec0')
            ax.set_ylabel(f'{stock_symbol} Returns (%)', fontsize=12, color='#a0aec0')
            ax.grid(True, linestyle='--', alpha=0.2, color=grid_color)

            ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', loc='upper right')

            for spine in ax.spines.values():
                spine.set_edgecolor(grid_color)

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
            img_buf.seek(0)
            plot_url = base64.b64encode(img_buf.getvalue()).decode()
            plt.close(fig)

            result = {
                'dependency_score': round(dep_score, 4),
                'composite_score': composite,
                'dependency_label': dep_label,
                'dependency_color': dep_color,
                'badge_text': badge_text,
                'badge_bg': badge_bg,
                'magnetism_plain': magnetism_plain,
                'diversification_note': diversification_note,
                'raw_hsic': round(raw_hsic, 8),
                'correlation': correlation,
                'beta': beta,
                'downside_beta': downside_beta,
                'mirror_verdict': mirror_verdict,
                'mirror_color': mirror_color,
                'mirror_explain': mirror_explain,
                'hidden_sync': hidden_sync,
                'data_points': n_used,
                'market_source': nifty_source,
                'trading_insight': trading_insight,
                'plot_url': plot_url
            }
            REGRESSION_CACHE.set(stock_symbol, result)
            del stock_data, nifty_data, rets
            gc.collect()
            return result

        except Exception as e:
            print(f"\n[FATAL ERROR] HSIC analysis failed for {stock_symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def fetch_dividend_data(self, symbols, limit_results=True, exclude_downtrend=True):
        """Fetch FY-based dividend yield, current price, and volatility for given symbols.

        Annual dividend uses a *sustainable* estimate: the last two completed Indian
        financial years (April 1 → March 31) are compared.  If the most recent FY's
        dividend is ≥ 2× the previous FY (indicating a one-time special dividend),
        the previous FY figure is used instead.  This prevents stocks like CHENNPETRO
        or Vedanta from appearing with inflated yields after a one-off payout.

        Stocks whose current price is below their 200-day moving average (long-term
        downtrend) are excluded from results by default so the dividend optimiser never
        recommends stocks in structural decline.  Pass exclude_downtrend=False to
        retrieve data for all stocks regardless (used by the single-stock verdict tab).
        """
        results = []
        dividend_found = 0

        if not symbols:
            return results, dividend_found

        # Resolve FY window once for this call
        fy_lbl = _fy_label(0)

        def _batched(iterable, size):
            for idx in range(0, len(iterable), size):
                yield iterable[idx:idx + size]

        for batch in _batched(symbols, 75):
            tickers = [f"{symbol}.NS" for symbol in batch]
            try:
                # 3y: covers last 2 complete FYs for sustainable-dividend comparison
                # AND provides enough history (~500 days) for a reliable 200-DMA
                data = yf.download(
                    tickers=tickers,
                    period='3y',
                    interval='1d',
                    group_by='column',
                    actions=True,
                    auto_adjust=False,
                    progress=False,
                    threads=False
                )
            except Exception:
                data = None

            for symbol in batch:
                try:
                    if data is None or data.empty:
                        continue
                    ticker_symbol = f"{symbol}.NS"
                    if isinstance(data.columns, pd.MultiIndex):
                        close_series = data['Close'][ticker_symbol].dropna()
                        dividends = data['Dividends'][ticker_symbol].dropna() if 'Dividends' in data.columns.get_level_values(0) else pd.Series(dtype=float)
                    else:
                        close_series = data['Close'].dropna()
                        dividends = data['Dividends'].dropna() if 'Dividends' in data.columns else pd.Series(dtype=float)
                    if close_series.empty or len(close_series) < 10:
                        continue
                    # Use latest close as current price (for yield denominator)
                    current_price = float(close_series.iloc[-1])
                    if current_price <= 0:
                        continue

                    # ── Long-term trend: 200-DMA check ──────────────────────────────
                    sma200_val = close_series.rolling(min(200, len(close_series))).mean().iloc[-1]
                    in_downtrend = bool(current_price < float(sma200_val)) if not pd.isna(sma200_val) else False
                    if exclude_downtrend and in_downtrend:
                        continue  # never recommend stocks in structural decline

                    # ── Sustainable dividend: compare last 2 FYs ─────────────────────
                    annual_dividend, latest_fy_div, prev_fy_div, was_capped, fy_count = \
                        _compute_sustainable_dividend(dividends, current_price)
                    if annual_dividend <= 0:
                        continue

                    dividend_yield = (annual_dividend / current_price) * 100
                    returns = close_series.pct_change().dropna()
                    volatility = float(returns.std() * np.sqrt(252) * 100) if len(returns) > 5 else 0.0

                    dividend_found += 1

                    results.append({
                        'symbol':          symbol,
                        'price':           round(current_price, 2),
                        'annual_dividend': round(annual_dividend, 2),
                        'dividend_yield':  round(dividend_yield, 2),
                        'volatility':      round(volatility, 2),
                        'fy_label':        fy_lbl,
                        'in_downtrend':    in_downtrend,
                        'latest_fy_dividend': round(latest_fy_div, 2),
                        'prev_fy_dividend':   round(prev_fy_div, 2),
                        'yield_capped':       was_capped,
                        'fy_count':           fy_count,
                    })
                except Exception:
                    continue

        results = sorted(results, key=lambda x: x['dividend_yield'], reverse=True)
        if limit_results and len(results) > DIVIDEND_MAX_RESULTS:
            results = results[:DIVIDEND_MAX_RESULTS]

        return results, dividend_found

    def optimize_dividend_portfolio(self, stocks_data, capital, risk_appetite):
        """Compute optimal portfolio allocation to maximize dividend income."""
        if not stocks_data or capital <= 0:
            return None

        n = len(stocks_data)
        yields_arr = np.array([s['dividend_yield'] for s in stocks_data])
        vols_arr = np.array([s['volatility'] for s in stocks_data])
        fy_counts = np.array([s.get('fy_count', 1) for s in stocks_data])

        params = {
            'conservative': {'max_weight': 0.08, 'vol_penalty': 0.15, 'min_yield': 1.0, 'min_fy_count': 2},
            'moderate':     {'max_weight': 0.15, 'vol_penalty': 0.05, 'min_yield': 0.5, 'min_fy_count': 2},
            'aggressive':   {'max_weight': 0.30, 'vol_penalty': 0.01, 'min_yield': 0.0, 'min_fy_count': 2}
        }
        p = params.get(risk_appetite, params['moderate'])

        valid = [i for i in range(n)
                 if yields_arr[i] >= p['min_yield'] and fy_counts[i] >= p['min_fy_count']]
        if len(valid) < 3:
            # Relax consistency requirement if not enough stocks qualify
            valid = [i for i in range(n) if yields_arr[i] >= p['min_yield']]
        if len(valid) < 3:
            valid = list(range(min(n, 10)))

        m = len(valid)
        v_yields = yields_arr[valid]
        v_vols = vols_arr[valid]
        v_stocks = [stocks_data[i] for i in valid]

        def neg_objective(w):
            port_yield = np.dot(w, v_yields)
            port_vol = np.sqrt(np.sum((w ** 2) * (v_vols ** 2)))
            return -(port_yield - p['vol_penalty'] * port_vol)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0, p['max_weight']) for _ in range(m)]
        w0 = np.ones(m) / m

        result = scipy_minimize(neg_objective, w0, method='SLSQP', bounds=bounds,
                                constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-10})

        if result.success:
            weights = result.x
        else:
            weights = v_yields / v_yields.sum()
            weights = np.minimum(weights, p['max_weight'])
            weights = weights / weights.sum()

        allocation = []
        total_dividend = 0
        total_invested = 0

        for i, w in enumerate(weights):
            if w < 0.005:
                continue
            stock = v_stocks[i]
            target_amount = capital * w
            shares = int(target_amount / stock['price']) if stock['price'] > 0 else 0
            if shares == 0:
                continue
            actual_amount = shares * stock['price']
            expected_div = actual_amount * stock['dividend_yield'] / 100
            total_dividend += expected_div
            total_invested += actual_amount
            allocation_entry = {
                'symbol': stock['symbol'],
                'weight': 0.0,
                'shares': shares,
                'amount': round(actual_amount, 2),
                'price': stock['price'],
                'dividend_yield': stock['dividend_yield'],
                'expected_dividend': round(expected_div, 2),
                'volatility': stock['volatility']
            }
            allocation.append(allocation_entry)

        remaining = capital - total_invested
        if remaining > 0 and allocation:
            ranked = sorted(allocation, key=lambda x: x['dividend_yield'], reverse=True)
            min_price = min(a['price'] for a in ranked if a['price'] > 0)
            for stock in ranked:
                if remaining < min_price:
                    break
                max_amount = capital * p['max_weight']
                current_amount = stock['amount']
                room = max_amount - current_amount
                if room <= 0:
                    continue
                buyable = int(min(remaining, room) / stock['price'])
                if buyable <= 0:
                    continue
                add_amount = buyable * stock['price']
                stock['shares'] += buyable
                stock['amount'] = round(current_amount + add_amount, 2)
                add_div = add_amount * stock['dividend_yield'] / 100
                stock['expected_dividend'] = round(stock['expected_dividend'] + add_div, 2)
                total_dividend += add_div
                total_invested += add_amount
                remaining -= add_amount

        for stock in allocation:
            stock['weight'] = round((stock['amount'] / capital) * 100, 2)

        allocation.sort(key=lambda x: x['expected_dividend'], reverse=True)
        portfolio_yield = (total_dividend / total_invested * 100) if total_invested > 0 else 0

        return {
            'allocation': allocation,
            'total_invested': round(total_invested, 2),
            'capital': capital,
            'unallocated': round(capital - total_invested, 2),
            'total_expected_dividend': round(total_dividend, 2),
            'portfolio_yield': round(portfolio_yield, 2),
            'num_stocks': len(allocation),
            'risk_appetite': risk_appetite
        }

    def dcf_valuation(self, symbol):
        """Fetch financial data needed for DCF valuation from yfinance."""
        try:
            base_symbols = [symbol] + YAHOO_TICKER_ALIASES.get(symbol, [])
            ticker_obj = None
            info = {}
            for base in base_symbols:
                for suffix in ['.NS', '.BO']:
                    try:
                        t = yf.Ticker(f"{base}{suffix}")
                        i = t.info
                        if i and (i.get('regularMarketPrice') or i.get('currentPrice')):
                            ticker_obj = t
                            info = i
                            break
                    except Exception:
                        continue
                if ticker_obj:
                    break

            if not ticker_obj or not info:
                return None

            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                return None

            shares = (info.get('sharesOutstanding') or
                      info.get('impliedSharesOutstanding') or
                      info.get('floatShares'))
            # Fallback: derive shares from market cap / price
            if not shares or shares <= 0:
                mktcap = info.get('marketCap')
                if mktcap and current_price and current_price > 0:
                    shares = mktcap / current_price
            if not shares or shares <= 0:
                return None

            # --- Parse Cash Flow Statement ---
            current_fcf = None
            historical_growth = None
            fcf_history = []
            try:
                cashflow = ticker_obj.cashflow
                if cashflow is not None and not cashflow.empty:
                    ocf = None
                    for row_name in ['Operating Cash Flow', 'Total Cash From Operating Activities',
                                     'Cash From Operating Activities', 'Cash Flow From Operations',
                                     'Net Cash Provided By Operating Activities']:
                        if row_name in cashflow.index:
                            ocf = cashflow.loc[row_name]
                            break

                    capex = None
                    for row_name in ['Capital Expenditure', 'Capital Expenditures',
                                     'Purchase Of Property Plant And Equipment',
                                     'Capital Expenditures Reported', 'Purchases Of Property']:
                        if row_name in cashflow.index:
                            capex = cashflow.loc[row_name]
                            break

                    if ocf is not None:
                        fcf_row = (ocf + capex) if capex is not None else ocf
                        fcf_row = fcf_row.dropna()
                        if not fcf_row.empty:
                            # oldest → newest chronological order
                            fcf_vals_chrono = list(reversed(fcf_row.values.tolist()))
                            dates_chrono = list(reversed(fcf_row.index.tolist()))
                            for val, dt in zip(fcf_vals_chrono, dates_chrono):
                                if pd.notna(val):
                                    year = dt.year if hasattr(dt, 'year') else str(dt)[:4]
                                    fcf_history.append({'year': int(year), 'fcf': int(round(float(val)))})

                            # Most recent FCF (first element, newest)
                            latest_vals = [v for v in fcf_row.values if pd.notna(v)]
                            if latest_vals:
                                current_fcf = float(latest_vals[0])

                            # CAGR from positive historical FCF values
                            pos_vals = [v for v in fcf_vals_chrono if pd.notna(v) and v > 0]
                            if len(pos_vals) >= 2:
                                periods = len(pos_vals) - 1
                                historical_growth = (pos_vals[-1] / pos_vals[0]) ** (1.0 / periods) - 1
            except Exception as e:
                print(f"DCF cashflow parse error for {symbol}: {e}")

            # Fallback chain when FCF from cashflow statement is unavailable/negative
            if current_fcf is None or current_fcf <= 0:
                # 1. freeCashflow field on info dict (yfinance pre-computes this)
                info_fcf = info.get('freeCashflow')
                if info_fcf and float(info_fcf) > 0:
                    current_fcf = float(info_fcf)
                    historical_growth = info.get('earningsGrowth') or info.get('revenueGrowth') or 0.10
                    fcf_history = []
                else:
                    # 2. operatingCashflow from info dict (common for capital-heavy sectors)
                    info_ocf = info.get('operatingCashflow')
                    if info_ocf and float(info_ocf) > 0:
                        current_fcf = float(info_ocf) * 0.75  # conservative capex haircut
                        historical_growth = info.get('revenueGrowth') or 0.08
                        fcf_history = []
                    else:
                        # 3. EBITDA proxy (ebitda * ~0.5 ≈ levered FCF for heavy industries)
                        ebitda = info.get('ebitda')
                        if ebitda and float(ebitda) > 0:
                            current_fcf = float(ebitda) * 0.50
                            historical_growth = info.get('revenueGrowth') or 0.08
                            fcf_history = []
                        else:
                            # 4. EPS proxy (trailingEps then forwardEps)
                            eps = info.get('trailingEps') or info.get('forwardEps')
                            if eps and float(eps) > 0:
                                current_fcf = float(eps) * float(shares)
                                historical_growth = info.get('earningsGrowth') or 0.10
                                fcf_history = []
                            else:
                                return None

            # --- Parse Balance Sheet for Debt/Cash ---
            total_debt = 0.0
            cash = 0.0
            balance_sheet = None  # initialise so RoCE block can safely reference it
            try:
                balance_sheet = ticker_obj.balance_sheet
                if balance_sheet is not None and not balance_sheet.empty:
                    for row_name in ['Total Debt', 'Long Term Debt', 'Long-Term Debt',
                                     'Short And Long Term Debt', 'Total Long Term Debt',
                                     'Net Debt']:
                        if row_name in balance_sheet.index:
                            val = balance_sheet.loc[row_name].iloc[0]
                            if pd.notna(val) and float(val) > 0:
                                total_debt = float(val)
                                break
                    for row_name in ['Cash And Cash Equivalents', 'Cash',
                                     'Cash And Short Term Investments',
                                     'Cash Cash Equivalents And Short Term Investments',
                                     'Cash And Cash Equivalents And Short Term Investments']:
                        if row_name in balance_sheet.index:
                            val = balance_sheet.loc[row_name].iloc[0]
                            if pd.notna(val):
                                cash = float(val)
                                break
            except Exception as e:
                print(f"DCF balance sheet parse error for {symbol}: {e}")

            # --- Compute RoCE (Return on Capital Employed) ---
            roce = None
            margin_trend = []
            try:
                income_stmt = ticker_obj.income_stmt
                if income_stmt is not None and not income_stmt.empty:
                    ebit_val = None
                    for row_name in ['EBIT', 'Operating Income', 'Operating Income Or Loss',
                                     'Normalized EBITDA', 'Pretax Income']:
                        if row_name in income_stmt.index:
                            v = income_stmt.loc[row_name].iloc[0]
                            if pd.notna(v):
                                ebit_val = float(v)
                                break
                    if ebit_val is not None and balance_sheet is not None and not balance_sheet.empty:
                        total_assets = None
                        curr_liabilities = None
                        for row_name in ['Total Assets', 'Total Assets Net Minority Interest']:
                            if row_name in balance_sheet.index:
                                v = balance_sheet.loc[row_name].iloc[0]
                                if pd.notna(v):
                                    total_assets = float(v)
                                    break
                        for row_name in ['Current Liabilities', 'Total Current Liabilities',
                                         'Current Liabilities Net Minority Interest']:
                            if row_name in balance_sheet.index:
                                v = balance_sheet.loc[row_name].iloc[0]
                                if pd.notna(v):
                                    curr_liabilities = float(v)
                                    break
                        if total_assets is not None and total_assets > 0:
                            cap_employed = total_assets - (curr_liabilities or 0)
                            if cap_employed > 0:
                                roce = round(ebit_val / cap_employed, 4)
                    # --- Margin and Revenue trend (all available years) ---
                    revenue_row = None
                    op_income_row = None
                    net_income_row = None
                    for r in ['Total Revenue', 'Revenue', 'Net Revenue',
                              'Operating Revenue', 'Gross Revenue']:
                        if r in income_stmt.index:
                            revenue_row = income_stmt.loc[r]
                            break
                    for r in ['Operating Income', 'EBIT', 'Operating Income Or Loss',
                              'Normalized EBITDA']:
                        if r in income_stmt.index:
                            op_income_row = income_stmt.loc[r]
                            break
                    for r in ['Net Income', 'Net Income From Continuing Operations',
                              'Net Income Common Stockholders', 'Normalized Income']:
                        if r in income_stmt.index:
                            net_income_row = income_stmt.loc[r]
                            break
                    if revenue_row is not None:
                        for col in revenue_row.dropna().index:
                            yr = col.year if hasattr(col, 'year') else int(str(col)[:4])
                            rev = float(revenue_row[col]) if pd.notna(revenue_row[col]) else None
                            op = float(op_income_row[col]) if (op_income_row is not None and col in op_income_row.index and pd.notna(op_income_row[col])) else None
                            net = float(net_income_row[col]) if (net_income_row is not None and col in net_income_row.index and pd.notna(net_income_row[col])) else None
                            if rev and rev > 0:
                                margin_trend.append({
                                    'year': yr,
                                    'revenue': int(round(rev)),
                                    'op_margin': round(op / rev * 100, 1) if op is not None else None,
                                    'net_margin': round(net / rev * 100, 1) if net is not None else None,
                                })
                        margin_trend.sort(key=lambda x: x['year'])
            except Exception as e:
                print(f"DCF income stmt parse error for {symbol}: {e}")

            # Suggested growth rate: clamp historical CAGR to [3%, 40%]
            if historical_growth is not None:
                suggested_growth = min(max(float(historical_growth), 0.03), 0.40)
            else:
                analyst_growth = (info.get('earningsGrowth') or
                                  info.get('revenueGrowth') or 0.10)
                suggested_growth = min(max(float(analyst_growth), 0.03), 0.40)

            return {
                'symbol': symbol,
                'name': info.get('longName') or info.get('shortName') or symbol,
                'current_price': round(float(current_price), 2),
                'current_fcf': int(round(float(current_fcf))),
                'shares_outstanding': int(round(float(shares))),
                'total_debt': int(round(float(total_debt))),
                'cash': int(round(float(cash))),
                'historical_fcf_growth': round(float(historical_growth), 4) if historical_growth is not None else None,
                'suggested_growth_rate': round(float(suggested_growth), 4),
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector') or '',
                'industry': info.get('industry') or '',
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'ev_ebitda': info.get('enterpriseToEbitda'),
                'roe': info.get('returnOnEquity'),
                'roce': roce,
                'margin_trend': margin_trend,
                'fcf_history': fcf_history,
            }
        except Exception as e:
            print(f"DCF valuation error for {symbol}: {e}")
            return None

    # ── Financial-sector detection (Banks / Financial Services / NBFC) ────────
    _FINANCIAL_SECTORS_SET = {'banking', 'financial services', 'nbfc'}
    _FINANCIAL_INDUSTRIES = {
        'banks—regional', 'banks—diversified', 'banks - regional',
        'banks - diversified', 'credit services', 'mortgage finance',
        'insurance—life', 'insurance—diversified', 'insurance—property & casualty',
        'insurance - life', 'insurance - diversified', 'insurance - property & casualty',
        'insurance brokers', 'capital markets', 'asset management',
        'financial data & stock exchanges', 'financial conglomerates',
        'insurance—reinsurance', 'insurance - reinsurance', 'insurance—specialty',
        'insurance - specialty', 'shell companies',
    }

    @classmethod
    def is_financial_sector(cls, sector, industry):
        """Return True if the company belongs to Banks / Financial Services / NBFC."""
        s = (sector or '').strip().lower()
        i = (industry or '').strip().lower()
        if s in cls._FINANCIAL_SECTORS_SET:
            return True
        if i in cls._FINANCIAL_INDUSTRIES:
            return True
        # Catch-all: yfinance sector 'Financial Services' or industry containing 'bank'
        if 'bank' in s or 'bank' in i:
            return True
        if 'financial' in s:
            return True
        if 'nbfc' in i or 'nbfc' in s:
            return True
        return False

    def excess_return_valuation(self, symbol):
        """Damodaran Excess Return (ROE-Book Value) model for financial firms."""
        try:
            base_symbols = [symbol] + YAHOO_TICKER_ALIASES.get(symbol, [])
            ticker_obj = None
            info = {}
            for base in base_symbols:
                for suffix in ['.NS', '.BO']:
                    try:
                        t = yf.Ticker(f"{base}{suffix}")
                        i = t.info
                        if i and (i.get('regularMarketPrice') or i.get('currentPrice')):
                            ticker_obj = t
                            info = i
                            break
                    except Exception:
                        continue
                if ticker_obj:
                    break

            if not ticker_obj or not info:
                return None

            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                return None

            shares = (info.get('sharesOutstanding') or
                      info.get('impliedSharesOutstanding') or
                      info.get('floatShares'))
            if not shares or shares <= 0:
                mktcap = info.get('marketCap')
                if mktcap and current_price and current_price > 0:
                    shares = mktcap / current_price
            if not shares or shares <= 0:
                return None

            # --- Book Value Per Share ---
            book_value_per_share = info.get('bookValue')
            total_book_value = None

            balance_sheet = None
            try:
                balance_sheet = ticker_obj.balance_sheet
                if balance_sheet is not None and not balance_sheet.empty:
                    for row_name in ['Stockholders Equity', 'Total Stockholder Equity',
                                     'Common Stock Equity', 'Total Equity Gross Minority Interest',
                                     'Stockholders\' Equity', 'Tangible Book Value',
                                     'Net Tangible Assets']:
                        if row_name in balance_sheet.index:
                            val = balance_sheet.loc[row_name].iloc[0]
                            if pd.notna(val) and float(val) > 0:
                                total_book_value = float(val)
                                break
            except Exception as e:
                print(f"Excess-return balance sheet parse error for {symbol}: {e}")

            if not book_value_per_share or book_value_per_share <= 0:
                if total_book_value and total_book_value > 0:
                    book_value_per_share = total_book_value / shares
                else:
                    return None

            if not total_book_value or total_book_value <= 0:
                total_book_value = book_value_per_share * shares

            # --- ROE (Return on Equity) ---
            roe = info.get('returnOnEquity')

            # Fallback: compute ROE from Net Income / Equity
            net_income = None
            roe_history = []
            try:
                income_stmt = ticker_obj.income_stmt
                if income_stmt is not None and not income_stmt.empty:
                    ni_row = None
                    for row_name in ['Net Income', 'Net Income From Continuing Operations',
                                     'Net Income Common Stockholders', 'Normalized Income']:
                        if row_name in income_stmt.index:
                            ni_row = income_stmt.loc[row_name]
                            break
                    if ni_row is not None:
                        latest_ni = ni_row.dropna()
                        if not latest_ni.empty:
                            net_income = float(latest_ni.iloc[0])
                            if roe is None and total_book_value and total_book_value > 0:
                                roe = net_income / total_book_value

                            # Build ROE history from available years
                            equity_row = None
                            if balance_sheet is not None and not balance_sheet.empty:
                                for row_name in ['Stockholders Equity', 'Total Stockholder Equity',
                                                 'Common Stock Equity', 'Total Equity Gross Minority Interest']:
                                    if row_name in balance_sheet.index:
                                        equity_row = balance_sheet.loc[row_name]
                                        break

                            ni_vals = ni_row.dropna()
                            for col in ni_vals.index:
                                yr = col.year if hasattr(col, 'year') else int(str(col)[:4])
                                ni_val = float(ni_vals[col])
                                eq_val = None
                                if equity_row is not None and col in equity_row.index and pd.notna(equity_row[col]):
                                    eq_val = float(equity_row[col])
                                if eq_val and eq_val > 0:
                                    roe_history.append({'year': yr, 'roe': round(ni_val / eq_val, 4),
                                                       'net_income': int(round(ni_val)),
                                                       'equity': int(round(eq_val))})
                            roe_history.sort(key=lambda x: x['year'])
            except Exception as e:
                print(f"Excess-return income stmt parse error for {symbol}: {e}")

            if roe is None or roe <= 0:
                # Cannot run excess return model without a positive ROE
                return None

            # --- Cost of Equity (Ke) ---
            # Use CAPM: Rf + beta * (Rm - Rf)
            # Defaults: Rf=7% (India 10-yr govt bond), market premium=6%
            beta = info.get('beta') or 1.0
            risk_free_rate = 0.07
            market_premium = 0.06
            cost_of_equity = risk_free_rate + float(beta) * market_premium

            # --- Book Value history ---
            bv_history = []
            try:
                if balance_sheet is not None and not balance_sheet.empty:
                    equity_row = None
                    for row_name in ['Stockholders Equity', 'Total Stockholder Equity',
                                     'Common Stock Equity', 'Total Equity Gross Minority Interest']:
                        if row_name in balance_sheet.index:
                            equity_row = balance_sheet.loc[row_name]
                            break
                    if equity_row is not None:
                        eq_vals = equity_row.dropna()
                        for col in eq_vals.index:
                            yr = col.year if hasattr(col, 'year') else int(str(col)[:4])
                            eq_val = float(eq_vals[col])
                            bvps = eq_val / shares if shares > 0 else 0
                            bv_history.append({'year': yr, 'equity': int(round(eq_val)),
                                               'bvps': round(bvps, 2)})
                        bv_history.sort(key=lambda x: x['year'])
            except Exception as e:
                print(f"Excess-return BV history parse error for {symbol}: {e}")

            # --- Suggested growth rate for book value ---
            bv_growth = None
            if len(bv_history) >= 2:
                first_eq = bv_history[0]['equity']
                last_eq = bv_history[-1]['equity']
                if first_eq > 0 and last_eq > 0:
                    periods = len(bv_history) - 1
                    bv_growth = (last_eq / first_eq) ** (1.0 / periods) - 1

            suggested_growth = None
            if bv_growth is not None:
                suggested_growth = min(max(float(bv_growth), 0.03), 0.30)
            else:
                analyst_growth = (info.get('earningsGrowth') or
                                  info.get('revenueGrowth') or 0.10)
                suggested_growth = min(max(float(analyst_growth), 0.03), 0.30)

            # --- Margin trend (reuse same logic as DCF) ---
            margin_trend = []
            try:
                income_stmt = ticker_obj.income_stmt
                if income_stmt is not None and not income_stmt.empty:
                    revenue_row = None
                    op_income_row = None
                    net_income_row = None
                    for r in ['Total Revenue', 'Revenue', 'Net Revenue',
                              'Operating Revenue', 'Gross Revenue',
                              'Interest Income', 'Net Interest Income',
                              'Total Interest Income']:
                        if r in income_stmt.index:
                            revenue_row = income_stmt.loc[r]
                            break
                    for r in ['Operating Income', 'EBIT', 'Operating Income Or Loss']:
                        if r in income_stmt.index:
                            op_income_row = income_stmt.loc[r]
                            break
                    for r in ['Net Income', 'Net Income From Continuing Operations',
                              'Net Income Common Stockholders']:
                        if r in income_stmt.index:
                            net_income_row = income_stmt.loc[r]
                            break
                    if revenue_row is not None:
                        for col in revenue_row.dropna().index:
                            yr = col.year if hasattr(col, 'year') else int(str(col)[:4])
                            rev = float(revenue_row[col]) if pd.notna(revenue_row[col]) else None
                            op = float(op_income_row[col]) if (op_income_row is not None and col in op_income_row.index and pd.notna(op_income_row[col])) else None
                            net = float(net_income_row[col]) if (net_income_row is not None and col in net_income_row.index and pd.notna(net_income_row[col])) else None
                            if rev and rev > 0:
                                margin_trend.append({
                                    'year': yr,
                                    'revenue': int(round(rev)),
                                    'op_margin': round(op / rev * 100, 1) if op is not None else None,
                                    'net_margin': round(net / rev * 100, 1) if net is not None else None,
                                })
                        margin_trend.sort(key=lambda x: x['year'])
            except Exception as e:
                print(f"Excess-return margin trend parse error for {symbol}: {e}")

            return {
                'valuation_model': 'excess_return',
                'symbol': symbol,
                'name': info.get('longName') or info.get('shortName') or symbol,
                'current_price': round(float(current_price), 2),
                'shares_outstanding': int(round(float(shares))),
                'book_value_per_share': round(float(book_value_per_share), 2),
                'total_book_value': int(round(float(total_book_value))),
                'roe': round(float(roe), 4),
                'cost_of_equity': round(float(cost_of_equity), 4),
                'beta': round(float(beta), 2),
                'suggested_growth_rate': round(float(suggested_growth), 4),
                'bv_growth': round(float(bv_growth), 4) if bv_growth is not None else None,
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector') or '',
                'industry': info.get('industry') or '',
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'net_income': int(round(float(net_income))) if net_income else None,
                'roe_history': roe_history,
                'bv_history': bv_history,
                'margin_trend': margin_trend,
                # Include these for verdict compatibility
                'current_fcf': None,
                'total_debt': 0,
                'cash': 0,
                'historical_fcf_growth': None,
                'roce': None,
                'ev_ebitda': None,
                'fcf_history': [],
            }
        except Exception as e:
            print(f"Excess return valuation error for {symbol}: {e}")
            return None


analyzer = Analyzer()

# ── Stock Alert Monitor (background thread) ───────────────────────────────────
alert_monitor = StockAlertMonitor(analyzer)
# Auto-start if a watchlist and email credentials are already set via env vars
if alert_monitor.config.get("watchlist") and alert_monitor.config.get("smtp_user"):
    alert_monitor.start()

# ===== HTML TEMPLATE (IDENTICAL TO WORKING VERSION) =====
# [Keeping your exact HTML - no changes needed]

# Pre-serialize ticker names JSON for the API endpoint (done once at startup)
_TICKER_NAMES_JSON = json.dumps(TICKER_TO_NAME, separators=(',', ':'))

@app.route('/api/ticker-names')
def ticker_names_api():
    resp = make_response(_TICKER_NAMES_JSON)
    resp.headers['Content-Type'] = 'application/json'
    resp.headers['Cache-Control'] = 'public, max-age=86400'
    return resp

@app.route('/')
def landing():
    html = '''<!DOCTYPE html>
<html lang="en-IN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Institutional-grade fundamental and technical analysis on 500+ NSE-listed securities. Fair value estimates, risk profiling, and actionable verdicts.">
    <title>Stock Analysis Pro — Equity Research & Analysis Platform</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400;1,700&family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <script>
    // Landing-page mobile-panel toggle. Loaded in <head> with document-level
    // event delegation so it works regardless of DOM-ready timing.
    (function () {
        function getEl(id) { return document.getElementById(id); }
        function setState(open) {
            var ham = getEl('hamburger');
            var mob = getEl('mob-panel');
            if (ham) { ham.classList.toggle('open', open); ham.setAttribute('aria-expanded', open ? 'true' : 'false'); }
            if (mob) mob.classList.toggle('open', open);
            document.body.style.overflow = open ? 'hidden' : '';
        }
        function toggle() {
            var mob = getEl('mob-panel');
            if (!mob) return;
            setState(!mob.classList.contains('open'));
        }
        function close() { setState(false); }
        document.addEventListener('click', function (e) {
            if (e.target.closest && e.target.closest('#hamburger')) {
                e.preventDefault(); toggle(); return;
            }
            if (e.target.closest && e.target.closest('#mob-panel a')) {
                close();
            }
        }, false);
        window.toggleMobPanel = toggle;
        window.closeMob = close;
    })();
    </script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root { --bg: #0d0f14; --bg-card: #131822; --bg-section: #0f1118; --gold: #C9A84C; --gold-light: rgba(201,168,76,0.15); --gold-border: rgba(201,168,76,0.25); --text: #E8EDF2; --text-sec: #90A4BE; --text-muted: #607B96; --border: #1e2535; }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }
        a { text-decoration: none; color: inherit; }

        /* NAV */
        nav { position: sticky; top: 0; z-index: 100; background: var(--bg); border-bottom: 1px solid var(--border); }
        .nav-inner { max-width: 1200px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; height: 64px; padding: 0 32px; }
        .brand { display: flex; align-items: center; gap: 10px; font-family: 'Space Grotesk', sans-serif; font-size: 1.1em; font-weight: 700; }
        .brand-icon { width: 36px; height: 36px; background: var(--gold); border-radius: 8px; display: flex; align-items: center; justify-content: center; }
        .brand-icon svg { width: 20px; height: 20px; }
        .nav-links { display: flex; align-items: center; gap: 6px; height: 100%; }
        .nav-link-item { position: relative; height: 100%; display: flex; align-items: center; }
        .nav-link-item > a, .nav-link-item > button { color: var(--text-sec); font-size: 0.88em; font-weight: 500; transition: color 0.2s; padding: 8px 14px; border-radius: 6px; background: none; border: none; cursor: pointer; font-family: 'Inter', sans-serif; display: flex; align-items: center; gap: 5px; }
        .nav-link-item > a:hover, .nav-link-item > button:hover { color: var(--text); background: rgba(255,255,255,0.04); }
        .nav-link-item > button svg { width: 12px; height: 12px; stroke: currentColor; fill: none; stroke-width: 2; transition: transform 0.2s; }
        .nav-link-item:hover > button svg { transform: rotate(180deg); }
        .nav-cta { padding: 10px 24px; background: transparent; border: 1.5px solid var(--gold) !important; color: var(--gold) !important; border-radius: 6px !important; font-weight: 600 !important; font-size: 0.88em; transition: all 0.25s; font-family: 'Space Grotesk', sans-serif !important; }
        .nav-cta:hover { background: var(--gold) !important; color: var(--bg) !important; }

        /* MEGA DROPDOWN */
        .mega-drop { position: absolute; top: 100%; left: 50%; transform: translateX(-50%); width: 640px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; padding: 8px; opacity: 0; visibility: hidden; transition: opacity 0.2s, visibility 0.2s, transform 0.2s; transform: translateX(-50%) translateY(8px); box-shadow: 0 20px 60px rgba(0,0,0,0.5); pointer-events: none; }
        .nav-link-item:hover .mega-drop { opacity: 1; visibility: visible; transform: translateX(-50%) translateY(0); pointer-events: auto; }
        .mega-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
        .mega-item { display: flex; gap: 14px; padding: 16px 18px; border-radius: 10px; transition: background 0.15s; }
        .mega-item:hover { background: rgba(255,255,255,0.04); }
        .mega-item-icon { flex-shrink: 0; width: 38px; height: 38px; border-radius: 8px; background: var(--gold-light); border: 1px solid var(--gold-border); display: flex; align-items: center; justify-content: center; }
        .mega-item-icon svg { width: 18px; height: 18px; stroke: var(--gold); fill: none; stroke-width: 1.5; }
        .mega-item-text h4 { font-family: 'Space Grotesk', sans-serif; font-size: 0.88em; font-weight: 700; color: var(--text); margin-bottom: 3px; }
        .mega-item-text p { font-size: 0.76em; color: var(--text-muted); line-height: 1.5; }
        .mega-footer { margin-top: 4px; padding: 14px 18px; border-top: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; border-radius: 0 0 10px 10px; }
        .mega-footer span { font-size: 0.78em; color: var(--text-muted); }
        .mega-footer a { font-size: 0.82em; font-weight: 600; color: var(--gold); display: flex; align-items: center; gap: 4px; }
        .mega-footer a:hover { text-decoration: underline; }

        /* HAMBURGER */
        .hamburger { display: none; background: none; border: 1px solid var(--border); border-radius: 8px; width: 42px; height: 42px; cursor: pointer; position: relative; }
        .hamburger span { display: block; width: 18px; height: 2px; background: var(--text); border-radius: 1px; position: absolute; left: 50%; transform: translateX(-50%); transition: all 0.3s; }
        .hamburger span:nth-child(1) { top: 13px; }
        .hamburger span:nth-child(2) { top: 20px; }
        .hamburger span:nth-child(3) { top: 27px; }
        .hamburger.open span:nth-child(1) { top: 20px; transform: translateX(-50%) rotate(45deg); }
        .hamburger.open span:nth-child(2) { opacity: 0; }
        .hamburger.open span:nth-child(3) { top: 20px; transform: translateX(-50%) rotate(-45deg); }

        /* MOBILE MENU */
        .mob-panel { display: none; position: fixed; top: 65px; left: 0; right: 0; bottom: 0; z-index: 99; background: var(--bg); overflow-y: auto; }
        .mob-panel.open { display: block; }
        .mob-section { padding: 16px 24px; border-bottom: 1px solid var(--border); }
        .mob-section-label { font-size: 0.7em; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); margin-bottom: 12px; }
        .mob-link { display: block; padding: 14px 0; color: var(--text-sec); font-size: 1em; font-weight: 500; border-bottom: 1px solid rgba(255,255,255,0.04); }
        .mob-link:last-child { border-bottom: none; }
        .mob-link:hover { color: var(--text); }
        .mob-tool { display: flex; gap: 14px; padding: 14px 0; align-items: flex-start; border-bottom: 1px solid rgba(255,255,255,0.04); }
        .mob-tool:last-child { border-bottom: none; }
        .mob-tool-icon { flex-shrink: 0; width: 36px; height: 36px; border-radius: 8px; background: var(--gold-light); border: 1px solid var(--gold-border); display: flex; align-items: center; justify-content: center; }
        .mob-tool-icon svg { width: 16px; height: 16px; stroke: var(--gold); fill: none; stroke-width: 1.5; }
        .mob-tool h4 { font-family: 'Space Grotesk', sans-serif; font-size: 0.92em; font-weight: 600; color: var(--text); }
        .mob-tool p { font-size: 0.78em; color: var(--text-muted); line-height: 1.4; margin-top: 2px; }
        .mob-cta { display: block; margin: 20px 24px; padding: 16px; text-align: center; background: var(--gold); color: var(--bg); border-radius: 10px; font-weight: 700; font-family: 'Space Grotesk', sans-serif; font-size: 1em; }

        @media (max-width: 860px) {
            .nav-links { display: none; }
            .hamburger { display: block; }
            .nav-inner { padding: 0 20px; }
        }

        /* HERO */
        .hero { text-align: center; padding: 100px 32px 80px; max-width: 900px; margin: 0 auto; }
        .hero-label { display: inline-block; font-size: 0.75em; font-weight: 600; letter-spacing: 2.5px; text-transform: uppercase; color: var(--gold); margin-bottom: 28px; }
        .hero h1 { font-family: 'Playfair Display', serif; font-size: 3.8em; font-weight: 700; line-height: 1.15; margin-bottom: 28px; letter-spacing: -0.5px; }
        .hero h1 em { font-style: italic; color: var(--gold); }
        .hero p { color: var(--text-sec); font-size: 1.1em; max-width: 600px; margin: 0 auto 40px; line-height: 1.75; }
        .hero-btns { display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; }
        .btn-primary { display: inline-flex; align-items: center; gap: 8px; padding: 16px 36px; background: var(--gold); color: var(--bg); border: none; border-radius: 8px; font-size: 1em; font-weight: 700; cursor: pointer; transition: all 0.25s; font-family: 'Space Grotesk', sans-serif; }
        .btn-primary:hover { filter: brightness(1.1); transform: translateY(-1px); box-shadow: 0 8px 30px rgba(201,168,76,0.3); }
        .btn-secondary { display: inline-flex; align-items: center; gap: 8px; padding: 16px 36px; background: transparent; color: var(--text); border: 1.5px solid var(--border); border-radius: 8px; font-size: 1em; font-weight: 600; cursor: pointer; transition: all 0.25s; font-family: 'Space Grotesk', sans-serif; }
        .btn-secondary:hover { border-color: var(--text-sec); background: rgba(255,255,255,0.03); }
        @media (max-width: 768px) {
            .hero { padding: 60px 20px 50px; }
            .hero h1 { font-size: 2.3em; }
            .hero p { font-size: 1em; }
        }

        /* STATS BAR */
        .stats { max-width: 1000px; margin: 0 auto; padding: 50px 32px; display: flex; justify-content: center; gap: 80px; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
        .stat { text-align: center; }
        .stat-val { font-family: 'Playfair Display', serif; font-size: 2.2em; font-weight: 700; color: var(--text); }
        .stat-label { font-size: 0.82em; color: var(--text-muted); margin-top: 4px; }
        @media (max-width: 768px) {
            .stats { gap: 30px; flex-wrap: wrap; padding: 30px 20px; }
            .stat-val { font-size: 1.6em; }
        }

        /* PILLARS */
        .pillars { background: var(--bg-section); padding: 90px 32px; }
        .pillars-inner { max-width: 1100px; margin: 0 auto; display: grid; grid-template-columns: repeat(3, 1fr); gap: 48px; }
        .pillar { text-align: center; }
        .pillar-icon { width: 56px; height: 56px; border-radius: 50%; border: 1.5px solid var(--border); display: flex; align-items: center; justify-content: center; margin: 0 auto 20px; }
        .pillar-icon svg { width: 24px; height: 24px; stroke: var(--gold); fill: none; stroke-width: 1.5; }
        .pillar h3 { font-family: 'Space Grotesk', sans-serif; font-size: 0.82em; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px; }
        .pillar p { color: var(--text-sec); font-size: 0.92em; line-height: 1.7; max-width: 320px; margin: 0 auto; }
        @media (max-width: 768px) {
            .pillars-inner { grid-template-columns: 1fr; gap: 40px; }
            .pillars { padding: 60px 20px; }
        }

        /* METHODOLOGY */
        .methodology { padding: 90px 32px; max-width: 1100px; margin: 0 auto; }
        .section-label { font-size: 0.75em; font-weight: 600; letter-spacing: 2.5px; text-transform: uppercase; color: var(--gold); margin-bottom: 14px; text-align: center; }
        .section-title { font-family: 'Playfair Display', serif; font-size: 2.4em; font-weight: 700; text-align: center; margin-bottom: 50px; font-style: italic; }
        .steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 32px; }
        .step { padding: 32px 28px; border-left: 2px solid var(--border); }
        .step-num { font-family: 'Space Grotesk', sans-serif; font-size: 0.85em; font-weight: 700; color: var(--gold); margin-bottom: 14px; }
        .step h3 { font-family: 'Space Grotesk', sans-serif; font-size: 1.15em; font-weight: 700; margin-bottom: 12px; }
        .step p { color: var(--text-sec); font-size: 0.9em; line-height: 1.7; }
        @media (max-width: 768px) {
            .methodology { padding: 60px 20px; }
            .section-title { font-size: 1.8em; margin-bottom: 36px; }
            .steps { grid-template-columns: 1fr; }
        }

        /* TOOLS */
        .tools { background: var(--bg-section); padding: 90px 32px; }
        .tools-inner { max-width: 1100px; margin: 0 auto; }
        .tool-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
        .tool-card { display: block; background: var(--bg-card); padding: 36px 30px; transition: background 0.25s; cursor: pointer; }
        .tool-card:hover { background: var(--bg); }
        .tool-icon { margin-bottom: 18px; }
        .tool-icon svg { width: 28px; height: 28px; stroke: var(--gold); fill: none; stroke-width: 1.5; }
        .tool-card h3 { font-family: 'Space Grotesk', sans-serif; font-size: 1.05em; font-weight: 700; margin-bottom: 10px; }
        .tool-card p { color: var(--text-sec); font-size: 0.88em; line-height: 1.65; }
        .tool-card .tool-arrow { display: inline-flex; align-items: center; gap: 6px; margin-top: 14px; font-size: 0.82em; font-weight: 600; color: var(--gold); opacity: 0; transition: opacity 0.2s, transform 0.2s; transform: translateX(-4px); }
        .tool-card:hover .tool-arrow { opacity: 1; transform: translateX(0); }
        @media (max-width: 768px) {
            .tools { padding: 60px 20px; }
            .tool-grid { grid-template-columns: 1fr; }
        }

        /* CTA */
        .cta { padding: 100px 32px; text-align: center; }
        .cta h2 { font-family: 'Playfair Display', serif; font-size: 2.4em; font-weight: 700; font-style: italic; margin-bottom: 18px; }
        .cta p { color: var(--text-sec); font-size: 1.05em; max-width: 560px; margin: 0 auto 36px; line-height: 1.7; }
        @media (max-width: 768px) {
            .cta { padding: 60px 20px; }
            .cta h2 { font-size: 1.8em; }
        }

        /* FOOTER */
        footer { border-top: 1px solid var(--border); padding: 32px; }
        .footer-inner { max-width: 1200px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center; }
        .footer-brand { display: flex; align-items: center; gap: 8px; font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.95em; }
        .footer-brand .brand-icon { width: 28px; height: 28px; border-radius: 6px; }
        .footer-brand .brand-icon svg { width: 16px; height: 16px; }
        .footer-note { color: var(--text-muted); font-size: 0.82em; }
        @media (max-width: 768px) {
            .footer-inner { flex-direction: column; gap: 12px; text-align: center; }
        }
    </style>
</head>
<body>
    <!-- NAV -->
    <nav>
        <div class="nav-inner">
            <a class="brand" href="/">
                <div class="brand-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="#0d0f14" stroke-width="2.5"><polyline points="4 16 8 11 13 14 20 7"/><polyline points="16 7 20 7 20 11"/></svg>
                </div>
                Stock Analysis Pro
            </a>
            <div class="nav-links">
                <div class="nav-link-item"><a href="#methodology">Methodology</a></div>
                <div class="nav-link-item">
                    <button>Research Tools <svg viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg></button>
                    <div class="mega-drop">
                        <div class="mega-grid">
                            <a class="mega-item" href="/app#verdict">
                                <div class="mega-item-icon"><svg viewBox="0 0 24 24"><polyline points="4 16 8 11 13 14 20 7"/><polyline points="16 7 20 7 20 11"/></svg></div>
                                <div class="mega-item-text"><h4>Investment Verdict</h4><p>Unified 4-in-1 score with actionable recommendation</p></div>
                            </a>
                            <a class="mega-item" href="/app#analysis">
                                <div class="mega-item-icon"><svg viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg></div>
                                <div class="mega-item-text"><h4>Technical Analysis</h4><p>RSI, MACD, Bollinger Bands &amp; momentum signals</p></div>
                            </a>
                            <a class="mega-item" href="/app#dcf">
                                <div class="mega-item-icon"><svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg></div>
                                <div class="mega-item-text"><h4>DCF Valuation</h4><p>Intrinsic value via discounted cash-flow models</p></div>
                            </a>
                            <a class="mega-item" href="/app#dividend">
                                <div class="mega-item-icon"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="12" x2="16" y2="12"/></svg></div>
                                <div class="mega-item-text"><h4>Dividend Analyzer</h4><p>Yield screening &amp; payout sustainability</p></div>
                            </a>
                            <a class="mega-item" href="/app#regression">
                                <div class="mega-item-icon"><svg viewBox="0 0 24 24"><circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="12" r="3"/><line x1="8.6" y1="7.5" x2="15.4" y2="10.5"/><line x1="8.6" y1="16.5" x2="15.4" y2="13.5"/></svg></div>
                                <div class="mega-item-text"><h4>Market Correlation</h4><p>HSIC non-linear dependency vs Nifty 50</p></div>
                            </a>
                            <a class="mega-item" href="/app#scanner">
                                <div class="mega-item-icon"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></div>
                                <div class="mega-item-text"><h4>Universe Scanner</h4><p>Filter 500+ stocks by your criteria</p></div>
                            </a>
                        </div>
                        <div class="mega-footer">
                            <span>6 research modules available</span>
                            <a href="/app">Open full platform &rarr;</a>
                        </div>
                    </div>
                </div>
                <div class="nav-link-item"><a href="#trust">Why Trust Us</a></div>
                <div class="nav-link-item"><a class="nav-cta" href="/app">Access Platform</a></div>
            </div>
            <button class="hamburger" id="hamburger" type="button" aria-label="Menu" aria-expanded="false">
                <span></span><span></span><span></span>
            </button>
        </div>
    </nav>

    <!-- MOBILE PANEL -->
    <div class="mob-panel" id="mob-panel">
        <div class="mob-section">
            <a class="mob-link" href="#methodology" onclick="closeMob()">Methodology</a>
            <a class="mob-link" href="#trust" onclick="closeMob()">Why Trust Us</a>
        </div>
        <div class="mob-section">
            <div class="mob-section-label">Research Tools</div>
            <a class="mob-tool" href="/app#verdict">
                <div class="mob-tool-icon"><svg viewBox="0 0 24 24"><polyline points="4 16 8 11 13 14 20 7"/><polyline points="16 7 20 7 20 11"/></svg></div>
                <div><h4>Investment Verdict</h4><p>Unified 4-in-1 score with actionable recommendation</p></div>
            </a>
            <a class="mob-tool" href="/app#analysis">
                <div class="mob-tool-icon"><svg viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg></div>
                <div><h4>Technical Analysis</h4><p>RSI, MACD, Bollinger Bands &amp; momentum signals</p></div>
            </a>
            <a class="mob-tool" href="/app#dcf">
                <div class="mob-tool-icon"><svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg></div>
                <div><h4>DCF Valuation</h4><p>Intrinsic value via discounted cash-flow models</p></div>
            </a>
            <a class="mob-tool" href="/app#dividend">
                <div class="mob-tool-icon"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="12" x2="16" y2="12"/></svg></div>
                <div><h4>Dividend Analyzer</h4><p>Yield screening &amp; payout sustainability</p></div>
            </a>
            <a class="mob-tool" href="/app#regression">
                <div class="mob-tool-icon"><svg viewBox="0 0 24 24"><circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="12" r="3"/><line x1="8.6" y1="7.5" x2="15.4" y2="10.5"/><line x1="8.6" y1="16.5" x2="15.4" y2="13.5"/></svg></div>
                <div><h4>Market Correlation</h4><p>HSIC non-linear dependency vs Nifty 50</p></div>
            </a>
            <a class="mob-tool" href="/app#scanner">
                <div class="mob-tool-icon"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></div>
                <div><h4>Universe Scanner</h4><p>Filter 500+ stocks by your criteria</p></div>
            </a>
        </div>
        <a class="mob-cta" href="/app">Access Platform &rarr;</a>
    </div>

    <!-- HERO -->
    <section class="hero">
        <div class="hero-label">Equity Research &amp; Analysis Platform</div>
        <h1>Informed decisions for <em>every investment.</em></h1>
        <p>Institutional-grade fundamental and technical analysis on 500+ NSE-listed securities. Fair value estimates, risk profiling, and actionable verdicts &mdash; distilled for clarity.</p>
        <div class="hero-btns">
            <a class="btn-primary" href="/app">Start Research &nbsp;&rarr;</a>
            <a class="btn-secondary" href="#tools">Explore Capabilities</a>
        </div>
    </section>

    <!-- STATS -->
    <section class="stats">
        <div class="stat"><div class="stat-val">500+</div><div class="stat-label">Securities Covered</div></div>
        <div class="stat"><div class="stat-val">4-in-1</div><div class="stat-label">Unified Score Model</div></div>
        <div class="stat"><div class="stat-val">Live</div><div class="stat-label">Market Data Feed</div></div>
        <div class="stat"><div class="stat-val">6</div><div class="stat-label">Research Modules</div></div>
    </section>

    <!-- TRUST PILLARS -->
    <section class="pillars" id="trust">
        <div class="pillars-inner">
            <div class="pillar">
                <div class="pillar-icon">
                    <svg viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                </div>
                <h3>Data Integrity</h3>
                <p>All analysis is derived from verified NSE market data, ensuring accuracy you can rely on for portfolio decisions.</p>
            </div>
            <div class="pillar">
                <div class="pillar-icon">
                    <svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
                </div>
                <h3>Institutional Methodology</h3>
                <p>Our models employ the same quantitative frameworks used by professional fund managers and research desks.</p>
            </div>
            <div class="pillar">
                <div class="pillar-icon">
                    <svg viewBox="0 0 24 24"><polyline points="4 16 8 11 13 14 20 7"/><polyline points="16 7 20 7 20 11"/></svg>
                </div>
                <h3>Transparent Signals</h3>
                <p>Every recommendation is fully explainable. No black boxes &mdash; you see exactly what drives each verdict.</p>
            </div>
        </div>
    </section>

    <!-- METHODOLOGY -->
    <section class="methodology" id="methodology">
        <div class="section-label">Methodology</div>
        <div class="section-title">Research in three steps</div>
        <div class="steps">
            <div class="step">
                <div class="step-num">01</div>
                <h3>Select a security</h3>
                <p>Enter any NSE ticker or browse by sector to begin analysis.</p>
            </div>
            <div class="step">
                <div class="step-num">02</div>
                <h3>Review the analysis</h3>
                <p>Our engine runs fundamental, technical, and risk models simultaneously.</p>
            </div>
            <div class="step">
                <div class="step-num">03</div>
                <h3>Act with confidence</h3>
                <p>Read the unified verdict, examine supporting data, and make your decision.</p>
            </div>
        </div>
    </section>

    <!-- TOOLS -->
    <section class="tools" id="tools">
        <div class="tools-inner">
            <div class="section-label">Research Tools</div>
            <div class="section-title">Comprehensive analysis suite</div>
            <div class="tool-grid">
                <a class="tool-card" href="/app#verdict">
                    <div class="tool-icon"><svg viewBox="0 0 24 24"><polyline points="4 16 8 11 13 14 20 7"/><polyline points="16 7 20 7 20 11"/></svg></div>
                    <h3>Investment Verdict</h3>
                    <p>A unified score combining technical signals, DCF valuation, dividend metrics, and market correlation into a single actionable recommendation.</p>
                    <span class="tool-arrow">Open tool &rarr;</span>
                </a>
                <a class="tool-card" href="/app#analysis">
                    <div class="tool-icon"><svg viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg></div>
                    <h3>Technical Analysis</h3>
                    <p>Comprehensive indicator suite including RSI, MACD, Bollinger Bands, and momentum oscillators &mdash; computed in real time across 500+ securities.</p>
                    <span class="tool-arrow">Open tool &rarr;</span>
                </a>
                <a class="tool-card" href="/app#dcf">
                    <div class="tool-icon"><svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg></div>
                    <h3>DCF Valuation</h3>
                    <p>Rigorous discounted cash-flow modelling to estimate intrinsic value and identify pricing inefficiencies in the market.</p>
                    <span class="tool-arrow">Open tool &rarr;</span>
                </a>
                <a class="tool-card" href="/app#dividend">
                    <div class="tool-icon"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="12" x2="16" y2="12"/></svg></div>
                    <h3>Dividend Analysis</h3>
                    <p>Sector-level yield screening and payout sustainability analysis to construct resilient income-generating portfolios.</p>
                    <span class="tool-arrow">Open tool &rarr;</span>
                </a>
                <a class="tool-card" href="/app#regression">
                    <div class="tool-icon"><svg viewBox="0 0 24 24"><circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="12" r="3"/><line x1="8.6" y1="7.5" x2="15.4" y2="10.5"/><line x1="8.6" y1="16.5" x2="15.4" y2="13.5"/></svg></div>
                    <h3>Market Correlation</h3>
                    <p>HSIC-powered non-linear dependence analysis revealing hidden systematic risk exposure relative to Nifty 50.</p>
                    <span class="tool-arrow">Open tool &rarr;</span>
                </a>
                <a class="tool-card" href="/app#scanner">
                    <div class="tool-icon"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></div>
                    <h3>Universe Scanner</h3>
                    <p>Filter the entire NSE universe by sector, fundamental metrics, and technical signals to surface high-conviction opportunities.</p>
                    <span class="tool-arrow">Open tool &rarr;</span>
                </a>
            </div>
        </div>
    </section>

    <!-- CTA -->
    <section class="cta">
        <h2>Begin your research</h2>
        <p>Access institutional-quality analysis on 500+ NSE-listed securities. No account required to get started.</p>
        <a class="btn-primary" href="/app">Launch Platform &nbsp;&rarr;</a>
    </section>

    <!-- FOOTER -->
    <footer>
        <div class="footer-inner">
            <div class="footer-brand">
                <div class="brand-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="#0d0f14" stroke-width="2.5"><polyline points="4 16 8 11 13 14 20 7"/><polyline points="16 7 20 7 20 11"/></svg>
                </div>
                Stock Analysis Pro
            </div>
            <div class="footer-note">Analysis based on historical data. Past performance does not guarantee future results.</div>
        </div>
    </footer>
</body>
</html>'''
    resp = make_response(html)
    resp.headers['Cache-Control'] = 'public, max-age=600, stale-while-revalidate=1200'
    return resp

@app.route('/app')
def dashboard():
    # Main analysis dashboard
    html = '''<!DOCTYPE html>
<html lang="en-IN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Advanced NSE stock analysis platform with technical indicators, Nifty 50 market correlation, HSIC dependency analysis, and dividend portfolio optimization. Analyze 2000+ Indian stocks with real-time insights and risk-reward metrics.">
    <title>NSE Stock Analysis & Dividend Portfolio Optimizer | Stock Analysis Pro</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="preload" as="style" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" onload="this.onload=null;this.rel='stylesheet'">
    <noscript><link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet"></noscript>
    <script>
    // Mobile-menu toggle, loaded in <head> with document-level event delegation
    // so it works regardless of DOM-ready timing or any later script failure.
    (function () {
        function getEl(id) { return document.getElementById(id); }
        function setState(open) {
            var ham = getEl('hamburger');
            var menu = getEl('mobile-menu');
            var overlay = getEl('mobile-overlay');
            if (ham) { ham.classList.toggle('open', open); ham.setAttribute('aria-expanded', open ? 'true' : 'false'); }
            if (menu) menu.classList.toggle('open', open);
            if (overlay) overlay.classList.toggle('open', open);
        }
        function toggle() {
            var menu = getEl('mobile-menu');
            if (!menu) return;
            setState(!menu.classList.contains('open'));
        }
        function close() { setState(false); }
        document.addEventListener('click', function (e) {
            if (e.target.closest && e.target.closest('#hamburger')) {
                e.preventDefault(); toggle(); return;
            }
            if (e.target.closest && e.target.closest('#mobile-overlay')) {
                close(); return;
            }
            var item = e.target.closest && e.target.closest('#mobile-menu .mobile-menu-item');
            if (item) {
                var tab = item.getAttribute('data-tab');
                if (tab && typeof window.switchTab === 'function') {
                    try { window.switchTab(tab, e); } catch (err) {}
                }
                close();
            }
        }, false);
        window.toggleMobileMenu = toggle;
        window.closeMobileMenu = close;
    })();
    </script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root { --bg-dark: #0d0f14; --bg-card: #131822; --bg-card-hover: #1a2030; --accent-cyan: #C9A84C; --accent-gold: #C9A84C; --accent-purple: #C9A84C; --accent-green: #2ECC8C; --text-primary: #E8EDF2; --text-secondary: #90A4BE; --text-muted: #607B96; --border-color: #1e2535; --success: #2ECC8C; --warning: #F59E0B; --danger: #EF4444; }
        body { font-family: 'Inter', sans-serif; background: var(--bg-dark); color: var(--text-primary); min-height: 100vh; line-height: 1.6; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

        /* ===== NAVBAR ===== */
        .navbar { position: sticky; top: 0; z-index: 100; background: var(--bg-dark); border-bottom: 1px solid var(--border-color); }
        .navbar-inner { max-width: 1400px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; height: 60px; padding: 0 20px; }
        .navbar-brand { font-family: 'Space Grotesk', sans-serif; font-size: 1.15em; font-weight: 700; color: var(--text-primary); text-decoration: none; letter-spacing: -0.3px; white-space: nowrap; }
        .navbar-brand span { color: var(--accent-cyan); }
        .nav-links { display: flex; gap: 0; height: 100%; align-items: stretch; }
        .nav-link { display: flex; align-items: center; padding: 0 16px; background: none; border: none; border-bottom: 3px solid transparent; color: var(--text-secondary); font-size: 0.88em; font-weight: 600; cursor: pointer; transition: color 0.2s, border-color 0.2s; font-family: 'Space Grotesk', sans-serif; white-space: nowrap; }
        .nav-link:hover { color: var(--text-primary); background: transparent !important; transform: none; border-color: transparent; border-bottom-color: rgba(201,168,76,0.4); }
        .nav-link.active { color: var(--text-primary); border-bottom-color: var(--accent-cyan); background: transparent !important; }
        .hamburger { display: none; background: none !important; border: 1px solid var(--border-color); border-radius: 6px; padding: 8px 10px; cursor: pointer; flex-direction: column; gap: 4px; align-items: center; }
        .hamburger:hover { background: var(--bg-card-hover) !important; border-color: var(--border-color) !important; }
        .hamburger span { display: block; width: 20px; height: 2px; background: var(--text-primary); border-radius: 1px; transition: all 0.3s; }
        .hamburger.open span:nth-child(1) { transform: rotate(45deg) translate(4px, 4px); }
        .hamburger.open span:nth-child(2) { opacity: 0; }
        .hamburger.open span:nth-child(3) { transform: rotate(-45deg) translate(4px, -4px); }
        .mobile-menu { display: none; position: fixed; top: 60px; left: 0; right: 0; background: var(--bg-card); border-bottom: 1px solid var(--border-color); z-index: 99; padding: 8px 0; box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
        .mobile-menu.open { display: block; }
        .mobile-menu-item { display: block; width: 100%; padding: 14px 24px; background: none; border: none; color: var(--text-secondary); font-size: 0.95em; font-weight: 600; cursor: pointer; text-align: left; transition: background 0.2s, color 0.2s; font-family: 'Space Grotesk', sans-serif; }
        .mobile-menu-item:hover, .mobile-menu-item.active { background: var(--bg-card-hover) !important; color: var(--text-primary) !important; border-color: transparent !important; }
        .mobile-overlay { display: none; position: fixed; top: 60px; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 98; }
        .mobile-overlay.open { display: block; }
        @media (max-width: 900px) { .nav-links { display: none; } .hamburger { display: flex; } }

        /* ===== HERO HEADER ===== */
        header { text-align: center; padding: 48px 0 40px; border-bottom: 1px solid var(--border-color); background: var(--bg-dark); }
        header h1 { font-family: 'Space Grotesk', sans-serif; font-size: 2.4em; font-weight: 700; margin-bottom: 14px; color: var(--text-primary); line-height: 1.25; letter-spacing: -0.5px; }
        header p { color: var(--text-secondary); font-size: 1.05em; max-width: 520px; margin: 0 auto; line-height: 1.65; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px; }
        .card { background: var(--bg-card); border-radius: 12px; padding: 25px; border: 1px solid var(--border-color); transition: all 0.3s; }
        .card:hover { background: var(--bg-card-hover); border-color: rgba(201,168,76,0.35); }
        .card h2 { color: var(--text-primary); margin-bottom: 15px; font-size: 1.3em; font-family: 'Space Grotesk', sans-serif; font-weight: 600; }
        #search, #regression-search, #verdict-search, #dcf-search { width: 100%; padding: 14px; border: 2px solid var(--border-color); border-radius: 8px; font-size: 1em; background: var(--bg-dark); color: var(--text-primary); transition: all 0.3s; font-family: 'Inter', sans-serif; }
        #search:focus, #regression-search:focus, #verdict-search:focus, #dcf-search:focus { outline: none; border-color: var(--accent-cyan); box-shadow: 0 0 0 3px rgba(201,168,76, 0.15); }
        #dividend-search:focus { outline: none; border-color: var(--accent-gold); box-shadow: 0 0 0 3px rgba(201,168,76, 0.12); }
        .suggestions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 15px; max-height: 300px; overflow-y: auto; }
        .category { margin-bottom: 20px; }
        .category h3 { color: var(--accent-cyan); font-size: 0.85em; margin-bottom: 8px; text-transform: uppercase; font-weight: 600; letter-spacing: 1px; }
        .stocks { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
        /* ===== Browse by Sector - Pill Tabs + Stock Cards ===== */
        .sector-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
        .sector-pill { padding: 7px 16px; border-radius: 20px; border: 1px solid var(--border-color); background: transparent; color: var(--text-secondary); font-size: 0.82em; font-weight: 600; cursor: pointer; transition: all 0.2s; font-family: 'Space Grotesk', sans-serif; white-space: nowrap; }
        .sector-pill:hover { border-color: var(--accent-cyan); color: var(--text-primary); background: transparent; }
        .sector-pill.active { border-color: var(--accent-cyan); color: var(--accent-cyan); background: rgba(201,168,76,0.1); }
        .stock-cards-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; min-height: 120px; }
        .stock-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 16px 18px; cursor: pointer; transition: all 0.25s; position: relative; overflow: hidden; }
        .stock-card:hover { border-color: rgba(201,168,76,0.45); background: var(--bg-card-hover); transform: translateY(-2px); }
        .stock-card .sc-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px; }
        .stock-card .sc-symbol { font-weight: 700; font-size: 0.95em; color: var(--text-primary); font-family: 'Space Grotesk', sans-serif; }
        .stock-card .sc-change { font-size: 0.78em; font-weight: 600; padding: 2px 8px; border-radius: 4px; white-space: nowrap; }
        .stock-card .sc-change.up { color: var(--accent-green); background: rgba(46,204,140,0.12); }
        .stock-card .sc-change.down { color: var(--danger); background: rgba(239,68,68,0.12); }
        .stock-card .sc-name { font-size: 0.78em; color: var(--text-muted); margin-bottom: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .stock-card .sc-price { font-size: 1.15em; font-weight: 700; color: var(--text-primary); font-family: 'Space Grotesk', sans-serif; }
        .stock-card .sc-mcap { font-size: 0.72em; color: var(--text-muted); text-align: right; margin-top: 2px; }
        .stock-card .sc-bottom { display: flex; justify-content: space-between; align-items: flex-end; }
        .stock-card.sc-loading .sc-price, .stock-card.sc-loading .sc-change, .stock-card.sc-loading .sc-mcap { color: transparent; background: linear-gradient(90deg, var(--border-color) 25%, var(--bg-card-hover) 50%, var(--border-color) 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite; border-radius: 4px; }
        @keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
        .sector-load-more { display: block; width: 100%; padding: 12px; margin-top: 12px; background: transparent; border: 1px dashed var(--border-color); border-radius: 8px; color: var(--text-secondary); font-size: 0.88em; font-weight: 600; cursor: pointer; transition: all 0.2s; }
        .sector-load-more:hover { border-color: var(--accent-cyan); color: var(--accent-cyan); background: transparent; }
        @media (max-width: 900px) { .stock-cards-grid { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 500px) { .stock-cards-grid { grid-template-columns: 1fr; } .sector-pills { gap: 6px; } .sector-pill { font-size: 0.75em; padding: 5px 12px; } }
        button { padding: 10px 16px; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 6px; cursor: pointer; font-weight: 500; transition: all 0.2s; color: var(--text-secondary); font-size: 0.9em; }
        button:hover { background: var(--accent-gold); color: var(--bg-dark); border-color: var(--accent-gold); }
        #result-view { display: none; }
        .loading { text-align: center; color: var(--accent-cyan); font-size: 1.3em; padding: 40px; font-family: 'Space Grotesk', sans-serif; }
        .error { background: rgba(239, 68, 68, 0.1); color: var(--danger); padding: 20px; border-radius: 8px; border-left: 4px solid var(--danger); }
    </style>
    <style media="not all" id="deferred-css">
        .result-card { background: var(--bg-card); border-radius: 12px; padding: 35px; border: 1px solid var(--border-color); }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 2px solid var(--border-color); padding-bottom: 20px; }
        .header h2 { color: var(--text-primary); font-size: 2.5em; font-family: 'Space Grotesk', sans-serif; font-weight: 700; }
        .signal-badge { font-size: 1.2em; font-weight: 700; padding: 12px 24px; border-radius: 8px; font-family: 'Space Grotesk', sans-serif; letter-spacing: 1px; }
        .signal-BUY { background: linear-gradient(135deg, #10b981, #059669); color: white; }
        .signal-SELL { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
        .signal-HOLD { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
        .action-banner { background: linear-gradient(135deg, #0d1018, #111520); color: var(--text-primary); padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; font-weight: 600; font-size: 1.2em; font-family: 'Space Grotesk', sans-serif; border: 1px solid rgba(201,168,76,0.25); box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
        .rec-box { background: var(--bg-card-hover); border-left: 4px solid var(--accent-green); color: var(--text-primary); padding: 20px; border-radius: 8px; margin-bottom: 25px; font-size: 1.05em; }
        .confidence-meter { margin: 25px 0; padding: 20px; background: var(--bg-card-hover); border-radius: 10px; border: 1px solid var(--border-color); }
        .confidence-label { font-size: 0.9em; color: var(--text-secondary); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
        .confidence-bar-container { background: var(--bg-dark); height: 30px; border-radius: 15px; overflow: hidden; margin-bottom: 10px; }
        .confidence-bar { height: 100%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.9em; transition: width 1s ease; }
        .confidence-text { color: var(--text-secondary); font-size: 0.95em; line-height: 1.6; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 15px; margin: 25px 0; }
        .metric { background: var(--bg-card-hover); padding: 18px; border-radius: 8px; border-left: 3px solid var(--accent-cyan); transition: all 0.3s; }
        .metric:hover { transform: translateY(-3px); border-left-color: var(--accent-gold); }
        .metric-label { font-size: 0.75em; color: var(--text-muted); text-transform: uppercase; margin-bottom: 5px; font-weight: 600; letter-spacing: 0.5px; }
        .metric-value { font-size: 1.5em; font-weight: 700; color: var(--text-primary); font-family: 'Space Grotesk', sans-serif; }
        .explanation-section { margin: 25px 0; }
        .explanation-section h3 { color: var(--accent-cyan); margin-bottom: 12px; font-size: 1em; font-family: 'Space Grotesk', sans-serif; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
        .explanation { background: var(--bg-card-hover); padding: 16px; border-radius: 8px; line-height: 1.7; color: var(--text-secondary); border-left: 3px solid var(--accent-cyan); }
        .trading-plan { background: var(--bg-card-hover); padding: 25px; border-radius: 10px; margin-top: 30px; border: 2px solid var(--accent-purple); }
        .trading-plan h3 { color: var(--accent-purple); margin-bottom: 20px; font-family: 'Space Grotesk', sans-serif; font-size: 1.3em; font-weight: 700; }
        .plan-item { display: grid; grid-template-columns: 140px 1fr; gap: 20px; margin-bottom: 15px; padding: 15px; background: var(--bg-dark); border-radius: 8px; transition: all 0.3s; }
        .plan-item:hover { background: var(--bg-card); transform: translateX(5px); }
        .plan-label { font-weight: 700; color: var(--accent-cyan); font-family: 'Space Grotesk', sans-serif; font-size: 0.9em; }
        .plan-value { color: var(--text-primary); font-weight: 500; }
        .back-btn { background: rgba(201,168,76,0.12); color: var(--accent-gold); padding: 12px 28px; margin-bottom: 20px; border: 1px solid rgba(201,168,76,0.3); font-weight: 600; font-size: 1em; }
        .back-btn:hover { transform: translateY(-2px); background: rgba(201,168,76,0.18); box-shadow: 0 5px 20px rgba(0,0,0,0.3); color: var(--accent-gold); }
        .regression-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 25px 0; }
        .regression-metric { background: var(--bg-card-hover); padding: 20px; border-radius: 10px; border-left: 3px solid var(--accent-purple); }
        .regression-metric-label { font-size: 0.85em; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; font-weight: 600; letter-spacing: 0.5px; }
        .regression-metric-value { font-size: 2em; font-weight: 700; color: var(--text-primary); font-family: 'Space Grotesk', sans-serif; margin-bottom: 8px; }
        .regression-metric-desc { font-size: 0.9em; color: var(--text-secondary); line-height: 1.5; }
        .plot-container { background: var(--bg-card-hover); padding: 20px; border-radius: 12px; margin-bottom: 30px; border: 1px solid var(--border-color); text-align: center; }
        .plot-img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .hsic-badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 0.85em; letter-spacing: 0.5px; color: #fff; }
        .hsic-hero { text-align: center; padding: 30px 20px; background: var(--bg-card-hover); border-radius: 14px; margin-bottom: 25px; border: 1px solid var(--border-color); position: relative; }
        .hsic-hero-score { font-size: 3.5em; font-weight: 700; font-family: 'Space Grotesk', sans-serif; line-height: 1.1; }
        .hsic-hero-label { font-size: 1.1em; margin-top: 6px; color: var(--text-secondary); }
        .hsic-hero-subtitle { font-size: 0.9em; margin-top: 12px; color: var(--text-muted); max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.6; }
        .hsic-tooltip { position: relative; cursor: help; border-bottom: 1px dashed var(--text-muted); }
        .hsic-tooltip .hsic-tooltip-text { visibility: hidden; opacity: 0; position: absolute; z-index: 10; bottom: 125%; left: 50%; transform: translateX(-50%); width: 300px; background: #0d1018; color: var(--text-secondary); padding: 14px; border-radius: 10px; font-size: 0.85em; line-height: 1.5; border: 1px solid var(--border-color); box-shadow: 0 8px 25px rgba(0,0,0,0.5); transition: opacity 0.2s; font-weight: 400; text-transform: none; letter-spacing: normal; }
        .hsic-tooltip .hsic-tooltip-text::after { content: ''; position: absolute; top: 100%; left: 50%; margin-left: -6px; border-width: 6px; border-style: solid; border-color: #0d1018 transparent transparent transparent; }
        .hsic-tooltip:hover .hsic-tooltip-text { visibility: visible; opacity: 1; }
        .insight-card { background: var(--bg-card-hover); border-radius: 12px; padding: 22px; margin-bottom: 15px; border-left: 4px solid var(--accent-purple); }
        .insight-card-title { font-size: 0.8em; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; color: var(--text-muted); margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
        .insight-card-body { font-size: 0.95em; color: var(--text-secondary); line-height: 1.7; }
        .mirror-verdict { font-weight: 700; font-family: 'Space Grotesk', sans-serif; font-size: 1.3em; margin-bottom: 6px; }
        .tech-details-toggle { background: none; border: 1px solid var(--border-color); color: var(--text-muted); padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 0.85em; font-weight: 600; transition: all 0.2s; width: 100%; text-align: left; display: flex; justify-content: space-between; align-items: center; margin-top: 20px; }
        .tech-details-toggle:hover { border-color: var(--accent-cyan); color: var(--text-secondary); }
        .tech-details-content { max-height: 0; overflow: hidden; transition: max-height 0.3s ease-out; }
        .tech-details-content.open { max-height: 500px; }
        .tech-details-inner { padding: 20px 0 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; }
        .tech-detail-item { background: var(--bg-card); padding: 15px; border-radius: 8px; border: 1px solid var(--border-color); }
        .tech-detail-label { font-size: 0.75em; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px; }
        .tech-detail-value { font-size: 1.4em; font-weight: 700; font-family: 'Space Grotesk', sans-serif; color: var(--text-primary); }
        .tech-detail-note { font-size: 0.8em; color: var(--text-muted); margin-top: 4px; }

        /* ===== TRADING SIGNAL CARD STYLES ===== */
        .tsc { background: var(--bg-card); border-radius: 16px; padding: 0; border: 1px solid var(--border-color); overflow: hidden; animation: slideIn 0.5s ease; }
        .tsc-header { display: flex; justify-content: space-between; align-items: flex-start; padding: 24px 28px 16px; }
        .tsc-ticker { font-family: 'Space Grotesk', sans-serif; font-size: 1.8em; font-weight: 700; color: var(--text-primary); letter-spacing: -0.5px; }
        .tsc-price-row { display: flex; align-items: center; gap: 10px; margin-top: 4px; }
        .tsc-price { font-size: 1.05em; color: var(--text-secondary); font-weight: 500; }
        .tsc-change { font-size: 0.9em; font-weight: 600; padding: 2px 8px; border-radius: 4px; }
        .tsc-change.up { color: var(--accent-green); background: rgba(46, 204, 140, 0.1); }
        .tsc-change.down { color: var(--danger); background: rgba(239, 68, 68, 0.1); }
        .tsc-header-right { display: flex; align-items: center; gap: 10px; }
        .tsc-badge { font-size: 0.85em; font-weight: 700; padding: 8px 20px; border-radius: 8px; font-family: 'Space Grotesk', sans-serif; letter-spacing: 0.5px; }
        .tsc-badge-BUY { background: #10b981; color: white; }
        .tsc-badge-SELL { background: #ef4444; color: white; }
        .tsc-badge-HOLD { background: #f59e0b; color: white; }
        .tsc-menu-btn { background: rgba(255,255,255,0.06); border: 1px solid var(--border-color); border-radius: 8px; padding: 8px 12px; color: var(--text-muted); cursor: pointer; font-size: 1.1em; transition: all 0.2s; }
        .tsc-menu-btn:hover { background: rgba(255,255,255,0.1); color: var(--text-primary); }
        .tsc-body { padding: 0 28px 28px; }
        .tsc-setup-banner { background: linear-gradient(135deg, rgba(201,168,76,0.08), rgba(61,122,181,0.08)); border: 1px solid rgba(201,168,76,0.2); color: var(--text-primary); padding: 12px 20px; border-radius: 10px; text-align: center; font-weight: 600; font-size: 0.95em; font-family: 'Space Grotesk', sans-serif; margin-bottom: 20px; }
        .tsc-confidence-card { background: var(--bg-card-hover); border-radius: 12px; padding: 22px 24px; margin-bottom: 20px; border: 1px solid var(--border-color); }
        .tsc-confidence-top { display: flex; align-items: center; gap: 16px; margin-bottom: 4px; }
        .tsc-signal-label { font-size: 1.1em; font-weight: 700; padding: 8px 20px; border-radius: 6px; font-family: 'Space Grotesk', sans-serif; }
        .tsc-signal-label-BUY { background: #10b981; color: white; }
        .tsc-signal-label-SELL { background: #ef4444; color: white; }
        .tsc-signal-label-HOLD { background: #f59e0b; color: white; }
        .tsc-confidence-info { flex: 1; }
        .tsc-confidence-pct { font-family: 'Space Grotesk', sans-serif; font-size: 1.1em; font-weight: 600; }
        .tsc-confidence-pct span { color: var(--text-secondary); font-weight: 400; font-size: 0.85em; }
        .tsc-confidence-hint { color: var(--text-muted); font-size: 0.85em; margin-top: 2px; }
        .tsc-rr-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0; margin-bottom: 20px; background: var(--bg-card-hover); border-radius: 12px; border: 1px solid var(--border-color); overflow: visible; }
        .tsc-rr-item { padding: 18px 16px; text-align: left; border-right: 1px solid var(--border-color); }
        .tsc-rr-item:first-child { border-radius: 12px 0 0 12px; }
        .tsc-rr-item:last-child { border-right: none; border-radius: 0 12px 12px 0; }
        .tsc-rr-label { font-size: 0.75em; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 6px; }
        .tsc-rr-value { font-size: 1.5em; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }
        .tsc-rr-value.green { color: var(--accent-green); }
        .tsc-rr-value.red { color: var(--danger); }
        .tsc-rr-value.neutral { color: var(--text-primary); }
        .tsc-rr-bar { height: 4px; border-radius: 2px; margin-top: 10px; display: flex; overflow: hidden; background: var(--bg-dark); }
        .tsc-rr-bar-fill { height: 100%; border-radius: 2px; }
        .tsc-why { background: var(--bg-card-hover); border-radius: 12px; padding: 22px 24px; margin-bottom: 20px; border: 1px solid var(--border-color); }
        .tsc-why h3 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.05em; color: var(--text-primary); margin-bottom: 10px; }
        .tsc-why p { color: var(--text-secondary); line-height: 1.7; font-size: 0.95em; }
        .tsc-regime { background: var(--bg-card-hover); border-radius: 12px; padding: 22px 24px; margin-bottom: 20px; border: 1px solid var(--border-color); }
        .tsc-regime h3 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.05em; color: var(--text-primary); margin-bottom: 14px; }
        .tsc-regime-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
        .tsc-regime-item { background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px 14px; border: 1px solid var(--border-color); }
        .tsc-regime-item-label { font-size: 0.8em; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; font-weight: 600; }
        .tsc-regime-item-value { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.05em; }
        .tsc-regime-bullish { color: #10b981; }
        .tsc-regime-bearish { color: #ef4444; }
        .tsc-regime-neutral { color: #f59e0b; }
        .tsc-regime-bar { width: 100%; height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px; margin-top: 14px; margin-bottom: 10px; }
        .tsc-regime-bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s ease; }
        .tsc-regime-meta { display: flex; justify-content: space-between; align-items: center; font-size: 0.85em; color: var(--text-muted); margin-bottom: 10px; }
        .tsc-regime-reason { color: var(--text-secondary); font-size: 0.9em; line-height: 1.6; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border-color); }
        .tsc-regime-override { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; font-size: 0.9em; color: #fca5a5; }
        @media (max-width: 600px) { .tsc-regime-grid { grid-template-columns: 1fr 1fr; } }
        .tsc-verdict { position: relative; border-radius: 14px; padding: 26px 28px; margin-bottom: 22px; border: 1px solid transparent; line-height: 1.8; }
        .tsc-verdict-BUY { background: linear-gradient(135deg, rgba(16, 185, 129, 0.10), rgba(46, 204, 140, 0.05)); border-color: rgba(16, 185, 129, 0.28); }
        .tsc-verdict-SELL { background: linear-gradient(135deg, rgba(239, 68, 68, 0.10), rgba(239, 68, 68, 0.06)); border-color: rgba(239, 68, 68, 0.30); }
        .tsc-verdict-HOLD { background: linear-gradient(135deg, rgba(245, 158, 11, 0.10), rgba(245, 158, 11, 0.06)); border-color: rgba(245, 158, 11, 0.30); }
        .tsc-verdict-header { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
        .tsc-verdict-icon { font-size: 1.5em; line-height: 1; }
        .tsc-verdict-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.15em; color: var(--text-primary); }
        .tsc-verdict-body { color: var(--text-secondary); font-size: 0.93em; line-height: 1.85; }
        .tsc-verdict-body ul { margin: 0; padding-left: 20px; list-style-type: disc; }
        .tsc-verdict-body li { margin-bottom: 8px; }
        .tsc-verdict-body li:last-child { margin-bottom: 0; }
        .tsc-verdict-body strong { color: var(--text-primary); font-weight: 600; }
        .tsc-calc { background: var(--bg-card-hover); border-radius: 12px; padding: 24px; margin-bottom: 20px; border: 1px solid var(--border-color); }
        .tsc-calc-header { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
        .tsc-calc-check { width: 20px; height: 20px; background: var(--accent-green); border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.7em; font-weight: 700; }
        .tsc-calc-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1em; }
        .tsc-calc-row { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; }
        .tsc-calc-input-wrap { flex: 1; position: relative; }
        .tsc-calc-input { width: 100%; padding: 14px 14px 14px 8px; border: 1px solid var(--border-color); border-radius: 8px; background: var(--bg-dark); color: var(--text-primary); font-size: 1em; font-family: 'Space Grotesk', sans-serif; font-weight: 600; transition: all 0.2s; }
        .tsc-calc-input:focus { outline: none; border-color: var(--accent-green); box-shadow: 0 0 0 3px rgba(46, 204, 140, 0.1); }
        .tsc-calc-info-box { background: var(--bg-dark); border: 1px solid var(--border-color); border-radius: 8px; padding: 14px 16px; display: flex; justify-content: space-between; align-items: center; }
        .tsc-calc-info-label { color: var(--text-muted); font-size: 0.85em; }
        .tsc-calc-info-value { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.1em; }
        .tsc-calc-result { background: linear-gradient(135deg, rgba(46,204,140,0.08), rgba(46,204,140,0.04)); border: 1px solid rgba(46,204,140,0.2); border-radius: 10px; padding: 14px 18px; display: flex; justify-content: space-between; align-items: center; margin-top: 4px; }
        .tsc-calc-result-label { color: var(--text-secondary); font-size: 0.9em; font-weight: 500; }
        .tsc-calc-result-value { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.3em; color: var(--accent-green); }
        .tsc-calc-details { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 14px; }
        .tsc-calc-detail-item { background: var(--bg-dark); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; }
        .tsc-calc-detail-label { font-size: 0.72em; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px; }
        .tsc-calc-detail-value { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1em; color: var(--text-primary); }
        .tsc-accordion { margin-bottom: 16px; }
        .tsc-accordion-toggle { background: var(--bg-card-hover); border: 1px solid var(--border-color); color: var(--text-secondary); padding: 14px 20px; border-radius: 10px; cursor: pointer; font-size: 0.9em; font-weight: 600; transition: all 0.2s; width: 100%; text-align: left; display: flex; justify-content: space-between; align-items: center; font-family: 'Space Grotesk', sans-serif; }
        .tsc-accordion-toggle:hover { border-color: var(--accent-cyan); color: var(--text-primary); background: var(--bg-card-hover); }
        .tsc-accordion-toggle .tsc-arrow { transition: transform 0.3s; font-size: 0.8em; }
        .tsc-accordion-toggle.open .tsc-arrow { transform: rotate(180deg); }
        .tsc-accordion-content { max-height: 0; overflow: hidden; transition: max-height 0.4s ease-out; }
        .tsc-accordion-content.open { max-height: 2000px; }
        .tsc-accordion-inner { padding: 16px 0 0; }
        .tsc-tech-item { background: var(--bg-card-hover); border: 1px solid var(--border-color); border-radius: 10px; padding: 18px 20px; margin-bottom: 12px; }
        .tsc-tech-item-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .tsc-tech-item-name { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 0.95em; color: var(--text-primary); }
        .tsc-tech-item-value { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.1em; }
        .tsc-tech-item-explain { color: var(--text-secondary); font-size: 0.85em; line-height: 1.7; margin-top: 6px; }
        .tsc-tech-item-example { color: var(--text-muted); font-size: 0.8em; line-height: 1.6; margin-top: 8px; padding: 10px 12px; background: var(--bg-dark); border-radius: 6px; border-left: 3px solid var(--accent-purple); }
        .tsc-capital-display { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.5em; color: var(--text-primary); text-align: right; min-width: 120px; }
        .tsc-tip { position: relative; cursor: help; }
        .tsc-tip .tsc-tip-text { visibility: hidden; opacity: 0; position: absolute; z-index: 20; bottom: calc(100% + 10px); left: 50%; transform: translateX(-50%); width: 340px; background: #0d1018; color: var(--text-secondary); padding: 16px; border-radius: 10px; font-size: 0.82em; line-height: 1.65; border: 1px solid var(--border-color); box-shadow: 0 10px 30px rgba(0,0,0,0.55); transition: opacity 0.2s, visibility 0.2s; font-weight: 400; text-transform: none; letter-spacing: normal; pointer-events: none; }
        .tsc-tip .tsc-tip-text::after { content: ''; position: absolute; top: 100%; left: 50%; margin-left: -7px; border-width: 7px; border-style: solid; border-color: #0d1018 transparent transparent transparent; }
        .tsc-tip:hover .tsc-tip-text { visibility: visible; opacity: 1; pointer-events: auto; }
        .tsc-tip .tsc-tip-text strong { color: var(--text-primary); }
        .tsc-tip .tsc-tip-formula { display: block; margin-top: 8px; padding: 8px 10px; background: var(--bg-dark); border-radius: 6px; font-family: 'Space Grotesk', monospace; font-size: 0.95em; color: var(--accent-cyan); word-break: break-word; }
        .tsc-rr-item .tsc-rr-label { display: inline-flex; align-items: center; gap: 5px; }
        .tsc-rr-item .tsc-rr-label .tsc-tip-icon { display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; border-radius: 50%; border: 1px solid var(--text-muted); font-size: 0.65em; color: var(--text-muted); flex-shrink: 0; transition: border-color 0.2s, color 0.2s; }
        .tsc-tip:hover .tsc-tip-icon { border-color: var(--accent-cyan); color: var(--accent-cyan); }
        .tsc-auto-risk { display: flex; align-items: center; gap: 12px; padding: 14px 16px; background: var(--bg-dark); border: 1px solid var(--border-color); border-radius: 10px; margin-bottom: 16px; }
        .tsc-auto-risk-badge { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.5em; color: var(--accent-green); min-width: 60px; }
        .tsc-auto-risk-label { font-size: 0.85em; color: var(--text-secondary); line-height: 1.5; }
        .tsc-auto-risk-label strong { color: var(--text-primary); }
        @media (max-width: 600px) {
            .tsc-tip .tsc-tip-text { width: 260px; font-size: 0.78em; padding: 12px; left: 0; transform: translateX(-20%); }
        }
        @media (max-width: 768px) {
            .tsc-header { padding: 16px 18px 12px; }
            .tsc-body { padding: 0 18px 18px; }
            .tsc-ticker { font-size: 1.4em; }
            .tsc-rr-grid { grid-template-columns: 1fr; }
            .tsc-rr-item { border-right: none; border-bottom: 1px solid var(--border-color); border-radius: 0; }
            .tsc-rr-item:first-child { border-radius: 12px 12px 0 0; }
            .tsc-rr-item:last-child { border-bottom: none; border-radius: 0 0 12px 12px; }
            .tsc-calc-details { grid-template-columns: 1fr; }
            .tsc-confidence-top { flex-wrap: wrap; }
            .tsc-calc-row { flex-direction: column; }
            .tsc-capital-display { text-align: left; }
        }

        .scope-btn, .risk-btn { padding: 10px 18px; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 6px; cursor: pointer; color: var(--text-secondary); font-size: 0.9em; font-weight: 500; transition: all 0.2s; }
        .scope-btn.active, .risk-btn.active { background: var(--accent-gold); color: var(--bg-dark); border-color: var(--accent-gold); }
        .scope-btn:hover, .risk-btn:hover { border-color: var(--accent-gold); color: var(--accent-gold); }
        .scope-btn.active:hover, .risk-btn.active:hover { color: var(--bg-dark); }
        .btn-group { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }
        .sector-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; margin-top: 10px; max-height: 300px; overflow-y: auto; padding-right: 5px; }
        .sector-grid label { display: flex; align-items: center; gap: 6px; cursor: pointer; padding: 6px 8px; border-radius: 4px; color: var(--text-secondary); font-size: 0.85em; transition: background 0.2s; }
        .sector-grid label:hover { background: var(--bg-card-hover); }
        .dividend-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
        .dividend-table th { padding: 12px 10px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--accent-gold); font-weight: 600; text-transform: uppercase; font-size: 0.8em; letter-spacing: 0.5px; position: sticky; top: 0; background: var(--bg-card); }
        .dividend-table td { padding: 10px; border-bottom: 1px solid var(--border-color); }
        .dividend-table tr:hover { background: var(--bg-card-hover); }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 25px 0; }
        .summary-card { background: var(--bg-card-hover); padding: 22px; border-radius: 10px; text-align: center; border: 1px solid var(--border-color); transition: all 0.3s; }
        .summary-card:hover { border-color: rgba(201,168,76,0.35); transform: translateY(-3px); }
        .summary-value { font-size: 1.8em; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }
        .summary-label { font-size: 0.85em; color: var(--text-secondary); margin-top: 8px; }
        .optimize-btn { width: 100%; padding: 16px; background: linear-gradient(135deg, var(--accent-green), #059669); color: white; border: none; border-radius: 8px; font-size: 1.1em; font-weight: 700; cursor: pointer; transition: all 0.3s; font-family: 'Space Grotesk', sans-serif; letter-spacing: 0.5px; }
        .optimize-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(46, 204, 140, 0.25); }
        .dividend-search-row { display: flex; gap: 10px; align-items: stretch; }
        .dividend-search-row .search-input-wrap { flex: 1; min-width: 0; position: relative; }
        .dividend-search-row .search-input-wrap input { width: 100%; padding: 12px 14px; border: 2px solid var(--border-color); border-radius: 8px; font-size: 0.95em; background: var(--bg-dark); color: var(--text-primary); transition: all 0.3s; box-sizing: border-box; }
        .dividend-search-row .search-input-wrap input:focus { outline: none; border-color: var(--accent-gold); box-shadow: 0 0 0 3px rgba(201,168,76, 0.12); }
        .search-btn { padding: 12px 22px; background: var(--accent-gold); color: var(--bg-dark); border: none; border-radius: 8px; font-size: 0.95em; font-weight: 700; cursor: pointer; transition: all 0.3s; font-family: 'Space Grotesk', sans-serif; white-space: nowrap; flex-shrink: 0; }
        .search-btn:hover { opacity: 0.9; box-shadow: 0 4px 15px rgba(201,168,76, 0.3); }
        .portfolio-action-btn { width: 100%; padding: 14px 20px; background: linear-gradient(135deg, #1C3A5E, #243F68); color: var(--text-primary); border: 1px solid rgba(61,122,181,0.35); border-radius: 10px; font-size: 1em; font-weight: 700; cursor: pointer; transition: all 0.3s; font-family: 'Space Grotesk', sans-serif; display: flex; align-items: center; justify-content: center; gap: 8px; letter-spacing: 0.3px; margin-top: 6px; }
        .portfolio-action-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(0,0,0,0.35); }
        .portfolio-action-btn:active { transform: translateY(0); }
        @media (max-width: 480px) {
            .dividend-search-row { flex-direction: column; }
            .search-btn { width: 100%; padding: 14px; text-align: center; }
            .portfolio-action-btn { font-size: 0.95em; padding: 14px 16px; }
        }
        .risk-desc { margin-top: 10px; padding: 12px; background: var(--bg-dark); border-radius: 6px; color: var(--text-muted); font-size: 0.85em; line-height: 1.5; border-left: 3px solid var(--accent-purple); }
        #capital-input { width: 100%; padding: 14px; border: 2px solid var(--border-color); border-radius: 8px; font-size: 1.1em; background: var(--bg-dark); color: var(--accent-green); font-weight: 600; transition: all 0.3s; font-family: 'Space Grotesk', sans-serif; }
        #capital-input:focus { outline: none; border-color: var(--accent-green); box-shadow: 0 0 0 3px rgba(46, 204, 140, 0.1); }
        @keyframes slideIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .result-card { animation: slideIn 0.5s ease; }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
            .header { flex-direction: column; gap: 15px; }
            .stocks, .suggestions { grid-template-columns: repeat(2, 1fr); }
            .plan-item { grid-template-columns: 1fr; gap: 5px; text-align: left; }
            .container { padding: 12px; }
            .result-card { padding: 20px; }
            .plan-label { font-size: 0.85em; opacity: 0.9; }
            header { padding: 28px 0 24px; }
            header h1 { font-size: 1.7em; }
            header p { font-size: 0.92em; }
            .vd-section-body { padding: 0 14px 18px; }
            .vd-score-card { padding: 18px 14px; }
            .tsc-verdict { padding: 20px 18px; }
        }
        @media (max-width: 480px) {
            .result-card { padding: 14px; }
            .tsc-header { padding: 14px 14px 10px; }
            .tsc-body { padding: 0 14px 14px; }
            .tsc-confidence-card, .tsc-why, .tsc-regime, .tsc-calc { padding: 16px 16px; }
            .vd-hero { padding: 18px 16px; }
            .vd-name { font-size: 1.3em; }
            .vd-price-val { font-size: 1.6em; }
            .vd-section-header { padding: 14px 16px; }
            .dcf-params-card, .dcf-results-card { padding: 18px; }
            .dcf-stock-hero { padding: 18px 16px; }
        }
        @media (max-width: 400px) {
            header h1 { font-size: 1.5em; }
        }

        /* ===== DCF VALUATION TAB STYLES ===== */
        #dcf-search { width: 100%; padding: 14px; border: 2px solid var(--border-color); border-radius: 8px; font-size: 1em; background: var(--bg-dark); color: var(--text-primary); transition: all 0.3s; }
        #dcf-search:focus { outline: none; border-color: var(--accent-gold); box-shadow: 0 0 0 3px rgba(201,168,76, 0.12); }
        .dcf-fetch-btn { width: 100%; margin-top: 14px; padding: 14px; background: linear-gradient(135deg, #1C3A5E, #243F68); color: var(--text-primary); border: 1px solid rgba(61,122,181,0.35); border-radius: 8px; font-size: 1em; font-weight: 700; cursor: pointer; font-family: 'Space Grotesk', sans-serif; transition: all 0.3s; letter-spacing: 0.3px; }
        .dcf-fetch-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(0,0,0,0.35); }
        .dcf-stock-hero { background: var(--bg-card-hover); border-radius: 14px; padding: 24px 28px; margin-bottom: 24px; border: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px; }
        .dcf-stock-name { font-family: 'Space Grotesk', sans-serif; font-size: 1.6em; font-weight: 700; color: var(--text-primary); }
        .dcf-stock-sub { color: var(--text-muted); font-size: 0.88em; margin-top: 4px; }
        .dcf-price-box { text-align: right; }
        .dcf-price-label { font-size: 0.78em; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.5px; }
        .dcf-price-val { font-family: 'Space Grotesk', sans-serif; font-size: 2em; font-weight: 700; color: var(--text-primary); }
        .dcf-valuation-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
        .dcf-params-card { background: var(--bg-card); border-radius: 14px; padding: 26px; border: 1px solid var(--border-color); }
        .dcf-params-card h3 { font-family: 'Space Grotesk', sans-serif; font-size: 1.05em; font-weight: 700; color: var(--accent-cyan); margin-bottom: 20px; text-transform: uppercase; letter-spacing: 0.5px; }
        .dcf-param-row { margin-bottom: 18px; }
        .dcf-param-label { display: flex; justify-content: space-between; align-items: center; font-size: 0.85em; color: var(--text-secondary); font-weight: 600; margin-bottom: 8px; }
        .dcf-param-value { font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: var(--accent-cyan); font-size: 0.95em; }
        .dcf-slider { width: 100%; -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: var(--bg-dark); outline: none; cursor: pointer; }
        .dcf-slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--accent-gold); cursor: pointer; border: 2px solid var(--bg-dark); box-shadow: 0 0 6px rgba(201,168,76, 0.4); transition: box-shadow 0.2s; }
        .dcf-slider::-webkit-slider-thumb:hover { box-shadow: 0 0 12px rgba(201,168,76, 0.65); }
        .dcf-slider::-moz-range-thumb { width: 18px; height: 18px; border-radius: 50%; background: var(--accent-cyan); cursor: pointer; border: 2px solid var(--bg-dark); }
        .dcf-param-hint { font-size: 0.76em; color: var(--text-muted); margin-top: 4px; }
        .dcf-years-group { display: flex; gap: 8px; }
        .dcf-year-btn { flex: 1; padding: 9px; background: var(--bg-dark); border: 1px solid var(--border-color); border-radius: 6px; color: var(--text-secondary); font-size: 0.88em; font-weight: 600; cursor: pointer; transition: all 0.2s; text-align: center; }
        .dcf-year-btn.active { background: var(--accent-cyan); color: var(--bg-dark); border-color: var(--accent-cyan); }
        .dcf-results-card { background: var(--bg-card); border-radius: 14px; padding: 26px; border: 1px solid var(--border-color); }
        .dcf-results-card h3 { font-family: 'Space Grotesk', sans-serif; font-size: 1.05em; font-weight: 700; color: var(--accent-purple); margin-bottom: 20px; text-transform: uppercase; letter-spacing: 0.5px; }
        .dcf-intrinsic-hero { text-align: center; padding: 24px 16px; background: var(--bg-card-hover); border-radius: 12px; margin-bottom: 20px; border: 1px solid var(--border-color); }
        .dcf-intrinsic-label { font-size: 0.8em; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.5px; margin-bottom: 8px; }
        .dcf-intrinsic-value { font-family: 'Space Grotesk', sans-serif; font-size: 3em; font-weight: 700; line-height: 1; }
        .dcf-intrinsic-undervalued { color: var(--accent-green); }
        .dcf-intrinsic-overvalued { color: var(--danger); }
        .dcf-margin-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: var(--bg-dark); border-radius: 8px; margin-bottom: 8px; }
        .dcf-margin-label { font-size: 0.85em; color: var(--text-secondary); }
        .dcf-margin-val { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 0.95em; }
        .dcf-upside { color: var(--accent-green); }
        .dcf-downside { color: var(--danger); }
        .dcf-verdict-banner { padding: 14px 18px; border-radius: 10px; text-align: center; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1em; margin-bottom: 16px; letter-spacing: 0.3px; }
        .dcf-verdict-buy { background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(6,255,165,0.1)); border: 1px solid rgba(16,185,129,0.4); color: #10b981; }
        .dcf-verdict-sell { background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(239,68,68,0.1)); border: 1px solid rgba(239,68,68,0.4); color: #ef4444; }
        .dcf-verdict-hold { background: linear-gradient(135deg, rgba(245,158,11,0.18), rgba(245,158,11,0.1)); border: 1px solid rgba(245,158,11,0.4); color: #f59e0b; }
        .dcf-breakdown-bar { margin: 18px 0; }
        .dcf-breakdown-label { font-size: 0.8em; color: var(--text-muted); font-weight: 600; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
        .dcf-bar-track { height: 28px; background: var(--bg-dark); border-radius: 6px; overflow: hidden; display: flex; }
        .dcf-bar-fcf { background: linear-gradient(90deg, var(--accent-cyan), #0891b2); display: flex; align-items: center; justify-content: center; font-size: 0.75em; font-weight: 700; color: white; transition: width 0.6s ease; white-space: nowrap; overflow: hidden; min-width: 0; }
        .dcf-bar-tv { background: linear-gradient(90deg, var(--accent-purple), #7c3aed); display: flex; align-items: center; justify-content: center; font-size: 0.75em; font-weight: 700; color: white; transition: width 0.6s ease; white-space: nowrap; overflow: hidden; min-width: 0; }
        .dcf-bar-legend { display: flex; gap: 16px; margin-top: 8px; }
        .dcf-bar-legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.78em; color: var(--text-muted); }
        .dcf-bar-legend-dot { width: 10px; height: 10px; border-radius: 2px; }
        .dcf-section { margin-top: 28px; }
        .dcf-section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1em; font-weight: 700; color: var(--text-primary); margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid var(--border-color); }
        .dcf-proj-table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
        .dcf-proj-table th { padding: 10px 12px; text-align: right; border-bottom: 2px solid var(--border-color); color: var(--accent-cyan); font-weight: 600; text-transform: uppercase; font-size: 0.78em; letter-spacing: 0.5px; }
        .dcf-proj-table th:first-child { text-align: left; }
        .dcf-proj-table td { padding: 9px 12px; border-bottom: 1px solid rgba(45,55,72,0.5); text-align: right; color: var(--text-secondary); }
        .dcf-proj-table td:first-child { text-align: left; color: var(--text-primary); font-weight: 600; }
        .dcf-proj-table tr:hover td { background: var(--bg-card-hover); }
        .dcf-proj-table tr.tv-row td { color: var(--accent-purple); border-top: 2px solid var(--border-color); }
        .dcf-proj-table tr.total-row td { color: var(--accent-green); font-weight: 700; font-family: 'Space Grotesk', sans-serif; border-top: 2px solid var(--border-color); }
        .dcf-proj-table tr.hist-row td { color: var(--text-secondary); background: rgba(255,200,100,0.04); }
        .dcf-proj-table tr.hist-row td:first-child { color: var(--warning); font-weight: 600; }
        .dcf-proj-table tr.hist-separator td { background: transparent; padding: 4px 12px; border-bottom: 2px dashed var(--border-color); border-top: 2px dashed var(--border-color); color: var(--text-muted); font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.8px; text-align: center; }
        .dcf-sensitivity { margin-top: 28px; }
        .dcf-sens-table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
        .dcf-sens-table th { padding: 8px 10px; text-align: center; border-bottom: 2px solid var(--border-color); color: var(--text-muted); font-weight: 600; font-size: 0.78em; letter-spacing: 0.5px; }
        .dcf-sens-table th:first-child { text-align: left; color: var(--accent-purple); }
        .dcf-sens-table td { padding: 7px 10px; text-align: center; border-bottom: 1px solid rgba(45,55,72,0.4); font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.88em; }
        .dcf-sens-table td:first-child { text-align: left; color: var(--text-muted); font-weight: 700; font-family: inherit; font-size: 0.8em; }
        .dcf-sens-undervalue { color: var(--accent-green); }
        .dcf-sens-overvalue { color: var(--danger); }
        .dcf-sens-near { color: var(--warning); }
        .dcf-sens-highlight { background: rgba(0,217,255,0.08); border: 1px solid rgba(0,217,255,0.2); border-radius: 4px; }
        .dcf-key-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
        .dcf-stat { background: var(--bg-card-hover); padding: 14px 16px; border-radius: 8px; border-left: 3px solid var(--border-color); }
        .dcf-stat-label { font-size: 0.72em; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px; }
        .dcf-stat-value { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.05em; color: var(--text-primary); }
        .dcf-fcf-hist { display: flex; align-items: flex-end; gap: 8px; height: 70px; margin: 14px 0 6px; }
        .dcf-fcf-bar-wrap { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px; }
        .dcf-fcf-bar-inner { width: 100%; border-radius: 3px 3px 0 0; min-height: 4px; transition: height 0.4s ease; }
        .dcf-fcf-bar-year { font-size: 0.68em; color: var(--text-muted); }
        .dcf-fcf-positive { background: linear-gradient(180deg, var(--accent-cyan), #0891b2); }
        .dcf-fcf-negative { background: linear-gradient(180deg, var(--danger), #dc2626); }
        .dcf-disclaimer { margin-top: 28px; padding: 14px 18px; background: rgba(245,158,11,0.07); border: 1px solid rgba(245,158,11,0.2); border-radius: 8px; color: var(--text-muted); font-size: 0.8em; line-height: 1.6; }
        @media (max-width: 768px) {
            .dcf-valuation-grid { grid-template-columns: 1fr; }
            .dcf-key-stats { grid-template-columns: 1fr 1fr; }
            .dcf-intrinsic-value { font-size: 2.2em; }
            .dcf-stock-hero { flex-direction: column; align-items: flex-start; }
            .dcf-price-box { text-align: left; }
        }
        @media (max-width: 480px) {
            .dcf-key-stats { grid-template-columns: 1fr; }
        }
        /* ===== INVESTMENT VERDICT TAB ===== */
        .verdict-fetch-btn { width:100%;margin-top:14px;padding:14px;background:linear-gradient(135deg,#1C3A5E,#2A5080);color:#E8EDF2;border:none;border-radius:10px;font-size:1em;font-weight:700;cursor:pointer;font-family:'Space Grotesk',sans-serif;transition:all 0.25s;letter-spacing:0.3px;box-shadow:0 4px 16px rgba(27,45,63,0.22); }
        .verdict-fetch-btn:hover { transform:translateY(-2px);box-shadow:0 8px 28px rgba(27,45,63,0.32);background:linear-gradient(135deg,#233F60,#316090); }
        .vd-hero { background:var(--bg-card-hover);border-radius:14px;padding:24px 28px;margin-bottom:22px;border:1px solid var(--border-color);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px; }
        .vd-name { font-family:'Space Grotesk',sans-serif;font-size:1.55em;font-weight:700;color:var(--text-primary); }
        .vd-sub { color:var(--text-muted);font-size:0.86em;margin-top:3px; }
        .vd-price-box { text-align:right; }
        .vd-price-label { font-size:0.76em;text-transform:uppercase;color:var(--text-muted);font-weight:600;letter-spacing:0.5px; }
        .vd-price-val { font-family:'Space Grotesk',sans-serif;font-size:1.9em;font-weight:700;color:var(--text-primary); }
        .vd-daily.up { color:var(--accent-green);font-weight:700;font-size:0.9em; }
        .vd-daily.down { color:var(--danger);font-weight:700;font-size:0.9em; }
        .vd-score-grid { display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:28px; }
        .vd-score-card { background:var(--bg-card);border-radius:14px;padding:22px 18px;border:1px solid var(--border-color);text-align:center;position:relative;overflow:hidden;transition:transform 0.2s; }
        .vd-score-card:hover { transform:translateY(-3px); }
        .vd-score-card.vd-best { border-color:var(--accent-green);box-shadow:0 0 20px rgba(6,255,165,0.12); }
        .vd-score-label { font-size:0.72em;text-transform:uppercase;letter-spacing:0.6px;font-weight:700;margin-bottom:12px; }
        .vd-score-ring { width:80px;height:80px;margin:0 auto 12px;position:relative; }
        .vd-score-ring svg { transform:rotate(-90deg); }
        .vd-score-ring-num { position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-family:'Space Grotesk',sans-serif;font-size:1.25em;font-weight:700; }
        .vd-verdict-badge { display:inline-block;padding:5px 14px;border-radius:20px;font-size:0.78em;font-weight:700;margin-top:8px; }
        .vd-badge-suitable { background:rgba(6,255,165,0.12);color:var(--accent-green);border:1px solid rgba(6,255,165,0.3); }
        .vd-badge-moderate { background:rgba(245,158,11,0.1);color:var(--warning);border:1px solid rgba(245,158,11,0.25); }
        .vd-badge-weak { background:rgba(239,68,68,0.1);color:var(--danger);border:1px solid rgba(239,68,68,0.25); }
        .vd-section { background:var(--bg-card);border-radius:14px;margin-bottom:14px;border:1px solid var(--border-color);overflow:hidden; }
        .vd-section-header { display:flex;justify-content:space-between;align-items:center;padding:18px 22px;cursor:pointer;user-select:none;transition:background 0.2s; }
        .vd-section-header:hover { background:var(--bg-card-hover); }
        .vd-section-header-left { display:flex;align-items:center;gap:12px; }
        .vd-section-title { font-family:'Space Grotesk',sans-serif;font-size:1.02em;font-weight:700; }
        .vd-section-score { font-family:'Space Grotesk',sans-serif;font-size:0.88em;font-weight:700; }
        .vd-section-chevron { color:var(--text-muted);font-size:0.85em;transition:transform 0.25s;margin-left:8px; }
        .vd-section.open .vd-section-chevron { transform:rotate(180deg); }
        .vd-section-body { display:none;padding:0 22px 22px; }
        .vd-section.open .vd-section-body { display:block; }
        .vd-metrics { display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px; }
        .vd-metric { background:var(--bg-dark);border-radius:8px;padding:10px 12px; }
        .vd-metric-label { font-size:0.72em;text-transform:uppercase;color:var(--text-muted);font-weight:600;letter-spacing:0.4px;margin-bottom:4px; }
        .vd-metric-value { font-family:'Space Grotesk',sans-serif;font-size:0.95em;font-weight:700; }
        .vd-metric-value.green { color:var(--accent-green); }
        .vd-metric-value.cyan { color:var(--accent-cyan); }
        .vd-metric-value.yellow { color:var(--warning); }
        .vd-metric-value.red { color:var(--danger); }
        .vd-metric-value.muted { color:var(--text-muted); }
        .vd-trade-levels { display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:14px 0; }
        .vd-level { background:var(--bg-dark);border-radius:8px;padding:12px 14px;text-align:center; }
        .vd-level-label { font-size:0.7em;text-transform:uppercase;color:var(--text-muted);font-weight:600;letter-spacing:0.5px;margin-bottom:4px; }
        .vd-level-val { font-family:'Space Grotesk',sans-serif;font-size:1.1em;font-weight:700; }
        .vd-narrative { background:var(--bg-dark);border-radius:10px;padding:14px 16px;font-size:0.86em;color:var(--text-secondary);line-height:1.7;margin-top:12px; }
        .vd-overall { background:linear-gradient(135deg,rgba(6,255,165,0.06),rgba(0,217,255,0.06));border:1px solid rgba(6,255,165,0.2);border-radius:14px;padding:26px;margin-bottom:20px;text-align:center; }
        .vd-overall-label { font-size:0.76em;text-transform:uppercase;color:var(--text-muted);font-weight:600;letter-spacing:0.5px;margin-bottom:8px; }
        .vd-overall-verdict { font-family:'Space Grotesk',sans-serif;font-size:1.8em;font-weight:700;margin-bottom:10px; }
        .vd-overall-reason { color:var(--text-secondary);font-size:0.88em;line-height:1.7;max-width:680px;margin:0 auto; }
        .vd-loading-grid { display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:20px 0; }
        .vd-load-item { background:var(--bg-card);border-radius:10px;padding:14px 16px;display:flex;align-items:center;gap:12px;border:1px solid var(--border-color); }
        .vd-load-icon { font-size:1.4em; }
        .vd-load-label { font-size:0.85em;color:var(--text-secondary);font-weight:600; }
        .vd-load-status { font-size:0.78em;margin-top:2px; }
        .vd-load-status.done { color:var(--accent-green); }
        .vd-load-status.loading { color:var(--warning); }
        .vd-load-status.error { color:var(--danger); }
        .vd-divider { height:1px;background:var(--border-color);margin:6px 0 14px; }
        /* Investor Profile */
        .pref-group { display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;margin-bottom:4px; }
        .pref-btn { background:var(--bg-dark);color:var(--text-secondary);border:1px solid var(--border-color);border-radius:20px;padding:6px 14px;font-size:0.82em;font-weight:600;cursor:pointer;transition:all 0.2s;font-family:'Inter',sans-serif; }
        .pref-btn:hover { border-color:var(--accent-cyan);color:var(--text-primary); }
        .pref-btn.active { background:rgba(201,168,76,0.15);border-color:var(--accent-cyan);color:var(--accent-cyan); }
        .pref-label { font-size:0.75em;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);font-weight:700;margin-top:14px;margin-bottom:2px; }
        /* ── HNI Questionnaire ───────────────────────────────────────────── */
        .qz-section { margin-bottom:18px; }
        .qz-header { display:flex;align-items:flex-start;gap:10px;margin-bottom:10px; }
        .qz-num { min-width:26px;height:26px;border-radius:50%;background:rgba(201,168,76,0.1);border:1px solid rgba(201,168,76,0.28);color:var(--accent-cyan);font-size:0.72em;font-weight:800;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px;transition:all 0.25s; }
        .qz-num.done { background:rgba(16,185,129,0.15);border-color:rgba(16,185,129,0.4);color:var(--accent-green); }
        .qz-q { font-size:0.9em;font-weight:600;color:var(--text-primary);line-height:1.45;padding-top:3px; }
        .qz-opts { display:flex;flex-wrap:wrap;gap:8px;padding-left:36px; }
        .qz-opt { flex:1 1 150px;background:var(--bg-dark);border:1px solid var(--border-color);border-radius:10px;padding:10px 13px;cursor:pointer;transition:all 0.2s;text-align:left;font-family:'Inter',sans-serif; }
        .qz-opt:hover { border-color:var(--accent-cyan);background:rgba(201,168,76,0.06); }
        .qz-opt.selected { background:rgba(201,168,76,0.13);border-color:var(--accent-cyan); }
        .qz-opt-label { font-size:0.82em;font-weight:700;color:var(--text-primary);margin-bottom:3px; }
        .qz-opt.selected .qz-opt-label { color:var(--accent-cyan); }
        .qz-opt-desc { font-size:0.71em;color:var(--text-muted);line-height:1.3; }
        .qz-divider { height:1px;background:var(--border-color);margin:4px 0 18px; }
        .qz-progress { font-size:0.75em;color:var(--text-muted);margin-bottom:12px;min-height:1.1em; }
        .qz-progress strong { color:var(--accent-cyan); }
        .qz-cta { width:100%;padding:14px;background:linear-gradient(90deg,rgba(201,168,76,0.18),rgba(139,92,246,0.12));border:1px solid rgba(201,168,76,0.4);color:var(--accent-cyan);border-radius:12px;font-size:0.95em;font-weight:700;cursor:pointer;font-family:'Space Grotesk',sans-serif;transition:all 0.25s;display:none; }
        .qz-cta.ready { display:block; }
        .qz-cta:hover { background:linear-gradient(90deg,rgba(201,168,76,0.28),rgba(139,92,246,0.22));box-shadow:0 0 18px rgba(201,168,76,0.22); }
        .suggest-stocks-btn { width:100%;margin-top:18px;padding:11px;background:rgba(201,168,76,0.12);border:1px solid rgba(201,168,76,0.35);color:var(--accent-cyan);border-radius:10px;font-size:0.9em;font-weight:700;cursor:pointer;font-family:'Space Grotesk',sans-serif;transition:all 0.2s; }
        .suggest-stocks-btn:hover { background:rgba(201,168,76,0.22); }
        /* Trend indicators */
        .trend-section-title { font-size:0.75em;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);font-weight:700;margin:16px 0 8px; }
        .trend-row { display:flex;align-items:center;gap:8px;padding:7px 12px;border-radius:8px;background:var(--bg-dark);margin-bottom:6px; }
        .trend-label { font-size:0.78em;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.4px;min-width:130px; }
        .trend-values { font-size:0.8em;color:var(--text-secondary);font-family:'Space Grotesk',sans-serif;flex:1; }
        .trend-arrow { font-size:1em;font-weight:700;min-width:20px;text-align:right; }
        .trend-arrow.up { color:var(--accent-green); }
        .trend-arrow.down { color:var(--danger); }
        .trend-arrow.flat { color:var(--warning); }
        /* Score card clickable */
        .vd-score-card { cursor:pointer; }
        /* Curated stocks modal */
        .curated-modal { position:fixed;inset:0;background:rgba(0,0,0,0.75);z-index:500;display:flex;align-items:center;justify-content:center;padding:16px; }
        .curated-modal-box { background:var(--bg-card);border-radius:16px;border:1px solid var(--border-color);padding:26px;max-width:680px;width:100%;max-height:82vh;overflow-y:auto; }
        .curated-modal-title { font-family:'Space Grotesk',sans-serif;font-size:1.2em;font-weight:700;margin-bottom:4px; }
        .curated-modal-sub { color:var(--text-muted);font-size:0.83em;margin-bottom:18px;line-height:1.5; }
        .curated-group { margin-bottom:18px; }
        .curated-group-title { font-size:0.74em;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);font-weight:700;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid var(--border-color); }
        .curated-stocks { display:flex;flex-wrap:wrap;gap:8px; }
        .curated-stock-btn { background:var(--bg-dark);border:1px solid var(--border-color);color:var(--text-secondary);border-radius:20px;padding:6px 14px;font-size:0.83em;cursor:pointer;transition:all 0.2s;font-family:'Space Grotesk',sans-serif;font-weight:600; }
        .curated-stock-btn:hover { border-color:var(--accent-cyan);color:var(--accent-cyan);background:rgba(201,168,76,0.1); }
        .curated-close-btn { width:100%;padding:10px;background:var(--bg-dark);border:1px solid var(--border-color);color:var(--text-secondary);border-radius:10px;font-size:0.88em;cursor:pointer;margin-top:16px;font-family:'Inter',sans-serif; }
        /* Scanner tab */
        .scanner-start-btn { width:100%;margin-top:18px;padding:11px;background:rgba(201,168,76,0.12);border:1px solid rgba(201,168,76,0.35);color:var(--accent-cyan);border-radius:10px;font-size:0.9em;font-weight:700;cursor:pointer;font-family:'Space Grotesk',sans-serif;transition:all 0.2s; }
        .scanner-start-btn:hover { background:rgba(201,168,76,0.22); }
        .scanner-stop-btn { width:100%;margin-top:8px;padding:9px;background:rgba(220,53,69,0.1);border:1px solid rgba(220,53,69,0.3);color:var(--danger);border-radius:10px;font-size:0.85em;font-weight:700;cursor:pointer;font-family:'Space Grotesk',sans-serif;transition:all 0.2s;display:none; }
        .scanner-stop-btn:hover { background:rgba(220,53,69,0.2); }
        .sc-stat-row { display:flex;flex-wrap:wrap;gap:16px;margin-bottom:14px;font-size:0.82em;color:var(--text-muted); }
        .sc-stat-row span { display:flex;align-items:center;gap:4px; }
        .sc-stat-row strong { color:var(--text-primary); }
        /* Live scan rows */
        .scan-row { display:flex;align-items:center;gap:12px;padding:12px 14px;background:var(--bg-dark);border:1px solid var(--border-color);border-radius:10px;margin-bottom:8px;transition:border-color 0.2s,background 0.2s; }
        .scan-row:hover { border-color:var(--accent-cyan);background:rgba(201,168,76,0.06); }
        .scan-rank { font-family:'Space Grotesk',sans-serif;font-size:0.82em;font-weight:700;color:var(--text-muted);min-width:22px;text-align:center; }
        .scan-info { flex:1;min-width:0; }
        .scan-name { font-family:'Space Grotesk',sans-serif;font-size:0.92em;font-weight:700;color:var(--text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
        .scan-symbol { font-size:0.74em;color:var(--text-muted); }
        .scan-scores { display:flex;gap:10px; }
        .scan-score-item { text-align:center;min-width:32px; }
        .scan-score-label { display:block;font-size:0.62em;text-transform:uppercase;color:var(--text-muted);letter-spacing:0.3px; }
        .scan-score-item span:last-child { font-family:'Space Grotesk',sans-serif;font-size:0.88em;font-weight:700; }
        .scan-main-score { font-family:'Space Grotesk',sans-serif;font-size:1.4em;font-weight:700;min-width:40px;text-align:right; }
        @media (max-width:480px) { .scan-scores { display:none; } .scan-main-score { font-size:1.2em; } }
        /* Profile badge on verdict overall */
        .profile-match-badge { display:inline-block;margin-top:10px;padding:4px 12px;border-radius:20px;font-size:0.78em;font-weight:700;background:rgba(201,168,76,0.15);color:var(--accent-cyan);border:1px solid rgba(201,168,76,0.3); }
        @media (max-width:768px) {
            .vd-score-grid { grid-template-columns:1fr; }
            .vd-metrics { grid-template-columns:1fr 1fr; }
            .vd-trade-levels { grid-template-columns:1fr 1fr 1fr; }
            .trend-label { min-width:100px; }
        }

        /* ===== AI Assistant Tab ===== */
        .ai-hero { text-align:center; padding:48px 20px 18px; max-width:680px; margin:0 auto; }
        .ai-pill { display:inline-flex; align-items:center; gap:7px; background:rgba(201,168,76,0.13); border:1px solid rgba(201,168,76,0.3); border-radius:20px; padding:5px 16px; font-size:0.76em; font-weight:700; color:var(--accent-cyan); letter-spacing:0.6px; text-transform:uppercase; margin-bottom:22px; }
        .ai-hero h1 { font-family:'Space Grotesk',sans-serif; font-size:2.6em; font-weight:800; color:var(--text-primary); line-height:1.15; margin-bottom:16px; letter-spacing:-0.5px; }
        .ai-hero p { color:var(--text-secondary); font-size:1em; line-height:1.75; max-width:560px; margin:0 auto; }
        .ai-shell { max-width:880px; margin:0 auto; padding:0 20px 60px; }
        .ai-suggest-row { display:flex; flex-wrap:wrap; gap:10px; justify-content:center; margin:22px auto 22px; max-width:780px; }
        .ai-suggest-chip { background:var(--bg-card); border:1px solid var(--border-color); color:var(--text-secondary); padding:9px 14px; border-radius:20px; font-size:0.83em; cursor:pointer; transition:border-color .2s,color .2s,transform .2s; font-family:inherit; }
        .ai-suggest-chip:hover { border-color:var(--accent-gold); color:var(--text-primary); transform:translateY(-1px); }
        .ai-chat-card { background:var(--bg-card); border:1px solid var(--border-color); border-radius:18px; overflow:hidden; box-shadow:0 8px 40px rgba(27,45,63,0.18); display:flex; flex-direction:column; min-height:520px; max-height:72vh; }
        .ai-chat-header { display:flex; align-items:center; justify-content:space-between; padding:16px 20px; border-bottom:1px solid var(--border-color); background:rgba(255,255,255,0.02); }
        .ai-chat-title { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:0.95em; color:var(--text-primary); display:flex; align-items:center; gap:10px; }
        .ai-status-dot { width:8px; height:8px; border-radius:50%; background:var(--accent-green); display:inline-block; box-shadow:0 0 0 4px rgba(46,204,113,0.18); }
        .ai-rate-tag { font-size:0.76em; color:var(--text-muted); }
        .ai-messages { flex:1; overflow-y:auto; padding:20px 22px; display:flex; flex-direction:column; gap:14px; min-height:0; }
        .ai-messages::-webkit-scrollbar { width:6px; }
        .ai-messages::-webkit-scrollbar-track { background:transparent; }
        .ai-messages::-webkit-scrollbar-thumb { background:var(--border-color); border-radius:3px; }
        .ai-msg { max-width:78%; padding:12px 16px; border-radius:14px; font-size:0.92em; line-height:1.6; word-break:break-word; white-space:pre-wrap; }
        .ai-msg.user { align-self:flex-end; background:rgba(201,168,76,0.18); color:var(--text-primary); border:1px solid rgba(201,168,76,0.28); border-bottom-right-radius:4px; }
        .ai-msg.agent { align-self:flex-start; background:var(--bg-card-hover,rgba(255,255,255,0.04)); color:var(--text-primary); border:1px solid var(--border-color); border-bottom-left-radius:4px; }
        .ai-msg.agent.streaming::after { content:'▋'; display:inline-block; animation:ai-blink .7s step-end infinite; margin-left:2px; }
        @keyframes ai-blink { 0%,100%{opacity:1} 50%{opacity:0} }
        .ai-msg.error { align-self:flex-start; background:rgba(239,68,68,0.1); color:var(--danger); border:1px solid rgba(239,68,68,0.25); border-bottom-left-radius:4px; }
        .ai-msg.thinking { align-self:flex-start; color:var(--text-muted); font-style:italic; background:transparent; border:none; padding:6px 4px; }
        .ai-thinking-block { align-self:flex-start; max-width:78%; margin-bottom:2px; }
        .ai-thinking-block details { border:1px solid var(--border-color); border-radius:10px; background:rgba(255,255,255,0.02); overflow:hidden; }
        .ai-thinking-block summary { list-style:none; cursor:pointer; padding:8px 12px; font-size:0.8em; color:var(--text-muted); display:flex; align-items:center; gap:8px; user-select:none; }
        .ai-thinking-block summary::-webkit-details-marker { display:none; }
        .ai-thinking-block summary .think-spinner { width:12px; height:12px; border:2px solid var(--border-color); border-top-color:var(--accent-gold); border-radius:50%; animation:ai-spin .7s linear infinite; flex-shrink:0; }
        .ai-thinking-block.done summary .think-spinner { display:none; }
        .ai-thinking-block.done summary::before { content:'✓'; color:var(--accent-gold); font-size:0.9em; }
        @keyframes ai-spin { to{transform:rotate(360deg)} }
        .ai-thinking-block .think-steps { padding:6px 12px 10px; display:flex; flex-direction:column; gap:4px; }
        .ai-thinking-step { font-size:0.78em; color:var(--text-muted); padding:3px 0; border-left:2px solid var(--border-color); padding-left:8px; }
        .ai-thinking-step.active { color:var(--text-secondary); border-color:var(--accent-gold); }
        @media (max-width:720px) { .ai-msg,.ai-thinking-block { max-width:90%; } }
        .ai-input-row { display:flex; gap:10px; padding:14px 16px; border-top:1px solid var(--border-color); background:rgba(255,255,255,0.02); }
        #ai-input { flex:1; background:var(--bg-dark); border:1px solid var(--border-color); border-radius:10px; padding:12px 14px; color:var(--text-primary); font-size:0.92em; font-family:'Inter',sans-serif; resize:none; height:46px; max-height:140px; line-height:1.45; transition:border-color .2s; }
        #ai-input:focus { outline:none; border-color:var(--accent-gold); }
        #ai-send-btn { background:var(--accent-gold); border:none; border-radius:10px; padding:0 22px; cursor:pointer; color:var(--bg-dark); font-size:0.92em; font-weight:700; transition:opacity .2s; flex-shrink:0; font-family:inherit; }
        #ai-send-btn:hover { opacity:0.88; }
        #ai-send-btn:disabled { opacity:0.4; cursor:not-allowed; }
        .ai-disclaimer { font-size:0.72em; color:var(--text-muted); text-align:center; padding:14px 16px 0; }
        @media (max-width:720px) {
            .ai-hero { padding:32px 16px 12px; }
            .ai-hero h1 { font-size:1.9em; }
            .ai-msg { max-width:90%; }
            .ai-chat-card { min-height:60vh; max-height:75vh; border-radius:14px; }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="navbar-inner">
            <a class="navbar-brand" href="/">Stock Analysis <span>Pro</span></a> <a href="/app" style="margin-left:auto;margin-right:16px;padding:8px 20px;background:var(--accent-gold);color:var(--bg-dark);border-radius:6px;font-size:0.85em;font-weight:700;text-decoration:none;font-family:'Space Grotesk',sans-serif;display:none;" class="dash-home-link">Dashboard</a>
            <div class="nav-links">
                <button class="nav-link active" data-tab="verdict" onclick="switchTab('verdict', event)">Investment Verdict</button>
                <button class="nav-link" data-tab="analysis" onclick="switchTab('analysis', event)">Technical Analysis</button>
                <button class="nav-link" data-tab="dcf" onclick="switchTab('dcf', event)">DCF Valuation</button>
                <button class="nav-link" data-tab="dividend" onclick="switchTab('dividend', event)">Dividend Analyzer</button>
                <button class="nav-link" data-tab="regression" onclick="switchTab('regression', event)">Market Connection</button>
                <button class="nav-link" data-tab="scanner" onclick="switchTab('scanner', event)">&#128269; Scanner</button>
                <button class="nav-link" data-tab="ai" onclick="switchTab('ai', event)">&#10024; AI Assistant</button>
            </div>
            <button class="hamburger" id="hamburger" type="button" aria-label="Menu" aria-expanded="false">
                <span></span><span></span><span></span>
            </button>
        </div>
    </nav>
    <div class="mobile-overlay" id="mobile-overlay"></div>
    <div class="mobile-menu" id="mobile-menu">
        <button class="mobile-menu-item active" data-tab="verdict">Investment Verdict</button>
        <button class="mobile-menu-item" data-tab="analysis">Technical Analysis</button>
        <button class="mobile-menu-item" data-tab="dcf">DCF Valuation</button>
        <button class="mobile-menu-item" data-tab="dividend">Dividend Analyzer</button>
        <button class="mobile-menu-item" data-tab="regression">Market Connection</button>
        <button class="mobile-menu-item" data-tab="scanner">&#128269; Scanner</button>
        <button class="mobile-menu-item" data-tab="ai">&#10024; AI Assistant</button>
    </div>
    <header>
        <div class="container">
            <h1>Smart analysis for<br>every NSE stock.</h1>
            <p>500+ stocks. Buy, hold, or sell signals with fair value estimates, dividend analysis, and market risk — explained simply.</p>
        </div>
    </header>
    <main class="container">
        <div id="analysis-tab" class="tab-content">
            <div id="search-view">
                <div class="card" style="margin-bottom: 20px;">
                    <h2>Search Any NSE Stock</h2>
                    <input type="text" id="search" placeholder="Search TCS, RELIANCE, INFY, or any NSE stock...">
                    <div class="suggestions" id="suggestions"></div>
                </div>
                <div class="card">
                    <h2>Browse by Sector</h2>
                    <div id="sector-pills-container" class="sector-pills"></div>
                    <div id="sector-cards-container" class="stock-cards-grid"></div>
                    <button id="sector-load-more-btn" class="sector-load-more" style="display:none;" onclick="loadMoreSectorStocks()">Load more stocks</button>
                </div>
            </div>
            <div id="result-view">
                <button class="back-btn" onclick="goBack()">← Back to Search</button>
                <div id="result"></div>
            </div>
        </div>
        <div id="regression-tab" class="tab-content">
            <div class="card">
                <h2>Market Connection Analysis</h2>
                <p style="color: var(--text-secondary); margin-bottom: 20px;">Find out how closely any NSE stock is tied to the Nifty 50, including hidden connections that simple charts don't show</p>
                <input type="text" id="regression-search" placeholder="Enter stock symbol (e.g., TCS, INFY, RELIANCE)">
                <div class="suggestions" id="regression-suggestions"></div>
                <button onclick="analyzeRegression()" style="margin-top: 15px; width: 100%; background: linear-gradient(135deg, #1C3A5E, #243F68); color: var(--text-primary); border: 1px solid rgba(61,122,181,0.35); font-weight: 600; padding: 14px; border-radius: 8px;">Analyze Connection</button>
            </div>
            <div class="card" style="margin-top: 20px; border-left: 3px solid var(--accent-purple); padding: 20px 25px;">
                <h3 style="color: var(--accent-purple); margin-bottom: 10px; font-family: 'Space Grotesk', sans-serif;">How to read your results</h3>
                <p style="color: var(--text-secondary); font-size: 0.92em; line-height: 1.8; margin: 0;">
                    This tool measures how connected a stock is to the Nifty 50 index. It goes beyond simple correlation by using a technique called <strong style="color: var(--text-primary);">HSIC</strong> (Hilbert-Schmidt Independence Criterion). Think of it as an X-ray that can detect both <em>obvious</em> and <em>hidden</em> links between a stock and the market.
                </p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; margin-top: 15px;">
                    <div style="background: var(--bg-dark); padding: 12px 15px; border-radius: 8px;">
                        <strong style="color: var(--text-primary);">Market Connection %</strong>
                        <p style="color: var(--text-muted); font-size: 0.85em; margin: 4px 0 0;">The main score. Higher = more tied to the market. A stock with a high score tends to fall when Nifty 50 falls, offering less portfolio protection.</p>
                    </div>
                    <div style="background: var(--bg-dark); padding: 12px 15px; border-radius: 8px;">
                        <strong style="color: var(--text-primary);">Mirror Test</strong>
                        <p style="color: var(--text-muted); font-size: 0.85em; margin: 4px 0 0;">Compares visible co-movement with deeper hidden links. If these disagree, the stock may surprise you during a crash, even if it looks independent on normal days.</p>
                    </div>
                    <div style="background: var(--bg-dark); padding: 12px 15px; border-radius: 8px;">
                        <strong style="color: var(--text-primary);">Downside Beta</strong>
                        <p style="color: var(--text-muted); font-size: 0.85em; margin: 4px 0 0;">How much the stock falls when the market falls. Above 1.0 means it drops harder than Nifty 50 on bad days. The most important number for crash protection.</p>
                    </div>
                </div>
                <p style="color: var(--text-muted); font-size: 0.82em; margin-top: 12px; margin-bottom: 0; font-style: italic;">All results are based on historical data (up to 1 year) and describe past behaviour. They do not guarantee future performance.</p>
            </div>
            <div id="regression-result" style="margin-top: 30px;"></div>
        </div>
        <div id="dividend-tab" class="tab-content">
            <div class="grid">
                <div class="card">
                    <h2>Stock Universe</h2>
                    <p style="color: var(--text-secondary); margin-bottom: 15px; font-size: 0.9em;">Select sectors to scan for dividend yields and optimize your portfolio</p>
                    <div id="sector-checkboxes" style="margin-top: 15px;">
                        <div style="margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border-color);">
                            <label style="cursor: pointer; color: var(--accent-cyan); font-weight: 600; font-size: 0.9em;">
                                <input type="checkbox" id="select-all-sectors" onchange="toggleAllSectors(this.checked)"> Select All Sectors
                            </label>
                        </div>
                        <div id="sector-grid" class="sector-grid"></div>
                        <button class="optimize-btn" onclick="analyzeDividends()" style="margin-top: 18px;">Scan Dividends & Optimize Portfolio</button>
                    </div>
                </div>
                <div class="card">
                    <h2>Portfolio Configuration</h2>
                    <div style="margin-bottom: 22px;">
                        <label style="display: block; color: var(--text-secondary); font-size: 0.85em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Investment Capital (INR)</label>
                        <input type="text" id="capital-input" inputmode="numeric" placeholder="e.g. 10,00,000" value="1,00,000">
                    </div>
                    <div style="margin-bottom: 22px;">
                        <label style="display: block; color: var(--text-secondary); font-size: 0.85em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Risk Appetite</label>
                        <div class="btn-group">
                            <button class="risk-btn" onclick="setRisk('conservative', this)">Conservative</button>
                            <button class="risk-btn active" onclick="setRisk('moderate', this)">Moderate</button>
                            <button class="risk-btn" onclick="setRisk('aggressive', this)">Aggressive</button>
                        </div>
                        <div class="risk-desc" id="risk-desc">Max 15% per stock. Balanced yield vs risk tradeoff. Good diversification across dividend payers.</div>
                    </div>
                    <div style="margin-top: 18px;">
                        <button class="portfolio-action-btn" onclick="analyzeDividends()" style="width:100%;padding:14px 20px;background:linear-gradient(135deg,#1C3A5E,#243F68);color:#E8EDF2;border:1px solid rgba(61,122,181,0.35);border-radius:10px;font-size:1em;font-weight:700;cursor:pointer;font-family:'Space Grotesk',sans-serif;display:flex;align-items:center;justify-content:center;gap:8px;">
                            <span>&#9881;</span>
                            <span>Optimize Portfolio</span>
                        </button>
                    </div>
                </div>
            </div>
            <div id="dividend-results"></div>
        </div>
        <div id="dcf-tab" class="tab-content">
            <!-- Search View -->
            <div id="dcf-search-view">
                <div class="grid">
                    <div class="card">
                        <h2>DCF Valuation</h2>
                        <p style="color:var(--text-secondary);margin-bottom:18px;font-size:0.92em;line-height:1.7;">
                            Estimate the intrinsic value of any NSE stock using Discounted Cash Flow analysis.
                            Enter a symbol to fetch its financial data, then adjust the assumptions to see how intrinsic value changes in real time.
                        </p>
                        <input type="text" id="dcf-search" placeholder="Search stock (e.g. TCS, INFY, RELIANCE)...">
                        <div class="suggestions" id="dcf-suggestions"></div>
                        <button class="dcf-fetch-btn" onclick="fetchDCFData()">Fetch Financial Data &amp; Value</button>
                    </div>
                    <div class="card">
                        <h2>How DCF Works</h2>
                        <div style="display:flex;flex-direction:column;gap:12px;margin-top:4px;">
                            <div style="background:var(--bg-dark);padding:14px 16px;border-radius:8px;">
                                <strong style="color:var(--text-primary);font-size:0.9em;">1. Project Free Cash Flows</strong>
                                <p style="color:var(--text-muted);font-size:0.82em;margin:5px 0 0;line-height:1.6;">Estimate how much cash the company will generate each year, growing at your chosen rate.</p>
                            </div>
                            <div style="background:var(--bg-dark);padding:14px 16px;border-radius:8px;">
                                <strong style="color:var(--text-primary);font-size:0.9em;">2. Discount to Present Value</strong>
                                <p style="color:var(--text-muted);font-size:0.82em;margin:5px 0 0;line-height:1.6;">A rupee today is worth more than a rupee tomorrow. We discount future cash using your WACC rate.</p>
                            </div>
                            <div style="background:var(--bg-dark);padding:14px 16px;border-radius:8px;">
                                <strong style="color:var(--text-primary);font-size:0.9em;">3. Add Terminal Value</strong>
                                <p style="color:var(--text-muted);font-size:0.82em;margin:5px 0 0;line-height:1.6;">Accounts for all cash flows beyond the projection period using the Gordon Growth Model.</p>
                            </div>
                            <div style="background:var(--bg-dark);padding:14px 16px;border-radius:8px;">
                                <strong style="color:var(--text-primary);font-size:0.9em;">4. Calculate Intrinsic Value</strong>
                                <p style="color:var(--text-muted);font-size:0.82em;margin:5px 0 0;line-height:1.6;">Sum all PV of FCFs + Terminal Value, subtract debt, add cash, divide by shares outstanding.</p>
                            </div>
                        </div>
                    </div>
                </div>
                <!-- Universe Screener Card -->
                <div class="card" style="margin-top:20px;border:1px solid rgba(61,122,181,0.25);background:linear-gradient(135deg,var(--bg-card) 0%,rgba(28,58,94,0.15) 100%);">
                    <h2 style="display:flex;align-items:center;gap:8px;">
                        <span style="font-size:1.1em;">&#x1F50D;</span>
                        DCF Universe Screener
                    </h2>
                    <p style="color:var(--text-secondary);margin-bottom:14px;font-size:0.9em;line-height:1.7;">
                        Run a DCF valuation on <strong style="color:var(--text-primary);">every stock</strong> in the universe and find the <strong style="color:var(--accent-green);">Top 50 most undervalued</strong> picks.
                        Uses conservative assumptions (12% WACC, 3% terminal growth, 10-year projection).
                        Stocks are processed iteratively to keep memory usage low.
                    </p>
                    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
                        <button id="dcf-screen-btn" class="dcf-fetch-btn" onclick="startDCFScreen()" style="margin:0;flex:0 0 auto;">
                            Screen All Stocks
                        </button>
                        <button id="dcf-screen-stop-btn" onclick="stopDCFScreen()" style="display:none;margin:0;flex:0 0 auto;padding:10px 22px;background:var(--danger);color:#fff;border:none;border-radius:8px;font-weight:700;cursor:pointer;font-family:'Space Grotesk',sans-serif;font-size:0.9em;">
                            Stop
                        </button>
                        <span id="dcf-screen-status" style="color:var(--text-muted);font-size:0.85em;"></span>
                    </div>
                    <div id="dcf-screen-progress" style="display:none;margin-top:16px;">
                        <div style="background:var(--bg-dark);border-radius:8px;height:8px;overflow:hidden;">
                            <div id="dcf-screen-bar" style="height:100%;width:0%;background:linear-gradient(90deg,var(--accent-cyan),var(--accent-green));border-radius:8px;transition:width 0.3s ease;"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-top:6px;">
                            <span id="dcf-screen-pct" style="font-size:0.8em;color:var(--text-muted);">0%</span>
                            <span id="dcf-screen-found" style="font-size:0.8em;color:var(--accent-green);">0 undervalued found</span>
                        </div>
                    </div>
                </div>
            </div>
            <!-- Screener Results View -->
            <div id="dcf-screen-results" style="display:none;margin-top:20px;"></div>
            <!-- Result View -->
            <div id="dcf-result-view" style="display:none;">
                <button class="back-btn" onclick="dcfGoBack()">← Back to Search</button>
                <div id="dcf-result"></div>
            </div>
        </div>
        <div id="verdict-tab" class="tab-content active">
            <div id="verdict-search-view">
                <div style="text-align:center;padding:64px 20px 28px;max-width:680px;margin:0 auto;">
                    <div style="display:inline-flex;align-items:center;gap:7px;background:rgba(201,168,76,0.13);border:1px solid rgba(201,168,76,0.3);border-radius:20px;padding:5px 16px;font-size:0.76em;font-weight:700;color:var(--accent-cyan);letter-spacing:0.6px;text-transform:uppercase;margin-bottom:22px;">&#9889; AI-Powered Analysis Engine</div>
                    <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.6em;font-weight:800;color:var(--text-primary);line-height:1.15;margin-bottom:16px;letter-spacing:-0.5px;">Investment Verdict</h1>
                    <p style="color:var(--text-secondary);font-size:1em;line-height:1.75;max-width:520px;margin:0 auto;">Combines <strong style="color:var(--text-primary);">Technical signals</strong>, <strong style="color:var(--text-primary);">DCF valuation</strong>, <strong style="color:var(--text-primary);">Market Connection</strong> and <strong style="color:var(--text-primary);">Dividend metrics</strong> into one unified score — telling you exactly what this stock is best suited for.</p>
                </div>
                <div style="max-width:540px;margin:0 auto;padding:0 20px 20px;">
                    <div style="background:var(--bg-card);border-radius:18px;padding:30px 28px;border:1px solid var(--border-color);box-shadow:0 8px 40px rgba(27,45,63,0.18);">
                        <label style="display:block;font-size:0.76em;font-weight:700;text-transform:uppercase;letter-spacing:0.7px;color:var(--text-muted);margin-bottom:8px;">Stock Symbol</label>
                        <input type="text" id="verdict-search" placeholder="e.g. TCS, INFY, RELIANCE, HDFC..." style="font-size:1.05em;padding:14px 18px;height:52px;border-radius:10px;">
                        <div class="suggestions" id="verdict-suggestions"></div>
                        <button class="verdict-fetch-btn" onclick="fetchVerdictData()" style="margin-top:16px;border-radius:10px;font-size:1em;height:52px;letter-spacing:0.3px;">&#128269;&nbsp; Analyse &amp; Get Verdict</button>
                    </div>
                    <div style="display:flex;justify-content:center;gap:28px;margin-top:28px;flex-wrap:wrap;">
                        <div style="text-align:center;"><div style="font-family:'Space Grotesk',sans-serif;font-size:1.3em;font-weight:800;color:var(--text-primary);">4-in-1</div><div style="font-size:0.75em;color:var(--text-muted);margin-top:2px;">Score Model</div></div>
                        <div style="width:1px;background:var(--border-color);"></div>
                        <div style="text-align:center;"><div style="font-family:'Space Grotesk',sans-serif;font-size:1.3em;font-weight:800;color:var(--text-primary);">292</div><div style="font-size:0.75em;color:var(--text-muted);margin-top:2px;">NSE Stocks</div></div>
                        <div style="width:1px;background:var(--border-color);"></div>
                        <div style="text-align:center;"><div style="font-family:'Space Grotesk',sans-serif;font-size:1.3em;font-weight:800;color:var(--text-primary);">Live</div><div style="font-size:0.75em;color:var(--text-muted);margin-top:2px;">Market Data</div></div>
                    </div>
                </div>
            </div>
            <div id="verdict-result-view" style="display:none;">
                <button class="back-btn" onclick="verdictGoBack()">&#8592; Back to Search</button>
                <div id="verdict-result"></div>
            </div>
        </div>
        <div id="scanner-tab" class="tab-content">
            <div class="card" style="max-width:860px;margin:0 auto 0;">
                <h2 style="margin-bottom:4px;">&#127919; Build Your Top-20 Portfolio</h2>
                <p style="color:var(--text-secondary);font-size:0.85em;margin-bottom:22px;line-height:1.5;">Answer 6 questions and we will run 292 NSE stocks through 3 progressive filters — returning your personalised top 20.</p>

                <!-- Q1 -->
                <div class="qz-section">
                    <div class="qz-header"><div class="qz-num" id="qn1">1</div><div class="qz-q">What is your primary investment objective?</div></div>
                    <div class="qz-opts">
                        <button class="qz-opt" data-q="q1" data-v="beat" onclick="setQ('q1','beat',this)"><div class="qz-opt-label">&#128200; Beat the Index</div><div class="qz-opt-desc">Alpha above Nifty — concentrated bets on high-conviction ideas</div></button>
                        <button class="qz-opt" data-q="q1" data-v="compounder" onclick="setQ('q1','compounder',this)"><div class="qz-opt-label">&#9889; Steady Compounder</div><div class="qz-opt-desc">12–18% CAGR — quality businesses held through business cycles</div></button>
                        <button class="qz-opt" data-q="q1" data-v="income" onclick="setQ('q1','income',this)"><div class="qz-opt-label">&#128176; Regular Income</div><div class="qz-opt-desc">Consistent dividends, capital safety, minimal drawdowns</div></button>
                        <button class="qz-opt" data-q="q1" data-v="tactical" onclick="setQ('q1','tactical',this)"><div class="qz-opt-label">&#127919; Tactical Play</div><div class="qz-opt-desc">Momentum &amp; technical setups — shorter-horizon event-driven trades</div></button>
                    </div>
                </div>
                <div class="qz-divider"></div>

                <!-- Q2 -->
                <div class="qz-section">
                    <div class="qz-header"><div class="qz-num" id="qn2">2</div><div class="qz-q">Your portfolio is down 25% from peak. What is your instinct?</div></div>
                    <div class="qz-opts">
                        <button class="qz-opt" data-q="q2" data-v="cut" onclick="setQ('q2','cut',this)"><div class="qz-opt-label">&#128683; Protect Capital</div><div class="qz-opt-desc">I exit or reduce — preserving principal is non-negotiable</div></button>
                        <button class="qz-opt" data-q="q2" data-v="calm" onclick="setQ('q2','calm',this)"><div class="qz-opt-label">&#128564; Hold Steady</div><div class="qz-opt-desc">I trust the fundamentals, stay the course and wait it out</div></button>
                        <button class="qz-opt" data-q="q2" data-v="buy" onclick="setQ('q2','buy',this)"><div class="qz-opt-label">&#128640; Average Down</div><div class="qz-opt-desc">Drawdowns are opportunities — I deploy more capital aggressively</div></button>
                    </div>
                </div>
                <div class="qz-divider"></div>

                <!-- Q3 -->
                <div class="qz-section">
                    <div class="qz-header"><div class="qz-num" id="qn3">3</div><div class="qz-q">What is your planned holding horizon for this deployment?</div></div>
                    <div class="qz-opts">
                        <button class="qz-opt" data-q="q3" data-v="s6m" onclick="setQ('q3','s6m',this)"><div class="qz-opt-label">&#9889; Under 6 Months</div><div class="qz-opt-desc">Event-driven plays, earnings season, technical breakouts</div></button>
                        <button class="qz-opt" data-q="q3" data-v="s2yr" onclick="setQ('q3','s2yr',this)"><div class="qz-opt-label">&#128337; 6 Months – 2 Years</div><div class="qz-opt-desc">Sectoral recovery, re-rating plays, medium business cycle</div></button>
                        <button class="qz-opt" data-q="q3" data-v="s5yr" onclick="setQ('q3','s5yr',this)"><div class="qz-opt-label">&#128200; 2 – 5 Years</div><div class="qz-opt-desc">Structural theme investing across a full business cycle</div></button>
                        <button class="qz-opt" data-q="q3" data-v="long" onclick="setQ('q3','long',this)"><div class="qz-opt-label">&#127381; 5+ Years</div><div class="qz-opt-desc">Long-term compounding — buy, hold, and let time do the work</div></button>
                    </div>
                </div>
                <div class="qz-divider"></div>

                <!-- Q4 -->
                <div class="qz-section">
                    <div class="qz-header"><div class="qz-num" id="qn4">4</div><div class="qz-q">Which market-cap range aligns with your strategy?</div></div>
                    <div class="qz-opts">
                        <button class="qz-opt" data-q="q4" data-v="large" onclick="setQ('q4','large',this)"><div class="qz-opt-label">&#127963; Large-Cap Only</div><div class="qz-opt-desc">Nifty 50 blue chips — deep liquidity, institutional grade</div></button>
                        <button class="qz-opt" data-q="q4" data-v="mid" onclick="setQ('q4','mid',this)"><div class="qz-opt-label">&#9878;&#65039; Large + Mid</div><div class="qz-opt-desc">Best of both — quality anchors with mid-cap alpha kickers</div></button>
                        <button class="qz-opt" data-q="q4" data-v="small" onclick="setQ('q4','small',this)"><div class="qz-opt-label">&#128293; Mid &amp; Small</div><div class="qz-opt-desc">Maximum alpha potential — higher short-term volatility accepted</div></button>
                    </div>
                </div>
                <div class="qz-divider"></div>

                <!-- Q5 -->
                <div class="qz-section">
                    <div class="qz-header"><div class="qz-num" id="qn5">5</div><div class="qz-q">Do you have a sector conviction right now?</div></div>
                    <div class="qz-opts">
                        <button class="qz-opt" data-q="q5" data-v="fin" onclick="setQ('q5','fin',this)"><div class="qz-opt-label">&#127981; Financials</div><div class="qz-opt-desc">Banks, NBFCs, insurance, AMCs</div></button>
                        <button class="qz-opt" data-q="q5" data-v="pharma" onclick="setQ('q5','pharma',this)"><div class="qz-opt-label">&#128138; Pharma &amp; Health</div><div class="qz-opt-desc">Pharma, hospitals, diagnostics, chemicals</div></button>
                        <button class="qz-opt" data-q="q5" data-v="infra" onclick="setQ('q5','infra',this)"><div class="qz-opt-label">&#127959;&#65039; Infra &amp; Capex</div><div class="qz-opt-desc">Infrastructure, defence, capital goods, power, cement</div></button>
                        <button class="qz-opt" data-q="q5" data-v="tech" onclick="setQ('q5','tech',this)"><div class="qz-opt-label">&#128187; Technology</div><div class="qz-opt-desc">IT services, software, digital platforms</div></button>
                        <button class="qz-opt" data-q="q5" data-v="consumer" onclick="setQ('q5','consumer',this)"><div class="qz-opt-label">&#128666; Consumer &amp; Auto</div><div class="qz-opt-desc">FMCG, auto, retail, hospitality, electronics</div></button>
                        <button class="qz-opt" data-q="q5" data-v="all" onclick="setQ('q5','all',this)"><div class="qz-opt-label">&#127758; No Preference</div><div class="qz-opt-desc">Purely quantitative — best ideas across all sectors</div></button>
                    </div>
                </div>
                <div class="qz-divider"></div>

                <!-- Q6 -->
                <div class="qz-section">
                    <div class="qz-header"><div class="qz-num" id="qn6">6</div><div class="qz-q">How are you positioned on the broader market right now?</div></div>
                    <div class="qz-opts">
                        <button class="qz-opt" data-q="q6" data-v="bull" onclick="setQ('q6','bull',this)"><div class="qz-opt-label">&#128640; Fully Deployed</div><div class="qz-opt-desc">Markets are going higher — I want to be fully invested</div></button>
                        <button class="qz-opt" data-q="q6" data-v="selective" onclick="setQ('q6','selective',this)"><div class="qz-opt-label">&#127919; Selective</div><div class="qz-opt-desc">Cautiously optimistic — quality at fair value, not at any price</div></button>
                        <button class="qz-opt" data-q="q6" data-v="defensive" onclick="setQ('q6','defensive',this)"><div class="qz-opt-label">&#128737;&#65039; Defensive</div><div class="qz-opt-desc">Capital preservation first — low volatility, earnings certainty</div></button>
                    </div>
                </div>

                <div class="qz-progress" id="qz-progress"></div>
                <button class="qz-cta" id="qz-cta" onclick="startScanInTab()">&#128269; Find My Top 20 Stocks</button>
                <button class="scanner-stop-btn" id="scanner-stop-btn" onclick="stopScanInTab()">&#9632; Stop Scan</button>
            </div>
            <div class="card" id="sc-status-card" style="display:none;max-width:860px;margin:18px auto 0;">
                <div id="sc-sub" style="font-size:0.88em;color:var(--text-muted);margin-bottom:12px;line-height:1.5;"></div>
                <div style="background:var(--bg-dark);border-radius:8px;height:6px;overflow:hidden;margin-bottom:10px;">
                    <div id="sc-progress-fill" style="height:100%;width:0%;background:var(--accent-cyan);transition:width 0.35s;border-radius:8px;"></div>
                </div>
                <div id="sc-status-text" style="font-size:0.78em;color:var(--text-muted);margin-bottom:10px;"></div>
                <div class="sc-stat-row" id="sc-stat-row"></div>
            </div>
            <div id="scanner-results-area" style="margin-top:18px;max-width:860px;margin-left:auto;margin-right:auto;"></div>
        </div>
        <div id="ai-tab" class="tab-content">
            <div class="ai-hero">
                <div class="ai-pill">&#10024; Powered by Groq AI</div>
                <h1>AI Research Assistant</h1>
                <p>Ask anything about NSE stocks. The assistant automatically runs all analyses &mdash; verdict, DCF valuation, technicals, dividends, and market correlation &mdash; and explains everything in plain English.</p>
            </div>
            <div class="ai-shell">
                <div class="ai-suggest-row">
                    <button class="ai-suggest-chip" type="button" onclick="aiUseSuggestion('Is TCS a buy at the current price?')">Is TCS a buy right now?</button>
                    <button class="ai-suggest-chip" type="button" onclick="aiUseSuggestion('Find undervalued large caps based on DCF.')">Undervalued large caps</button>
                    <button class="ai-suggest-chip" type="button" onclick="aiUseSuggestion('Compare Reliance and TCS on momentum and valuation.')">Compare Reliance vs TCS</button>
                    <button class="ai-suggest-chip" type="button" onclick="aiUseSuggestion('High dividend yield stocks in the banking sector.')">High dividend banking stocks</button>
                </div>
                <div class="ai-chat-card">
                    <div class="ai-chat-header">
                        <div class="ai-chat-title"><span class="ai-status-dot"></span> Live Research Session</div>
                    </div>
                    <div class="ai-messages" id="ai-messages">
                        <div class="ai-msg agent">Welcome. I can pull live data for any of the 292 NSE stocks tracked here &mdash; verdicts, intrinsic value, momentum, dividends, market correlation, or screen ideas. Ask away.</div>
                    </div>
                    <div class="ai-input-row">
                        <textarea id="ai-input" placeholder="e.g. Compare HDFC Bank and ICICI Bank on dividends and momentum" rows="1" aria-label="Message input"></textarea>
                        <button id="ai-send-btn" type="button" onclick="aiSendQuery()">Send</button>
                    </div>
                </div>
                <div class="ai-disclaimer">For research and educational use only. Not investment advice.</div>
            </div>
        </div>
    </main>
    <script>
        const stocks = ''' + json.dumps(STOCKS, separators=(',', ':')) + ''';
        const nifty50List = ''' + json.dumps(NIFTY_50_STOCKS, separators=(',', ':')) + ''';
        let tickerNames = {};
        fetch('/api/ticker-names').then(r=>r.json()).then(d=>{tickerNames=d;});
        function getStockName(symbol) {
            return tickerNames[symbol] || symbol;
        }
        function getStockSector(symbol) {
            const skipSectors = new Set(["All NSE", "Nifty 50", "Nifty Next 50", "Others", "Conglomerate"]);
            for (const [sector, list] of Object.entries(stocks)) {
                if (skipSectors.has(sector)) continue;
                if (list.includes(symbol)) return sector;
            }
            return "";
        }
        const allTickers = [...new Set(Object.values(stocks).flat())];

        function getPeerStocks(symbol, limit) {
            limit = limit || 6;
            var sector = getStockSector(symbol);
            if (!sector || !stocks[sector]) return [];
            return stocks[sector].filter(function(s) { return s !== symbol; }).slice(0, limit);
        }
        function peerOnclick(p, tabContext) {
            if (tabContext === 'technical') { analyze(p); }
            else if (tabContext === 'regression') { document.getElementById('regression-search').value = p; analyzeRegression(); }
            else if (tabContext === 'dcf') { document.getElementById('dcf-search').value = p; fetchDCFData(); }
            else if (tabContext === 'verdict') { document.getElementById('verdict-search').value = p; fetchVerdictData(); }
            else if (tabContext === 'dividend-to-verdict') { switchTab('verdict'); document.getElementById('verdict-search').value = p; fetchVerdictData(); }
        }
        function buildPeerStocksHTML(symbol, tabContext) {
            var peers = getPeerStocks(symbol, 6);
            if (peers.length === 0) return '';
            var sector = getStockSector(symbol);
            var h = `<div class="peer-stocks-section" style="margin-top:20px;padding:18px 20px;background:var(--bg-card-hover);border:1px solid var(--border-color);border-radius:12px;">`;
            h += `<div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:0.95em;color:var(--accent-cyan);margin-bottom:12px;">Peer Stocks${sector ? ' \u2014 ' + sector : ''}</div>`;
            h += `<div style="display:flex;flex-wrap:wrap;gap:8px;">`;
            for (var i = 0; i < peers.length; i++) {
                var p = peers[i];
                var name = getStockName(p);
                h += `<button onclick="peerOnclick('${p}','${tabContext}')" style="background:rgba(0,217,255,0.08);border:1px solid var(--border-color);border-radius:8px;padding:8px 14px;cursor:pointer;transition:all 0.2s;color:var(--text-primary);font-size:0.85em;line-height:1.3;text-align:left;" onmouseover="this.style.borderColor='var(--accent-cyan)';this.style.background='rgba(0,217,255,0.15)'" onmouseout="this.style.borderColor='var(--border-color)';this.style.background='rgba(0,217,255,0.08)'">`
                + `<div style="font-weight:600;color:var(--accent-cyan);">${p}</div>`
                + `<div style="font-size:0.8em;color:var(--text-muted);">${name}</div>`
                + `</button>`;
            }
            h += `</div></div>`;
            return h;
        }


        // ===== INVESTOR PROFILE =====
        var investorProfile = { horizon: 'long', risk: 'medium', goal: 'growth', mcap: 'mid', sector: 'all', view: 'selective' };

        // ===== QUESTIONNAIRE STATE =====
        var questionnaire = { q1: null, q2: null, q3: null, q4: null, q5: null, q6: null };
        var QZ_TOTAL = 6;

        function loadProfile() {
            try {
                var p = JSON.parse(localStorage.getItem('stockProProfile'));
                if (p && typeof p === 'object') {
                    investorProfile = Object.assign(investorProfile, p);
                    if (p.qz && typeof p.qz === 'object') questionnaire = Object.assign(questionnaire, p.qz);
                }
            } catch(e) {}
            updateProfileUI();
            updateQzUI();
        }
        function saveProfile() {
            try { localStorage.setItem('stockProProfile', JSON.stringify(Object.assign({}, investorProfile, { qz: questionnaire }))); } catch(e) {}
        }
        function setPref(key, val, el) {
            // Used by verdict tab profile toggles only
            var grp = el.closest('.pref-group');
            if (investorProfile[key] === val) {
                investorProfile[key] = null;
                saveProfile();
                grp.querySelectorAll('.pref-btn').forEach(function(b) { b.classList.remove('active'); });
                return;
            }
            investorProfile[key] = val;
            saveProfile();
            grp.querySelectorAll('.pref-btn').forEach(function(b) { b.classList.remove('active'); });
            el.classList.add('active');
        }
        function updateProfileUI() {
            // Sync verdict-tab profile toggles only
            ['horizon', 'risk', 'goal'].forEach(function(key) {
                var grp = document.getElementById('pref-' + key);
                if (!grp) return;
                grp.querySelectorAll('.pref-btn').forEach(function(b) {
                    b.classList.toggle('active', b.dataset.value === investorProfile[key]);
                });
            });
        }

        // ── Questionnaire functions ──────────────────────────────────────────
        function setQ(key, val, el) {
            questionnaire[key] = val;
            document.querySelectorAll('[data-q="' + key + '"]').forEach(function(b) {
                b.classList.toggle('selected', b.dataset.v === val);
            });
            var badge = document.getElementById('qn' + key.replace('q', ''));
            if (badge) badge.classList.add('done');
            resolveProfile();
            updateQzProgress();
            saveProfile();
        }
        function resolveProfile() {
            var a = questionnaire;
            var goalMap  = { beat: 'growth', compounder: 'balanced', income: 'income', tactical: 'growth' };
            var riskMap  = { cut: 'low', calm: 'medium', buy: 'high' };
            var horizMap = { s6m: 'short', s2yr: 'medium', s5yr: 'long', long: 'long' };
            if (a.q1) investorProfile.goal    = goalMap[a.q1]  || investorProfile.goal;
            if (a.q2) investorProfile.risk    = riskMap[a.q2]  || investorProfile.risk;
            if (a.q3) investorProfile.horizon = horizMap[a.q3] || investorProfile.horizon;
            // Tactical + no horizon answer → default to short
            if (a.q1 === 'tactical' && !a.q3) investorProfile.horizon = 'short';
            if (a.q4) investorProfile.mcap   = a.q4;
            if (a.q5) investorProfile.sector = a.q5;
            if (a.q6) investorProfile.view   = a.q6;
        }
        function qzAnswered() {
            return Object.values(questionnaire).filter(function(v) { return v !== null; }).length;
        }
        function updateQzProgress() {
            var n = qzAnswered();
            var pEl  = document.getElementById('qz-progress');
            var cBtn = document.getElementById('qz-cta');
            if (pEl) {
                pEl.innerHTML = n < QZ_TOTAL
                    ? '<strong>' + n + '</strong> of ' + QZ_TOTAL + ' questions answered'
                    : '<strong style="color:var(--accent-green);">&#10003; All questions answered</strong> — your picks are ready';
            }
            if (cBtn) cBtn.classList.toggle('ready', n === QZ_TOTAL);
        }
        function updateQzUI() {
            Object.keys(questionnaire).forEach(function(key) {
                var val = questionnaire[key];
                if (!val) return;
                document.querySelectorAll('[data-q="' + key + '"]').forEach(function(b) {
                    b.classList.toggle('selected', b.dataset.v === val);
                });
                var badge = document.getElementById('qn' + key.replace('q', ''));
                if (badge) badge.classList.add('done');
            });
            updateQzProgress();
        }

        // ===== LIVE STOCK SCANNER BY PROFILE =====
        var _scanAbort = null;
        function getProfileLabels() {
            return {
                horizon: { short: 'Short-Term', medium: 'Medium-Term', long: 'Long-Term' }[investorProfile.horizon] || 'Long-Term',
                risk:    { low: 'Capital Preserving', medium: 'Balanced Risk', high: 'High Risk' }[investorProfile.risk] || 'Balanced Risk',
                goal:    { growth: 'Alpha Growth', income: 'Dividend Income', balanced: 'Steady Compounder' }[investorProfile.goal] || 'Alpha Growth',
                mcap:    { large: 'Large-Cap', mid: 'Large+Mid', small: 'Mid & Small' }[investorProfile.mcap] || 'Large+Mid',
                sector:  { fin: 'Financials', pharma: 'Pharma & Health', infra: 'Infra & Capex', tech: 'Technology', consumer: 'Consumer & Auto', all: 'All Sectors' }[investorProfile.sector] || 'All Sectors',
                view:    { bull: 'Bullish', selective: 'Selective', defensive: 'Defensive' }[investorProfile.view] || 'Selective',
            };
        }
        function getRelevantScore(stScore, ltScore, divScore) {
            var g = investorProfile.goal, h = investorProfile.horizon;
            if (g === 'income') return divScore;
            if (g === 'balanced') return Math.round((stScore + ltScore + divScore) / 3);
            if (h === 'short') return stScore;
            if (h === 'medium') return Math.round((stScore + ltScore) / 2);
            return ltScore;
        }
        function getRelevantLabel() {
            var g = investorProfile.goal, h = investorProfile.horizon;
            if (g === 'income') return 'Dividend';
            if (g === 'balanced') return 'Balanced';
            if (h === 'short') return 'Short-Term';
            if (h === 'medium') return 'Medium-Term';
            return 'Long-Term';
        }
        function passesRiskFilter(stRes, ltRes, dcfD, regr) {
            var r = investorProfile.risk;
            if (!r) return true;
            if (r === 'low') {
                if (dcfD && dcfD.pb_ratio && dcfD.pb_ratio > 8) return false;
                if (regr && regr.beta && regr.beta > 1.5) return false;
            }
            return true;
        }
        // ===== SCANNER TAB — persistent, no popup =====
        var _prefilterES = null;
        var _scanRunning = false;

        function showCuratedStocks() {
            // "Get Stock Suggestions" button in verdict tab → go to Scanner
            switchTab('scanner');
            // Auto-start only if questionnaire is fully answered
            if (!_scanRunning && qzAnswered() === QZ_TOTAL) startScanInTab();
        }
        function startScanInTab() {
            if (_prefilterES) { _prefilterES.close(); _prefilterES = null; }
            _scanAbort = false;
            _scanRunning = true;
            resolveProfile();  // make sure profile reflects latest questionnaire answers
            var labels   = getProfileLabels();
            var relLabel = getRelevantLabel();
            var universeTotal = allTickers.length;
            // Show status card; hide CTA, show stop
            var statusCard = document.getElementById('sc-status-card');
            if (statusCard) statusCard.style.display = '';
            var startBtn = document.getElementById('qz-cta');
            var stopBtn  = document.getElementById('scanner-stop-btn');
            if (startBtn) startBtn.style.display = 'none';
            if (stopBtn)  stopBtn.style.display  = '';
            // Helper refs
            function setSub(msg)   { var el = document.getElementById('sc-sub');          if (el) el.innerHTML  = msg; }
            function setFill(pct)  { var el = document.getElementById('sc-progress-fill'); if (el) el.style.width = pct + '%'; }
            function setStatus(msg){ var el = document.getElementById('sc-status-text');   if (el) el.innerHTML  = msg; }
            function setStats(chk, s1, s2, s3done, s3tot) {
                var el = document.getElementById('sc-stat-row');
                if (!el) return;
                el.innerHTML =
                    '<span>Universe <strong>' + chk + '/' + universeTotal + '</strong></span>'
                  + '<span style="color:var(--accent-cyan);">S1 &#8594; <strong>' + s1 + '</strong></span>'
                  + '<span style="color:var(--accent-green);">S2 &#8594; <strong>' + s2 + '</strong></span>'
                  + '<span>Deep <strong>' + s3done + '/' + s3tot + '</strong></span>';
            }
            setSub('Scanning <strong>' + universeTotal + ' NSE stocks</strong> &mdash; '
                   + '<strong style="color:var(--accent-cyan);">' + labels.goal + '</strong>'
                   + ' &middot; ' + labels.risk
                   + ' &middot; ' + labels.horizon
                   + ' &middot; ' + labels.mcap
                   + ' &middot; ' + labels.sector
                   + ' &middot; ' + labels.view
                   + ' &mdash; surfacing your <strong>top 20</strong>');
            setFill(0);
            setStatus('Connecting to data stream&hellip;');
            // Clear previous results
            var resEl = document.getElementById('scanner-results-area');
            if (resEl) resEl.innerHTML = '';

            var risk    = investorProfile.risk    || 'medium';
            var horizon = investorProfile.horizon || 'long';
            var goal    = investorProfile.goal    || 'growth';
            var mcap    = investorProfile.mcap    || 'mid';
            var sector  = investorProfile.sector  || 'all';
            var view    = investorProfile.view    || 'selective';

            // ── Stage 1: SSE price-action prefilter ─────────────────────────────
            var pfChecked = 0, pfPassed = 0, pfDone = false;

            // ── Stage 2: Mid-filter (quick fundamentals via /midfilter) ─────────
            var midQueue = [], midActive = 0, midCompleted = 0, midTotal = 0, midPassed = 0;
            var midDone  = false;
            var MAX_MID  = 6;

            // ── Stage 3: Deep scan (/analyze + /dcf-data + /dividend-info) ──────
            var deepQueue = [], deepActive = 0, deepCompleted = 0, deepTotal = 0;
            var deepResults = [];
            var MAX_DEEP  = 5;

            var MAX_DISPLAY = 20;  // show top 20 only

            function finishScan() {
                _scanRunning = false;
                if (startBtn) { startBtn.style.display = ''; startBtn.classList.add('ready'); startBtn.innerHTML = '&#128269; Re-scan with This Profile'; }
                if (stopBtn)  stopBtn.style.display = 'none';
            }
            function updateProgress() {
                var bar;
                if (!pfDone) {
                    // Stage 1: 0 – 45 %
                    bar = Math.min(44, Math.round((pfChecked / universeTotal) * 45));
                } else if (midCompleted < midTotal) {
                    // Stage 2: 45 – 70 %
                    bar = 45 + Math.round((midCompleted / Math.max(midTotal, 1)) * 25);
                } else {
                    // Stage 3: 70 – 100 %
                    bar = 70 + Math.round((deepCompleted / Math.max(deepTotal, 1)) * 30);
                }
                setFill(bar);
                setStats(pfChecked, pfPassed, midPassed, deepCompleted, deepTotal);
                if (!pfDone) {
                    setStatus('Stage 1 of 3 &mdash; Applying 6 price-action gates&hellip;');
                } else if (midCompleted < midTotal) {
                    setStatus('Stage 2 of 3 &mdash; Fundamental filter &mdash; '
                              + midCompleted + '/' + midTotal + ' checked, '
                              + midPassed + ' passed&hellip;');
                } else if (deepCompleted < deepTotal) {
                    setStatus('Stage 3 of 3 &mdash; Deep scanning '
                              + deepCompleted + '/' + deepTotal + '&hellip;');
                } else {
                    setFill(100);
                    setStatus('&#9989; Complete &mdash; <strong>' + deepResults.length
                              + '</strong> stocks ranked by <strong>' + relLabel + '</strong> score');
                    finishScan();
                }
            }
            function renderTabResults() {
                if (!resEl) return;
                var display = deepResults.slice(0, MAX_DISPLAY);
                var h = '';
                if (display.length > 0) {
                    h += '<div style="font-size:0.78em;color:var(--text-muted);margin-bottom:10px;text-align:right;">Showing top <strong style="color:var(--accent-cyan);">' + display.length + '</strong> stocks ranked by <strong>' + relLabel + '</strong> score</div>';
                }
                display.forEach(function(r, i) {
                    var scoreColor = r.relevantScore >= 65 ? 'var(--accent-green)' : r.relevantScore >= 40 ? 'var(--warning)' : 'var(--danger)';
                    h += '<div class="scan-row" onclick="openScannerStock(\\'' + r.symbol + '\\')" style="cursor:pointer;">';
                    h += '<div class="scan-rank">' + (i + 1) + '</div>';
                    h += '<div class="scan-info"><div class="scan-name">' + r.name + '</div>';
                    h += '<div class="scan-symbol">' + r.symbol + ' &middot; Best for: <span style="color:' + r.bestColor + ';">' + r.bestLabel + '</span></div></div>';
                    h += '<div class="scan-scores">';
                    h += '<div class="scan-score-item"><span class="scan-score-label">ST</span><span style="color:' + verdictScoreColor(r.stScore) + ';">' + r.stScore + '</span></div>';
                    h += '<div class="scan-score-item"><span class="scan-score-label">LT</span><span style="color:' + verdictScoreColor(r.ltScore) + ';">' + r.ltScore + '</span></div>';
                    h += '<div class="scan-score-item"><span class="scan-score-label">Div</span><span style="color:' + verdictScoreColor(r.divScore) + ';">' + r.divScore + '</span></div>';
                    h += '</div>';
                    h += '<div class="scan-main-score" style="color:' + scoreColor + ';">' + r.relevantScore + '</div>';
                    h += '</div>';
                });
                if (display.length === 0 && pfDone && midDone && deepTotal === 0)
                    h = '<div style="text-align:center;padding:32px;color:var(--text-muted);">No stocks matched your profile. Try adjusting your sector or market view.</div>';
                resEl.innerHTML = h;
            }

            // Stage 3 launcher
            function launchDeepNext() {
                while (deepActive < MAX_DEEP && deepQueue.length > 0 && !_scanAbort) {
                    var sym = deepQueue.shift();
                    deepActive++;
                    scanOneStock(sym).then(function(res) {
                        deepActive--;
                        deepCompleted++;
                        if (_scanAbort) return;
                        if (res && !(investorProfile.goal === 'income' && res.divScore <= 5)) deepResults.push(res);
                        deepResults.sort(function(a, b) { return b.relevantScore - a.relevantScore; });
                        renderTabResults();
                        updateProgress();
                        launchDeepNext();
                    });
                }
            }

            // Stage 2 launcher — uses IIFE to capture sym per iteration
            function launchMidNext() {
                while (midActive < MAX_MID && midQueue.length > 0 && !_scanAbort) {
                    (function(capturedSym) {
                        midActive++;
                        fetch('/midfilter?symbol=' + encodeURIComponent(capturedSym)
                              + '&risk='    + encodeURIComponent(risk)
                              + '&goal='    + encodeURIComponent(goal)
                              + '&horizon=' + encodeURIComponent(horizon)
                              + '&sector='  + encodeURIComponent(sector))
                        .then(function(r) { return r.json(); })
                        .catch(function()  { return { passed: true, symbol: capturedSym }; })
                        .then(function(mf) {
                            midActive--;
                            midCompleted++;
                            if (_scanAbort) return;
                            if (mf && mf.passed) {
                                midPassed++;
                                deepTotal++;
                                deepQueue.push(capturedSym);
                                launchDeepNext();
                            }
                            updateProgress();
                            launchMidNext();
                            // If Stage 1 is done and all mid-filter work is finished
                            if (pfDone && midCompleted >= midTotal) {
                                midDone = true;
                                if (deepTotal === 0) { renderTabResults(); finishScan(); }
                            }
                        });
                    })(midQueue.shift());
                }
            }

            // Stage 1: SSE prefilter stream
            var es = new EventSource('/prefilter-stream?risk=' + encodeURIComponent(risk)
                                    + '&horizon=' + encodeURIComponent(horizon)
                                    + '&goal='    + encodeURIComponent(goal)
                                    + '&mcap='    + encodeURIComponent(mcap)
                                    + '&view='    + encodeURIComponent(view));
            _prefilterES = es;
            es.onmessage = function(evt) {
                if (_scanAbort) { es.close(); _prefilterES = null; return; }
                var d = JSON.parse(evt.data);
                if (d.type === 'pass') {
                    pfPassed++;
                    midTotal++;
                    midQueue.push(d.symbol);
                    launchMidNext();
                    updateProgress();
                } else if (d.type === 'progress') {
                    pfChecked = d.checked;
                    updateProgress();
                } else if (d.type === 'done') {
                    pfDone    = true;
                    pfChecked = d.checked;
                    pfPassed  = d.passed;
                    es.close();
                    _prefilterES = null;
                    updateProgress();
                    if (midTotal === 0) { midDone = true; renderTabResults(); finishScan(); }
                }
            };
            es.onerror = function() {
                es.close(); _prefilterES = null;
                if (!pfDone) {
                    pfDone = true;
                    updateProgress();
                    if (midTotal === 0) { midDone = true; renderTabResults(); finishScan(); }
                }
            };
        }
        function stopScanInTab() {
            _scanAbort = true;
            _scanRunning = false;
            if (_prefilterES) { _prefilterES.close(); _prefilterES = null; }
            var startBtn = document.getElementById('qz-cta');
            var stopBtn  = document.getElementById('scanner-stop-btn');
            if (startBtn) { startBtn.style.display = ''; startBtn.classList.add('ready'); startBtn.innerHTML = '&#128269; Re-scan with This Profile'; }
            if (stopBtn)  stopBtn.style.display = 'none';
            var el = document.getElementById('sc-status-text');
            if (el) el.innerHTML = 'Scan stopped. Click Re-scan to start again.';
        }
        var _verdictReturnTab = null;
        function openScannerStock(symbol) {
            // Remember that we came from the scanner so Back returns there
            _verdictReturnTab = 'scanner';
            switchTab('verdict');
            document.getElementById('verdict-search').value = symbol;
            fetchVerdictData();
        }
        function scanOneStock(symbol) {
            return Promise.all([
                fetch('/analyze?symbol=' + encodeURIComponent(symbol)).then(function(r){return r.json();}).catch(function(){return null;}),
                fetch('/dcf-data?symbol=' + encodeURIComponent(symbol)).then(function(r){return r.json();}).catch(function(){return null;}),
                fetch('/dividend-info?symbol=' + encodeURIComponent(symbol)).then(function(r){return r.json();}).catch(function(){return null;})
            ]).then(function(data) {
                var tech = data[0], dcfD = data[1], divD = data[2];
                if (!tech || tech.error) return null;
                var stRes = scoreShortTerm(tech);
                var ltRes = scoreLongTerm(tech, dcfD, null);
                var divRes = scoreDividend(divD);
                var relevantScore = getRelevantScore(stRes.score, ltRes.score, divRes.score);
                if (!passesRiskFilter(stRes, ltRes, dcfD, null)) relevantScore = Math.round(relevantScore * 0.6);
                var name = (dcfD && dcfD.name) ? dcfD.name : getStockName(symbol);
                var mx = Math.max(stRes.score, ltRes.score, divRes.score);
                var bestLabel, bestColor;
                if (investorProfile.goal === 'income' && divRes.score > 5) {
                    bestLabel = 'Dividend'; bestColor = 'var(--warning)';
                } else {
                    bestLabel = stRes.score === mx ? 'Short-Term' : ltRes.score === mx ? 'Long-Term' : 'Dividend';
                    bestColor = stRes.score === mx ? 'var(--accent-cyan)' : ltRes.score === mx ? 'var(--accent-green)' : 'var(--warning)';
                }
                return { symbol: symbol, name: name, stScore: stRes.score, ltScore: ltRes.score, divScore: divRes.score, relevantScore: relevantScore, bestLabel: bestLabel, bestColor: bestColor };
            }).catch(function() { return null; });
        }

        // ===== BROWSE BY SECTOR - Rich Card UI =====
        const _sectorSkip = new Set(['Nifty 50', 'Nifty Next 50', 'Conglomerate']);
        let _activeSector = null;
        let _sectorOffset = 0;
        const _sectorPageSize = 20;
        const _sectorQuoteCache = {};

        function _formatMcap(mcap) {
            if (!mcap) return '';
            if (mcap >= 1e12) return '\u20B9' + (mcap / 1e12).toFixed(1) + 'L Cr';
            if (mcap >= 1e9) return '\u20B9' + (mcap / 1e7 / 100).toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ',') + ' Cr';
            if (mcap >= 1e7) return '\u20B9' + (mcap / 1e7).toFixed(0) + ' Cr';
            return '';
        }

        function _stockCardHtml(sym, quote) {
            const name = (quote && quote.name) || getStockName(sym);
            const price = (quote && quote.price) ? '\u20B9' + quote.price.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '';
            const chg = (quote && quote.change_pct) || 0;
            const chgStr = chg >= 0 ? '+' + chg.toFixed(2) + '%' : chg.toFixed(2) + '%';
            const chgClass = chg >= 0 ? 'up' : 'down';
            const chgIcon = chg >= 0 ? '\u2197' : '\u2198';
            const mcap = (quote && quote.mcap) ? _formatMcap(quote.mcap) : '';
            const loadingClass = quote ? '' : ' sc-loading';
            return `<div class="stock-card${loadingClass}" onclick="analyze('${sym}')" data-sym="${sym}">
                <div class="sc-top"><span class="sc-symbol">${sym}</span><span class="sc-change ${chgClass}">${chgIcon} ${chgStr}</span></div>
                <div class="sc-name">${name}</div>
                <div class="sc-bottom"><span class="sc-price">${price || '\u20B9---'}</span><span class="sc-mcap">${mcap}</span></div>
            </div>`;
        }

        function initSectorBrowser() {
            const pillsEl = document.getElementById('sector-pills-container');
            if (!pillsEl) return;
            // Build sector list: All NSE first, real sectors, then Others last
            const sectorOrder = [];
            const othersEntry = [];
            Object.keys(stocks).forEach(s => {
                if (_sectorSkip.has(s)) return;
                if (s === 'All NSE') return;          // handled separately below
                if (s === 'Others') { othersEntry.push(s); return; }
                sectorOrder.push(s);
            });
            // "All NSE" first if it exists
            if (stocks['All NSE']) sectorOrder.unshift('All NSE');
            sectorOrder.push(...othersEntry);

            let pillsHtml = '';
            sectorOrder.forEach((s, i) => {
                const active = i === 0 ? ' active' : '';
                pillsHtml += `<button class="sector-pill${active}" data-sector="${s}" onclick="switchSector('${s.replace(/'/g, "\\'")}')">${s} (${stocks[s].length})</button>`;
            });
            pillsEl.innerHTML = pillsHtml;

            // Load first sector
            if (sectorOrder.length) switchSector(sectorOrder[0]);
        }

        function switchSector(sector) {
            _activeSector = sector;
            _sectorOffset = 0;
            // Update pill active state
            document.querySelectorAll('.sector-pill').forEach(p => {
                p.classList.toggle('active', p.dataset.sector === sector);
            });
            const container = document.getElementById('sector-cards-container');
            container.innerHTML = '';
            _renderSectorPage(sector, true);
        }

        function _renderSectorPage(sector, fresh) {
            const container = document.getElementById('sector-cards-container');
            const btn = document.getElementById('sector-load-more-btn');
            const tickers = stocks[sector] || [];
            const page = tickers.slice(_sectorOffset, _sectorOffset + _sectorPageSize);

            if (!page.length) { btn.style.display = 'none'; return; }

            // Render placeholder cards immediately
            let cardsHtml = '';
            page.forEach(sym => {
                const cached = _sectorQuoteCache[sym];
                cardsHtml += _stockCardHtml(sym, cached || null);
            });
            if (fresh) container.innerHTML = cardsHtml;
            else container.insertAdjacentHTML('beforeend', cardsHtml);

            _sectorOffset += page.length;
            btn.style.display = (_sectorOffset < tickers.length) ? 'block' : 'none';
            btn.textContent = 'Load more (' + (tickers.length - _sectorOffset) + ' remaining)';

            // Fetch live quotes for uncached symbols
            const uncached = page.filter(s => !_sectorQuoteCache[s]);
            if (uncached.length) {
                fetch('/sector-quotes?sector=' + encodeURIComponent(sector) + '&offset=' + (_sectorOffset - page.length) + '&limit=' + page.length)
                    .then(r => r.json())
                    .then(data => {
                        if (_activeSector !== sector) return; // user switched away
                        (data.quotes || []).forEach(q => {
                            _sectorQuoteCache[q.symbol] = q;
                            const card = container.querySelector('[data-sym="' + q.symbol + '"]');
                            if (card) {
                                card.outerHTML = _stockCardHtml(q.symbol, q);
                            }
                        });
                    }).catch(() => {});
            }
        }

        function loadMoreSectorStocks() {
            if (_activeSector) _renderSectorPage(_activeSector, false);
        }

        let currentTab = 'analysis';
        let loadedTabs = new Set();
        function ensureTabLoaded(tab) {
            if (loadedTabs.has(tab)) return;
            if (tab === 'analysis') {
                initSectorBrowser();
                setupAutocomplete('search', 'suggestions', 'analyze');
            } else if (tab === 'regression') {
                setupAutocomplete('regression-search', 'regression-suggestions', 'analyzeRegression');
                document.getElementById('regression-search').addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') analyzeRegression();
                });
            } else if (tab === 'dividend') {
            } else if (tab === 'dcf') {
                setupAutocomplete('dcf-search', 'dcf-suggestions', 'dcfAutocomplete');
                document.getElementById('dcf-search').addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') fetchDCFData();
                });
            } else if (tab === 'verdict') {
                setupAutocomplete('verdict-search', 'verdict-suggestions', 'verdictAutocomplete');
                document.getElementById('verdict-search').addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') fetchVerdictData();
                });
            }
            loadedTabs.add(tab);
        }
        function switchTab(tab, event) {
            currentTab = tab;
            // Update desktop nav links
            document.querySelectorAll('.nav-link').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.nav-link[data-tab="' + tab + '"]').forEach(t => t.classList.add('active'));
            // Update mobile menu items
            document.querySelectorAll('.mobile-menu-item').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.mobile-menu-item[data-tab="' + tab + '"]').forEach(t => t.classList.add('active'));
            // Update tab content
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(tab + '-tab').classList.add('active');
            ensureTabLoaded(tab);
        }
        // toggleMobileMenu / closeMobileMenu are wired up by the head-loaded
        // script using document-level click delegation, so the hamburger keeps
        // working even if this larger script block fails to load.
        function setupAutocomplete(inputId, suggestionId, callbackName) {
            const input = document.getElementById(inputId);
            if(!input) return;
            input.addEventListener('input', (e) => {
                const raw = e.target.value.trim();
                const q = raw.toUpperCase();
                const sug = document.getElementById(suggestionId);
                if (q.length === 0) { sug.innerHTML = ''; return; }
                const filtered = allTickers.filter(s => {
                    const name = getStockName(s).toUpperCase();
                    return s.includes(q) || name.includes(q);
                }).slice(0, 12);
                sug.innerHTML = filtered.map(s => {
                    const label = `${s} <span style='font-size:0.8em;color:var(--text-muted);'>${getStockName(s)}</span>`;
                    if(callbackName === 'analyzeRegression') return `<button onclick="document.getElementById('${inputId}').value = '${s}'; analyzeRegression();">${label}</button>`;
                    else if(callbackName === 'dcfAutocomplete') return `<button onclick="document.getElementById('${inputId}').value = '${s}'; fetchDCFData();">${label}</button>`;
                    else if(callbackName === 'verdictAutocomplete') return `<button onclick="document.getElementById('${inputId}').value = '${s}'; fetchVerdictData();">${label}</button>`;
                    else return `<button onclick="analyze('${s}')">${label}</button>`;
                }).join('');
            });
        }
        function init() {
            ensureTabLoaded('verdict');
        }
        function analyze(symbol) {
            document.getElementById('search-view').style.display = 'none';
            document.getElementById('result-view').style.display = 'block';
            document.getElementById('result').innerHTML = '<div class="loading">⏳ Analyzing ' + symbol + '...</div>';
            fetch(`/analyze?symbol=${symbol}`)
                .then(r => r.json())
                .then(data => {
                    if (data.error) document.getElementById('result').innerHTML = `<div class="error">❌ ${data.error}</div>`;
                    else showResult(data, symbol);
                })
                .catch(e => document.getElementById('result').innerHTML = `<div class="error">❌ ${e.message}</div>`);
        }
        function showResult(data, symbol) {
            if (!data || !data.signal) { document.getElementById('result').innerHTML = '<div class="error">Invalid response data</div>'; return; }
            const s = data.signal || {};
            const d = data.details || {};
            const dailyRaw = d.daily_raw || parseFloat(d.daily) || 0;
            const dailyClass = dailyRaw >= 0 ? 'up' : 'down';
            const dailySign = dailyRaw >= 0 ? '+' : '';
            const riskPer = s.risk_per_share || Math.abs((d.price_raw || 0) - (s.stop_raw || 0));
            const expectedPct = s.expected_move_pct || 0;  // always positive magnitude
            const expectedSigned = s.expected_move_signed != null ? s.expected_move_signed : expectedPct;
            const expectedColor = expectedSigned >= 0 ? 'green' : 'red';
            const expectedSign = expectedSigned >= 0 ? '+' : '';
            const maxRiskPct = s.max_risk_pct || 0;
            const rrRatio = s.risk_reward || 0;
            const rrGreenWidth = rrRatio > 0 ? Math.min((expectedPct / (expectedPct + maxRiskPct)) * 100, 95) : 50;
            // SMA status text
            const smaAbove = [];
            if (d.above_sma20) smaAbove.push('20');
            if (d.above_sma50) smaAbove.push('50');
            const smaStatus = smaAbove.length === 2 ? 'Above 20 & 50 SMA' : smaAbove.length === 1 ? 'Above ' + smaAbove[0] + ' SMA' : 'Below 20 & 50 SMA';
            const html = `
                <div class="tsc">
                    <!-- HEADER -->
                    <div class="tsc-header">
                        <div>
                            <h2 class="tsc-ticker">${symbol}</h2>
                            <div class="tsc-price-row">
                                <span class="tsc-price"><span style="font-size:0.75em;color:var(--text-muted);font-weight:400;">CMP</span> ${d.price}</span>
                                <span class="tsc-change ${dailyClass} tsc-tip">${dailySign}${dailyRaw.toFixed(2)}%<span class="tsc-tip-text" style="width:220px;bottom:calc(100% + 8px);">Change from previous trading day's closing price.</span></span>
                            </div>
                        </div>
                        <div class="tsc-header-right">
                            <div class="tsc-badge tsc-badge-${s.signal}">${s.signal}</div>
                        </div>
                    </div>
                    <div class="tsc-body">
                        <!-- SETUP BANNER -->
                        <div class="tsc-setup-banner">${s.setup_duration || 'Short Term Setup'} &bull; ${s.days_to_target} Days</div>

                        <!-- THE BOTTOM LINE (verdict) -->
                        ${s.verdict_text ? '<div class="tsc-verdict tsc-verdict-' + s.signal + '">' +
                            '<div class="tsc-verdict-header">' +
                                '<span class="tsc-verdict-title">The Bottom Line</span>' +
                            '</div>' +
                            '<div class="tsc-verdict-body">' + s.verdict_text + '</div>' +
                        '</div>' : ''}

                        <!-- CONFIDENCE CARD -->
                        <div class="tsc-confidence-card">
                            <div class="tsc-confidence-top">
                                <div class="tsc-signal-label tsc-signal-label-${s.signal}">${s.signal}</div>
                                <div class="tsc-confidence-info">
                                    <div class="tsc-confidence-pct">Confidence <strong>${s.confidence}%</strong></div>
                                    <div class="tsc-confidence-hint">${s.confidence_oneliner || ''}</div>
                                </div>
                            </div>
                        </div>

                        <!-- RISK-REWARD GRID -->
                        <div class="tsc-rr-grid">
                            <div class="tsc-rr-item tsc-tip">
                                <div class="tsc-rr-label">Expected Move <span class="tsc-tip-icon">?</span></div>
                                <div class="tsc-rr-value ${expectedColor}">${expectedSign}${expectedSigned}%</div>
                                <div class="tsc-rr-bar"><div class="tsc-rr-bar-fill" style="width:${rrGreenWidth}%;background:${expectedSigned >= 0 ? 'var(--accent-green)' : 'var(--danger)'};"></div></div>
                                <div class="tsc-tip-text">${s.expected_move_tooltip || ''}</div>
                            </div>
                            <div class="tsc-rr-item tsc-tip">
                                <div class="tsc-rr-label">Max Risk <span class="tsc-tip-icon">?</span></div>
                                <div class="tsc-rr-value red">-${maxRiskPct}%</div>
                                <div class="tsc-rr-bar"><div class="tsc-rr-bar-fill" style="width:${100-rrGreenWidth}%;background:var(--danger);"></div></div>
                                <div class="tsc-tip-text">${s.max_risk_tooltip || ''}</div>
                            </div>
                            <div class="tsc-rr-item tsc-tip">
                                <div class="tsc-rr-label">Risk-Reward <span class="tsc-tip-icon">?</span></div>
                                <div class="tsc-rr-value ${rrRatio >= 1.5 ? 'green' : rrRatio >= 1 ? 'neutral' : 'red'}">${rrRatio}x</div>
                                <div class="tsc-rr-bar"><div class="tsc-rr-bar-fill" style="width:${Math.min(rrRatio/(rrRatio+1)*100,95)}%;background:linear-gradient(90deg,var(--accent-green),var(--accent-cyan));"></div></div>
                                <div class="tsc-tip-text">${s.risk_reward_tooltip || ''}</div>
                            </div>
                        </div>

                        <!-- WHY THIS MAKES SENSE -->
                        <div class="tsc-why">
                            <h3>Why This Makes Sense</h3>
                            <p>${s.why_makes_sense || s.rec}</p>
                        </div>

                        <!-- REGIME LAYER -->
                        ${(function(){
                            if (!s.market_regime) return '';
                            const mRegime = s.market_regime || 'neutral';
                            const sRegime = s.sector_regime || 'neutral';
                            const regimeScore = s.regime_score != null ? (s.regime_score * 100).toFixed(0) : '--';
                            const regimeFactor = s.regime_factor != null ? s.regime_factor.toFixed(1) : '1.0';
                            const mClass = 'tsc-regime-' + mRegime;
                            const sClass = 'tsc-regime-' + sRegime;
                            const barColor = s.regime_score >= 0.6 ? 'var(--accent-green)' : s.regime_score >= 0.4 ? 'var(--warning)' : 'var(--danger)';
                            const barWidth = Math.max(5, Math.min((s.regime_score || 0.5) * 100, 100));
                            const origSig = s.original_signal || '';
                            const overrideHtml = origSig ? '<div class="tsc-regime-override">Signal overridden: <strong>' + origSig + ' \\u2192 ' + s.signal + '</strong> due to regime conflict.</div>' : '';
                            return '<div class="tsc-regime">'
                                + '<h3>Market &amp; Sector Regime</h3>'
                                + overrideHtml
                                + '<div class="tsc-regime-grid">'
                                + '  <div class="tsc-regime-item">'
                                + '    <div class="tsc-regime-item-label">Market (Nifty 50)</div>'
                                + '    <div class="tsc-regime-item-value ' + mClass + '">' + mRegime.toUpperCase() + '</div>'
                                + '  </div>'
                                + '  <div class="tsc-regime-item">'
                                + '    <div class="tsc-regime-item-label">Sector</div>'
                                + '    <div class="tsc-regime-item-value ' + sClass + '">' + sRegime.toUpperCase() + '</div>'
                                + '  </div>'
                                + '</div>'
                                + '<div class="tsc-regime-meta">'
                                + '  <span>Regime Score: <strong>' + regimeScore + '%</strong></span>'
                                + '  <span>Risk Factor: <strong>x' + regimeFactor + '</strong></span>'
                                + '</div>'
                                + '<div class="tsc-regime-bar"><div class="tsc-regime-bar-fill" style="width:' + barWidth + '%;background:' + barColor + ';"></div></div>'
                                + '<div class="tsc-regime-reason">' + (s.regime_reason_text || '') + '</div>'
                                + '</div>';
                        })()}

                        <!-- TRADE LEVELS (moved up for visibility) -->
                        <div class="tsc-calc-details" style="margin-bottom:20px;">
                            <div class="tsc-calc-detail-item">
                                <div class="tsc-calc-detail-label">Exit Price</div>
                                <div class="tsc-calc-detail-value" style="color:var(--accent-green);">${s.target}</div>
                            </div>
                            <div class="tsc-calc-detail-item">
                                <div class="tsc-calc-detail-label">Stop Loss</div>
                                <div class="tsc-calc-detail-value" style="color:var(--danger);">${s.stop}</div>
                            </div>
                            <div class="tsc-calc-detail-item">
                                <div class="tsc-calc-detail-label">Time Frame</div>
                                <div class="tsc-calc-detail-value">${s.days_to_target} days</div>
                            </div>
                        </div>

                        <!-- PRICE PROJECTION CHART -->
                        <div id="tsc-projection-chart" style="display:none;"></div>

                        <!-- CAPITAL CALCULATOR -->
                        <div class="tsc-calc">
                            <div class="tsc-calc-header">
                                <div class="tsc-calc-check">&#10003;</div>
                                <div class="tsc-calc-title">Capital Calculator</div>
                            </div>

                            <div class="tsc-auto-risk tsc-tip">
                                <div class="tsc-auto-risk-badge" id="tsc-risk-pct-label">${s.rec_risk_pct || 1}%</div>
                                <div class="tsc-auto-risk-label">
                                    <strong>Recommended risk per trade</strong><br>
                                    <span style="font-size:0.9em;">Computed from confidence, risk-reward &amp; volatility. Hover for details.</span>
                                </div>
                                <div class="tsc-tip-text">${s.risk_reason_text || 'Standard 1% risk per trade.'}</div>
                            </div>

                            <div class="tsc-calc-row">
                                <div class="tsc-calc-input-wrap">
                                    <input type="text" class="tsc-calc-input" id="tsc-capital-input" placeholder="Enter total capital (e.g. 5,00,000)" oninput="formatTscCapital(this); updateCapitalCalc()">
                                </div>
                                <div class="tsc-capital-display" id="tsc-capital-display"></div>
                            </div>

                            <div class="tsc-calc-info-box" style="margin-bottom:10px;">
                                <span class="tsc-calc-info-label">Risk per share:</span>
                                <span class="tsc-calc-info-value" style="color:var(--danger);" id="tsc-risk-per-share">&#8377;${riskPer.toFixed(2)}</span>
                            </div>

                            <div class="tsc-calc-result">
                                <span class="tsc-calc-result-label">Suggested quantity:</span>
                                <span class="tsc-calc-result-value" id="tsc-qty-result">-- shares</span>
                            </div>
                        </div>

                        <!-- TECHNICAL DETAILS ACCORDION -->
                        <div class="tsc-accordion">
                            <button class="tsc-accordion-toggle" onclick="toggleTscAccordion(this)">
                                Technical Details (click to expand) <span class="tsc-arrow">&#9660;</span>
                            </button>
                            <div class="tsc-accordion-content" id="tsc-tech-details">
                                <div class="tsc-accordion-inner">
                                    <!-- LOCAL MINIMA/MAXIMA CONFIDENCE SCORE -->
                                    <div class="tsc-tech-item" style="border:1px solid ${d.minmax_type === 'BOTTOM' ? 'var(--accent-green)' : d.minmax_type === 'TOP' ? 'var(--danger)' : 'var(--warning)'};border-radius:10px;padding:16px;">
                                        <div class="tsc-tech-item-header">
                                            <span class="tsc-tech-item-name" style="font-size:1.05em;">Turning Point Confidence Score</span>
                                            <span class="tsc-tech-item-value" style="font-size:1.2em;font-weight:700;color:${d.minmax_type === 'BOTTOM' ? 'var(--accent-green)' : d.minmax_type === 'TOP' ? 'var(--danger)' : 'var(--warning)'};">
                                                ${d.minmax_type === 'BOTTOM' ? 'Bottom' : d.minmax_type === 'TOP' ? 'Top' : 'Mixed Signals'}: ${d.minmax_score}/5
                                            </span>
                                        </div>
                                        <div style="margin:10px 0 6px;font-size:0.88em;line-height:1.7;color:var(--text-secondary);">
                                            ${(function(){
                                                var checks = d.confidence_checks || {};
                                                var lines = '';
                                                var order = ['rsi','williams','bb','volume','divergence'];
                                                for(var ci=0;ci<order.length;ci++){
                                                    var ck = checks[order[ci]];
                                                    if(ck) lines += '<div>' + (ck.met ? '✅' : '❌') + ' ' + ck.label + '</div>';
                                                }
                                                return lines;
                                            })()}
                                        </div>
                                        <div class="tsc-tech-item-example">
                                            <strong>What is this score?</strong> This score counts how many independent signals are agreeing that a turning point may be near. No single indicator is reliable alone — but when 4 or 5 are all flashing the same warning at the same time, the probability of a local bottom or top rises significantly. Think of it like a weather forecast: one cloud doesn't mean rain, but five clouds, falling pressure, and humidity all together usually do.
                                        </div>
                                    </div>
                                    <!-- RSI -->
                                    <div class="tsc-tech-item">
                                        <div class="tsc-tech-item-header">
                                            <span class="tsc-tech-item-name">RSI (Relative Strength Index)</span>
                                            <span class="tsc-tech-item-value" style="color:${d.rsi_raw > 70 ? 'var(--danger)' : d.rsi_raw < 30 ? 'var(--accent-green)' : 'var(--text-primary)'};">${d.rsi}</span>
                                        </div>
                                        <div class="tsc-tech-item-explain">
                                            ${s.rsi_explain}
                                        </div>
                                        <div class="tsc-tech-item-example">
                                            <strong>What is RSI?</strong> Think of RSI like a speedometer for a stock. It measures how fast the price has been going up or down on a scale of 0-100. Above 70 means the stock has been running too fast (overbought) and might need to rest. Below 30 means it's been beaten down too much (oversold) and might bounce back. Between 30-70 is normal cruising speed.
                                        </div>
                                    </div>
                                    <!-- MACD -->
                                    <div class="tsc-tech-item">
                                        <div class="tsc-tech-item-header">
                                            <span class="tsc-tech-item-name">MACD (Moving Average Convergence Divergence)</span>
                                            <span class="tsc-tech-item-value" style="color:${d.macd_bullish ? 'var(--accent-green)' : 'var(--danger)'};">${d.macd}</span>
                                        </div>
                                        <div class="tsc-tech-item-explain">
                                            ${s.macd_text}
                                        </div>
                                        <div class="tsc-tech-item-example">
                                            <strong>What is MACD?</strong> Imagine two runners, one fast and one slow. MACD tracks the gap between them. When the fast runner pulls ahead (Bullish), it means momentum is building upward, like a car accelerating. When the slow runner catches up (Bearish), the stock is losing steam. It's one of the most reliable ways to spot when a trend is gaining or losing strength.
                                        </div>
                                    </div>
                                    <!-- SMA Status -->
                                    <div class="tsc-tech-item">
                                        <div class="tsc-tech-item-header">
                                            <span class="tsc-tech-item-name">Moving Averages (SMA 20 & 50)</span>
                                            <span class="tsc-tech-item-value" style="color:${d.above_sma20 && d.above_sma50 ? 'var(--accent-green)' : !d.above_sma20 && !d.above_sma50 ? 'var(--danger)' : 'var(--warning)'};">${smaStatus}</span>
                                        </div>
                                        <div class="tsc-tech-item-explain">
                                            ${s.trend_explain}
                                        </div>
                                        <div class="tsc-tech-item-example">
                                            <strong>What are Moving Averages?</strong> A moving average smooths out daily price noise to show the real trend. The 20-day SMA shows the short-term trend (like last month's direction), while the 50-day SMA shows the bigger picture. When the price is above both, it's like a boat sailing with the current and the trend is your friend. Below both means you're swimming against the tide.
                                        </div>
                                    </div>
                                    <!-- Z-Score -->
                                    <div class="tsc-tech-item">
                                        <div class="tsc-tech-item-header">
                                            <span class="tsc-tech-item-name">Z-Score (Mean Reversion)</span>
                                            <span class="tsc-tech-item-value" style="color:${d.zscore_raw > 2 ? 'var(--danger)' : d.zscore_raw < -2 ? 'var(--accent-green)' : 'var(--text-primary)'};">${d.zscore}</span>
                                        </div>
                                        <div class="tsc-tech-item-explain">
                                            ${s.zscore_explain}
                                        </div>
                                        <div class="tsc-tech-item-example">
                                            <strong>What is Z-Score?</strong> Think of a rubber band stretched between your fingers. The further you pull it, the harder it snaps back. Z-Score measures how far a stock price has stretched from its average. A score above +2 means it's stretched too far up (likely to snap back down). Below -2 means it's pulled too far down (likely to bounce up). Near 0 means it's at its comfortable resting point.
                                        </div>
                                    </div>
                                    <!-- Bollinger Bands -->
                                    <div class="tsc-tech-item">
                                        <div class="tsc-tech-item-header">
                                            <span class="tsc-tech-item-name">Bollinger Bands</span>
                                            <span class="tsc-tech-item-value" style="color:${d.bb_position > 80 ? 'var(--danger)' : d.bb_position < 20 ? 'var(--accent-green)' : 'var(--text-primary)'};">${d.bb_label || 'Middle'}</span>
                                        </div>
                                        <div class="tsc-tech-item-explain">
                                            ${s.bb_explain}
                                        </div>
                                        <div class="tsc-tech-item-example">
                                            <strong>What are Bollinger Bands?</strong> Picture a highway with lanes. The middle lane is the average price, and the outer lanes (upper and lower bands) represent where the price "usually" stays. When the price drives onto the shoulder (touches the upper band), it's probably going too fast and will merge back. When it drifts to the other shoulder (lower band), it's likely to bounce back toward the center. About 95% of price action stays within these bands.
                                        </div>
                                    </div>
                                    <!-- Williams %R -->
                                    <div class="tsc-tech-item">
                                        <div class="tsc-tech-item-header">
                                            <span class="tsc-tech-item-name">Williams %R</span>
                                            <span class="tsc-tech-item-value" style="color:${d.williams_r_signal === 'POTENTIAL BOTTOM' ? 'var(--accent-green)' : d.williams_r_signal === 'POTENTIAL TOP' ? 'var(--danger)' : 'var(--text-primary)'};">${d.williams_r_signal}</span>
                                        </div>
                                        <div class="tsc-tech-item-explain">
                                            ${s.williams_r_explain}
                                        </div>
                                        <div class="tsc-tech-item-example">
                                            <strong>What is Williams %R?</strong> Think of Williams %R like a bungee cord. When the cord is fully stretched down (below -80), the snap back upward is likely. When it's fully stretched upward (above -20), gravity tends to pull it back down. It's one of the most sensitive indicators for spotting turning points before they happen.
                                        </div>
                                    </div>
                                    <!-- Bollinger %B -->
                                    <div class="tsc-tech-item">
                                        <div class="tsc-tech-item-header">
                                            <span class="tsc-tech-item-name">Bollinger Band %B</span>
                                            <span class="tsc-tech-item-value" style="color:${d.percent_b_signal === 'OVERSOLD' ? 'var(--accent-green)' : d.percent_b_signal === 'OVERBOUGHT' ? 'var(--danger)' : d.percent_b_signal === 'LOWER HALF' ? 'var(--warning)' : 'var(--text-primary)'};">${d.percent_b_signal}</span>
                                        </div>
                                        <div class="tsc-tech-item-explain">
                                            ${s.percent_b_explain}
                                        </div>
                                        <div class="tsc-tech-item-example">
                                            <strong>What is Bollinger %B?</strong> Bollinger %B tells you exactly where price sits within its normal range, expressed as a number between 0 and 1. Zero means you're at the bottom boundary, 1 means you're at the top. When it goes negative, price has broken below its statistical range — like a ball pushed underwater, it tends to float back up.
                                        </div>
                                    </div>
                                    <!-- RSI Divergence -->
                                    <div class="tsc-tech-item">
                                        <div class="tsc-tech-item-header">
                                            <span class="tsc-tech-item-name">RSI Divergence</span>
                                            <span class="tsc-tech-item-value" style="color:${d.rsi_divergence === 'BULLISH DIVERGENCE' ? 'var(--accent-green)' : d.rsi_divergence === 'BEARISH DIVERGENCE' ? 'var(--danger)' : 'var(--text-primary)'};">${d.rsi_divergence === 'NONE' ? 'NO DIVERGENCE' : d.rsi_divergence}</span>
                                        </div>
                                        <div class="tsc-tech-item-explain">
                                            ${s.rsi_divergence_explain}
                                        </div>
                                        <div class="tsc-tech-item-example">
                                            <strong>What is RSI Divergence?</strong> Divergence is when price and momentum stop agreeing. If price falls to a new low but RSI refuses to follow, it means sellers are losing energy even as price drops — like a wave that looks big but has no force behind it. This is often the earliest warning sign of a reversal.
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- SECOND ACCORDION: RAW DATA -->
                        <div class="tsc-accordion">
                            <button class="tsc-accordion-toggle" onclick="toggleTscAccordion(this)">
                                Raw Data & Price Levels <span class="tsc-arrow">&#9660;</span>
                            </button>
                            <div class="tsc-accordion-content">
                                <div class="tsc-accordion-inner">
                                    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px;">
                                        <div class="tsc-calc-detail-item"><div class="tsc-calc-detail-label">Current Price</div><div class="tsc-calc-detail-value">${d.price}</div></div>
                                        <div class="tsc-calc-detail-item"><div class="tsc-calc-detail-label">Daily Change</div><div class="tsc-calc-detail-value" style="color:${dailyRaw>=0?'var(--accent-green)':'var(--danger)'};">${d.daily}</div></div>
                                        <div class="tsc-calc-detail-item"><div class="tsc-calc-detail-label">20-Day Mean</div><div class="tsc-calc-detail-value">${d.mean}</div></div>
                                        <div class="tsc-calc-detail-item"><div class="tsc-calc-detail-label">SMA 9</div><div class="tsc-calc-detail-value">${d.sma9}</div></div>
                                        <div class="tsc-calc-detail-item"><div class="tsc-calc-detail-label">SMA 20</div><div class="tsc-calc-detail-value">${d.sma20 || d.sma9}</div></div>
                                        <div class="tsc-calc-detail-item"><div class="tsc-calc-detail-label">SMA 50</div><div class="tsc-calc-detail-value">${d.sma50 || d.sma9}</div></div>
                                        <div class="tsc-calc-detail-item"><div class="tsc-calc-detail-label">18-Day High</div><div class="tsc-calc-detail-value">${d.high}</div></div>
                                        <div class="tsc-calc-detail-item"><div class="tsc-calc-detail-label">18-Day Low</div><div class="tsc-calc-detail-value">${d.low}</div></div>
                                        <div class="tsc-calc-detail-item"><div class="tsc-calc-detail-label">BB Upper</div><div class="tsc-calc-detail-value">${d.bb_upper}</div></div>
                                        <div class="tsc-calc-detail-item"><div class="tsc-calc-detail-label">BB Lower</div><div class="tsc-calc-detail-value">${d.bb_lower}</div></div>
                                        <div class="tsc-calc-detail-item"><div class="tsc-calc-detail-label">% from Mean</div><div class="tsc-calc-detail-value" style="color:${parseFloat(d.pct_deviation)>=0?'var(--accent-cyan)':'var(--accent-purple)'};">${d.pct_deviation}</div></div>
                                        <div class="tsc-calc-detail-item"><div class="tsc-calc-detail-label">Volatility</div><div class="tsc-calc-detail-value">${d.volatility}</div></div>
                                        <div class="tsc-calc-detail-item"><div class="tsc-calc-detail-label">Bollinger %B</div><div class="tsc-calc-detail-value" style="color:${d.percent_b < 0 ? 'var(--accent-green)' : d.percent_b > 1 ? 'var(--danger)' : 'var(--text-primary)'};">${d.percent_b !== undefined ? d.percent_b.toFixed(4) : 'N/A'}</div></div>
                                        <div class="tsc-calc-detail-item"><div class="tsc-calc-detail-label">Williams %R</div><div class="tsc-calc-detail-value" style="color:${d.williams_r < -80 ? 'var(--accent-green)' : d.williams_r > -20 ? 'var(--danger)' : 'var(--text-primary)'};">${d.williams_r !== undefined ? d.williams_r.toFixed(1) : 'N/A'}</div></div>
                                    </div>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            `;
            document.getElementById('result').innerHTML = html + buildPeerStocksHTML(symbol, 'technical');
            // Store signal data for capital calculator
            window._tscSignalData = { riskPerShare: riskPer, price: d.price_raw || 0, stop: s.stop_raw || 0, target: s.target_raw || 0, recRiskPct: s.rec_risk_pct || 1 };
            // Render projection chart if available
            if (data.projection_chart) {
                const chartDiv = document.getElementById('tsc-projection-chart');
                if (chartDiv) {
                    chartDiv.style.display = 'block';
                    chartDiv.innerHTML = '<div style="background:var(--bg-card-hover);border-radius:12px;border:1px solid var(--border-color);padding:18px 20px;margin-bottom:20px;">'
                        + '<img src="data:image/png;base64,' + data.projection_chart + '" style="width:100%;border-radius:8px;" alt="Price projection chart">'
                        + '<p style="color:var(--text-muted);font-size:0.78em;margin:10px 0 0;line-height:1.5;font-style:italic;">Forecast assumes current trend conditions remain unchanged. The shaded band shows the probable price range based on recent volatility. This is not a guarantee of future movement.</p>'
                        + '</div>';
                }
            }
        }
        function toggleTscAccordion(btn) {
            const content = btn.nextElementSibling;
            btn.classList.toggle('open');
            content.classList.toggle('open');
        }
        function formatTscCapital(input) {
            let raw = input.value.replace(/,/g, '').replace(/[^0-9]/g, '');
            if (raw === '') { input.value = ''; return; }
            // Indian number formatting
            let num = raw;
            let lastThree = num.substring(num.length - 3);
            let otherNumbers = num.substring(0, num.length - 3);
            if (otherNumbers !== '') lastThree = ',' + lastThree;
            input.value = otherNumbers.replace(/\\B(?=(\\d{2})+(?!\\d))/g, ',') + lastThree;
        }
        function updateCapitalCalc() {
            const inp = document.getElementById('tsc-capital-input');
            const display = document.getElementById('tsc-capital-display');
            const qtyEl = document.getElementById('tsc-qty-result');
            if (!inp || !qtyEl) return;
            const riskPct = window._tscSignalData ? (window._tscSignalData.recRiskPct || 1) : 1;
            const raw = inp.value.replace(/,/g, '');
            const capital = parseFloat(raw);
            if (!capital || capital <= 0 || !window._tscSignalData) {
                display.textContent = '';
                qtyEl.textContent = '-- shares';
                return;
            }
            // Format display in Indian numbering
            let formatted = Math.round(capital).toString();
            let lastThree = formatted.substring(formatted.length - 3);
            let otherNumbers = formatted.substring(0, formatted.length - 3);
            if (otherNumbers !== '') lastThree = ',' + lastThree;
            const indianFormatted = otherNumbers.replace(/\\B(?=(\\d{2})+(?!\\d))/g, ',') + lastThree;
            display.innerHTML = '&#8377;' + indianFormatted;
            const riskPerShare = window._tscSignalData.riskPerShare;
            if (riskPerShare > 0) {
                const riskAmount = capital * (riskPct / 100);
                const qty = Math.floor(riskAmount / riskPerShare);
                qtyEl.textContent = qty.toLocaleString('en-IN') + ' shares';
            } else {
                qtyEl.textContent = '-- shares';
            }
        }
        function analyzeRegression() {
            const symbol = document.getElementById('regression-search').value.toUpperCase().trim();
            if (!symbol) { alert('Please enter a stock symbol'); return; }
            const target = document.getElementById('regression-result');
            target.innerHTML = '<div class="loading">⏳ Analysing market connection for ' + symbol + '...<br><small style="font-size: 0.8em; color: var(--text-secondary);">Heavy calculations run async on free-tier. Showing result when ready.</small></div>';

            const poll = () => fetch(`/regression?symbol=${symbol}`)
                .then(async r => ({status: r.status, data: await r.json()}))
                .then(({status, data}) => {
                    if (status === 202 || data.status === 'computing') {
                        target.innerHTML = '<div class="loading">⏳ Still computing ' + symbol + ' market connection...<br><small style="font-size: 0.8em; color: var(--text-secondary);">Returning partial UI to keep response fast.</small></div>';
                        setTimeout(poll, 1500);
                        return;
                    }
                    if (data.error) target.innerHTML = `<div class="error">❌ ${data.error}</div>`;
                    else showRegressionResult(data, symbol);
                })
                .catch(e => target.innerHTML = `<div class="error">❌ ${e.message}</div>`);
            poll();
        }
        function showRegressionResult(data, symbol) {
            const marketInfo = data.market_source ? `<div style="background: rgba(201,168,76, 0.07); padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 3px solid var(--accent-gold);"><strong>Market Benchmark:</strong> ${data.market_source} ${data.market_source !== 'Nifty 50 Index' ? '<br><small style="color: var(--text-muted);">Note: Using alternative benchmark due to Nifty 50 data availability.</small>' : ''}</div>` : '';
            const scorePercent = (data.dependency_score * 100).toFixed(1);
            const compositePercent = (data.composite_score * 100).toFixed(1);
            const cs = data.composite_score;
            const hiddenSyncTag = data.hidden_sync ? '<span style="background:rgba(255,107,107,0.15); color:#ff6b6b; padding:3px 10px; border-radius:12px; font-size:0.75em; font-weight:600; margin-left:8px;">HIDDEN SYNC</span>' : '';
            const protColor = cs >= 0.35 ? '#ff6b6b' : '#10b981';
            const protLabel = cs >= 0.55 ? 'Weak Protection' : cs >= 0.35 ? 'Partial Protection' : 'Good Protection';
            const dbSign = data.downside_beta >= 0 ? '' : '';
            const html = `
                <div class="result-card">
                    <div class="header"><h2>${symbol} vs Market</h2><div style="color: var(--accent-cyan); font-size: 1.2em;">Market Connection Analysis</div></div>
                    <div style="margin: -20px 0 20px; font-size: 1.1em; color: var(--text-secondary);">${getStockName(symbol)} <span style="background: var(--bg-card-hover); padding: 3px 10px; border-radius: 4px; font-size: 0.8em; color: var(--accent-cyan); border: 1px solid var(--border-color);">${getStockSector(symbol)}</span></div>
                    ${marketInfo}
                    <div class="action-banner">${data.trading_insight}</div>

                    <!-- HERO: Composite Connection Score -->
                    <div class="hsic-hero">
                        <div style="margin-bottom: 10px;">
                            <span class="hsic-tooltip" style="font-size: 0.85em; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">
                                Market Connection
                                <span class="hsic-tooltip-text">A composite score combining three measures: the Magnetism Score (HSIC, which detects hidden non-linear ties), Pearson correlation (daily co-movement), and beta (sensitivity to market swings). Higher = more connected to Nifty 50.</span>
                            </span>
                        </div>
                        <div class="hsic-hero-score" style="color: ${data.dependency_color};">${compositePercent}%</div>
                        <div class="hsic-hero-label">
                            <span class="hsic-badge" style="background: ${data.badge_bg};">${data.badge_text}</span>
                            ${hiddenSyncTag}
                        </div>
                        <div class="hsic-hero-subtitle">${data.magnetism_plain}</div>
                        <div style="margin-top: 16px; display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
                            <div style="text-align:center;">
                                <div style="font-size:0.7em; text-transform:uppercase; color:var(--text-muted); letter-spacing:0.5px;">
                                    <span class="hsic-tooltip">Magnetism (HSIC)<span class="hsic-tooltip-text">The Magnetism Score uses a kernel method (HSIC) to detect all types of statistical ties, including non-linear ones that correlation misses. Think of it as an X-ray for hidden market connections.</span></span>
                                </div>
                                <div style="font-size:1.4em; font-weight:700; font-family:'Space Grotesk',sans-serif; color:var(--text-primary);">${scorePercent}%</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="font-size:0.7em; text-transform:uppercase; color:var(--text-muted); letter-spacing:0.5px;">
                                    <span class="hsic-tooltip">Correlation<span class="hsic-tooltip-text">How closely the stock's daily returns move with the market. +1 = perfect match, 0 = no pattern, -1 = opposite.</span></span>
                                </div>
                                <div style="font-size:1.4em; font-weight:700; font-family:'Space Grotesk',sans-serif; color:var(--text-primary);">${data.correlation >= 0 ? '+' : ''}${data.correlation.toFixed(2)}</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="font-size:0.7em; text-transform:uppercase; color:var(--text-muted); letter-spacing:0.5px;">
                                    <span class="hsic-tooltip">Beta<span class="hsic-tooltip-text">If Nifty 50 moves 1%, this stock historically moves about ${Math.abs(data.beta).toFixed(1)}% in the ${data.beta >= 0 ? 'same' : 'opposite'} direction. Beta > 1 means it amplifies market moves.</span></span>
                                </div>
                                <div style="font-size:1.4em; font-weight:700; font-family:'Space Grotesk',sans-serif; color:var(--text-primary);">${data.beta.toFixed(2)}</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="font-size:0.7em; text-transform:uppercase; color:var(--text-muted); letter-spacing:0.5px;">
                                    <span class="hsic-tooltip">Downside Beta<span class="hsic-tooltip-text">Same as beta, but measured only on days when the market fell. A high downside beta means the stock tends to fall harder than the market during sell-offs. This is the most relevant measure for crash protection.</span></span>
                                </div>
                                <div style="font-size:1.4em; font-weight:700; font-family:'Space Grotesk',sans-serif; color:${data.downside_beta > 1.2 ? '#ff6b6b' : data.downside_beta > 0.8 ? '#ffa94d' : 'var(--text-primary)'};">${data.downside_beta.toFixed(2)}</div>
                            </div>
                        </div>
                    </div>

                    <!-- SCATTER PLOT -->
                    <div class="plot-container">
                        <h3 style="color: var(--accent-purple); margin-bottom: 5px;">Daily Returns: ${symbol} vs ${data.market_source || 'Market'}</h3>
                        <p style="color: var(--text-muted); font-size: 0.85em; margin-bottom: 15px;">Each dot is one trading day over the last ${data.data_points} sessions. A tight diagonal cluster means the stock and market often move together.</p>
                        <img src="data:image/png;base64,${data.plot_url}" class="plot-img" alt="Stock vs Market scatter plot">
                    </div>

                    <!-- INSIGHT CARDS -->
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 15px; margin-bottom: 20px;">
                        <!-- Mirror Test Card -->
                        <div class="insight-card" style="border-left-color: ${data.mirror_color};">
                            <div class="insight-card-title">
                                <span class="hsic-tooltip">
                                    Mirror Test
                                    <span class="hsic-tooltip-text">Compares simple correlation with the deeper HSIC analysis. If correlation is low but HSIC is high, the stock has a "Hidden Sync", meaning it follows the market in complex, non-obvious ways that only appear during stress events.</span>
                                </span>
                            </div>
                            <div class="mirror-verdict" style="color: ${data.mirror_color};">${data.mirror_verdict}</div>
                            <div class="insight-card-body">${data.mirror_explain}</div>
                        </div>

                        <!-- Diversification Card -->
                        <div class="insight-card" style="border-left-color: ${protColor};">
                            <div class="insight-card-title">
                                <span class="hsic-tooltip">
                                    Portfolio Protection
                                    <span class="hsic-tooltip-text">Based on the composite score and downside beta. A stock with high market connection and high downside beta offers little crash protection. Genuinely uncorrelated stocks (where both HSIC and correlation are low) are the best diversifiers.</span>
                                </span>
                            </div>
                            <div style="font-weight: 700; font-family: 'Space Grotesk', sans-serif; font-size: 1.3em; margin-bottom: 6px; color: ${protColor};">
                                ${protLabel}
                            </div>
                            <div class="insight-card-body">${data.diversification_note}</div>
                        </div>
                    </div>

                    <!-- WHAT DOES THIS MEAN FOR ME -->
                    <div class="trading-plan" style="margin-top: 10px;">
                        <h3>What Does This Mean For Me?</h3>
                        <div class="plan-item"><span class="plan-label">If Nifty 50 crashes</span><span class="plan-value">${cs >= 0.55 ? (data.downside_beta > 1.0 ? 'Based on historical data, this stock has tended to fall as much or more than the market (downside beta ' + data.downside_beta.toFixed(2) + '). Holding both concentrates your risk.' : 'This stock has historically moved with the market. In a sharp downturn, it is likely to decline as well, though perhaps not as sharply (downside beta ' + data.downside_beta.toFixed(2) + ').') : cs >= 0.35 ? 'Moderate market connection means this stock may get pulled down during a sell-off, especially if the decline is severe. It offers some, but not full, cushioning.' : cs >= 0.18 ? 'Historical data suggests limited market linkage. This stock may be less affected than the broader market, though no stock is fully immune to a severe crash.' : 'Over the analysis window, this stock showed little connection to Nifty 50. It may behave differently during a market downturn, but correlations can spike during extreme events.'}</span></div>
                        <div class="plan-item"><span class="plan-label">If I already own Nifty funds</span><span class="plan-value">${cs >= 0.55 ? 'Adding this stock provides limited additional diversification since it is already closely linked to the same market forces driving Nifty 50.' : cs >= 0.35 ? 'Adds some variety, but the moderate connection to Nifty means overlapping risk. Consider pairing with lower-connection stocks for better spread.' : 'Historically independent enough to add genuine diversification to a Nifty-heavy portfolio. A reasonable choice for spreading risk.'}</span></div>
                        <div class="plan-item"><span class="plan-label">Hidden surprises?</span><span class="plan-value">${data.hidden_sync ? 'The Mirror Test flagged a discrepancy: daily correlation appears low, but the deeper HSIC analysis detected non-linear coupling. This stock may seem independent on calm days but move with the market during crises. Factor this into hedging decisions.' : 'No significant discrepancy between linear and non-linear measures. The visible relationship is a fair representation of the underlying connection.'}</span></div>
                    </div>

                    <!-- TECHNICAL DETAILS (collapsible) -->
                    <button class="tech-details-toggle" onclick="this.nextElementSibling.classList.toggle('open'); this.querySelector('.arrow').textContent = this.nextElementSibling.classList.contains('open') ? '▲' : '▼';">
                        Technical Details (For Advanced Users) <span class="arrow">▼</span>
                    </button>
                    <div class="tech-details-content">
                        <div class="tech-details-inner">
                            <div class="tech-detail-item">
                                <div class="tech-detail-label">Composite Score</div>
                                <div class="tech-detail-value" style="color: ${data.dependency_color};">${compositePercent}%</div>
                                <div class="tech-detail-note">0.5 × HSIC + 0.3 × |corr| + 0.2 × min(|β|,2)/2</div>
                            </div>
                            <div class="tech-detail-item">
                                <div class="tech-detail-label">HSIC (Normalised)</div>
                                <div class="tech-detail-value">${scorePercent}%</div>
                                <div class="tech-detail-note">HSIC / sqrt(HSIC_xx × HSIC_yy), range 0–1.</div>
                            </div>
                            <div class="tech-detail-item">
                                <div class="tech-detail-label">Raw HSIC (Unfiltered Signal)</div>
                                <div class="tech-detail-value">${data.raw_hsic.toFixed(8)}</div>
                                <div class="tech-detail-note">tr(K_c L_c) / n&sup2;, the unnormalised kernel statistic.</div>
                            </div>
                            <div class="tech-detail-item">
                                <div class="tech-detail-label">Pearson Correlation</div>
                                <div class="tech-detail-value">${data.correlation >= 0 ? '+' : ''}${data.correlation.toFixed(4)}</div>
                                <div class="tech-detail-note">Linear co-movement. Basis for Mirror Test.</div>
                            </div>
                            <div class="tech-detail-item">
                                <div class="tech-detail-label">Beta (OLS)</div>
                                <div class="tech-detail-value">${data.beta.toFixed(4)}</div>
                                <div class="tech-detail-note">Slope of stock vs market returns.</div>
                            </div>
                            <div class="tech-detail-item">
                                <div class="tech-detail-label">Downside Beta</div>
                                <div class="tech-detail-value" style="color: ${data.downside_beta > 1.2 ? '#ff6b6b' : 'var(--text-primary)'};">${data.downside_beta.toFixed(4)}</div>
                                <div class="tech-detail-note">Beta on market-down days only. Key crash metric.</div>
                            </div>
                            <div class="tech-detail-item">
                                <div class="tech-detail-label">Observations</div>
                                <div class="tech-detail-value">${data.data_points}</div>
                                <div class="tech-detail-note">Trading days analysed (up to 252).</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            document.getElementById('regression-result').innerHTML = html + buildPeerStocksHTML(symbol, 'regression');
        }
        function goBack() {
            document.getElementById('search-view').style.display = 'block';
            document.getElementById('result-view').style.display = 'none';
            document.getElementById('search').value = '';
            document.getElementById('suggestions').innerHTML = '';
        }
        let dividendScope = 'custom';
        let dividendRisk = 'moderate';
        function formatIndianNumber(num) {
            num = num.toString();
            let lastThree = num.substring(num.length - 3);
            let otherNumbers = num.substring(0, num.length - 3);
            if (otherNumbers !== '') lastThree = ',' + lastThree;
            return otherNumbers.replace(/\\B(?=(\\d{2})+(?!\\d))/g, ',') + lastThree;
        }
        function setupCapitalInput() {
            const inp = document.getElementById('capital-input');
            inp.addEventListener('input', function() {
                let raw = this.value.replace(/,/g, '').replace(/[^0-9]/g, '');
                if (raw === '') { this.value = ''; return; }
                this.value = formatIndianNumber(raw);
            });
        }
        function getCapitalValue() {
            return parseFloat(document.getElementById('capital-input').value.replace(/,/g, ''));
        }
        function toggleAllNifty(checked) {
            document.querySelectorAll('.nifty-cb').forEach(cb => cb.checked = checked);
        }
        function initNifty50Checkboxes() {
            const grid = document.getElementById('nifty50-grid');
            if (!grid) return;
            nifty50List.forEach(symbol => {
                const label = document.createElement('label');
                label.innerHTML = '<input type="checkbox" class="nifty-cb" value="' + symbol + '" checked> ' + symbol + ' <span style="color:var(--text-muted);font-size:0.85em;">(' + getStockName(symbol) + ')</span>';
                grid.appendChild(label);
            });
        }
        function setRisk(risk, btn) {
            dividendRisk = risk;
            document.querySelectorAll('.risk-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const descs = {
                conservative: 'Max 8% per stock. High penalty for volatility. Only stocks with yield above 1%. Prefers stable, consistent dividend payers.',
                moderate: 'Max 15% per stock. Balanced yield vs risk tradeoff. Good diversification across dividend payers.',
                aggressive: 'Max 30% per stock. Pure yield maximization with minimal volatility penalty. Higher concentration allowed.'
            };
            document.getElementById('risk-desc').innerText = descs[risk];
        }
        function toggleAllSectors(checked) {
            document.querySelectorAll('.sector-cb').forEach(cb => cb.checked = checked);
        }
        let dividendStream = null;
        let liveDividendEntries = [];
        let liveDividendMax = 0;
        let liveRenderTimer = null;
        function renderLiveDividendRows() {
            const tbody = document.getElementById('live-dividend-body');
            if (!tbody) return;
            const fmt = (n) => Number(n).toLocaleString('en-IN', {maximumFractionDigits: 2});
            tbody.innerHTML = liveDividendEntries.map((s, idx) => {
                const fc = s.fy_count || 0;
                const fcColor = fc >= 2 ? 'var(--accent-green)' : fc === 1 ? 'var(--warning)' : 'var(--accent-red, #ef4444)';
                const fcLabel = fc >= 2 ? fc + '/2 FY' : fc === 1 ? '1/2 FY' : 'None';
                return `<tr>
                    <td>${idx + 1}. ${s.symbol}<br><span style="font-size:0.8em; color: var(--text-muted);">${getStockName(s.symbol)}</span></td>
                    <td style="font-size:0.8em; color: var(--text-muted);">${getStockSector(s.symbol)}</td>
                    <td style="text-align: right;">${fmt(s.price)}</td>
                    <td style="text-align: right;">${fmt(s.annual_dividend)}</td>
                    <td style="color: var(--accent-green); font-weight: 600;">${s.dividend_yield}%</td>
                    <td style="color: ${fcColor}; font-weight: 600;">${fcLabel}</td>
                    <td>${s.volatility}%</td>
                </tr>`;
            }).join('');
        }
        function scheduleLiveDividendRender() {
            if (liveRenderTimer) return;
            liveRenderTimer = setTimeout(() => {
                liveRenderTimer = null;
                renderLiveDividendRows();
            }, 250);
        }
        function updateLiveStatus(scanned, dividendFound) {
            const scannedEl = document.getElementById('live-scan-count');
            const foundEl = document.getElementById('live-dividend-count');
            if (scannedEl) scannedEl.textContent = scanned;
            if (foundEl) foundEl.textContent = dividendFound;
        }
        let dividendHighlight = '';
        let dividendSectorLabel = '';
        function scanStockSector(symbol) {
            const sector = getStockSector(symbol);
            dividendHighlight = symbol;
            dividendSectorLabel = sector;
            dividendScope = '_single_sector';
            const capital = getCapitalValue();
            if (!capital || capital <= 0) { alert('Please enter a valid capital amount'); return; }
            _runDividendScan(sector, capital, '');
        }
        function analyzeDividends() {
            const capital = getCapitalValue();
            if (!capital || capital <= 0) { alert('Please enter a valid capital amount'); return; }
            dividendHighlight = '';
            let sectors = '';
            let symbolsParam = '';
            if (dividendScope === 'custom') {
                const checked = document.querySelectorAll('.sector-cb:checked');
                if (checked.length === 0) { alert('Please select at least one sector'); return; }
                const sectorList = Array.from(checked).map(c => c.value);
                sectors = sectorList.join(',');
                if (sectorList.length === 1) dividendSectorLabel = sectorList[0];
                else if (sectorList.length <= 3) dividendSectorLabel = sectorList.join(' | ');
                else dividendSectorLabel = 'Multi-Sector';
            }
            else { sectors = 'all'; dividendSectorLabel = 'All NSE'; }
            _runDividendScan(sectors, capital, symbolsParam);
        }
        function _runDividendScan(sectors, capital, symbolsParam) {
            const resultsDiv = document.getElementById('dividend-results');
            resultsDiv.innerHTML = `
                <div class="result-card" style="margin-top: 30px;">
                    <div class="header">
                        <h2>Live Dividend Scan</h2>
                        <div class="signal-badge" style="background: var(--accent-cyan); color: white;">LIVE</div>
                    </div>
                    <div class="action-banner">Scanning <span id="live-scan-count">0</span> / <span id="live-scan-total">0</span> stocks | <span id="live-dividend-count">0</span> dividend payers found</div>
                    <div style="color: var(--text-secondary); font-size: 0.9em; margin-top: 8px;">New dividend yielders appear in the table below and move up if higher yield is found.</div>
                    <h3 style="color: var(--accent-purple); margin: 20px 0 10px; font-family: 'Space Grotesk', sans-serif; font-weight: 700;">Top Dividend Payers (Live)</h3>
                    <div style="overflow-x: auto; max-height: 400px; border: 1px solid var(--border-color); border-radius: 8px;">
                        <table class="dividend-table">
                            <thead><tr>
                                <th>Stock</th><th>Sector</th><th style="text-align:right;">Price (INR)</th><th style="text-align:right;">Annual Div (INR)</th><th>Div Yield</th><th>Consistency</th><th>Volatility</th>
                            </tr></thead>
                            <tbody id="live-dividend-body"></tbody>
                        </table>
                    </div>
                    <div id="live-portfolio" style="margin-top: 25px;"></div>
                    <div style="color: var(--text-muted); font-size: 0.8em; margin-top: 10px;">This may take 30-120 seconds for large universes. Please wait.</div>
                </div>`;

            if (dividendStream) dividendStream.close();
            liveDividendEntries = [];
            liveDividendMax = 0;
            document.getElementById('live-scan-total').textContent = '0';

            let streamUrl = `/dividend-optimize-stream?capital=${capital}&risk=${dividendRisk}&sectors=${encodeURIComponent(sectors)}`;
            if (symbolsParam) streamUrl += `&symbols=${encodeURIComponent(symbolsParam)}`;
            dividendStream = new EventSource(streamUrl);
            dividendStream.onmessage = (event) => {
                const payload = JSON.parse(event.data);
                if (payload.type === 'meta') {
                    liveDividendMax = payload.max_results || 0;
                    document.getElementById('live-scan-total').textContent = payload.total_scanned || 0;
                    return;
                }
                if (payload.type === 'stock') {
                    updateLiveStatus(payload.scanned, payload.dividend_found);
                    liveDividendEntries.push(payload.entry);
                    liveDividendEntries.sort((a, b) => b.dividend_yield - a.dividend_yield);
                    if (liveDividendMax && liveDividendEntries.length > liveDividendMax) {
                        liveDividendEntries = liveDividendEntries.slice(0, liveDividendMax);
                    }
                    scheduleLiveDividendRender();
                    return;
                }
                if (payload.type === 'portfolio') {
                    renderLivePortfolio(payload.portfolio, capital, payload.partial, payload.scanned, payload.dividend_found);
                    return;
                }
                if (payload.type === 'progress') {
                    updateLiveStatus(payload.scanned, payload.dividend_found);
                    return;
                }
                if (payload.type === 'error') {
                    dividendStream.close();
                    resultsDiv.innerHTML = `<div class="error">${payload.message}</div>`;
                    return;
                }
                if (payload.type === 'done') {
                    dividendStream.close();
                    showDividendResults(payload.result, capital);
                }
            };
            dividendStream.onerror = () => {
                dividendStream.close();
                resultsDiv.innerHTML = `<div class="error">Live stream failed. Retrying with standard request...</div>`;
                fetch(`/dividend-optimize?capital=${capital}&risk=${dividendRisk}&sectors=${encodeURIComponent(sectors)}${symbolsParam ? '&symbols=' + encodeURIComponent(symbolsParam) : ''}`)
                    .then(r => r.json())
                    .then(data => {
                        if (data.error) resultsDiv.innerHTML = `<div class="error">${data.error}</div>`;
                        else showDividendResults(data, capital);
                    })
                    .catch(e => resultsDiv.innerHTML = `<div class="error">Request failed: ${e.message}. Try a smaller universe or retry.</div>`);
            };
        }
        function showDividendResults(data, capital) {
            const fmt = (n) => Number(n).toLocaleString('en-IN', {maximumFractionDigits: 2});
            let allocRows = data.allocation.map((a, idx) => `
                <tr>
                    <td style="font-weight: 600; color: var(--accent-cyan);">${idx + 1}. ${a.symbol}<br><span style="font-size:0.8em; font-weight:400; color: var(--text-muted);">${getStockName(a.symbol)}</span></td>
                    <td style="font-size:0.85em; color: var(--text-muted);">${getStockSector(a.symbol)}</td>
                    <td>${a.weight}%</td>
                    <td style="text-align: right;">${a.shares}</td>
                    <td style="text-align: right;">${fmt(a.price)}</td>
                    <td style="text-align: right;">${fmt(a.amount)}</td>
                    <td style="color: var(--accent-green); font-weight: 600;">${a.dividend_yield}%</td>
                    <td style="color: var(--accent-green); font-weight: 700; text-align: right;">${fmt(a.expected_dividend)}</td>
                    <td>${a.volatility}%</td>
                </tr>`).join('');
            let sortedStocks = [...data.all_dividend_stocks];
            if (dividendHighlight) {
                sortedStocks.sort((a, b) => {
                    if (a.symbol === dividendHighlight) return -1;
                    if (b.symbol === dividendHighlight) return 1;
                    return b.dividend_yield - a.dividend_yield;
                });
            }
            let allStockRows = sortedStocks.map((s, idx) => {
                const isHL = s.symbol === dividendHighlight;
                const cappedNote = s.yield_capped ? `<br><span style="font-size:0.75em; color: var(--warning);" title="Latest FY div ₹${fmt(s.latest_fy_dividend)} was ≥2x prev FY ₹${fmt(s.prev_fy_dividend)}. Using prev FY for sustainable yield.">⚠ yield adjusted</span>` : '';
                const divHistory = (s.prev_fy_dividend > 0) ? `<br><span style="font-size:0.75em; color: var(--text-muted);">prev FY: ₹${fmt(s.prev_fy_dividend)}</span>` : '';
                const fc = s.fy_count || 0;
                const fcColor = fc >= 2 ? 'var(--accent-green)' : fc === 1 ? 'var(--warning)' : 'var(--accent-red, #ef4444)';
                const fcLabel = fc >= 2 ? fc + '/2 FY' : fc === 1 ? '1/2 FY' : 'None';
                return `<tr style="${isHL ? 'background: rgba(0,217,255,0.1); border-left: 3px solid var(--accent-cyan);' : ''}">
                    <td>${idx + 1}. ${s.symbol}<br><span style="font-size:0.8em; color: var(--text-muted);">${getStockName(s.symbol)}</span>${cappedNote}</td>
                    <td style="font-size:0.85em; color: var(--text-muted);">${getStockSector(s.symbol)}</td>
                    <td style="text-align: right;">${fmt(s.price)}</td>
                    <td style="text-align: right;">${fmt(s.annual_dividend)}${divHistory}</td>
                    <td style="color: var(--accent-green); font-weight: 600;">${s.dividend_yield}%</td>
                    <td style="color: ${fcColor}; font-weight: 600;">${fcLabel}</td>
                    <td>${s.volatility}%</td>
                </tr>`;
            }).join('');
            const riskColors = { conservative: '#10b981', moderate: '#f59e0b', aggressive: '#ef4444' };
            const riskColor = riskColors[data.risk_appetite] || '#f59e0b';
            const html = `
                <div class="result-card" style="margin-top: 30px;">
                    <div class="header">
                        <h2>Dividend Portfolio <span style="font-size: 0.5em; color: var(--accent-cyan); font-weight: 400;">| ${dividendSectorLabel}</span></h2>
                        <div class="signal-badge" style="background: ${riskColor}; color: white;">${data.risk_appetite.toUpperCase()}</div>
                    </div>
                    <div class="action-banner">Optimized for Maximum Dividend Income | ${data.stocks_scanned} stocks scanned | ${data.dividend_stocks_found} pay dividends</div>
                    <div class="summary-grid">
                        <div class="summary-card">
                            <div class="summary-value" style="color: var(--accent-green);">${data.portfolio_yield.toFixed(2)}%</div>
                            <div class="summary-label">Portfolio Dividend Yield</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-value" style="color: var(--accent-cyan);">${fmt(data.total_expected_dividend)}</div>
                            <div class="summary-label">Expected Annual Dividend (INR)</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-value">${data.num_stocks}</div>
                            <div class="summary-label">Stocks in Portfolio</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-value" style="color: var(--accent-purple);">${fmt(data.total_invested)}</div>
                            <div class="summary-label">Capital Deployed (INR)</div>
                        </div>
                    </div>
                    ${data.unallocated > 100 ? `<div style="margin: 15px 0; padding: 12px 15px; background: rgba(245, 158, 11, 0.1); border-left: 3px solid var(--warning); border-radius: 8px; font-size: 0.9em;"><strong style="color: var(--warning);">Unallocated:</strong> <span style="color: var(--text-secondary);">INR ${fmt(data.unallocated)} (due to rounding to whole shares)</span></div>` : ''}
                    <h3 style="color: var(--accent-cyan); margin: 30px 0 15px; font-family: 'Space Grotesk', sans-serif; font-weight: 700;">Optimized Allocation</h3>
                    <div style="overflow-x: auto;">
                        <table class="dividend-table">
                            <thead><tr>
                                <th>Stock</th><th>Sector</th><th>Weight</th><th style="text-align:right;">Shares</th><th style="text-align:right;">Price (INR)</th><th style="text-align:right;">Investment (INR)</th><th>Div Yield</th><th style="text-align:right;">Expected Div (INR)</th><th>Volatility</th>
                            </tr></thead>
                            <tbody>${allocRows}</tbody>
                        </table>
                    </div>
                    <div style="margin: 15px 0; padding: 10px 14px; background: rgba(0,217,255,0.06); border-left: 3px solid var(--accent-cyan); border-radius: 6px; font-size: 0.85em; color: var(--text-secondary);">
                        <strong style="color: var(--accent-cyan);">Dividend Sustainability Check:</strong> Yields are cross-checked against the previous financial year. If a stock's latest FY dividend is ≥2x the prior FY (indicating a one-time special dividend), the yield is adjusted down to the sustainable level. Stocks marked <span style="color: var(--warning);">⚠ yield adjusted</span> had their yield capped. The <strong>Consistency</strong> column shows how many of the last 2 FYs had dividends — <span style="color: var(--accent-green);">2/2 FY</span> = consistent payer. Only stocks paying in both FYs are selected for the optimized allocation across all risk levels.
                    </div>
                    <h3 style="color: var(--accent-purple); margin: 35px 0 15px; font-family: 'Space Grotesk', sans-serif; font-weight: 700;">All Dividend-Paying Stocks (${data.all_dividend_stocks.length} shown)</h3>
                    ${data.dividend_results_truncated ? `<div style="margin-bottom: 10px; color: var(--warning); font-size: 0.85em;">Showing top ${data.all_dividend_stocks.length} dividend payers to reduce memory usage. ${data.dividend_stocks_found} total dividend-paying stocks found.</div>` : ''}
                    <div style="overflow-x: auto; max-height: 400px; border: 1px solid var(--border-color); border-radius: 8px;">
                        <table class="dividend-table">
                            <thead><tr>
                                <th>Stock</th><th>Sector</th><th style="text-align:right;">Price (INR)</th><th style="text-align:right;">Annual Div (INR)</th><th>Div Yield</th><th>Consistency</th><th>Volatility</th>
                            </tr></thead>
                            <tbody>${allStockRows}</tbody>
                        </table>
                    </div>
                </div>`;
            if (data.allocation && data.allocation.length > 0) {
                html += buildPeerStocksHTML(data.allocation[0].symbol, 'dividend-to-verdict');
            }
            document.getElementById('dividend-results').innerHTML = html;
        }
        function renderLivePortfolio(data, capital, partial, scanned, dividendFound) {
            if (!data || !data.allocation || data.allocation.length === 0) return;
            const fmt = (n) => Number(n).toLocaleString('en-IN', {maximumFractionDigits: 2});
            const status = partial ? 'LIVE (PARTIAL)' : 'LIVE';
            const subText = partial ? `Updated after scanning ${scanned} stocks (${dividendFound} dividend payers)` : '';
            const rows = data.allocation.slice(0, 10).map((a, idx) => `
                <tr>
                    <td style="font-weight: 600; color: var(--accent-cyan);">${idx + 1}. ${a.symbol}<br><span style="font-size:0.8em; font-weight:400; color: var(--text-muted);">${getStockName(a.symbol)}</span></td>
                    <td style="font-size:0.85em; color: var(--text-muted);">${getStockSector(a.symbol)}</td>
                    <td>${a.weight}%</td>
                    <td style="text-align: right;">${a.shares}</td>
                    <td style="text-align: right;">${fmt(a.price)}</td>
                    <td style="text-align: right;">${fmt(a.amount)}</td>
                    <td style="color: var(--accent-green); font-weight: 600;">${a.dividend_yield}%</td>
                    <td style="color: var(--accent-green); font-weight: 700; text-align: right;">${fmt(a.expected_dividend)}</td>
                    <td>${a.volatility}%</td>
                </tr>`).join('');
            const html = `
                <div class="result-card">
                    <div class="header">
                        <h2>Optimized Portfolio (Live)</h2>
                        <div class="signal-badge" style="background: var(--accent-purple); color: white;">${status}</div>
                    </div>
                    ${subText ? `<div style="color: var(--text-secondary); font-size: 0.85em; margin-bottom: 10px;">${subText}</div>` : ''}
                    <div class="summary-grid">
                        <div class="summary-card">
                            <div class="summary-value" style="color: var(--accent-green);">${Number(data.portfolio_yield).toFixed(2)}%</div>
                            <div class="summary-label">Portfolio Dividend Yield</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-value" style="color: var(--accent-cyan);">${fmt(data.total_expected_dividend)}</div>
                            <div class="summary-label">Expected Annual Dividend (INR)</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-value">${data.num_stocks}</div>
                            <div class="summary-label">Stocks in Portfolio</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-value" style="color: var(--accent-purple);">${fmt(data.total_invested)}</div>
                            <div class="summary-label">Capital Deployed (INR)</div>
                        </div>
                    </div>
                    <h3 style="color: var(--accent-cyan); margin: 20px 0 10px; font-family: 'Space Grotesk', sans-serif; font-weight: 700;">Top Allocation (Live)</h3>
                    <div style="overflow-x: auto;">
                        <table class="dividend-table">
                            <thead><tr>
                                <th>Stock</th><th>Sector</th><th>Weight</th><th style="text-align:right;">Shares</th><th style="text-align:right;">Price (INR)</th><th style="text-align:right;">Investment (INR)</th><th>Div Yield</th><th style="text-align:right;">Expected Div (INR)</th><th>Volatility</th>
                            </tr></thead>
                            <tbody>${rows}</tbody>
                        </table>
                    </div>
                    <div style="color: var(--text-muted); font-size: 0.8em; margin-top: 8px;">Live portfolio updates every 50 stocks scanned to reduce memory usage.</div>
                </div>`;
            const target = document.getElementById('live-portfolio');
            if (target) target.innerHTML = html;
        }
        function initDividendSectors() {
            const grid = document.getElementById('sector-grid');
            if (!grid) return;
            const skipDiv = new Set(['All NSE', 'Nifty 50', 'Nifty Next 50', 'Conglomerate']);
            Object.keys(stocks).forEach(sector => {
                if (skipDiv.has(sector)) return;
                const label = document.createElement('label');
                label.innerHTML = '<input type="checkbox" class="sector-cb" value="' + sector + '"> ' + sector + ' (' + stocks[sector].length + ')';
                grid.appendChild(label);
            });
        }
        // ===== DCF VALUATION =====
        let dcfData = null;
        let _dcfYears = 10;

        function dcfGoBack() {
            document.getElementById('dcf-result-view').style.display = 'none';
            document.getElementById('dcf-search-view').style.display = 'block';
        }

        function fetchDCFData() {
            var symbol = document.getElementById('dcf-search').value.trim().toUpperCase();
            if (!symbol) { alert('Please enter a stock symbol'); return; }
            document.getElementById('dcf-suggestions').innerHTML = '';
            document.getElementById('dcf-search-view').style.display = 'none';
            document.getElementById('dcf-result-view').style.display = 'block';
            document.getElementById('dcf-result').innerHTML = '<div class="loading">\u23f3 Fetching financial data for ' + symbol + '...</div>';
            fetch('/dcf-data?symbol=' + encodeURIComponent(symbol))
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.error) {
                        document.getElementById('dcf-result').innerHTML = '<div class="error">\u274c ' + data.error + '</div>';
                        setTimeout(function() { document.getElementById('dcf-result-view').style.display = 'none'; document.getElementById('dcf-search-view').style.display = 'block'; }, 3500);
                    } else if (data.valuation_model === 'excess_return') { dcfData = data; renderExcessReturnResult(data); } else { dcfData = data; _dcfYears = 10; renderDCFResult(data); }
                })
                .catch(function(e) { document.getElementById('dcf-result').innerHTML = '<div class="error">\u274c ' + e.message + '</div>'; });
        }

        function fmtCr(n) {
            if (n === null || n === undefined || isNaN(n)) return 'N/A';
            var abs = Math.abs(n), sign = n < 0 ? '-' : '';
            if (abs >= 1e12) return sign + '\u20b9' + (abs / 1e12).toFixed(2) + 'L Cr';
            if (abs >= 1e7)  return sign + '\u20b9' + (abs / 1e7).toFixed(2) + ' Cr';
            if (abs >= 1e5)  return sign + '\u20b9' + (abs / 1e5).toFixed(2) + ' L';
            return sign + '\u20b9' + Math.round(abs).toLocaleString('en-IN');
        }

        var _FINANCIAL_SECTORS = ['financial services', 'banking', 'insurance', 'financial'];
        function isFinancialSector(sector) {
            if (!sector) return false;
            var s = sector.toLowerCase();
            return _FINANCIAL_SECTORS.some(function(fs) { return s.indexOf(fs) !== -1; });
        }

        function runDCF(fcf, g1, g2, wacc, tg, years, debt, cash, shares) {
            var phase1Yrs = Math.min(5, years);
            var projections = [], currentFCF = fcf, sumPV = 0;
            for (var yr = 1; yr <= years; yr++) {
                var g = yr <= phase1Yrs ? g1 : g2;
                currentFCF = currentFCF * (1 + g);
                var pv = currentFCF / Math.pow(1 + wacc, yr);
                projections.push({ year: yr, fcf: currentFCF, pv: pv, growth: g });
                sumPV += pv;
            }
            var terminalFCF = currentFCF * (1 + tg);
            var terminalValue = (tg < wacc) ? terminalFCF / (wacc - tg) : currentFCF * 15;
            var terminalPV = terminalValue / Math.pow(1 + wacc, years);
            var enterpriseValue = sumPV + terminalPV;
            var equityValue = Math.max(enterpriseValue - debt + cash, 0);
            var intrinsic = shares > 0 ? equityValue / shares : 0;
            return { projections: projections, sumPV: sumPV, terminalValue: terminalValue, terminalPV: terminalPV, enterpriseValue: enterpriseValue, equityValue: equityValue, intrinsic: intrinsic };
        }

        function renderFCFHistory(history) {
            if (!history || history.length === 0) return '';
            var vals = history.map(function(h) { return h.fcf; });
            var maxAbs = Math.max.apply(null, vals.map(Math.abs));
            if (maxAbs === 0) return '';
            var bars = history.map(function(h) {
                var pct = Math.abs(h.fcf) / maxAbs * 100;
                var cls = h.fcf >= 0 ? 'dcf-fcf-positive' : 'dcf-fcf-negative';
                return '<div class="dcf-fcf-bar-wrap"><div class="dcf-fcf-bar-inner ' + cls + '" style="height:' + pct + '%;"></div><div class="dcf-fcf-bar-year">' + String(h.year).slice(-2) + '</div></div>';
            }).join('');
            return '<div style="background:var(--bg-card-hover);border-radius:12px;padding:18px 22px;margin-bottom:22px;border:1px solid var(--border-color);"><div style="font-size:0.78em;text-transform:uppercase;color:var(--text-muted);font-weight:600;letter-spacing:0.5px;margin-bottom:10px;">Historical Free Cash Flow</div><div class="dcf-fcf-hist">' + bars + '</div></div>';
        }

        function dcfSetYears(n, btn) {
            _dcfYears = n;
            document.querySelectorAll('.dcf-year-btn').forEach(function(b) { b.classList.remove('active'); });
            btn.classList.add('active');
            updateDCFDisplay();
        }
        function dcfSliderChange(el) {
            var targetId = el.dataset.target || (el.id + '-val');
            var display = document.getElementById(targetId);
            if (display) display.textContent = el.value + '%';
            updateDCFDisplay();
        }

        function renderDCFResult(data) {
            var sugG = Math.round(data.suggested_growth_rate * 100);
            var phase2G = Math.max(Math.round(sugG * 0.5), 5);
            var defaultWACC = 12, defaultTG = 3;
            var histCagr = data.historical_fcf_growth !== null ? (data.historical_fcf_growth * 100).toFixed(1) + '%' : 'N/A';
            var mktCap = data.market_cap ? fmtCr(data.market_cap) : 'N/A';
            var netCash = fmtCr(data.cash - data.total_debt);
            var pe = data.pe_ratio ? data.pe_ratio.toFixed(1) + 'x' : 'N/A';
            var pb = data.pb_ratio ? data.pb_ratio.toFixed(2) + 'x' : 'N/A';
            var roce = data.roce ? (data.roce * 100).toFixed(1) + '%' : 'N/A';
            var roe = data.roe ? (data.roe * 100).toFixed(1) + '%' : 'N/A';
            var priceStr = '\u20b9' + data.current_price.toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2});
            var fcfStr = fmtCr(data.current_fcf);
            var h = '';
            h += '<div class="dcf-stock-hero"><div>';
            h += '<div class="dcf-stock-name">' + (data.name || data.symbol) + '</div>';
            h += '<div class="dcf-stock-sub">' + data.symbol + (data.sector ? ' &bull; ' + data.sector : '') + (data.industry ? ' &bull; ' + data.industry : '') + '</div>';
            h += '</div><div class="dcf-price-box"><div class="dcf-price-label">Current Price</div><div class="dcf-price-val">' + priceStr + '</div></div></div>';
            h += '<div class="dcf-key-stats">';
            h += '<div class="dcf-stat" style="border-left-color:var(--accent-cyan);"><div class="dcf-stat-label">Latest FCF</div><div class="dcf-stat-value">' + fcfStr + '</div></div>';
            h += '<div class="dcf-stat" style="border-left-color:var(--accent-purple);"><div class="dcf-stat-label">Market Cap</div><div class="dcf-stat-value">' + mktCap + '</div></div>';
            h += '<div class="dcf-stat" style="border-left-color:var(--accent-green);"><div class="dcf-stat-label">Net Cash</div><div class="dcf-stat-value">' + netCash + '</div></div>';
            h += '<div class="dcf-stat" style="border-left-color:var(--warning);"><div class="dcf-stat-label">P/E Ratio</div><div class="dcf-stat-value">' + pe + '</div></div>';
            h += '<div class="dcf-stat" style="border-left-color:var(--danger);"><div class="dcf-stat-label">P/B Ratio</div><div class="dcf-stat-value">' + pb + '</div></div>';
            h += '<div class="dcf-stat" style="border-left-color:var(--text-muted);"><div class="dcf-stat-label">Hist. FCF CAGR</div><div class="dcf-stat-value">' + histCagr + '</div></div>';
            h += '<div class="dcf-stat" style="border-left-color:var(--accent-green);"><div class="dcf-stat-label">RoCE</div><div class="dcf-stat-value" style="color:' + (data.roce ? (data.roce*100>=20?'var(--accent-green)':data.roce*100>=12?'var(--warning)':'var(--danger)') : 'var(--text-muted)') + ';">' + roce + '</div></div>';
            h += '<div class="dcf-stat" style="border-left-color:var(--accent-cyan);"><div class="dcf-stat-label">RoE</div><div class="dcf-stat-value" style="color:' + (data.roe ? (data.roe*100>=15?'var(--accent-green)':data.roe*100>=8?'var(--warning)':'var(--danger)') : 'var(--text-muted)') + ';">' + roe + '</div></div>';
            h += '</div>';
            if (data.fcf_history && data.fcf_history.length > 1) h += renderFCFHistory(data.fcf_history);
            h += '<div class="dcf-valuation-grid">';
            h += '<div class="dcf-params-card"><h3>Assumptions</h3>';
            h += '<div class="dcf-param-row"><div class="dcf-param-label"><span>Projection Years</span></div>';
            h += '<div class="dcf-years-group"><button class="dcf-year-btn" onclick="dcfSetYears(5, this)">5 Years</button><button class="dcf-year-btn active" onclick="dcfSetYears(10, this)">10 Years</button></div></div>';
            h += '<div class="dcf-param-row"><div class="dcf-param-label"><span>Phase 1 Growth (Yr 1\u20135)</span><span class="dcf-param-value" id="dcf-g1-val">' + sugG + '%</span></div>';
            h += '<input type="range" class="dcf-slider" id="dcf-g1" min="0" max="50" step="0.5" value="' + sugG + '" oninput="dcfSliderChange(this)">';
            h += '<div class="dcf-param-hint">Based on historical FCF CAGR: ' + histCagr + '. Adjust as needed.</div></div>';
            h += '<div class="dcf-param-row"><div class="dcf-param-label"><span>Phase 2 Growth (Yr 6\u201310)</span><span class="dcf-param-value" id="dcf-g2-val">' + phase2G + '%</span></div>';
            h += '<input type="range" class="dcf-slider" id="dcf-g2" min="0" max="30" step="0.5" value="' + phase2G + '" oninput="dcfSliderChange(this)">';
            h += '<div class="dcf-param-hint">Growth typically decelerates as companies mature.</div></div>';
            h += '<div class="dcf-param-row"><div class="dcf-param-label"><span>Discount Rate / WACC</span><span class="dcf-param-value" id="dcf-wacc-val">' + defaultWACC + '%</span></div>';
            h += '<input type="range" class="dcf-slider" id="dcf-wacc" min="6" max="25" step="0.5" value="' + defaultWACC + '" oninput="dcfSliderChange(this)">';
            h += '<div class="dcf-param-hint">Your required rate of return. Typically 10\u201315% for NSE stocks.</div></div>';
            h += '<div class="dcf-param-row"><div class="dcf-param-label"><span>Terminal Growth Rate</span><span class="dcf-param-value" id="dcf-tg-val">' + defaultTG + '%</span></div>';
            h += '<input type="range" class="dcf-slider" id="dcf-tg" min="1" max="6" step="0.25" value="' + defaultTG + '" oninput="dcfSliderChange(this)">';
            h += '<div class="dcf-param-hint">Perpetual growth after projection period. Should not exceed long-run GDP growth.</div></div>';
            h += '<div class="dcf-param-row"><div class="dcf-param-label"><span>Base FCF Adjustment</span><span class="dcf-param-value" id="dcf-fcf-display">100%</span></div>';
            h += '<input type="range" class="dcf-slider" id="dcf-fcf-adj" min="50" max="200" step="5" value="100" data-target="dcf-fcf-display" oninput="dcfSliderChange(this)">';
            h += '<div class="dcf-param-hint">Scale latest FCF. 100% = ' + fcfStr + ' (as reported).</div></div>';
            h += '</div>';
            h += '<div class="dcf-results-card"><h3>Valuation Output</h3>';
            h += '<div class="dcf-intrinsic-hero"><div class="dcf-intrinsic-label">Intrinsic Value Per Share</div><div class="dcf-intrinsic-value" id="dcf-intrinsic-val">\u2014</div></div>';
            h += '<div id="dcf-verdict-banner" class="dcf-verdict-banner"></div>';
            h += '<div class="dcf-margin-row"><span class="dcf-margin-label">Current Market Price</span><span class="dcf-margin-val">' + priceStr + '</span></div>';
            h += '<div class="dcf-margin-row"><span class="dcf-margin-label">Margin of Safety / Premium</span><span class="dcf-margin-val" id="dcf-mos">\u2014</span></div>';
            h += '<div class="dcf-margin-row"><span class="dcf-margin-label">Upside / Downside</span><span class="dcf-margin-val" id="dcf-updown">\u2014</span></div>';
            h += '<div class="dcf-margin-row"><span class="dcf-margin-label">Enterprise Value</span><span class="dcf-margin-val" id="dcf-ev">\u2014</span></div>';
            h += '<div class="dcf-margin-row"><span class="dcf-margin-label">Terminal Value (% of EV)</span><span class="dcf-margin-val" id="dcf-tv-pct">\u2014</span></div>';
            h += '<div class="dcf-breakdown-bar"><div class="dcf-breakdown-label">Value Breakdown: FCFs vs Terminal Value</div>';
            h += '<div class="dcf-bar-track"><div class="dcf-bar-fcf" id="dcf-bar-fcf" style="width:40%;">FCF PVs</div><div class="dcf-bar-tv" id="dcf-bar-tv" style="width:60%;">Terminal</div></div>';
            h += '<div class="dcf-bar-legend"><div class="dcf-bar-legend-item"><div class="dcf-bar-legend-dot" style="background:var(--accent-cyan);"></div>PV of FCFs</div><div class="dcf-bar-legend-item"><div class="dcf-bar-legend-dot" style="background:var(--accent-purple);"></div>Terminal Value</div></div>';
            h += '</div></div>';
            h += '</div>';
            h += '<div class="dcf-section"><div class="dcf-section-title">Year-by-Year Free Cash Flow Projection</div><div style="overflow-x:auto;"><table class="dcf-proj-table"><thead><tr><th>Year</th><th>FCF (\u20b9 Cr)</th><th>Growth</th><th>Discount Factor</th><th>PV (\u20b9 Cr)</th></tr></thead><tbody id="dcf-proj-tbody"></tbody></table></div></div>';
            h += '<div class="dcf-sensitivity dcf-section"><div class="dcf-section-title">Sensitivity Analysis \u2014 Intrinsic Value vs WACC &amp; Terminal Growth</div>';
            h += '<p style="color:var(--text-muted);font-size:0.82em;margin-bottom:12px;">Green = undervalued vs current price &bull; Red = overvalued &bull; Yellow = within \u00b115%.</p>';
            h += '<div style="overflow-x:auto;"><table class="dcf-sens-table" id="dcf-sens-table"></table></div></div>';
            h += buildPeerStocksHTML(data.symbol, 'dcf');
            h += '<div class="dcf-disclaimer">\u26a0\ufe0f <strong>Disclaimer:</strong> DCF valuations are highly sensitive to input assumptions. Small changes in growth rate or WACC can materially affect the intrinsic value. This tool is for educational and informational purposes only and does not constitute investment advice. Always do your own due diligence.</div>';
            document.getElementById('dcf-result').innerHTML = h;
            updateDCFDisplay();
        }

        function updateDCFDisplay() {
            if (!dcfData) return;
            var g1   = parseFloat(document.getElementById('dcf-g1').value) / 100;
            var g2   = parseFloat(document.getElementById('dcf-g2').value) / 100;
            var wacc = parseFloat(document.getElementById('dcf-wacc').value) / 100;
            var tg   = parseFloat(document.getElementById('dcf-tg').value) / 100;
            var adj  = parseFloat(document.getElementById('dcf-fcf-adj').value) / 100;
            var fcf  = dcfData.current_fcf * adj;
            var _isFin = isFinancialSector(dcfData.sector);
            var _debt = _isFin ? 0 : dcfData.total_debt;
            var _cash = _isFin ? 0 : dcfData.cash;
            var res  = runDCF(fcf, g1, g2, wacc, tg, _dcfYears, _debt, _cash, dcfData.shares_outstanding);
            var price = dcfData.current_price;
            var intrinsic = res.intrinsic;
            var ivEl = document.getElementById('dcf-intrinsic-val');
            var isUnder = intrinsic > price;
            ivEl.textContent = '\u20b9' + intrinsic.toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2});
            ivEl.className = 'dcf-intrinsic-value ' + (isUnder ? 'dcf-intrinsic-undervalued' : 'dcf-intrinsic-overvalued');
            var mosEl = document.getElementById('dcf-mos');
            var mos = ((intrinsic - price) / Math.max(intrinsic, 1)) * 100;
            mosEl.textContent = isUnder ? mos.toFixed(1) + '% Margin of Safety' : Math.abs(mos).toFixed(1) + '% Premium to Fair Value';
            mosEl.className = 'dcf-margin-val ' + (isUnder ? 'dcf-upside' : 'dcf-downside');
            var ud = ((intrinsic - price) / price) * 100;
            var udEl = document.getElementById('dcf-updown');
            udEl.textContent = (ud >= 0 ? '+' : '') + ud.toFixed(1) + '%';
            udEl.className = 'dcf-margin-val ' + (ud >= 0 ? 'dcf-upside' : 'dcf-downside');
            document.getElementById('dcf-ev').textContent = fmtCr(res.enterpriseValue);
            var tvPct = res.enterpriseValue > 0 ? (res.terminalPV / res.enterpriseValue * 100) : 0;
            document.getElementById('dcf-tv-pct').textContent = tvPct.toFixed(1) + '%';
            var fcfPct = Math.max(0, Math.min(100 - tvPct, 100));
            var tvBar  = Math.max(0, Math.min(tvPct, 100));
            var bFcf = document.getElementById('dcf-bar-fcf'), bTv = document.getElementById('dcf-bar-tv');
            bFcf.style.width = fcfPct.toFixed(1) + '%'; bTv.style.width = tvBar.toFixed(1) + '%';
            bFcf.textContent = fcfPct >= 18 ? 'FCF PVs ' + fcfPct.toFixed(0) + '%' : '';
            bTv.textContent  = tvBar  >= 18 ? 'Terminal ' + tvBar.toFixed(0)  + '%' : '';
            var vEl = document.getElementById('dcf-verdict-banner');
            if (ud >= 25)       { vEl.className = 'dcf-verdict-banner dcf-verdict-buy';  vEl.textContent = '\u2705 Potentially Undervalued \u2014 Trading at a significant discount to intrinsic value'; }
            else if (ud >= -15) { vEl.className = 'dcf-verdict-banner dcf-verdict-hold'; vEl.textContent = '\u2696\ufe0f Fairly Valued \u2014 Price is near the estimated intrinsic value'; }
            else                { vEl.className = 'dcf-verdict-banner dcf-verdict-sell'; vEl.textContent = '\u26a0\ufe0f Potentially Overvalued \u2014 Trading at a premium to intrinsic value'; }
            var tbody = document.getElementById('dcf-proj-tbody');
            if (tbody) {
                var rows = '';
                // Historical FCF rows (last 5 years)
                if (dcfData.fcf_history && dcfData.fcf_history.length > 0) {
                    var hist = dcfData.fcf_history.slice(-5);
                    hist.forEach(function(h, i) {
                        var growthCell = '<span style="color:var(--text-muted);">—</span>';
                        if (i > 0) {
                            var prev = hist[i - 1].fcf;
                            if (prev && prev !== 0) {
                                var g = ((h.fcf - prev) / Math.abs(prev)) * 100;
                                growthCell = '<span style="color:' + (g >= 0 ? 'var(--accent-green)' : 'var(--danger)') + ';">' + (g >= 0 ? '+' : '') + g.toFixed(1) + '%</span>';
                            }
                        }
                        rows += '<tr class="hist-row"><td>FY ' + h.year + '</td><td>' + fmtCr(h.fcf) + '</td><td>' + growthCell + '</td><td style="color:var(--text-muted);font-size:0.8em;">Actual</td><td style="color:var(--text-muted);font-size:0.8em;">Actual</td></tr>';
                    });
                    rows += '<tr class="hist-separator"><td colspan="5">\u25bc\u25bc Projections \u25bc\u25bc</td></tr>';
                }
                res.projections.forEach(function(p) {
                    rows += '<tr><td>Year ' + p.year + '</td><td>' + fmtCr(p.fcf) + '</td><td style="color:var(--text-secondary);">' + (p.growth * 100).toFixed(1) + '%</td><td style="color:var(--text-muted);">' + (1 / Math.pow(1 + wacc, p.year)).toFixed(4) + '</td><td>' + fmtCr(p.pv) + '</td></tr>';
                });
                rows += '<tr class="tv-row"><td>Terminal Value</td><td>' + fmtCr(res.terminalValue) + '</td><td style="color:var(--text-secondary);">@TG ' + (tg*100).toFixed(2) + '%</td><td style="color:var(--text-muted);">' + (1 / Math.pow(1 + wacc, _dcfYears)).toFixed(4) + '</td><td>' + fmtCr(res.terminalPV) + '</td></tr>';
                rows += '<tr class="total-row"><td>Enterprise Value</td><td colspan="3"></td><td>' + fmtCr(res.enterpriseValue) + '</td></tr>';
                rows += '<tr class="total-row"><td>' + (_isFin ? 'Equity Value (Financial \u2014 no debt adj.)' : 'Equity Value (\u2212Debt +Cash)') + '</td><td colspan="3"></td><td>' + fmtCr(res.equityValue) + '</td></tr>';
                rows += '<tr class="total-row"><td>Intrinsic Value / Share</td><td colspan="3"></td><td>\u20b9' + intrinsic.toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2}) + '</td></tr>';
                tbody.innerHTML = rows;
            }
            renderDCFSensitivity(price, wacc, tg, fcf, g1, g2, _dcfYears, _debt, _cash, dcfData.shares_outstanding);
        }

        function renderDCFSensitivity(price, baseWACC, baseTG, fcf, g1, g2, years, debt, cash, shares) {
            var waccSteps = [-0.04, -0.02, 0, 0.02, 0.04];
            var tgSteps   = [0.01, 0.02, 0.03, 0.04, 0.05];
            var tableEl = document.getElementById('dcf-sens-table');
            if (!tableEl) return;
            var html = '<thead><tr><th>WACC \\ Term Growth</th>';
            tgSteps.forEach(function(tg) { html += '<th>' + (tg * 100).toFixed(0) + '%</th>'; });
            html += '</tr></thead><tbody>';
            waccSteps.forEach(function(dw) {
                var w = baseWACC + dw;
                if (w <= 0) return;
                html += '<tr><td>WACC ' + (w * 100).toFixed(0) + '%</td>';
                tgSteps.forEach(function(tg) {
                    if (tg >= w) { html += '<td style="color:var(--text-muted);">N/A</td>'; return; }
                    var r = runDCF(fcf, g1, g2, w, tg, years, debt, cash, shares);
                    var ud = ((r.intrinsic - price) / price) * 100;
                    var isCurrent = (Math.abs(dw) < 0.001 && Math.abs(tg - baseTG) < 0.001);
                    var cls = ud >= 15 ? 'dcf-sens-undervalue' : ud <= -15 ? 'dcf-sens-overvalue' : 'dcf-sens-near';
                    if (isCurrent) cls += ' dcf-sens-highlight';
                    html += '<td class="' + cls + '">\u20b9' + Math.round(r.intrinsic).toLocaleString('en-IN') + '</td>';
                });
                html += '</tr>';
            });
            html += '</tbody>';
            tableEl.innerHTML = html;
        }


        // ===== DAMODARAN EXCESS RETURN (ROE - BOOK VALUE) MODEL =====
        function runExcessReturn(bvps, roe, coe, growthRate, years, shares) {
            var projections = [], currentBV = bvps, sumPV = 0;
            for (var yr = 1; yr <= years; yr++) {
                var excessReturn = (roe - coe) * currentBV;
                var pv = excessReturn / Math.pow(1 + coe, yr);
                projections.push({ year: yr, bv: currentBV, excessReturn: excessReturn, pv: pv });
                sumPV += pv;
                currentBV = currentBV * (1 + growthRate);
            }
            // Terminal value of excess returns beyond projection period
            var terminalER = (roe - coe) * currentBV;
            var terminalValue = (coe > growthRate * 0.5) ? terminalER / (coe - growthRate * 0.5) : terminalER * 15;
            var terminalPV = terminalValue / Math.pow(1 + coe, years);
            var intrinsic = bvps + sumPV + terminalPV;
            return { projections: projections, sumPV: sumPV, terminalValue: terminalValue, terminalPV: terminalPV, intrinsic: Math.max(intrinsic, 0) };
        }

        function renderExcessReturnResult(data) {
            var roe = data.roe, coe = data.cost_of_equity, bvps = data.book_value_per_share;
            var sugG = Math.round(data.suggested_growth_rate * 100);
            var defaultKe = Math.round(coe * 100);
            var priceStr = '\u20b9' + data.current_price.toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2});
            var mktCap = data.market_cap ? fmtCr(data.market_cap) : 'N/A';
            var pe = data.pe_ratio ? data.pe_ratio.toFixed(1) + 'x' : 'N/A';
            var pb = data.pb_ratio ? data.pb_ratio.toFixed(2) + 'x' : 'N/A';
            var roeStr = (roe * 100).toFixed(1) + '%';
            var bvGrowth = data.bv_growth !== null ? (data.bv_growth * 100).toFixed(1) + '%' : 'N/A';
            var excessSpread = ((roe - coe) * 100).toFixed(1);
            var h = '';
            h += '<div class="dcf-stock-hero"><div>';
            h += '<div class="dcf-stock-name">' + (data.name || data.symbol) + '</div>';
            h += '<div class="dcf-stock-sub">' + data.symbol + (data.sector ? ' &bull; ' + data.sector : '') + (data.industry ? ' &bull; ' + data.industry : '') + '</div>';
            h += '<div style="margin-top:6px;padding:4px 10px;background:var(--accent-purple);color:#fff;border-radius:6px;display:inline-block;font-size:0.78em;font-weight:600;">Damodaran Excess Return Model (Financial Firm)</div>';
            h += '</div><div class="dcf-price-box"><div class="dcf-price-label">Current Price</div><div class="dcf-price-val">' + priceStr + '</div></div></div>';
            h += '<div class="dcf-key-stats">';
            h += '<div class="dcf-stat" style="border-left-color:var(--accent-green);"><div class="dcf-stat-label">Book Value / Share</div><div class="dcf-stat-value">\u20b9' + bvps.toFixed(2) + '</div></div>';
            h += '<div class="dcf-stat" style="border-left-color:var(--accent-cyan);"><div class="dcf-stat-label">Return on Equity</div><div class="dcf-stat-value" style="color:' + (roe*100>=15?'var(--accent-green)':roe*100>=8?'var(--warning)':'var(--danger)') + ';">' + roeStr + '</div></div>';
            h += '<div class="dcf-stat" style="border-left-color:var(--accent-purple);"><div class="dcf-stat-label">Cost of Equity</div><div class="dcf-stat-value">' + (coe * 100).toFixed(1) + '%</div></div>';
            h += '<div class="dcf-stat" style="border-left-color:' + (parseFloat(excessSpread)>0?'var(--accent-green)':'var(--danger)') + ';"><div class="dcf-stat-label">Excess Spread (ROE\u2212Ke)</div><div class="dcf-stat-value" style="color:' + (parseFloat(excessSpread)>0?'var(--accent-green)':'var(--danger)') + ';">' + (parseFloat(excessSpread)>0?'+':'') + excessSpread + '%</div></div>';
            h += '<div class="dcf-stat" style="border-left-color:var(--accent-purple);"><div class="dcf-stat-label">Market Cap</div><div class="dcf-stat-value">' + mktCap + '</div></div>';
            h += '<div class="dcf-stat" style="border-left-color:var(--warning);"><div class="dcf-stat-label">P/E Ratio</div><div class="dcf-stat-value">' + pe + '</div></div>';
            h += '<div class="dcf-stat" style="border-left-color:var(--danger);"><div class="dcf-stat-label">P/B Ratio</div><div class="dcf-stat-value">' + pb + '</div></div>';
            h += '<div class="dcf-stat" style="border-left-color:var(--text-muted);"><div class="dcf-stat-label">Hist. BV CAGR</div><div class="dcf-stat-value">' + bvGrowth + '</div></div>';
            h += '</div>';
            // ROE history chart
            if (data.roe_history && data.roe_history.length > 1) {
                var roeVals = data.roe_history.map(function(r) { return r.roe * 100; });
                var maxRoe = Math.max.apply(null, roeVals.map(Math.abs));
                if (maxRoe === 0) maxRoe = 1;
                var bars = data.roe_history.map(function(r) {
                    var pct = Math.abs(r.roe * 100) / maxRoe * 100;
                    var cls = r.roe >= 0 ? 'dcf-fcf-positive' : 'dcf-fcf-negative';
                    return '<div class="dcf-fcf-bar-wrap"><div class="dcf-fcf-bar-inner ' + cls + '" style="height:' + pct + '%;"></div><div class="dcf-fcf-bar-year">' + String(r.year).slice(-2) + '</div></div>';
                }).join('');
                h += '<div style="background:var(--bg-card-hover);border-radius:12px;padding:18px 22px;margin-bottom:22px;border:1px solid var(--border-color);"><div style="font-size:0.78em;text-transform:uppercase;color:var(--text-muted);font-weight:600;letter-spacing:0.5px;margin-bottom:10px;">Historical Return on Equity</div><div class="dcf-fcf-hist">' + bars + '</div></div>';
            }
            h += '<div class="dcf-valuation-grid">';
            h += '<div class="dcf-params-card"><h3>Assumptions</h3>';
            h += '<div class="dcf-param-row"><div class="dcf-param-label"><span>Projection Years</span></div>';
            h += '<div class="dcf-years-group"><button class="dcf-year-btn" onclick="erSetYears(5, this)">5 Years</button><button class="dcf-year-btn active" onclick="erSetYears(10, this)">10 Years</button></div></div>';
            h += '<div class="dcf-param-row"><div class="dcf-param-label"><span>Return on Equity (ROE)</span><span class="dcf-param-value" id="er-roe-val">' + Math.round(roe*100) + '%</span></div>';
            h += '<input type="range" class="dcf-slider" id="er-roe" min="1" max="40" step="0.5" value="' + Math.round(roe*100) + '" oninput="erSliderChange(this)">';
            h += '<div class="dcf-param-hint">Current ROE: ' + roeStr + '. Higher ROE means more excess returns above cost of equity.</div></div>';
            h += '<div class="dcf-param-row"><div class="dcf-param-label"><span>Cost of Equity (Ke)</span><span class="dcf-param-value" id="er-ke-val">' + defaultKe + '%</span></div>';
            h += '<input type="range" class="dcf-slider" id="er-ke" min="6" max="25" step="0.5" value="' + defaultKe + '" oninput="erSliderChange(this)">';
            h += '<div class="dcf-param-hint">CAPM-derived: Rf(' + (0.07*100).toFixed(0) + '%) + \u03b2(' + data.beta.toFixed(2) + ') \u00d7 MRP(6%). Your required return.</div></div>';
            h += '<div class="dcf-param-row"><div class="dcf-param-label"><span>Book Value Growth Rate</span><span class="dcf-param-value" id="er-g-val">' + sugG + '%</span></div>';
            h += '<input type="range" class="dcf-slider" id="er-g" min="0" max="30" step="0.5" value="' + sugG + '" oninput="erSliderChange(this)">';
            h += '<div class="dcf-param-hint">Rate at which book value per share grows. Historical BV CAGR: ' + bvGrowth + '.</div></div>';
            h += '</div>';
            h += '<div class="dcf-results-card"><h3>Valuation Output</h3>';
            h += '<div class="dcf-intrinsic-hero"><div class="dcf-intrinsic-label">Intrinsic Value Per Share</div><div class="dcf-intrinsic-value" id="er-intrinsic-val">\u2014</div></div>';
            h += '<div id="er-verdict-banner" class="dcf-verdict-banner"></div>';
            h += '<div class="dcf-margin-row"><span class="dcf-margin-label">Current Market Price</span><span class="dcf-margin-val">' + priceStr + '</span></div>';
            h += '<div class="dcf-margin-row"><span class="dcf-margin-label">Book Value / Share</span><span class="dcf-margin-val">\u20b9' + bvps.toFixed(2) + '</span></div>';
            h += '<div class="dcf-margin-row"><span class="dcf-margin-label">Upside / Downside</span><span class="dcf-margin-val" id="er-updown">\u2014</span></div>';
            h += '<div class="dcf-margin-row"><span class="dcf-margin-label">PV of Excess Returns</span><span class="dcf-margin-val" id="er-pv-excess">\u2014</span></div>';
            h += '<div class="dcf-margin-row"><span class="dcf-margin-label">Terminal Value of Excess Returns</span><span class="dcf-margin-val" id="er-tv">\u2014</span></div>';
            h += '<div class="dcf-breakdown-bar"><div class="dcf-breakdown-label">Value Breakdown: Book Value vs Excess Returns</div>';
            h += '<div class="dcf-bar-track"><div class="dcf-bar-fcf" id="er-bar-bv" style="width:50%;">Book Value</div><div class="dcf-bar-tv" id="er-bar-er" style="width:50%;">Excess Returns</div></div>';
            h += '<div class="dcf-bar-legend"><div class="dcf-bar-legend-item"><div class="dcf-bar-legend-dot" style="background:var(--accent-cyan);"></div>Book Value</div><div class="dcf-bar-legend-item"><div class="dcf-bar-legend-dot" style="background:var(--accent-purple);"></div>PV of Excess Returns</div></div>';
            h += '</div></div>';
            h += '</div>';
            h += '<div class="dcf-section"><div class="dcf-section-title">Year-by-Year Excess Return Projection</div><div style="overflow-x:auto;"><table class="dcf-proj-table"><thead><tr><th>Year</th><th>Book Value / Sh</th><th>Excess Return / Sh</th><th>Discount Factor</th><th>PV / Sh</th></tr></thead><tbody id="er-proj-tbody"></tbody></table></div></div>';
            h += '<div class="dcf-sensitivity dcf-section"><div class="dcf-section-title">Sensitivity Analysis \u2014 Intrinsic Value vs Cost of Equity &amp; ROE</div>';
            h += '<p style="color:var(--text-muted);font-size:0.82em;margin-bottom:12px;">Green = undervalued vs current price &bull; Red = overvalued &bull; Yellow = within \u00b115%.</p>';
            h += '<div style="overflow-x:auto;"><table class="dcf-sens-table" id="er-sens-table"></table></div></div>';
            h += '<div class="dcf-section" style="background:var(--bg-card-hover);border-radius:12px;padding:18px 22px;border:1px solid var(--border-color);">';
            h += '<div class="dcf-section-title" style="margin-bottom:10px;">\U0001f4d6 About This Model</div>';
            h += '<p style="color:var(--text-secondary);font-size:0.85em;line-height:1.7;margin:0;">The <strong>Damodaran Excess Return Model</strong> values financial firms (banks, NBFCs, insurance) using <strong>Book Value + PV of future Excess Returns</strong>. Unlike a traditional DCF, it recognises that financial firms\u2019 \u201cdebt\u201d is their raw material (deposits, borrowings), not a financing choice. If ROE exceeds the Cost of Equity, the firm creates value above its book; if ROE &lt; Ke, the stock should trade below book.</p>';
            h += '</div>';
            h += buildPeerStocksHTML(data.symbol, 'dcf');
            h += '<div class="dcf-disclaimer">\u26a0\ufe0f <strong>Disclaimer:</strong> Excess return valuations are sensitive to ROE sustainability and cost of equity assumptions. This tool is for educational and informational purposes only and does not constitute investment advice.</div>';
            document.getElementById('dcf-result').innerHTML = h;
            _erYears = 10;
            updateExcessReturnDisplay();
        }

        var _erYears = 10;
        function erSetYears(n, btn) {
            _erYears = n;
            document.querySelectorAll('.dcf-year-btn').forEach(function(b) { b.classList.remove('active'); });
            btn.classList.add('active');
            updateExcessReturnDisplay();
        }
        function erSliderChange(el) {
            var targetId = el.id + '-val';
            var display = document.getElementById(targetId);
            if (display) display.textContent = el.value + '%';
            updateExcessReturnDisplay();
        }

        function updateExcessReturnDisplay() {
            if (!dcfData || dcfData.valuation_model !== 'excess_return') return;
            var roe  = parseFloat(document.getElementById('er-roe').value) / 100;
            var ke   = parseFloat(document.getElementById('er-ke').value) / 100;
            var g    = parseFloat(document.getElementById('er-g').value) / 100;
            var bvps = dcfData.book_value_per_share;
            var shares = dcfData.shares_outstanding;
            var price = dcfData.current_price;
            var res  = runExcessReturn(bvps, roe, ke, g, _erYears, shares);
            var intrinsic = res.intrinsic;
            var ivEl = document.getElementById('er-intrinsic-val');
            var isUnder = intrinsic > price;
            ivEl.textContent = '\u20b9' + intrinsic.toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2});
            ivEl.className = 'dcf-intrinsic-value ' + (isUnder ? 'dcf-intrinsic-undervalued' : 'dcf-intrinsic-overvalued');
            var ud = ((intrinsic - price) / price) * 100;
            var udEl = document.getElementById('er-updown');
            udEl.textContent = (ud >= 0 ? '+' : '') + ud.toFixed(1) + '%';
            udEl.className = 'dcf-margin-val ' + (ud >= 0 ? 'dcf-upside' : 'dcf-downside');
            document.getElementById('er-pv-excess').textContent = '\u20b9' + res.sumPV.toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2}) + ' / share';
            document.getElementById('er-tv').textContent = '\u20b9' + res.terminalPV.toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2}) + ' / share';
            // Breakdown bar
            var totalExcess = res.sumPV + res.terminalPV;
            var bvPct = intrinsic > 0 ? Math.max(0, Math.min(bvps / intrinsic * 100, 100)) : 50;
            var erPct = 100 - bvPct;
            var bBv = document.getElementById('er-bar-bv'), bEr = document.getElementById('er-bar-er');
            bBv.style.width = bvPct.toFixed(1) + '%'; bEr.style.width = erPct.toFixed(1) + '%';
            bBv.textContent = bvPct >= 18 ? 'Book Value ' + bvPct.toFixed(0) + '%' : '';
            bEr.textContent = erPct >= 18 ? 'Excess Returns ' + erPct.toFixed(0) + '%' : '';
            // Verdict banner
            var vEl = document.getElementById('er-verdict-banner');
            if (ud >= 25)       { vEl.className = 'dcf-verdict-banner dcf-verdict-buy';  vEl.textContent = '\u2705 Potentially Undervalued \u2014 ROE generates significant excess returns over cost of equity'; }
            else if (ud >= -15) { vEl.className = 'dcf-verdict-banner dcf-verdict-hold'; vEl.textContent = '\u2696\ufe0f Fairly Valued \u2014 Price is near the estimated intrinsic value'; }
            else                { vEl.className = 'dcf-verdict-banner dcf-verdict-sell'; vEl.textContent = '\u26a0\ufe0f Potentially Overvalued \u2014 Market premium exceeds estimated excess returns'; }
            // Projection table
            var tbody = document.getElementById('er-proj-tbody');
            if (tbody) {
                var rows = '';
                // Historical ROE rows
                if (dcfData.roe_history && dcfData.roe_history.length > 0) {
                    var hist = dcfData.roe_history.slice(-5);
                    hist.forEach(function(r) {
                        var eqPerShare = r.equity / shares;
                        rows += '<tr class="hist-row"><td>FY ' + r.year + '</td><td>\u20b9' + eqPerShare.toFixed(2) + '</td><td style="color:' + (r.roe>=0?'var(--accent-green)':'var(--danger)') + ';">ROE ' + (r.roe*100).toFixed(1) + '%</td><td style="color:var(--text-muted);font-size:0.8em;">Actual</td><td style="color:var(--text-muted);font-size:0.8em;">Actual</td></tr>';
                    });
                    rows += '<tr class="hist-separator"><td colspan="5">\u25bc\u25bc Projections \u25bc\u25bc</td></tr>';
                }
                res.projections.forEach(function(p) {
                    rows += '<tr><td>Year ' + p.year + '</td><td>\u20b9' + p.bv.toFixed(2) + '</td><td style="color:' + (p.excessReturn>=0?'var(--accent-green)':'var(--danger)') + ';">\u20b9' + p.excessReturn.toFixed(2) + '</td><td style="color:var(--text-muted);">' + (1 / Math.pow(1 + ke, p.year)).toFixed(4) + '</td><td>\u20b9' + p.pv.toFixed(2) + '</td></tr>';
                });
                rows += '<tr class="total-row"><td>PV of Excess Returns</td><td colspan="3"></td><td>\u20b9' + res.sumPV.toFixed(2) + '</td></tr>';
                rows += '<tr class="total-row"><td>Terminal Excess Return (PV)</td><td colspan="3"></td><td>\u20b9' + res.terminalPV.toFixed(2) + '</td></tr>';
                rows += '<tr class="total-row"><td>Book Value / Share</td><td colspan="3"></td><td>\u20b9' + bvps.toFixed(2) + '</td></tr>';
                rows += '<tr class="total-row"><td>Intrinsic Value / Share</td><td colspan="3"></td><td>\u20b9' + intrinsic.toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2}) + '</td></tr>';
                tbody.innerHTML = rows;
            }
            // Sensitivity table: ROE vs Cost of Equity
            renderERSensitivity(price, roe, ke, g, _erYears, bvps, shares);
        }

        function renderERSensitivity(price, baseROE, baseKe, g, years, bvps, shares) {
            var keSteps  = [-0.03, -0.015, 0, 0.015, 0.03];
            var roeSteps = [-0.04, -0.02, 0, 0.02, 0.04];
            var tableEl = document.getElementById('er-sens-table');
            if (!tableEl) return;
            var html = '<thead><tr><th>Ke \\ ROE</th>';
            roeSteps.forEach(function(dr) { html += '<th>' + ((baseROE + dr) * 100).toFixed(0) + '%</th>'; });
            html += '</tr></thead><tbody>';
            keSteps.forEach(function(dk) {
                var k = baseKe + dk;
                if (k <= 0) return;
                html += '<tr><td>Ke ' + (k * 100).toFixed(1) + '%</td>';
                roeSteps.forEach(function(dr) {
                    var r = baseROE + dr;
                    if (r <= 0) { html += '<td style="color:var(--text-muted);">N/A</td>'; return; }
                    var res = runExcessReturn(bvps, r, k, g, years, shares);
                    var ud = ((res.intrinsic - price) / price) * 100;
                    var isCurrent = (Math.abs(dk) < 0.001 && Math.abs(dr) < 0.001);
                    var cls = ud >= 15 ? 'dcf-sens-undervalue' : ud <= -15 ? 'dcf-sens-overvalue' : 'dcf-sens-near';
                    if (isCurrent) cls += ' dcf-sens-highlight';
                    html += '<td class="' + cls + '">\u20b9' + Math.round(res.intrinsic).toLocaleString('en-IN') + '</td>';
                });
                html += '</tr>';
            });
            html += '</tbody>';
            tableEl.innerHTML = html;
        }


        // ===== DCF UNIVERSE SCREENER =====
        var _dcfScreenES = null;   // EventSource handle
        function startDCFScreen() {
            if (_dcfScreenES) { _dcfScreenES.close(); _dcfScreenES = null; }
            document.getElementById('dcf-screen-btn').disabled = true;
            document.getElementById('dcf-screen-btn').textContent = 'Screening\u2026';
            document.getElementById('dcf-screen-stop-btn').style.display = 'inline-block';
            document.getElementById('dcf-screen-progress').style.display = 'block';
            document.getElementById('dcf-screen-results').style.display = 'none';
            document.getElementById('dcf-screen-results').innerHTML = '';
            document.getElementById('dcf-screen-bar').style.width = '0%';
            document.getElementById('dcf-screen-pct').textContent = '0%';
            document.getElementById('dcf-screen-found').textContent = '0 undervalued found';
            document.getElementById('dcf-screen-status').textContent = 'Connecting\u2026';

            _dcfScreenES = new EventSource('/dcf-screen');
            _dcfScreenES.onmessage = function(ev) {
                var d;
                try { d = JSON.parse(ev.data); } catch(e) { return; }
                if (d.type === 'start') {
                    document.getElementById('dcf-screen-status').textContent = 'Screening ' + d.total + ' stocks\u2026';
                } else if (d.type === 'progress') {
                    var pct = Math.round(d.done / d.total * 100);
                    document.getElementById('dcf-screen-bar').style.width = pct + '%';
                    document.getElementById('dcf-screen-pct').textContent = pct + '% (' + d.done + '/' + d.total + ')';
                    document.getElementById('dcf-screen-found').textContent = d.found + ' undervalued found';
                    document.getElementById('dcf-screen-status').textContent = 'Processing ' + d.symbol + '\u2026';
                } else if (d.type === 'complete') {
                    _dcfScreenES.close();
                    _dcfScreenES = null;
                    document.getElementById('dcf-screen-btn').disabled = false;
                    document.getElementById('dcf-screen-btn').textContent = 'Screen All Stocks';
                    document.getElementById('dcf-screen-stop-btn').style.display = 'none';
                    document.getElementById('dcf-screen-bar').style.width = '100%';
                    document.getElementById('dcf-screen-pct').textContent = '100%';
                    document.getElementById('dcf-screen-status').textContent = 'Done! ' + d.total_screened + ' screened, ' + d.total_undervalued + ' undervalued, ' + d.skipped + ' skipped, ' + d.errors + ' errors';
                    renderDCFScreenResults(d.results, d.total_screened, d.total_undervalued);
                }
            };
            _dcfScreenES.onerror = function() {
                if (_dcfScreenES) { _dcfScreenES.close(); _dcfScreenES = null; }
                document.getElementById('dcf-screen-btn').disabled = false;
                document.getElementById('dcf-screen-btn').textContent = 'Screen All Stocks';
                document.getElementById('dcf-screen-stop-btn').style.display = 'none';
                document.getElementById('dcf-screen-status').textContent = 'Connection lost. Results shown below (if any).';
            };
        }
        function stopDCFScreen() {
            if (_dcfScreenES) { _dcfScreenES.close(); _dcfScreenES = null; }
            document.getElementById('dcf-screen-btn').disabled = false;
            document.getElementById('dcf-screen-btn').textContent = 'Screen All Stocks';
            document.getElementById('dcf-screen-stop-btn').style.display = 'none';
            document.getElementById('dcf-screen-status').textContent = 'Stopped by user.';
        }

        function renderDCFScreenResults(results, totalScreened, totalUndervalued) {
            var container = document.getElementById('dcf-screen-results');
            if (!results || results.length === 0) {
                container.innerHTML = '<div class="card" style="text-align:center;padding:40px;"><p style="color:var(--text-muted);font-size:1.1em;">No undervalued stocks found with current assumptions.</p></div>';
                container.style.display = 'block';
                return;
            }
            var h = '';
            // Summary header
            h += '<div class="card" style="margin-bottom:20px;border:1px solid rgba(61,181,120,0.3);background:linear-gradient(135deg,var(--bg-card) 0%,rgba(16,185,129,0.06) 100%);">';
            h += '<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">';
            h += '<div style="flex:1;min-width:200px;">';
            h += '<h2 style="margin:0 0 6px;font-size:1.3em;">Top 50 Undervalued Stocks</h2>';
            h += '<p style="color:var(--text-secondary);margin:0;font-size:0.88em;">Screened ' + totalScreened + ' stocks &bull; ' + totalUndervalued + ' appear undervalued &bull; Showing top 50 by upside</p>';
            h += '</div>';
            h += '<div style="display:flex;gap:16px;flex-wrap:wrap;">';
            h += '<div style="text-align:center;"><div style="font-size:1.8em;font-weight:800;color:var(--accent-green);">' + results.length + '</div><div style="font-size:0.72em;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;">Shown</div></div>';
            var avgUpside = results.reduce(function(s, r) { return s + r.upside_pct; }, 0) / results.length;
            h += '<div style="text-align:center;"><div style="font-size:1.8em;font-weight:800;color:var(--accent-cyan);">+' + avgUpside.toFixed(0) + '%</div><div style="font-size:0.72em;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;">Avg Upside</div></div>';
            h += '</div></div>';
            h += '<p style="color:var(--text-muted);font-size:0.78em;margin-top:12px;margin-bottom:0;">Assumptions: 12% WACC, 3% terminal growth, 10-year projection, historical growth rates. Financial firms use Damodaran Excess Return model.</p>';
            h += '</div>';

            // Results table
            h += '<div class="card" style="padding:0;overflow:hidden;">';
            h += '<div style="overflow-x:auto;">';
            h += '<table style="width:100%;border-collapse:collapse;font-size:0.88em;">';
            h += '<thead><tr style="background:var(--bg-dark);text-align:left;">';
            h += '<th style="padding:12px 14px;color:var(--text-muted);font-size:0.78em;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;">#</th>';
            h += '<th style="padding:12px 14px;color:var(--text-muted);font-size:0.78em;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;">Stock</th>';
            h += '<th style="padding:12px 14px;color:var(--text-muted);font-size:0.78em;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;">Sector</th>';
            h += '<th style="padding:12px 14px;color:var(--text-muted);font-size:0.78em;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;text-align:right;">CMP</th>';
            h += '<th style="padding:12px 14px;color:var(--text-muted);font-size:0.78em;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;text-align:right;">Intrinsic</th>';
            h += '<th style="padding:12px 14px;color:var(--text-muted);font-size:0.78em;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;text-align:right;">Upside</th>';
            h += '<th style="padding:12px 14px;color:var(--text-muted);font-size:0.78em;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;text-align:right;">P/E</th>';
            h += '<th style="padding:12px 14px;color:var(--text-muted);font-size:0.78em;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;text-align:right;">RoCE/RoE</th>';
            h += '<th style="padding:12px 14px;color:var(--text-muted);font-size:0.78em;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;">Model</th>';
            h += '</tr></thead><tbody>';

            results.forEach(function(r, i) {
                var bgColor = i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)';
                var upsideColor = r.upside_pct >= 50 ? 'var(--accent-green)' : r.upside_pct >= 25 ? '#10b981' : 'var(--accent-cyan)';
                var peStr = r.pe_ratio ? r.pe_ratio.toFixed(1) + 'x' : 'N/A';
                var roceVal = r.roce ? r.roce : r.roe;
                var roceStr = roceVal ? (roceVal * 100).toFixed(1) + '%' : 'N/A';
                var roceColor = roceVal ? (roceVal * 100 >= 15 ? 'var(--accent-green)' : roceVal * 100 >= 8 ? 'var(--warning)' : 'var(--danger)') : 'var(--text-muted)';
                var modelBadge = r.model === 'excess_return'
                    ? '<span style="padding:2px 8px;background:var(--accent-purple);color:#fff;border-radius:4px;font-size:0.75em;font-weight:600;">Excess Return</span>'
                    : '<span style="padding:2px 8px;background:var(--accent-cyan);color:#0a1628;border-radius:4px;font-size:0.75em;font-weight:600;">DCF</span>';
                h += '<tr style="background:' + bgColor + ';border-bottom:1px solid var(--border-color);cursor:pointer;" onclick="toggleScreenWriteup(' + i + ')">';
                h += '<td style="padding:12px 14px;color:var(--text-muted);font-weight:600;">' + (i + 1) + '</td>';
                h += '<td style="padding:12px 14px;"><div style="font-weight:700;color:var(--text-primary);">' + r.symbol + '</div><div style="font-size:0.78em;color:var(--text-muted);margin-top:2px;">' + (r.name || '') + '</div></td>';
                h += '<td style="padding:12px 14px;color:var(--text-secondary);font-size:0.85em;">' + (r.sector || '') + '</td>';
                h += '<td style="padding:12px 14px;text-align:right;color:var(--text-primary);font-weight:600;">\u20b9' + r.current_price.toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2}) + '</td>';
                h += '<td style="padding:12px 14px;text-align:right;color:var(--accent-green);font-weight:700;">\u20b9' + r.intrinsic_value.toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2}) + '</td>';
                h += '<td style="padding:12px 14px;text-align:right;font-weight:800;color:' + upsideColor + ';font-size:1.05em;">+' + r.upside_pct.toFixed(1) + '%</td>';
                h += '<td style="padding:12px 14px;text-align:right;color:var(--text-secondary);">' + peStr + '</td>';
                h += '<td style="padding:12px 14px;text-align:right;color:' + roceColor + ';font-weight:600;">' + roceStr + '</td>';
                h += '<td style="padding:12px 14px;">' + modelBadge + '</td>';
                h += '</tr>';
                // Hidden write-up row
                h += '<tr id="screen-writeup-' + i + '" style="display:none;"><td colspan="9" style="padding:0;">';
                h += '<div style="padding:16px 20px 16px 44px;background:rgba(61,122,181,0.06);border-left:3px solid var(--accent-cyan);">';
                h += '<div style="font-size:0.76em;text-transform:uppercase;color:var(--accent-cyan);font-weight:700;letter-spacing:0.5px;margin-bottom:8px;">Investment Thesis</div>';
                h += '<p style="color:var(--text-secondary);font-size:0.88em;line-height:1.75;margin:0;">' + (r.writeup || 'No write-up available.') + '</p>';
                h += `<div style="margin-top:10px;"><button onclick="event.stopPropagation();fetchDCFFromScreen('${r.symbol}')" style="padding:6px 14px;background:var(--accent-cyan);color:#0a1628;border:none;border-radius:6px;font-size:0.8em;font-weight:700;cursor:pointer;font-family:'Space Grotesk',sans-serif;">View Full DCF Analysis</button></div>`;
                h += '</div></td></tr>';
            });
            h += '</tbody></table></div></div>';

            h += '<div class="dcf-disclaimer" style="margin-top:16px;">\u26a0\ufe0f <strong>Disclaimer:</strong> This screener uses automated DCF assumptions (12% WACC, 3% terminal growth) which may not suit every stock. High upside figures can result from optimistic growth extrapolation. Always verify with your own analysis before making investment decisions.</div>';

            container.innerHTML = h;
            container.style.display = 'block';
            container.scrollIntoView({behavior: 'smooth', block: 'start'});
        }

        function toggleScreenWriteup(idx) {
            var el = document.getElementById('screen-writeup-' + idx);
            if (el) el.style.display = el.style.display === 'none' ? 'table-row' : 'none';
        }
        function fetchDCFFromScreen(symbol) {
            document.getElementById('dcf-search').value = symbol;
            document.getElementById('dcf-screen-results').style.display = 'none';
            fetchDCFData();
        }


        // ===== INVESTMENT VERDICT TAB =====
        function verdictGoBack() {
            document.getElementById('verdict-result-view').style.display = 'none';
            var returnTab = _verdictReturnTab;
            _verdictReturnTab = null;
            if (returnTab) {
                switchTab(returnTab);
            } else {
                document.getElementById('verdict-search-view').style.display = 'block';
            }
        }

        function verdictScoreColor(score) {
            if (score >= 65) return 'var(--accent-green)';
            if (score >= 40) return 'var(--warning)';
            return 'var(--danger)';
        }
        function verdictBadge(score) {
            if (score >= 65) return ['Suitable', 'vd-badge-suitable'];
            if (score >= 40) return ['Moderate', 'vd-badge-moderate'];
            return ['Not Ideal', 'vd-badge-weak'];
        }
        function metColor(c) {
            if (c === 'green') return 'var(--accent-green)';
            if (c === 'cyan')  return 'var(--accent-cyan)';
            if (c === 'yellow') return 'var(--warning)';
            if (c === 'red')   return 'var(--danger)';
            return 'var(--text-secondary)';
        }

        function scoreShortTerm(tech) {
            if (!tech || tech.error || !tech.signal) return { score: 0, items: [], error: true };
            const s = tech.signal || {};
            const d = tech.details || {};
            let score = 45;
            const items = [];
            const sig = s.signal || 'HOLD';
            if (sig === 'BUY') { score += 20; items.push({ label: 'Signal', value: 'BUY', color: 'green' }); }
            else if (sig === 'SELL') { score -= 20; items.push({ label: 'Signal', value: 'SELL', color: 'red' }); }
            else { items.push({ label: 'Signal', value: 'HOLD', color: 'yellow' }); }
            const conf = parseFloat(s.confidence) || 0;
            if (conf >= 75) score += 10; else if (conf >= 50) score += 5;
            items.push({ label: 'Confidence', value: conf.toFixed(0) + '%', color: conf >= 65 ? 'green' : conf >= 40 ? 'yellow' : 'red' });
            const rsi = parseFloat(d.rsi_raw) || 50;
            if (rsi >= 40 && rsi <= 65) score += 10; else if (rsi > 78 || rsi < 22) score -= 10;
            items.push({ label: 'RSI', value: rsi.toFixed(0), color: rsi >= 40 && rsi <= 65 ? 'green' : rsi > 72 || rsi < 28 ? 'red' : 'yellow' });
            if (d.macd_bullish) { score += 8; items.push({ label: 'MACD', value: 'Bullish', color: 'green' }); }
            else { items.push({ label: 'MACD', value: 'Bearish', color: 'red' }); }
            if (d.above_sma20 && d.above_sma50) { score += 8; items.push({ label: 'Trend', value: 'Above 20 & 50 SMA', color: 'green' }); }
            else if (d.above_sma20) { score += 4; items.push({ label: 'Trend', value: 'Above 20 SMA', color: 'yellow' }); }
            else { items.push({ label: 'Trend', value: 'Below SMAs', color: 'red' }); }
            const rr = parseFloat(s.risk_reward) || 0;
            if (rr >= 2.5) score += 10; else if (rr >= 1.5) score += 5;
            items.push({ label: 'Risk / Reward', value: rr.toFixed(1) + 'x', color: rr >= 2 ? 'green' : rr >= 1.5 ? 'yellow' : 'red' });
            const regime = parseFloat(s.regime_score) || 0.5;
            if (regime >= 0.7) score += 8; else if (regime >= 0.4) score += 3; else score -= 5;
            const mr = (s.market_regime || 'unknown').charAt(0).toUpperCase() + (s.market_regime || 'unknown').slice(1);
            items.push({ label: 'Market Regime', value: mr, color: regime >= 0.6 ? 'green' : regime >= 0.4 ? 'yellow' : 'red' });
            return { score: Math.max(0, Math.min(100, score)), items };
        }

        function scoreLongTerm(tech, dcfD, regr) {
            let score = 40;
            const items = [];
            if (dcfD && !dcfD.error && dcfD.valuation_model === 'excess_return' && dcfD.book_value_per_share && dcfD.shares_outstanding) {
                // --- Excess Return model scoring for financial firms ---
                const roe = dcfD.roe || 0;
                const ke = dcfD.cost_of_equity || 0.13;
                const g = dcfD.suggested_growth_rate || 0.1;
                const res = runExcessReturn(dcfD.book_value_per_share, roe, ke, g, 10, dcfD.shares_outstanding);
                const up = ((res.intrinsic - dcfD.current_price) / Math.max(dcfD.current_price, 1)) * 100;
                if (up >= 30) score += 25; else if (up >= 15) score += 12; else if (up >= 0) score += 5; else if (up < -20) score -= 15; else score -= 5;
                items.push({ label: 'Excess Return Upside', value: (up >= 0 ? '+' : '') + up.toFixed(0) + '%', color: up >= 15 ? 'green' : up >= 0 ? 'yellow' : 'red' });
                const roePct = roe * 100;
                if (roePct >= 15) score += 12; else if (roePct >= 8) score += 6; else if (roePct >= 0) score += 2; else score -= 5;
                items.push({ label: 'RoE', value: roePct.toFixed(1) + '%', color: roePct >= 15 ? 'green' : roePct >= 8 ? 'yellow' : 'red' });
                const spread = (roe - ke) * 100;
                if (spread >= 5) score += 15; else if (spread >= 0) score += 6; else if (spread >= -3) score += 0; else score -= 8;
                items.push({ label: 'Excess Spread (ROE\u2212Ke)', value: (spread >= 0 ? '+' : '') + spread.toFixed(1) + '%', color: spread >= 3 ? 'green' : spread >= 0 ? 'yellow' : 'red' });
                const pe = dcfD.pe_ratio;
                if (pe && pe > 0) {
                    if (pe < 15) score += 12; else if (pe < 25) score += 6; else if (pe > 50) score -= 5;
                    items.push({ label: 'P/E Ratio', value: pe.toFixed(1) + 'x', color: pe < 20 ? 'green' : pe < 35 ? 'yellow' : 'red' });
                }
                const pb = dcfD.pb_ratio;
                if (pb && pb > 0) {
                    if (pb < 2) score += 6; else if (pb < 4) score += 3;
                    items.push({ label: 'P/B Ratio', value: pb.toFixed(2) + 'x', color: pb < 2 ? 'green' : pb < 4 ? 'yellow' : 'red' });
                }
                const bvg = dcfD.bv_growth;
                if (bvg !== null && bvg !== undefined) {
                    const bvgPct = bvg * 100;
                    if (bvgPct >= 15) score += 10; else if (bvgPct >= 8) score += 5; else if (bvgPct >= 0) score += 2;
                    items.push({ label: 'BV Growth', value: bvgPct.toFixed(1) + '%/yr', color: bvgPct >= 12 ? 'green' : bvgPct >= 5 ? 'yellow' : 'red' });
                }
            } else if (dcfD && !dcfD.error && dcfD.current_fcf && dcfD.shares_outstanding) {
                // --- Standard DCF scoring ---
                const g1 = dcfD.suggested_growth_rate || 0.1;
                const g2 = Math.max(g1 * 0.5, 0.04);
                const _isFin2 = isFinancialSector(dcfD.sector);
                const res = runDCF(dcfD.current_fcf, g1, g2, 0.12, 0.03, 10, _isFin2 ? 0 : (dcfD.total_debt || 0), _isFin2 ? 0 : (dcfD.cash || 0), dcfD.shares_outstanding);
                const up = ((res.intrinsic - dcfD.current_price) / Math.max(dcfD.current_price, 1)) * 100;
                if (up >= 30) score += 25; else if (up >= 15) score += 12; else if (up >= 0) score += 5; else if (up < -20) score -= 15; else score -= 5;
                items.push({ label: 'DCF Upside', value: (up >= 0 ? '+' : '') + up.toFixed(0) + '%', color: up >= 15 ? 'green' : up >= 0 ? 'yellow' : 'red' });
                const pe = dcfD.pe_ratio;
                if (pe && pe > 0) {
                    if (pe < 15) score += 12; else if (pe < 25) score += 6; else if (pe > 50) score -= 5;
                    items.push({ label: 'P/E Ratio', value: pe.toFixed(1) + 'x', color: pe < 20 ? 'green' : pe < 35 ? 'yellow' : 'red' });
                }
                const pb = dcfD.pb_ratio;
                if (pb && pb > 0) {
                    if (pb < 2) score += 6; else if (pb < 4) score += 3;
                    items.push({ label: 'P/B Ratio', value: pb.toFixed(2) + 'x', color: pb < 2 ? 'green' : pb < 4 ? 'yellow' : 'red' });
                }
                const fcgr = dcfD.historical_fcf_growth;
                if (fcgr !== null && fcgr !== undefined) {
                    const pct = fcgr * 100;
                    if (pct >= 20) score += 15; else if (pct >= 10) score += 8; else if (pct >= 0) score += 3;
                    items.push({ label: 'FCF Growth', value: pct.toFixed(1) + '%/yr', color: pct >= 15 ? 'green' : pct >= 5 ? 'yellow' : 'red' });
                }
                const roce = dcfD.roce;
                if (roce !== null && roce !== undefined) {
                    const rocePct = roce * 100;
                    if (rocePct >= 20) score += 12; else if (rocePct >= 12) score += 6; else if (rocePct >= 0) score += 2; else score -= 4;
                    items.push({ label: 'RoCE', value: rocePct.toFixed(1) + '%', color: rocePct >= 20 ? 'green' : rocePct >= 12 ? 'yellow' : 'red' });
                }
                const roe = dcfD.roe;
                if (roe !== null && roe !== undefined) {
                    const roePct = roe * 100;
                    if (roePct >= 15) score += 8; else if (roePct >= 8) score += 4; else if (roePct < 0) score -= 5;
                    items.push({ label: 'RoE', value: roePct.toFixed(1) + '%', color: roePct >= 15 ? 'green' : roePct >= 8 ? 'yellow' : 'red' });
                }
            }
            if (regr && !regr.error && regr.beta !== undefined) {
                const beta = regr.beta || 1;
                if (beta < 0.7) score += 10; else if (beta < 1.0) score += 5;
                items.push({ label: 'Beta', value: beta.toFixed(2), color: beta < 0.8 ? 'green' : beta < 1.2 ? 'yellow' : 'red' });
                const dep = regr.dependency_score || 0.5;
                if (dep < 0.3) score += 8; else if (dep < 0.5) score += 4;
                items.push({ label: 'Market Dependency', value: (dep * 100).toFixed(0) + '%', color: dep < 0.35 ? 'green' : dep < 0.6 ? 'yellow' : 'red' });
            }
            if (tech && tech.details) {
                const d = tech.details;
                const bb = d.bb_position || 50;
                if (bb < 30) { score += 5; items.push({ label: 'Bollinger Band', value: 'Oversold zone', color: 'green' }); }
                else if (bb > 80) { score -= 5; items.push({ label: 'Bollinger Band', value: 'Overbought zone', color: 'red' }); }
                else { items.push({ label: 'Bollinger Band', value: 'Normal zone', color: 'yellow' }); }
            }
            return { score: Math.max(0, Math.min(100, score)), items };
        }

        function scoreDividend(divD) {
            const items = [];
            if (!divD || divD.error || !divD.found || !divD.dividend_yield) {
                return { score: 5, items: [{ label: 'Dividend', value: 'No dividend found', color: 'red' }] };
            }
            let score = 0;
            const fyLbl = divD.fy_label || 'FY';

            // ── Long-term trend guard ────────────────────────────────────────────
            if (divD.in_downtrend) {
                // Stock is below 200-DMA — never a safe dividend recommendation
                items.push({ label: 'Long-term Trend', value: 'Downtrend \u2014 price below 200-DMA', color: 'red' });
                const yld = parseFloat(divD.dividend_yield) || 0;
                if (yld > 0) items.push({ label: 'Annual Yield', value: yld.toFixed(2) + '% (' + fyLbl + ')', color: 'red' });
                items.push({ label: 'Caution', value: 'Not suitable for income investing', color: 'red' });
                return { score: 8, items };
            }

            // ── Uptrend confirmed ────────────────────────────────────────────────
            items.push({ label: 'Long-term Trend', value: 'Uptrend \u2014 above 200-DMA', color: 'green' });
            score += 10;

            // ── Dividend yield (FY-based) ────────────────────────────────────────
            const yld = parseFloat(divD.dividend_yield) || 0;
            if (yld >= 4) score += 36; else if (yld >= 2) score += 22; else if (yld >= 0.5) score += 10;
            items.push({ label: 'Annual Yield', value: yld.toFixed(2) + '% (' + fyLbl + ')', color: yld >= 3 ? 'green' : yld >= 1.5 ? 'yellow' : 'red' });

            // ── Consistency: consecutive FYs with dividends ──────────────────────
            const yrsC = parseInt(divD.years_consistent) || 0;
            if      (yrsC >= 7) score += 22;
            else if (yrsC >= 5) score += 16;
            else if (yrsC >= 3) score += 8;
            else if (yrsC >= 1) score += 2;
            else                score -= 5;
            if (yrsC > 0) items.push({ label: 'Consistency', value: yrsC + ' consecutive FY' + (yrsC > 1 ? 's' : ''), color: yrsC >= 5 ? 'green' : yrsC >= 3 ? 'yellow' : 'red' });
            else items.push({ label: 'Consistency', value: 'No consistent history', color: 'red' });

            // ── Price volatility ─────────────────────────────────────────────────
            const vol = parseFloat(divD.volatility) || 50;
            if (vol < 22) score += 18; else if (vol < 32) score += 10; else if (vol < 42) score += 4;
            items.push({ label: 'Price Volatility', value: vol.toFixed(1) + '%', color: vol < 25 ? 'green' : vol < 38 ? 'yellow' : 'red' });

            // ── Dates ────────────────────────────────────────────────────────────
            if (divD.ex_dividend_date) { score += 8; items.push({ label: 'Ex-Div Date', value: divD.ex_dividend_date, color: 'cyan' }); }
            if (divD.payment_date)     { score += 4; items.push({ label: 'Payment Date', value: divD.payment_date, color: 'cyan' }); }

            const annDiv = parseFloat(divD.annual_dividend) || 0;
            if (annDiv > 0) items.push({ label: 'Annual Div/Share', value: '\u20b9' + annDiv.toFixed(2) + ' (' + fyLbl + ')', color: 'cyan' });

            // ── Sustainability check: flag if yield was capped ─────────────────
            if (divD.yield_capped) {
                score -= 10;
                const latFy = parseFloat(divD.latest_fy_dividend) || 0;
                const prvFy = parseFloat(divD.prev_fy_dividend) || 0;
                items.push({ label: 'Dividend Spike', value: 'Latest FY \u20b9' + latFy.toFixed(2) + ' vs prev FY \u20b9' + prvFy.toFixed(2) + ' \u2014 yield adjusted down', color: 'red' });
            } else if (divD.prev_fy_dividend > 0) {
                items.push({ label: 'Prev FY Div', value: '\u20b9' + parseFloat(divD.prev_fy_dividend).toFixed(2), color: 'cyan' });
            }

            return { score: Math.max(0, Math.min(100, score)), items };
        }

        function buildScoreRing(score, color) {
            var r = 32, circ = 2 * Math.PI * r;
            var dash = (score / 100) * circ;
            var gap = circ - dash;
            return `<svg width="80" height="80" viewBox="0 0 80 80"><circle cx="40" cy="40" r="${r}" fill="none" stroke="var(--bg-dark)" stroke-width="8"/><circle cx="40" cy="40" r="${r}" fill="none" stroke="${color}" stroke-width="8" stroke-dasharray="${dash.toFixed(1)} ${gap.toFixed(1)}" stroke-linecap="round"/></svg>`;
        }

        function buildMetricsHtml(items) {
            if (!items || items.length === 0) return '';
            return '<div class="vd-metrics">' + items.map(function(it) {
                return `<div class="vd-metric"><div class="vd-metric-label">${it.label}</div><div class="vd-metric-value ${it.color || 'muted'}">${it.value}</div></div>`;
            }).join('') + '</div>';
        }

        function buildScoreGrid(stRes, ltRes, divRes) {
            var cards = [
                { title: '⚡ Short-Term Trading', res: stRes, color: 'var(--accent-cyan)',   sectionId: 'vd-st-section' },
                { title: '📈 Long-Term Holding',  res: ltRes, color: 'var(--accent-green)',  sectionId: 'vd-lt-section' },
                { title: '💰 Dividend Income',    res: divRes, color: 'var(--warning)',       sectionId: 'vd-div-section' }
            ];
            var maxScore = Math.max(stRes.score, ltRes.score, divRes.score);
            var html = '<div class="vd-score-grid">';
            cards.forEach(function(c) {
                var [badgeText, badgeClass] = verdictBadge(c.res.score);
                var isBest = (c.res.score === maxScore && c.res.score >= 55);
                html += `<div class="vd-score-card${isBest ? ' vd-best' : ''}" title="Click to expand details" onclick="scrollToVdSection('${c.sectionId}')">
                    <div class="vd-score-label" style="color:${c.color};">${c.title}</div>
                    <div class="vd-score-ring">
                        ${buildScoreRing(c.res.score, c.color)}
                        <div class="vd-score-ring-num" style="color:${c.color};">${c.res.score}</div>
                    </div>
                    <div class="vd-verdict-badge ${badgeClass}">${isBest ? '&#9733; ' : ''}${badgeText}</div>
                    <div style="font-size:0.72em;color:var(--text-muted);margin-top:6px;">Tap to expand &#8595;</div>
                </div>`;
            });
            html += '</div>';
            return html;
        }

        function scrollToVdSection(id) {
            var el = document.getElementById(id);
            if (!el) return;
            if (!el.classList.contains('open')) el.classList.add('open');
            setTimeout(function() { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 50);
        }

        function verdictToggle(el) {
            el.closest('.vd-section').classList.toggle('open');
        }

        function vdSectionHeader(title, titleColor, badge, bdgClass, score) {
            return '<div class="vd-section-header" onclick="verdictToggle(this)">'
                 + '<div class="vd-section-header-left">'
                 + '<div class="vd-section-title" style="color:' + titleColor + ';">' + title + '</div>'
                 + '<div class="vd-section-score ' + bdgClass + '">' + badge + ' &bull; ' + score + '/100</div>'
                 + '</div>'
                 + '<span class="vd-section-chevron">&#9660;</span>'
                 + '</div>'
                 + '<div class="vd-section-body">';
        }

        function buildSTSection(tech, stRes) {
            var s = tech && tech.signal ? tech.signal : {};
            var d = tech && tech.details ? tech.details : {};
            var [badge, bdgClass] = verdictBadge(stRes.score);
            var h = '<div class="vd-section" id="vd-st-section">';
            h += vdSectionHeader('&#9889; Short-Term Trading', 'var(--accent-cyan)', badge, bdgClass, stRes.score);
            if (stRes.error) { h += '<p style="color:var(--text-muted);margin:0;">Technical data unavailable.</p>'; }
            else {
                h += buildMetricsHtml(stRes.items);
                if (s.target || s.stop) {
                    h += '<div class="vd-trade-levels">';
                    h += `<div class="vd-level"><div class="vd-level-label">Entry / CMP</div><div class="vd-level-val" style="color:var(--text-primary);">${d.price || 'N/A'}</div></div>`;
                    h += `<div class="vd-level"><div class="vd-level-label">Target</div><div class="vd-level-val" style="color:var(--accent-green);">${s.target || 'N/A'}</div></div>`;
                    h += `<div class="vd-level"><div class="vd-level-label">Stop Loss</div><div class="vd-level-val" style="color:var(--danger);">${s.stop || 'N/A'}</div></div>`;
                    h += '</div>';
                }
                if (s.setup_duration || s.days_to_target) {
                    h += '<div style="display:flex;gap:10px;flex-wrap:wrap;margin:10px 0;">';
                    if (s.setup_duration) h += `<div style="background:var(--bg-dark);padding:6px 13px;border-radius:20px;font-size:0.8em;color:var(--text-secondary);">&#9203; ${s.setup_duration}</div>`;
                    if (s.days_to_target) h += `<div style="background:var(--bg-dark);padding:6px 13px;border-radius:20px;font-size:0.8em;color:var(--text-secondary);">&#128197; ~${s.days_to_target} days to target</div>`;
                    if (s.risk_reward) h += `<div style="background:var(--bg-dark);padding:6px 13px;border-radius:20px;font-size:0.8em;color:var(--accent-cyan);">R:R ${s.risk_reward}x</div>`;
                    h += '</div>';
                }
                if (s.why_makes_sense) {
                    h += `<div class="vd-narrative"><strong style="color:var(--text-primary);">Why this setup makes sense:</strong><br>${s.why_makes_sense}</div>`;
                } else if (s.verdict_text) {
                    h += `<div class="vd-narrative">${s.verdict_text}</div>`;
                }
                if (s.regime_reason_text) {
                    h += `<div class="vd-narrative" style="margin-top:8px;border-left:3px solid var(--warning);padding-left:14px;"><strong style="color:var(--warning);">Market Regime:</strong> ${s.regime_reason_text}</div>`;
                }
            }
            h += '</div></div>';
            return h;
        }

        function buildLTSection(tech, dcfD, regr, ltRes) {
            var [badge, bdgClass] = verdictBadge(ltRes.score);
            var h = '<div class="vd-section" id="vd-lt-section">';
            h += vdSectionHeader('&#128200; Long-Term Holding', 'var(--accent-green)', badge, bdgClass, ltRes.score);
            h += buildMetricsHtml(ltRes.items);
            var narrativeParts = [];
            if (dcfD && !dcfD.error && dcfD.valuation_model === 'excess_return' && dcfD.book_value_per_share && dcfD.shares_outstanding) {
                // --- Excess Return model narrative for financial firms ---
                var roe = dcfD.roe || 0;
                var ke = dcfD.cost_of_equity || 0.13;
                var g = dcfD.suggested_growth_rate || 0.1;
                var res = runExcessReturn(dcfD.book_value_per_share, roe, ke, g, 10, dcfD.shares_outstanding);
                var up = ((res.intrinsic - dcfD.current_price) / Math.max(dcfD.current_price, 1)) * 100;
                var ivStr = '\u20b9' + res.intrinsic.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
                h += `<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 0 14px;">
                    <div class="vd-level"><div class="vd-level-label">Intrinsic Value (Excess Return)</div><div class="vd-level-val" style="color:${up >= 0 ? 'var(--accent-green)' : 'var(--danger)'};">${ivStr}</div></div>
                    <div class="vd-level"><div class="vd-level-label">Upside / Downside</div><div class="vd-level-val" style="color:${up >= 0 ? 'var(--accent-green)' : 'var(--danger)'};">${(up >= 0 ? '+' : '') + up.toFixed(1)}%</div></div>
                </div>`;
                var spread = ((roe - ke) * 100).toFixed(1);
                if (up >= 20) narrativeParts.push('Trading at a <strong style="color:var(--accent-green);">' + up.toFixed(0) + '% discount</strong> to its estimated intrinsic value (Damodaran Excess Return model) \u2014 potentially strong margin of safety for long-term investors.');
                else if (up >= 5) narrativeParts.push('Modestly undervalued by ~' + up.toFixed(0) + '% on the Excess Return model \u2014 reasonable entry if ROE sustainability is sound.');
                else if (up < -20) narrativeParts.push('Excess Return model suggests the stock is priced <strong style="color:var(--danger);">' + Math.abs(up).toFixed(0) + '% above</strong> its fair value \u2014 long-term upside may be limited.');
                else narrativeParts.push('Trading near its estimated fair value according to the Damodaran Excess Return model.');
                if (parseFloat(spread) > 0) narrativeParts.push('ROE exceeds the cost of equity by <strong>' + spread + '%</strong>, indicating the firm <strong style="color:var(--accent-green);">creates value</strong> above its book.');
                else narrativeParts.push('ROE is <strong style="color:var(--danger);">below the cost of equity</strong> by ' + Math.abs(parseFloat(spread)).toFixed(1) + '% \u2014 the firm currently destroys value relative to its book.');
                if (dcfD.bv_growth) {
                    var bvPct = (dcfD.bv_growth * 100).toFixed(1);
                    narrativeParts.push('Book value has grown at <strong>' + bvPct + '% per year</strong> \u2014 ' + (parseFloat(bvPct) >= 12 ? 'strong equity compounding.' : parseFloat(bvPct) >= 5 ? 'moderate book growth.' : 'slow equity expansion, worth monitoring.'));
                }
            } else if (dcfD && !dcfD.error && dcfD.current_fcf && dcfD.shares_outstanding) {
                var g1 = dcfD.suggested_growth_rate || 0.1;
                var g2 = Math.max(g1 * 0.5, 0.04);
                var _isF3 = isFinancialSector(dcfD.sector);
                var res = runDCF(dcfD.current_fcf, g1, g2, 0.12, 0.03, 10, _isF3 ? 0 : (dcfD.total_debt || 0), _isF3 ? 0 : (dcfD.cash || 0), dcfD.shares_outstanding);
                var up = ((res.intrinsic - dcfD.current_price) / Math.max(dcfD.current_price, 1)) * 100;
                var ivStr = '\u20b9' + res.intrinsic.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
                h += `<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 0 14px;">
                    <div class="vd-level"><div class="vd-level-label">Intrinsic Value (DCF)</div><div class="vd-level-val" style="color:${up >= 0 ? 'var(--accent-green)' : 'var(--danger)'};">${ivStr}</div></div>
                    <div class="vd-level"><div class="vd-level-label">Upside / Downside</div><div class="vd-level-val" style="color:${up >= 0 ? 'var(--accent-green)' : 'var(--danger)'};">${(up >= 0 ? '+' : '') + up.toFixed(1)}%</div></div>
                </div>`;
                if (up >= 20) narrativeParts.push('Trading at a <strong style="color:var(--accent-green);">' + up.toFixed(0) + '% discount</strong> to its estimated intrinsic value — potentially strong margin of safety for long-term investors.');
                else if (up >= 5) narrativeParts.push('Modestly undervalued by ~' + up.toFixed(0) + '% on DCF — reasonable entry if other fundamentals are sound.');
                else if (up < -20) narrativeParts.push('DCF model suggests the stock is priced <strong style="color:var(--danger);">' + Math.abs(up).toFixed(0) + '% above</strong> its estimated fair value — long-term upside may be limited.');
                else narrativeParts.push('Trading near its estimated fair value according to DCF analysis.');
                if (dcfD.historical_fcf_growth) {
                    var pct = (dcfD.historical_fcf_growth * 100).toFixed(1);
                    narrativeParts.push('Historical free cash flow has grown at <strong>' + pct + '% per year</strong> — ' + (parseFloat(pct) >= 15 ? 'a strong compounding track record.' : parseFloat(pct) >= 5 ? 'moderate growth momentum.' : 'slow FCF expansion, worth monitoring.'));
                }
            }
            if (regr && !regr.error) {
                var beta = regr.beta || 1;
                var dep = regr.dependency_score || 0.5;
                if (beta < 0.85) narrativeParts.push('Low beta of <strong>' + beta.toFixed(2) + '</strong> means this stock tends to be defensive — it typically falls less than the market in downturns, suiting a long-term portfolio.');
                else if (beta > 1.3) narrativeParts.push('High beta of <strong>' + beta.toFixed(2) + '</strong> — amplifies market moves. Factor in higher volatility for long-term holding.');
                if (dep < 0.4) narrativeParts.push('Low market dependency (' + (dep * 100).toFixed(0) + '%) — provides <strong>good diversification</strong> from broad market swings.');
                else if (dep > 0.7) narrativeParts.push('High market dependency (' + (dep * 100).toFixed(0) + '%) — performance closely tracks the Nifty 50.');
            }
            if (narrativeParts.length > 0) h += '<div class="vd-narrative">' + narrativeParts.join(' ') + '</div>';
            // --- Trend section ---
            if (dcfD && dcfD.margin_trend && dcfD.margin_trend.length >= 2) {
                h += buildTrendSection(dcfD.margin_trend);
            }
            h += '</div></div>';
            return h;
        }

        function buildTrendSection(trend) {
            if (!trend || trend.length < 2) return '';
            var h = '<div class="trend-section-title">&#128200; Key Trends</div>';
            // Operating margin trend
            var opMargins = trend.filter(function(t){ return t.op_margin !== null && t.op_margin !== undefined; });
            if (opMargins.length >= 2) {
                var first = opMargins[0].op_margin, last = opMargins[opMargins.length-1].op_margin;
                var diff = last - first;
                var arrowClass = diff > 1 ? 'up' : diff < -1 ? 'down' : 'flat';
                var arrowSymbol = diff > 1 ? '&#8593;' : diff < -1 ? '&#8595;' : '&#8594;';
                var valStr = opMargins.map(function(t){ return t.year + ': ' + t.op_margin + '%'; }).join(' &rarr; ');
                h += '<div class="trend-row"><span class="trend-label">Operating Margin</span><span class="trend-values">' + valStr + '</span><span class="trend-arrow ' + arrowClass + '">' + arrowSymbol + '</span></div>';
            }
            // Net margin trend
            var netMargins = trend.filter(function(t){ return t.net_margin !== null && t.net_margin !== undefined; });
            if (netMargins.length >= 2) {
                var first = netMargins[0].net_margin, last = netMargins[netMargins.length-1].net_margin;
                var diff = last - first;
                var arrowClass = diff > 1 ? 'up' : diff < -1 ? 'down' : 'flat';
                var arrowSymbol = diff > 1 ? '&#8593;' : diff < -1 ? '&#8595;' : '&#8594;';
                var valStr = netMargins.map(function(t){ return t.year + ': ' + t.net_margin + '%'; }).join(' &rarr; ');
                h += '<div class="trend-row"><span class="trend-label">Net Profit Margin</span><span class="trend-values">' + valStr + '</span><span class="trend-arrow ' + arrowClass + '">' + arrowSymbol + '</span></div>';
            }
            // Revenue trend
            var revs = trend.filter(function(t){ return t.revenue > 0; });
            if (revs.length >= 2) {
                var revFirst = revs[0].revenue, revLast = revs[revs.length-1].revenue;
                var revGrowth = ((revLast / revFirst) - 1) * 100;
                var arrowClass = revGrowth > 5 ? 'up' : revGrowth < -5 ? 'down' : 'flat';
                var arrowSymbol = revGrowth > 5 ? '&#8593;' : revGrowth < -5 ? '&#8595;' : '&#8594;';
                var revStr = revs.map(function(t){ return t.year + ': ' + fmtCr(t.revenue); }).join(' &rarr; ');
                h += '<div class="trend-row"><span class="trend-label">Revenue</span><span class="trend-values">' + revStr + '</span><span class="trend-arrow ' + arrowClass + '">' + arrowSymbol + ' ' + revGrowth.toFixed(0) + '%</span></div>';
            }
            return h;
        }

        function buildDivSection(divD, divRes) {
            var [badge, bdgClass] = verdictBadge(divRes.score);
            var h = '<div class="vd-section" id="vd-div-section">';
            h += vdSectionHeader('&#128176; Dividend Income', 'var(--warning)', badge, bdgClass, divRes.score);
            h += buildMetricsHtml(divRes.items);
            if (!divD || !divD.found || !divD.dividend_yield) {
                h += '<div class="vd-narrative" style="color:var(--text-muted);">No dividend history found for this stock. It may be a growth-oriented company that reinvests earnings rather than paying dividends.</div>';
            } else {
                var yld   = parseFloat(divD.dividend_yield) || 0;
                var vol   = parseFloat(divD.volatility) || 50;
                var yrsC  = parseInt(divD.years_consistent) || 0;
                var fyLbl = divD.fy_label || 'the last financial year';
                var narrative = '';

                // Downtrend warning overrides everything else
                if (divD.in_downtrend) {
                    narrative += '<span style="color:var(--danger);"><strong>&#9888; Long-term Downtrend:</strong> This stock is trading below its 200-day moving average. '
                               + 'A ' + yld.toFixed(2) + '% dividend yield (' + fyLbl + ') does NOT compensate for sustained capital erosion — '
                               + 'it is <strong>not recommended</strong> as a dividend income holding until the trend reverses.</span> ';
                } else {
                    // Yield narrative
                    if (yld >= 3) narrative += 'With a yield of <strong style="color:var(--warning);">' + yld.toFixed(2) + '%</strong> (' + fyLbl + '), this stock offers above-average dividend income. ';
                    else if (yld >= 1) narrative += 'Provides a modest yield of <strong>' + yld.toFixed(2) + '%</strong> (' + fyLbl + ') — not a primary income play but contributes to total return. ';
                    else narrative += 'Very low dividend yield (' + yld.toFixed(2) + '%, ' + fyLbl + '). Primarily a capital-appreciation story, not a dividend income candidate. ';

                    // Consistency narrative
                    if (yrsC >= 7) narrative += 'Exceptionally consistent payer — <strong>' + yrsC + ' consecutive financial years</strong> of dividends signals a reliable income stream. ';
                    else if (yrsC >= 5) narrative += '<strong>' + yrsC + ' consecutive years</strong> of dividend payments reflect a well-established payout policy. ';
                    else if (yrsC >= 3) narrative += yrsC + ' consecutive years of dividends — a reasonable track record, though longer history preferred for income portfolios. ';
                    else if (yrsC >= 1) narrative += 'Only ' + yrsC + ' year(s) of consistent dividend history — limited track record; treat with caution for income strategies. ';
                    else narrative += 'No consistent multi-year dividend history found. Dividend continuity is uncertain. ';

                    // Volatility narrative
                    if (vol < 25) narrative += 'Low price volatility (' + vol.toFixed(1) + '%) supports dividend sustainability.';
                    else if (vol > 40) narrative += 'High price volatility (' + vol.toFixed(1) + '%) may offset dividend income with capital loss risk.';
                }
                h += '<div class="vd-narrative">' + narrative + '</div>';
            }
            h += '</div></div>';
            return h;
        }

        function buildOverallVerdict(symbol, stRes, ltRes, divRes, bestFor, bestColor, tech, dcfD, regr) {
            var scores = [stRes.score, ltRes.score, divRes.score];
            var avgScore = (scores[0] + scores[1] + scores[2]) / 3;
            var parts = [];
            var maxScore = Math.max(stRes.score, ltRes.score, divRes.score);
            if (ltRes.score >= 65 && ltRes.score === maxScore) {
                parts.push('Strong DCF fundamentals and compounding metrics make this a candidate for <strong>patient long-term investors</strong>.');
            } else if (stRes.score >= 65 && stRes.score === maxScore) {
                var s = tech && tech.signal ? tech.signal : {};
                parts.push('Technical setup favours <strong>near-term active traders</strong>' + (s.signal === 'BUY' ? ' with a clear buy signal.' : ' though signal conviction is moderate.'));
            } else if (divRes.score >= 55 && divRes.score === maxScore) {
                parts.push('Dividend yield and low volatility profile suits <strong>income-focused investors</strong>.');
            } else {
                parts.push('No single strategy shows a dominant edge right now. Consider waiting for a clearer setup or a more attractive valuation.');
            }
            if (stRes.score < 40 && ltRes.score < 40) parts.push('Both short and long-term scores are weak — exercise caution.');
            if (avgScore >= 60) parts.push('Overall profile is <strong style="color:var(--accent-green);">positive</strong> across multiple dimensions.');
            else if (avgScore < 35) parts.push('Overall profile is <strong style="color:var(--danger);">weak</strong> — high risk across dimensions.');
            // Personalised note based on investor profile
            var profileNote = '';
            var ph = investorProfile.horizon, pr = investorProfile.risk, pg = investorProfile.goal;
            var profileScoreMap = { short: stRes.score, medium: Math.round((stRes.score + ltRes.score) / 2), long: ltRes.score };
            var relevantScore = profileScoreMap[ph] || ltRes.score;
            if (pg === 'income') relevantScore = divRes.score;
            if (pg === 'balanced') relevantScore = Math.round(avgScore);
            var profileLabels = { short: 'short-term trader', medium: 'medium-term investor', long: 'long-term investor' };
            var riskLabels = { low: 'low-risk', medium: 'moderate-risk', high: 'high-risk' };
            profileNote = 'For your profile (<strong style="color:var(--accent-cyan);">' + riskLabels[pr] + ' ' + profileLabels[ph] + '</strong>): ';
            if (relevantScore >= 65) profileNote += 'this stock <strong style="color:var(--accent-green);">aligns well</strong> with your investment style.';
            else if (relevantScore >= 45) profileNote += 'this stock is a <strong style="color:var(--warning);">partial fit</strong> — review the detailed sections before deciding.';
            else profileNote += 'this stock <strong style="color:var(--danger);">may not align</strong> with your current profile. Consider exploring alternatives.';
            // Risk warning for high-risk takers
            if (pr === 'low' && (dcfD && dcfD.pb_ratio && dcfD.pb_ratio > 5)) profileNote += ' Note: high P/B may signal elevated risk for conservative investors.';
            var h = `<div class="vd-overall">
                <div class="vd-overall-label">Overall Investment Verdict for ${symbol}</div>
                <div class="vd-overall-verdict" style="color:${bestColor};">Best For: ${bestFor}</div>
                <div class="vd-overall-reason">${parts.join(' ')}</div>
                <div class="vd-overall-reason" style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.06);">${profileNote}</div>
                <div style="margin-top:16px;display:flex;justify-content:center;gap:20px;flex-wrap:wrap;">
                    <div style="text-align:center;"><div style="font-size:0.72em;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;">Short-Term</div><div style="font-family:'Space Grotesk',sans-serif;font-size:1.3em;font-weight:700;color:${verdictScoreColor(stRes.score)};">${stRes.score}</div></div>
                    <div style="text-align:center;"><div style="font-size:0.72em;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;">Long-Term</div><div style="font-family:'Space Grotesk',sans-serif;font-size:1.3em;font-weight:700;color:${verdictScoreColor(ltRes.score)};">${ltRes.score}</div></div>
                    <div style="text-align:center;"><div style="font-size:0.72em;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;">Dividend</div><div style="font-family:'Space Grotesk',sans-serif;font-size:1.3em;font-weight:700;color:${verdictScoreColor(divRes.score)};">${divRes.score}</div></div>
                </div>
            </div>`;
            return h;
        }

        function renderVerdictResult(symbol, tech, regr, dcfD, divD) {
            var d = tech && tech.details ? tech.details : {};
            var price = d.price || 'N/A';
            var dailyRaw = parseFloat(d.daily_raw) || 0;
            var daily = d.daily || '';
            var name = (dcfD && dcfD.name) ? dcfD.name : getStockName(symbol);
            var sector = (dcfD && dcfD.sector) ? dcfD.sector : '';
            var stRes  = scoreShortTerm(tech);
            var ltRes  = scoreLongTerm(tech, dcfD, regr);
            var divRes = scoreDividend(divD);
            var maxScore = Math.max(stRes.score, ltRes.score, divRes.score);
            var bestFor, bestColor;
            if (stRes.score === maxScore && stRes.score >= 55)      { bestFor = 'Short-Term Trading'; bestColor = 'var(--accent-cyan)'; }
            else if (ltRes.score === maxScore && ltRes.score >= 55) { bestFor = 'Long-Term Holding';  bestColor = 'var(--accent-green)'; }
            else if (divRes.score === maxScore && divRes.score >= 45){ bestFor = 'Dividend Income';   bestColor = 'var(--warning)'; }
            else                                                     { bestFor = 'Needs Further Research'; bestColor = 'var(--text-muted)'; }
            var h = '';
            h += '<div class="vd-hero">';
            h += '<div><div class="vd-name">' + name + '</div>';
            h += '<div class="vd-sub">' + symbol + (sector ? ' &bull; ' + sector : '') + '</div></div>';
            h += '<div class="vd-price-box"><div class="vd-price-label">Current Price</div>';
            h += '<div class="vd-price-val">' + price + '</div>';
            h += '<div class="vd-daily ' + (dailyRaw >= 0 ? 'up' : 'down') + '">' + daily + '</div></div>';
            h += '</div>';
            h += buildOverallVerdict(symbol, stRes, ltRes, divRes, bestFor, bestColor, tech, dcfD, regr);
            h += buildScoreGrid(stRes, ltRes, divRes);
            h += buildSTSection(tech, stRes);
            h += buildLTSection(tech, dcfD, regr, ltRes);
            h += buildDivSection(divD, divRes);
            h += buildPeerStocksHTML(symbol, 'verdict');
            h += '<div style="margin-top:16px;padding:12px 16px;background:rgba(113,128,150,0.07);border:1px solid var(--border-color);border-radius:8px;color:var(--text-muted);font-size:0.78em;line-height:1.6;">';
            h += '&#9888;&#65039; <strong>Disclaimer:</strong> This analysis combines multiple quantitative signals and is for educational purposes only. Scores are model-based estimates — not financial advice. Always do your own due diligence before investing.';
            h += '</div>';
            document.getElementById('verdict-result').innerHTML = h;
        }

        var _vd = {};  // state store for current verdict fetch
        function verdictSetResult(key, value, statusId, ok, msg) {
            _vd[key] = value;
            var el = document.getElementById(statusId);
            if (el) { el.textContent = msg; el.className = 'vd-load-status ' + (ok ? 'done' : (ok === false ? 'error' : 'loading')); }
            if (_vd.tech !== undefined && _vd.regr !== undefined && _vd.dcf !== undefined && _vd.div !== undefined) {
                renderVerdictResult(_vd.symbol, _vd.tech, _vd.regr, _vd.dcf, _vd.div);
            }
        }

        function verdictPollRegression(symbol, attempt) {
            if (_vd.symbol !== symbol) return;  // stale request
            fetch('/regression?symbol=' + encodeURIComponent(symbol))
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (_vd.symbol !== symbol) return;
                    if (data.status === 'computing' && attempt < 12) {
                        var el = document.getElementById('vd-load-regr');
                        if (el) el.textContent = '\u23f3 Computing (' + (attempt + 1) + '/12)\u2026';
                        setTimeout(function() { verdictPollRegression(symbol, attempt + 1); }, 2000);
                    } else {
                        var result = (data.status === 'computing') ? null : data;
                        verdictSetResult('regr', result, 'vd-load-regr', result ? true : null, result ? '\u2705 Done' : '\u2139\ufe0f Timed out \u2014 skipped');
                    }
                })
                .catch(function(e) {
                    if (_vd.symbol !== symbol) return;
                    verdictSetResult('regr', null, 'vd-load-regr', false, '\u274c ' + e.message);
                });
        }

        function fetchVerdictData() {
            var symbol = document.getElementById('verdict-search').value.trim().toUpperCase();
            if (!symbol) { alert('Please enter a stock symbol'); return; }
            document.getElementById('verdict-suggestions').innerHTML = '';
            document.getElementById('verdict-search-view').style.display = 'none';
            document.getElementById('verdict-result-view').style.display = 'block';
            _vd = { symbol: symbol };  // reset state, mark 4 keys as pending (undefined)
            var loadHtml = '<div class="vd-hero"><div><div class="vd-name">' + symbol + '</div><div class="vd-sub">Fetching comprehensive analysis\u2026</div></div></div>';
            loadHtml += '<div class="vd-loading-grid">';
            loadHtml += '<div class="vd-load-item"><div class="vd-load-icon">&#9889;</div><div><div class="vd-load-label">Technical Analysis</div><div class="vd-load-status loading" id="vd-load-tech">\u23f3 Fetching\u2026</div></div></div>';
            loadHtml += '<div class="vd-load-item"><div class="vd-load-icon">&#128200;</div><div><div class="vd-load-label">Market Connection</div><div class="vd-load-status loading" id="vd-load-regr">\u23f3 Computing\u2026</div></div></div>';
            loadHtml += '<div class="vd-load-item"><div class="vd-load-icon">&#128202;</div><div><div class="vd-load-label">DCF Valuation</div><div class="vd-load-status loading" id="vd-load-dcf">\u23f3 Fetching\u2026</div></div></div>';
            loadHtml += '<div class="vd-load-item"><div class="vd-load-icon">&#128176;</div><div><div class="vd-load-label">Dividend Data</div><div class="vd-load-status loading" id="vd-load-div">\u23f3 Fetching\u2026</div></div></div>';
            loadHtml += '</div>';
            document.getElementById('verdict-result').innerHTML = loadHtml;
            fetch('/analyze?symbol=' + encodeURIComponent(symbol))
                .then(function(r) { return r.json(); })
                .then(function(data) { verdictSetResult('tech', data, 'vd-load-tech', !data.error, data.error ? '\u274c ' + data.error : '\u2705 Done'); })
                .catch(function(e) { verdictSetResult('tech', null, 'vd-load-tech', false, '\u274c ' + e.message); });
            fetch('/dcf-data?symbol=' + encodeURIComponent(symbol))
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    var doneMsg = '\u2705 Done';
                    if (!data.error && data.valuation_model === 'excess_return') {
                        doneMsg = '\u2705 Excess Return Model';
                        var lbl = document.querySelector('#vd-load-dcf')
                        if (lbl) { var p = lbl.previousElementSibling; if (p) p.textContent = 'Excess Return Valuation'; }
                    }
                    verdictSetResult('dcf', data, 'vd-load-dcf', !data.error, data.error ? '\u274c ' + data.error : doneMsg);
                })
                .catch(function(e) { verdictSetResult('dcf', null, 'vd-load-dcf', false, '\u274c ' + e.message); });
            fetch('/dividend-info?symbol=' + encodeURIComponent(symbol))
                .then(function(r) { return r.json(); })
                .then(function(data) { verdictSetResult('div', data, 'vd-load-div', true, data.found ? '\u2705 Done' : '\u2139\ufe0f No dividend data'); })
                .catch(function(e) { verdictSetResult('div', null, 'vd-load-div', false, '\u274c ' + e.message); });
            verdictPollRegression(symbol, 0);
        }


        window.addEventListener('DOMContentLoaded', () => {
            init(); initDividendSectors(); setupCapitalInput(); loadProfile();
            requestAnimationFrame(()=>{const ds=document.getElementById('deferred-css');if(ds)ds.media='all';});
            const hash = window.location.hash.replace('#','');
            const validTabs = ['verdict','analysis','dcf','dividend','regression','scanner','ai'];
            if (hash && validTabs.includes(hash)) { switchTab(hash); }
        });

        // ── AI Assistant tab ─────────────────────────────────────────────
        let aiHistory = [];
        let aiSending = false;
        let aiCooldownUntil = 0;
        let aiCooldownTimer = null;

        function aiAppend(role, text) {
            const box = document.getElementById('ai-messages');
            if (!box) return null;
            const el = document.createElement('div');
            el.className = 'ai-msg ' + role;
            el.textContent = text;
            box.appendChild(el);
            box.scrollTop = box.scrollHeight;
            return el;
        }
        function aiScrollBottom() {
            const box = document.getElementById('ai-messages');
            if (box) box.scrollTop = box.scrollHeight;
        }
        function aiCreateThinkingBlock() {
            const box = document.getElementById('ai-messages');
            if (!box) return null;
            const wrap = document.createElement('div');
            wrap.className = 'ai-thinking-block';
            const det = document.createElement('details');
            det.open = true;
            const sum = document.createElement('summary');
            const spinner = document.createElement('span');
            spinner.className = 'think-spinner';
            const label = document.createElement('span');
            label.className = 'think-label';
            label.textContent = 'Researching…';
            sum.appendChild(spinner);
            sum.appendChild(label);
            det.appendChild(sum);
            const steps = document.createElement('div');
            steps.className = 'think-steps';
            det.appendChild(steps);
            wrap.appendChild(det);
            box.appendChild(wrap);
            aiScrollBottom();
            return wrap;
        }
        function aiAddThinkingStep(block, text) {
            if (!block) return;
            const det = block.querySelector('details');
            const steps = block.querySelector('.think-steps');
            const label = block.querySelector('.think-label');
            if (!steps) return;
            // Mark previous step done
            const prev = steps.querySelector('.active');
            if (prev) prev.classList.remove('active');
            const step = document.createElement('div');
            step.className = 'ai-thinking-step active';
            step.textContent = text;
            steps.appendChild(step);
            if (label) label.textContent = text;
            aiScrollBottom();
        }
        function aiFinishThinkingBlock(block, stepCount) {
            if (!block) return;
            block.classList.add('done');
            const prev = block.querySelector('.active');
            if (prev) prev.classList.remove('active');
            const label = block.querySelector('.think-label');
            if (label) label.textContent = stepCount + ' data source' + (stepCount !== 1 ? 's' : '') + ' checked ▾';
            const det = block.querySelector('details');
            if (det) det.open = false;
        }
        function aiUseSuggestion(text) {
            const inp = document.getElementById('ai-input');
            if (!inp) return;
            inp.value = text;
            inp.focus();
        }
        function aiStartCooldown(seconds, errEl) {
            const secs = Math.max(1, parseInt(seconds, 10) || 0);
            aiCooldownUntil = Date.now() + secs * 1000;
            const sendBtn = document.getElementById('ai-send-btn');
            const inp = document.getElementById('ai-input');
            if (sendBtn) sendBtn.disabled = true;
            if (inp) inp.disabled = true;
            if (aiCooldownTimer) clearInterval(aiCooldownTimer);
            const tick = function() {
                const left = Math.max(0, Math.ceil((aiCooldownUntil - Date.now()) / 1000));
                if (errEl && errEl.parentNode) {
                    errEl.dataset.cooldown = '1';
                    errEl.textContent = left > 0
                        ? 'Rate-limited. Retrying in ' + left + 's.'
                        : 'Ready. Try again.';
                }
                if (left <= 0) {
                    clearInterval(aiCooldownTimer);
                    aiCooldownTimer = null;
                    if (sendBtn) sendBtn.disabled = false;
                    if (inp) inp.disabled = false;
                }
            };
            tick();
            aiCooldownTimer = setInterval(tick, 1000);
        }
        async function aiSendQuery() {
            if (aiSending) return;
            if (Date.now() < aiCooldownUntil) return;
            const inp = document.getElementById('ai-input');
            if (!inp) return;
            const message = inp.value.trim();
            if (!message) return;
            inp.value = '';
            aiAppend('user', message);
            aiHistory.push({ role: 'user', content: message });
            aiSending = true;
            const sendBtn = document.getElementById('ai-send-btn');
            if (sendBtn) sendBtn.disabled = true;
            inp.disabled = true;

            const thinkBlock = aiCreateThinkingBlock();
            let agentEl = null;
            let fullReply = '';
            let stepCount = 0;
            let committed = false;

            try {
                const res = await fetch('/api/agent/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message, history: aiHistory.slice(0, -1) })
                });
                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}));
                    if (thinkBlock && thinkBlock.parentNode) thinkBlock.parentNode.removeChild(thinkBlock);
                    const errEl = aiAppend('error', errData.error || 'Request failed. Please try again.');
                    aiHistory.pop();
                    if (res.status === 429) aiStartCooldown(errData.retryAfter || 30, errEl);
                    return;
                }
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buf = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buf += decoder.decode(value, { stream: true });
                    const lines = buf.split('\\n');
                    buf = lines.pop();
                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        let evt;
                        try { evt = JSON.parse(line.slice(6)); } catch { continue; }
                        if (evt.type === 'thinking') {
                            stepCount++;
                            aiAddThinkingStep(thinkBlock, evt.text);
                        } else if (evt.type === 'token') {
                            if (!agentEl) {
                                aiFinishThinkingBlock(thinkBlock, stepCount);
                                agentEl = aiAppend('agent', '');
                                agentEl.classList.add('streaming');
                            }
                            fullReply += evt.text;
                            agentEl.textContent = fullReply;
                            aiScrollBottom();
                        } else if (evt.type === 'done') {
                            if (agentEl) agentEl.classList.remove('streaming');
                            if (!committed && fullReply) {
                                aiHistory.push({ role: 'assistant', content: fullReply });
                                committed = true;
                            }
                            if (stepCount === 0 && thinkBlock && thinkBlock.parentNode) {
                                thinkBlock.parentNode.removeChild(thinkBlock);
                            }
                        } else if (evt.type === 'error') {
                            if (thinkBlock && thinkBlock.parentNode) thinkBlock.parentNode.removeChild(thinkBlock);
                            if (agentEl && agentEl.parentNode) agentEl.parentNode.removeChild(agentEl);
                            const errEl = aiAppend('error', evt.text || 'Request failed. Please try again.');
                            aiHistory.pop();
                            if (evt.retryAfter) aiStartCooldown(evt.retryAfter, errEl);
                        }
                    }
                }
                // Ensure cleanup if stream ended without explicit done event
                if (agentEl) agentEl.classList.remove('streaming');
                if (!committed && fullReply) {
                    aiHistory.push({ role: 'assistant', content: fullReply });
                }
                if (stepCount === 0 && thinkBlock && thinkBlock.parentNode) {
                    thinkBlock.parentNode.removeChild(thinkBlock);
                }
            } catch(e) {
                if (thinkBlock && thinkBlock.parentNode) thinkBlock.parentNode.removeChild(thinkBlock);
                aiAppend('error', 'Network error. Please try again.');
                aiHistory.pop();
            } finally {
                aiSending = false;
                const onCooldown = Date.now() < aiCooldownUntil;
                if (sendBtn) sendBtn.disabled = onCooldown;
                inp.disabled = onCooldown;
                if (!onCooldown) inp.focus();
            }
        }
        document.addEventListener('DOMContentLoaded', function() {
            const inp = document.getElementById('ai-input');
            if (inp) inp.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); aiSendQuery(); }
            });
        });
    </script>
</body>
</html>'''
    resp = make_response(html)
    resp.headers['Cache-Control'] = 'public, max-age=300, stale-while-revalidate=600'
    return resp

@app.route('/analyze')
def analyze_route():
    symbol = request.args.get('symbol', '').strip()
    if not symbol: return jsonify({'error': 'No symbol provided'})
    normalized_symbol, original = Analyzer.normalize_symbol(symbol)
    if not normalized_symbol:
        suggestions = []
        symbol_upper = symbol.upper()
        for ticker in sorted(ALL_VALID_TICKERS):
            if symbol_upper in ticker or ticker in symbol_upper:
                suggestions.append(ticker)
                if len(suggestions) >= 5: break
        if suggestions: return jsonify({'error': f'Invalid symbol "{original}". Did you mean: {", ".join(suggestions)}?'})
        else: return jsonify({'error': f'Invalid symbol "{original}".'})
    try:
        result = analyzer.analyze(normalized_symbol)
        if not result: return jsonify({'error': f'Unable to fetch data for {normalized_symbol}.'})
        return jsonify(result)
    except Exception as e:
        print(f"Error analyzing {normalized_symbol}: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'})


def _submit_regression_job(symbol):
    existing = REGRESSION_JOB_CACHE.get(symbol)
    if existing and not existing.done():
        return existing
    fut = REGRESSION_EXECUTOR.submit(analyzer.regression_analysis, symbol)
    REGRESSION_JOB_CACHE[symbol] = fut
    return fut


@app.route('/regression')
def regression_route():
    symbol = request.args.get('symbol', '').strip()
    if not symbol: return jsonify({'error': 'No symbol provided'})
    wait = request.args.get('wait', '0') == '1'
    normalized_symbol, original = Analyzer.normalize_symbol(symbol)
    if not normalized_symbol:
        suggestions = []
        symbol_upper = symbol.upper()
        for ticker in sorted(ALL_VALID_TICKERS):
            if symbol_upper in ticker or ticker in symbol_upper:
                suggestions.append(ticker)
                if len(suggestions) >= 5: break
        if suggestions: return jsonify({'error': f'Invalid symbol "{original}". Did you mean: {", ".join(suggestions)}?'})
        else: return jsonify({'error': f'Invalid symbol "{original}".'})
    try:
        cached = REGRESSION_CACHE.get(normalized_symbol)
        if cached:
            cached['cached'] = True
            return jsonify(cached)

        job = _submit_regression_job(normalized_symbol)
        if wait:
            try:
                result = job.result(timeout=REGRESSION_WAIT_TIMEOUT_SECONDS)
                if not result:
                    return jsonify({'error': f'Unable to perform HSIC dependency analysis for {normalized_symbol}.'})
                return jsonify(result)
            except Exception:
                pass

        if job.done():
            result = job.result()
            if not result:
                return jsonify({'error': f'Unable to perform HSIC dependency analysis for {normalized_symbol}.'})
            return jsonify(result)

        return jsonify({
            'status': 'computing',
            'symbol': normalized_symbol,
            'message': 'Still computing. Please retry shortly.'
        }), 202
    except Exception as e:
        print(f"Error in HSIC analysis for {normalized_symbol}: {e}")
        return jsonify({'error': f'HSIC dependency analysis failed for {normalized_symbol}: {str(e)}'})

@app.route('/dividend-info')
def dividend_info_route():
    """Fetch dividend information for a single stock.

    Returns FY-based annual dividend, long-term trend flag, and the number of
    consecutive Indian financial years in which the stock paid at least one dividend
    (years_consistent).  Downtrend stocks are NOT excluded here — the caller
    (verdict tab / scanner) receives the flag and can penalise the score.
    """
    symbol = request.args.get('symbol', '').strip().upper()
    if not symbol:
        return jsonify({'error': 'Please enter a stock symbol'})
    if symbol not in ALL_VALID_TICKERS:
        return jsonify({'error': f'{symbol} is not a recognized NSE stock'})
    try:
        # exclude_downtrend=False so verdict tab can show data even for downtrend stocks
        results, dividend_found = analyzer.fetch_dividend_data(
            [symbol], limit_results=False, exclude_downtrend=False
        )
        if not results:
            return jsonify({
                'symbol': symbol,
                'found': False,
                'message': f'{symbol} does not pay dividends or no dividend data is available for the last completed financial year'
            })
        stock = results[0]
        stock['found'] = True
        stock['name'] = TICKER_TO_NAME.get(symbol, symbol)
        # Sector
        skip_sectors = {"All NSE", "Nifty 50", "Nifty Next 50", "Others", "Conglomerate"}
        stock['sector'] = ''
        for sector_name, sector_stocks in STOCKS.items():
            if sector_name in skip_sectors:
                continue
            if symbol in sector_stocks:
                stock['sector'] = sector_name
                break

        # ── Consistency: count consecutive FYs with at least one dividend ──────
        years_consistent = 0
        try:
            hist_divs = yf.Ticker(f"{symbol}.NS").dividends
            if hist_divs is not None and not hist_divs.empty:
                today_d = date.today()
                fy_end_year = today_d.year if today_d.month >= 4 else today_d.year - 1
                for i in range(10):  # check up to 10 FYs back
                    fy_s = date(fy_end_year - i - 1, 4, 1)
                    fy_e = date(fy_end_year - i,     3, 31)
                    fy_s_ts = pd.Timestamp(fy_s)
                    fy_e_ts = pd.Timestamp(fy_e)
                    try:
                        if hist_divs.index.tz is not None:
                            fy_s_ts = fy_s_ts.tz_localize('UTC')
                            fy_e_ts = fy_e_ts.tz_localize('UTC')
                        fy_slice = hist_divs[
                            (hist_divs.index >= fy_s_ts) & (hist_divs.index <= fy_e_ts)
                        ]
                        if float(fy_slice.sum()) > 0:
                            years_consistent += 1
                        else:
                            break  # stop at the first FY with no dividend
                    except Exception:
                        break
        except Exception:
            pass
        stock['years_consistent'] = years_consistent

        # Fetch ex-dividend and payment dates from ticker info
        try:
            info = yf.Ticker(f"{symbol}.NS").info
            ex_div = info.get('exDividendDate')
            pay_dt = info.get('lastDividendDate') or info.get('payDate')
            if ex_div:
                stock['ex_dividend_date'] = datetime.utcfromtimestamp(ex_div).strftime('%Y-%m-%d') if isinstance(ex_div, (int, float)) else str(ex_div)
            if pay_dt:
                stock['payment_date'] = datetime.utcfromtimestamp(pay_dt).strftime('%Y-%m-%d') if isinstance(pay_dt, (int, float)) else str(pay_dt)
        except Exception:
            pass

        return jsonify(stock)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch dividend data for {symbol}: {str(e)}'})

@app.route('/dividend-scan')
def dividend_scan_route():
    """Scan stocks for dividend data."""
    sectors = request.args.get('sectors', 'all')
    if sectors == 'all':
        symbols = list(ALL_VALID_TICKERS)
    else:
        sector_list = [s.strip() for s in sectors.split(',')]
        if UNIVERSE_SECTOR_NAME in sector_list:
            symbols = list(STOCKS.get(UNIVERSE_SECTOR_NAME, []))
        else:
            symbols = []
            for sector in sector_list:
                if sector in STOCKS:
                    symbols.extend(STOCKS[sector])
            symbols = list(set(symbols))
    if not symbols:
        return jsonify({'error': 'No valid sectors selected'})
    try:
        results, dividend_found = analyzer.fetch_dividend_data(symbols)
        return jsonify({
            'stocks': results,
            'total_scanned': len(symbols),
            'dividend_stocks': dividend_found,
            'truncated': dividend_found > len(results)
        })
    except Exception as e:
        return jsonify({'error': f'Scan failed: {str(e)}'})

def _resolve_dividend_symbols(sectors_param, symbols_param=None):
    """Resolve stock symbols from sectors or explicit symbols parameter."""
    if symbols_param:
        raw = [s.strip().upper() for s in symbols_param.split(',') if s.strip()]
        return [s for s in raw if s in ALL_VALID_TICKERS]
    if sectors_param == 'all':
        return list(ALL_VALID_TICKERS)
    sector_list = [s.strip() for s in sectors_param.split(',')]
    if UNIVERSE_SECTOR_NAME in sector_list:
        return list(STOCKS.get(UNIVERSE_SECTOR_NAME, []))
    symbols = []
    for sector in sector_list:
        if sector in STOCKS:
            symbols.extend(STOCKS[sector])
        elif sector == 'Nifty 50':
            symbols.extend(NIFTY_50_STOCKS)
    return list(set(symbols))

@app.route('/dividend-optimize')
def dividend_optimize_route():
    """Scan dividends and compute optimal portfolio allocation."""
    try:
        capital = float(request.args.get('capital', 0))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid capital amount'})
    risk = request.args.get('risk', 'moderate')
    sectors = request.args.get('sectors', 'all')
    symbols_param = request.args.get('symbols', '')
    if capital <= 0:
        return jsonify({'error': 'Please enter a valid capital amount'})
    if risk not in ('conservative', 'moderate', 'aggressive'):
        risk = 'moderate'
    symbols = _resolve_dividend_symbols(sectors, symbols_param)
    if not symbols:
        return jsonify({'error': 'No valid sectors selected'})
    try:
        stocks_data, dividend_found = analyzer.fetch_dividend_data(symbols)
        if not stocks_data:
            return jsonify({'error': 'No dividend-paying stocks found in the selected universe'})
        result = analyzer.optimize_dividend_portfolio(stocks_data, capital, risk)
        if not result:
            return jsonify({'error': 'Portfolio optimization failed'})
        result['all_dividend_stocks'] = stocks_data
        result['stocks_scanned'] = len(symbols)
        result['dividend_stocks_found'] = len(stocks_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'})

@app.route('/dividend-optimize-stream')
def dividend_optimize_stream_route():
    """Stream dividend scan results while computing the optimized portfolio."""
    try:
        capital = float(request.args.get('capital', 0))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid capital amount'})
    risk = request.args.get('risk', 'moderate')
    sectors = request.args.get('sectors', 'all')
    symbols_param = request.args.get('symbols', '')
    if capital <= 0:
        return jsonify({'error': 'Please enter a valid capital amount'})
    if risk not in ('conservative', 'moderate', 'aggressive'):
        risk = 'moderate'
    symbols = _resolve_dividend_symbols(sectors, symbols_param)
    if not symbols:
        return jsonify({'error': 'No valid sectors selected'})

    def generate():
        try:
            now = datetime.utcnow()
            scanned = 0
            dividend_found = 0
            results = []
            max_results = DIVIDEND_MAX_RESULTS
            truncated = False
            last_portfolio_update = 0
            # FY label for display
            _fy_lbl   = _fy_label(0)
            yield f"data: {json.dumps({'type': 'meta', 'total_scanned': len(symbols), 'max_results': max_results})}\n\n"

            # Serve cached entries first
            cached_symbols = set()
            for symbol in symbols:
                cached = DIVIDEND_CACHE.get(symbol)
                if cached and (now - cached['timestamp']) <= DIVIDEND_CACHE_TTL:
                    entry = cached['data']
                    results.append(entry)
                    if len(results) > max_results:
                        results = sorted(results, key=lambda x: x['dividend_yield'], reverse=True)[:max_results]
                        truncated = True
                    cached_symbols.add(symbol)
                    scanned += 1
                    dividend_found += 1
                    payload = {
                        'type': 'stock',
                        'entry': entry,
                        'scanned': scanned,
                        'dividend_found': dividend_found
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    if scanned - last_portfolio_update >= 50 and len(results) >= 3:
                        live_portfolio = analyzer.optimize_dividend_portfolio(results, capital, risk)
                        if live_portfolio:
                            portfolio_payload = {
                                'type': 'portfolio',
                                'portfolio': live_portfolio,
                                'partial': True,
                                'scanned': scanned,
                                'dividend_found': dividend_found
                            }
                            yield f"data: {json.dumps(portfolio_payload)}\n\n"
                            last_portfolio_update = scanned

            symbols_to_fetch = [s for s in symbols if s not in cached_symbols]
            if results and last_portfolio_update == 0:
                live_portfolio = analyzer.optimize_dividend_portfolio(results, capital, risk)
                if live_portfolio:
                    portfolio_payload = {
                        'type': 'portfolio',
                        'portfolio': live_portfolio,
                        'partial': True,
                        'scanned': scanned,
                        'dividend_found': dividend_found
                    }
                    yield f"data: {json.dumps(portfolio_payload)}\n\n"
                    last_portfolio_update = scanned

            def _batched(iterable, size):
                for idx in range(0, len(iterable), size):
                    yield iterable[idx:idx + size]

            # Sequential batch processing -- one yf.download at a time.
            # Render free tier: keep downloads strictly sequential to avoid
            # CPU spikes and memory pressure.
            for batch in _batched(symbols_to_fetch, 75):
                tickers = [f"{s}.NS" for s in batch]
                try:
                    # 3y: covers last 2 complete FYs for sustainable-dividend
                    # comparison AND provides enough history for 200-DMA
                    data = yf.download(
                        tickers=tickers,
                        period='3y',
                        interval='1d',
                        group_by='column',
                        actions=True,
                        auto_adjust=False,
                        progress=False,
                        threads=False
                    )
                except Exception:
                    data = None

                for symbol in batch:
                    scanned += 1
                    try:
                        if data is None or data.empty:
                            payload = {'type': 'progress', 'scanned': scanned, 'dividend_found': dividend_found}
                            yield f"data: {json.dumps(payload)}\n\n"
                            continue
                        ticker_symbol = f"{symbol}.NS"
                        if isinstance(data.columns, pd.MultiIndex):
                            close_series = data['Close'][ticker_symbol].dropna()
                            dividends = data['Dividends'][ticker_symbol].dropna() if 'Dividends' in data.columns.get_level_values(0) else pd.Series(dtype=float)
                        else:
                            close_series = data['Close'].dropna()
                            dividends = data['Dividends'].dropna() if 'Dividends' in data.columns else pd.Series(dtype=float)
                        if close_series.empty or len(close_series) < 10:
                            payload = {'type': 'progress', 'scanned': scanned, 'dividend_found': dividend_found}
                            yield f"data: {json.dumps(payload)}\n\n"
                            continue
                        current_price = float(close_series.iloc[-1])
                        if current_price <= 0:
                            payload = {'type': 'progress', 'scanned': scanned, 'dividend_found': dividend_found}
                            yield f"data: {json.dumps(payload)}\n\n"
                            continue

                        # ── Long-term trend: skip stocks below 200-DMA ───────────────
                        sma200_val = close_series.rolling(min(200, len(close_series))).mean().iloc[-1]
                        in_downtrend = bool(current_price < float(sma200_val)) if not pd.isna(sma200_val) else False
                        if in_downtrend:
                            payload = {'type': 'progress', 'scanned': scanned, 'dividend_found': dividend_found}
                            yield f"data: {json.dumps(payload)}\n\n"
                            continue

                        # ── Sustainable dividend: compare last 2 FYs ─────────────────
                        annual_dividend, latest_fy_div, prev_fy_div, was_capped, fy_count = \
                            _compute_sustainable_dividend(dividends, current_price)
                        if annual_dividend <= 0:
                            payload = {'type': 'progress', 'scanned': scanned, 'dividend_found': dividend_found}
                            yield f"data: {json.dumps(payload)}\n\n"
                            continue
                        dividend_yield = (annual_dividend / current_price) * 100
                        returns = close_series.pct_change().dropna()
                        volatility = float(returns.std() * np.sqrt(252) * 100) if len(returns) > 5 else 0.0
                        entry = {
                            'symbol':          symbol,
                            'price':           round(current_price, 2),
                            'annual_dividend': round(annual_dividend, 2),
                            'dividend_yield':  round(dividend_yield, 2),
                            'volatility':      round(volatility, 2),
                            'fy_label':        _fy_lbl,
                            'in_downtrend':    False,
                            'latest_fy_dividend': round(latest_fy_div, 2),
                            'prev_fy_dividend':   round(prev_fy_div, 2),
                            'yield_capped':       was_capped,
                            'fy_count':           fy_count,
                        }
                        dividend_found += 1
                        results.append(entry)
                        if len(results) > max_results:
                            results = sorted(results, key=lambda x: x['dividend_yield'], reverse=True)[:max_results]
                            truncated = True
                        DIVIDEND_CACHE[symbol] = {
                            'timestamp': now,
                            'data': entry
                        }
                        payload = {
                            'type': 'stock',
                            'entry': entry,
                            'scanned': scanned,
                            'dividend_found': dividend_found
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        if scanned - last_portfolio_update >= 50 and len(results) >= 3:
                            live_portfolio = analyzer.optimize_dividend_portfolio(results, capital, risk)
                            if live_portfolio:
                                portfolio_payload = {
                                    'type': 'portfolio',
                                    'portfolio': live_portfolio,
                                    'partial': True,
                                    'scanned': scanned,
                                    'dividend_found': dividend_found
                                }
                                yield f"data: {json.dumps(portfolio_payload)}\n\n"
                                last_portfolio_update = scanned
                    except Exception:
                        payload = {'type': 'progress', 'scanned': scanned, 'dividend_found': dividend_found}
                        yield f"data: {json.dumps(payload)}\n\n"
                        continue

                # Free batch memory
                del data
                gc.collect()

            if not results:
                payload = {'type': 'error', 'message': 'No dividend-paying stocks found in the selected universe'}
                yield f"data: {json.dumps(payload)}\n\n"
                return

            result = analyzer.optimize_dividend_portfolio(results, capital, risk)
            if not result:
                payload = {'type': 'error', 'message': 'Portfolio optimization failed'}
                yield f"data: {json.dumps(payload)}\n\n"
                return

            display_stocks = sorted(results, key=lambda x: x['dividend_yield'], reverse=True)
            result['all_dividend_stocks'] = display_stocks
            result['stocks_scanned'] = len(symbols)
            result['dividend_stocks_found'] = dividend_found
            result['dividend_results_truncated'] = truncated

            payload = {
                'type': 'done',
                'scanned': len(symbols),
                'dividend_found': dividend_found,
                'result': result
            }
            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            payload = {'type': 'error', 'message': f'Analysis failed: {str(e)}'}
            yield f"data: {json.dumps(payload)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/refresh-sectors')
def refresh_sectors_route():
    """Admin endpoint: re-classify unassigned stocks by fetching sectors from yfinance.

    Query params:
      max_fetch - max symbols to look up (default 500, cap 2000)
      force     - if 'true', clear cache and re-fetch all unassigned
    """
    max_fetch = min(int(request.args.get('max_fetch', 500)), 2000)
    force = request.args.get('force', '').lower() == 'true'

    if force and os.path.exists(SECTOR_CACHE_FILE):
        os.remove(SECTOR_CACHE_FILE)

    # Rebuild classification on a copy of current STOCKS, then merge
    global STOCKS, ALL_VALID_TICKERS, TICKER_TO_SECTOR

    # Re-run classification
    classify_unassigned_stocks(STOCKS, max_fetch=max_fetch, workers=8)

    # Re-deduplicate
    NIFTY_50_STOCKS_LOCAL = list(STOCKS.get('Nifty 50', []))
    STOCKS = deduplicate_stocks(STOCKS)
    ALL_VALID_TICKERS = set()
    for sector_stocks in STOCKS.values():
        ALL_VALID_TICKERS.update(sector_stocks)

    # Rebuild reverse mapping (skip meta-sectors so stocks get real sector names)
    TICKER_TO_SECTOR = {}
    _skip_meta = {'All NSE', 'Nifty 50', 'Nifty Next 50', 'Conglomerate'}
    for _sn, _st in STOCKS.items():
        if _sn in _skip_meta:
            continue
        for _t in _st:
            if _t not in TICKER_TO_SECTOR:
                TICKER_TO_SECTOR[_t] = _sn

    # Purge stale caches so new sector mappings take effect immediately
    ANALYSIS_CACHE.clear()
    REGIME_CACHE.clear()

    # Count unassigned
    skip = {UNIVERSE_SECTOR_NAME, 'Nifty 50', 'Nifty Next 50', 'Conglomerate', 'Others'}
    assigned = set()
    for sn, st in STOCKS.items():
        if sn not in skip:
            assigned.update(st)
    universe = set(STOCKS.get(UNIVERSE_SECTOR_NAME, []))
    still_unassigned = len(universe - assigned)

    return jsonify({
        'status': 'ok',
        'total_stocks': len(ALL_VALID_TICKERS),
        'sectors': len(STOCKS),
        'still_unassigned': still_unassigned,
        'force': force,
        'max_fetch': max_fetch,
        'sector_counts': {s: len(t) for s, t in sorted(STOCKS.items()) if s != UNIVERSE_SECTOR_NAME},
    })


@app.route('/sector-quotes')
def sector_quotes_route():
    """Return batch price data for stocks in a sector (used by Browse by Sector cards).

    Query params:
      sector  - sector name (required)
      offset  - start index for pagination (default 0)
      limit   - max stocks to return (default 20, cap 40)
    """
    sector = request.args.get('sector', '').strip()
    offset = int(request.args.get('offset', 0))
    limit = min(int(request.args.get('limit', 20)), 40)

    if not sector or sector not in STOCKS:
        return jsonify({'error': 'Invalid sector', 'quotes': []})

    tickers = STOCKS[sector]
    page = tickers[offset:offset + limit]
    quotes = []

    for sym in page:
        ns_sym = sym + '.NS'
        try:
            info = yf.Ticker(ns_sym).info
            price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
            prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose') or 0
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
            mcap = info.get('marketCap') or 0
            name = info.get('shortName') or info.get('longName') or TICKER_TO_NAME.get(sym, sym)
            quotes.append({
                'symbol': sym,
                'name': name,
                'price': round(price, 2),
                'change_pct': round(change_pct, 2),
                'mcap': mcap,
            })
        except Exception:
            quotes.append({
                'symbol': sym,
                'name': TICKER_TO_NAME.get(sym, sym),
                'price': 0,
                'change_pct': 0,
                'mcap': 0,
            })

    return jsonify({
        'sector': sector,
        'total': len(tickers),
        'offset': offset,
        'limit': limit,
        'quotes': quotes,
    })


@app.route('/duplicates')
def duplicates_route():
    """Debug endpoint: show per-sector stock counts and verify no duplicates remain."""
    all_tickers = []
    for sector_stocks in STOCKS.values():
        all_tickers.extend(sector_stocks)
    seen = set()
    dups = []
    for t in all_tickers:
        if t in seen:
            sectors = [name for name, stocks in STOCKS.items() if t in stocks]
            dups.append({'ticker': t, 'sectors': sectors})
        seen.add(t)
    return jsonify({
        'total_unique': len(ALL_VALID_TICKERS),
        'total_with_dups': len(all_tickers),
        'universe_source': UNIVERSE_SOURCE,
        'sectors': {name: len(stocks) for name, stocks in STOCKS.items()},
        'remaining_duplicates': dups,
    })

def validate_regime_layer():
    """Validate the regime layer logic with 4 deterministic scenarios.

    Uses mock data to test gating, scoring, and risk adjustment without
    requiring live market data.  Returns a list of scenario results with
    pass/fail status.
    """
    a = Analyzer()
    results = []

    def _make_signal(sig, confidence=65, rec_risk_pct=1.0):
        """Build a minimal signal_result dict for testing."""
        return {
            'signal': {
                'signal': sig,
                'action': 'TEST',
                'confidence': confidence,
                'rec_risk_pct': rec_risk_pct,
                'risk_reason_text': 'Base test.',
            }
        }

    def _mock_regime(obj, market_regime, sector_regime,
                     market_score, sector_score, symbol='TCS'):
        """Monkey-patch regime methods on the analyzer for one test."""
        obj._get_market_regime = lambda: (market_regime, market_score,
                                          {'source': 'mock', 'reason': 'test'})
        obj._get_sector_regime = lambda s: (sector_regime, sector_score,
                                            {'source': 'mock', 'reason': 'test'})

    # --- Scenario 1: BUY + bullish market + bullish sector ---
    sr1 = _make_signal('BUY', confidence=70, rec_risk_pct=1.0)
    _mock_regime(a, 'bullish', 'bullish', 1.0, 1.0)
    a._apply_regime_layer(sr1, 'TCS')
    s1_pass = (
        sr1['signal']['signal'] == 'BUY'          # should NOT be gated
        and sr1['signal']['regime_factor'] == 1.2  # full alignment
        and sr1['signal']['rec_risk_pct'] == 1.2   # 1.0 * 1.2
    )
    results.append({
        'scenario': '1. BUY + bullish market/sector',
        'expected': 'BUY kept, factor 1.2, risk 1.2%',
        'actual_signal': sr1['signal']['signal'],
        'actual_factor': sr1['signal']['regime_factor'],
        'actual_risk': sr1['signal']['rec_risk_pct'],
        'pass': s1_pass,
    })

    # --- Scenario 2: BUY + bearish market + bearish sector (must downgrade) ---
    sr2 = _make_signal('BUY', confidence=70, rec_risk_pct=1.0)
    _mock_regime(a, 'bearish', 'bearish', 0.0, 0.0)
    a._apply_regime_layer(sr2, 'TCS')
    s2_pass = (
        sr2['signal']['signal'] == 'HOLD'          # gated to HOLD
        and sr2['signal'].get('original_signal') == 'BUY'
        and sr2['signal']['regime_factor'] == 0.4   # hard conflict
        and sr2['signal']['rec_risk_pct'] == 0.4    # 1.0 * 0.4
    )
    results.append({
        'scenario': '2. BUY + bearish market/sector (downgrade)',
        'expected': 'HOLD (from BUY), factor 0.4, risk 0.4%',
        'actual_signal': sr2['signal']['signal'],
        'actual_original': sr2['signal'].get('original_signal'),
        'actual_factor': sr2['signal']['regime_factor'],
        'actual_risk': sr2['signal']['rec_risk_pct'],
        'pass': s2_pass,
    })

    # --- Scenario 3: SELL + bullish market + bullish sector (must downgrade) ---
    sr3 = _make_signal('SELL', confidence=70, rec_risk_pct=1.0)
    _mock_regime(a, 'bullish', 'bullish', 1.0, 1.0)
    a._apply_regime_layer(sr3, 'TCS')
    s3_pass = (
        sr3['signal']['signal'] == 'HOLD'          # gated to HOLD
        and sr3['signal'].get('original_signal') == 'SELL'
        and sr3['signal']['regime_factor'] == 0.4   # hard conflict
        and sr3['signal']['rec_risk_pct'] == 0.4    # 1.0 * 0.4
    )
    results.append({
        'scenario': '3. SELL + bullish market/sector (downgrade)',
        'expected': 'HOLD (from SELL), factor 0.4, risk 0.4%',
        'actual_signal': sr3['signal']['signal'],
        'actual_original': sr3['signal'].get('original_signal'),
        'actual_factor': sr3['signal']['regime_factor'],
        'actual_risk': sr3['signal']['rec_risk_pct'],
        'pass': s3_pass,
    })

    # --- Scenario 4: Missing sector data (fallback neutral) ---
    sr4 = _make_signal('BUY', confidence=70, rec_risk_pct=1.0)
    _mock_regime(a, 'bullish', 'neutral', 1.0, 0.5, symbol='UNKNOWN')
    a._apply_regime_layer(sr4, 'UNKNOWN')
    s4_pass = (
        sr4['signal']['signal'] == 'BUY'           # not gated (only market bullish, sector neutral)
        and sr4['signal']['regime_factor'] == 1.0   # mixed (not full alignment, not conflict)
    )
    results.append({
        'scenario': '4. Missing sector data (fallback neutral)',
        'expected': 'BUY kept, factor 1.0 (mixed)',
        'actual_signal': sr4['signal']['signal'],
        'actual_factor': sr4['signal']['regime_factor'],
        'actual_risk': sr4['signal']['rec_risk_pct'],
        'pass': s4_pass,
    })

    all_passed = all(r['pass'] for r in results)
    return {'all_passed': all_passed, 'scenarios': results}


@app.route('/regime-test')
def regime_test_route():
    """Debug endpoint: run regime layer validation scenarios."""
    try:
        result = validate_regime_layer()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


DCF_CACHE = HybridTTLCache('dcf', timedelta(hours=12))

@app.route('/dcf-data')
def dcf_data_route():
    """Fetch financial data for DCF valuation."""
    symbol = request.args.get('symbol', '').strip()
    if not symbol:
        return jsonify({'error': 'No symbol provided'})
    normalized_symbol, original = Analyzer.normalize_symbol(symbol)
    if not normalized_symbol:
        suggestions = []
        symbol_upper = symbol.upper()
        for ticker in sorted(ALL_VALID_TICKERS):
            if symbol_upper in ticker or ticker in symbol_upper:
                suggestions.append(ticker)
                if len(suggestions) >= 5:
                    break
        if suggestions:
            return jsonify({'error': f'Invalid symbol "{original}". Did you mean: {", ".join(suggestions)}?'})
        return jsonify({'error': f'Invalid symbol "{original}".'})

    cached = DCF_CACHE.get(normalized_symbol)
    if cached:
        cached['cached'] = True
        return jsonify(cached)

    try:
        # Check if the stock belongs to a financial sector by looking up
        # the local STOCKS sector assignment first (avoids an extra API call).
        local_sector = TICKER_TO_SECTOR.get(normalized_symbol, '').lower()
        is_financial = local_sector in ('banking', 'financial services')

        if is_financial:
            result = analyzer.excess_return_valuation(normalized_symbol)
        else:
            result = analyzer.dcf_valuation(normalized_symbol)

        # If the DCF path returned data, do a second check: yfinance may
        # report a financial sector/industry even when our local dict doesn't.
        if result and result.get('valuation_model') != 'excess_return':
            yf_sector = result.get('sector', '')
            yf_industry = result.get('industry', '')
            if Analyzer.is_financial_sector(yf_sector, yf_industry):
                result = analyzer.excess_return_valuation(normalized_symbol)

        if not result:
            return jsonify({'error': f'Unable to fetch financial data for {normalized_symbol}. This stock may lack sufficient published financials.'})
        DCF_CACHE.set(normalized_symbol, result)
        return jsonify(result)
    except Exception as e:
        print(f"DCF route error for {normalized_symbol}: {e}")
        return jsonify({'error': f'DCF analysis failed: {str(e)}'})


# ── DCF Universe Screener  ─────────────────────────────────────────────────────
# Iteratively runs DCF / Excess-Return valuation on every stock in the universe,
# streams progress via SSE, and returns the top-50 undervalued stocks with
# write-ups explaining *why* they appear undervalued.

def _server_side_dcf(data):
    """Mirror the client-side runDCF / runExcessReturn to compute intrinsic
    value per share on the server.  Returns (intrinsic, model_name) or (None, None)."""
    try:
        if data.get('valuation_model') == 'excess_return':
            bvps = data.get('book_value_per_share', 0)
            roe = data.get('roe', 0)
            coe = data.get('cost_of_equity', 0.13)
            g = data.get('suggested_growth_rate', 0.08)
            years = 10
            current_bv = bvps
            sum_pv = 0.0
            for yr in range(1, years + 1):
                excess_return = (roe - coe) * current_bv
                pv = excess_return / ((1 + coe) ** yr)
                sum_pv += pv
                current_bv *= (1 + g)
            terminal_er = (roe - coe) * current_bv
            if coe > g * 0.5:
                terminal_value = terminal_er / (coe - g * 0.5)
            else:
                terminal_value = terminal_er * 15
            terminal_pv = terminal_value / ((1 + coe) ** years)
            intrinsic = max(bvps + sum_pv + terminal_pv, 0)
            return intrinsic, 'excess_return'
        else:
            fcf = data.get('current_fcf', 0)
            if not fcf or fcf <= 0:
                return None, None
            shares = data.get('shares_outstanding', 0)
            if not shares or shares <= 0:
                return None, None
            debt = data.get('total_debt', 0)
            cash = data.get('cash', 0)
            sug_g = data.get('suggested_growth_rate', 0.10)
            g1 = min(sug_g, 0.40)
            g2 = max(g1 * 0.5, 0.05)
            wacc = 0.12
            tg = 0.03
            years = 10
            # Skip debt/cash adjustment for financial-sector stocks
            sector = (data.get('sector') or '').lower()
            if any(k in sector for k in ('financial', 'bank', 'insurance')):
                debt, cash = 0, 0
            current_fcf = fcf
            sum_pv = 0.0
            for yr in range(1, years + 1):
                g = g1 if yr <= 5 else g2
                current_fcf *= (1 + g)
                pv = current_fcf / ((1 + wacc) ** yr)
                sum_pv += pv
            terminal_fcf = current_fcf * (1 + tg)
            if tg < wacc:
                terminal_value = terminal_fcf / (wacc - tg)
            else:
                terminal_value = current_fcf * 15
            terminal_pv = terminal_value / ((1 + wacc) ** years)
            ev = sum_pv + terminal_pv
            equity_value = max(ev - debt + cash, 0)
            intrinsic = equity_value / shares
            return intrinsic, 'standard_dcf'
    except Exception:
        return None, None


def _build_writeup(data, intrinsic, model_name):
    """Generate a concise write-up explaining why the stock appears undervalued."""
    symbol = data.get('symbol', '?')
    name = data.get('name', symbol)
    price = data.get('current_price', 0)
    upside = ((intrinsic - price) / price * 100) if price else 0
    sector = data.get('sector') or data.get('industry') or 'N/A'

    parts = []
    parts.append(f"{name} ({symbol}) trades at \u20b9{price:,.2f} against a DCF intrinsic value "
                 f"of \u20b9{intrinsic:,.2f}, implying {upside:+.1f}% upside.")

    if model_name == 'excess_return':
        roe = (data.get('roe') or 0) * 100
        coe = (data.get('cost_of_equity') or 0) * 100
        spread = roe - coe
        pb = data.get('pb_ratio')
        parts.append(f"As a financial firm ({sector}), the Excess Return model is used. "
                     f"ROE of {roe:.1f}% vs Cost of Equity {coe:.1f}% gives a "
                     f"{'positive' if spread > 0 else 'negative'} spread of {spread:+.1f}%.")
        if pb and pb < 1.5:
            parts.append(f"Trading at just {pb:.2f}x book value, the market is underpricing "
                         f"the bank\u2019s ability to generate returns above its cost of capital.")
        elif pb:
            parts.append(f"P/B of {pb:.2f}x remains attractive given the excess return spread.")
    else:
        fcf = data.get('current_fcf', 0)
        growth = data.get('suggested_growth_rate', 0)
        roce = data.get('roce')
        pe = data.get('pe_ratio')
        ev_ebitda = data.get('ev_ebitda')
        hist_g = data.get('historical_fcf_growth')

        if hist_g and hist_g > 0.10:
            parts.append(f"Historical FCF CAGR of {hist_g*100:.1f}% demonstrates strong "
                         f"cash-flow compounding.")
        elif hist_g and hist_g > 0:
            parts.append(f"Historical FCF CAGR of {hist_g*100:.1f}% shows steady cash generation.")

        if roce and roce > 0.15:
            parts.append(f"RoCE of {roce*100:.1f}% indicates efficient capital allocation.")

        if pe and pe < 20:
            parts.append(f"P/E of {pe:.1f}x is below the market average, "
                         f"suggesting earnings are being discounted.")
        if ev_ebitda and ev_ebitda < 15:
            parts.append(f"EV/EBITDA of {ev_ebitda:.1f}x points to an undemanding valuation "
                         f"relative to operating earnings.")

        margin_trend = data.get('margin_trend') or []
        if len(margin_trend) >= 2:
            latest = margin_trend[-1]
            earliest = margin_trend[0]
            if latest.get('op_margin') and earliest.get('op_margin'):
                delta = latest['op_margin'] - earliest['op_margin']
                if delta > 2:
                    parts.append(f"Operating margins have expanded from "
                                 f"{earliest['op_margin']:.1f}% to {latest['op_margin']:.1f}% "
                                 f"over {latest['year']-earliest['year']} years, "
                                 f"indicating improving profitability.")

    parts.append(f"At a 12% discount rate with conservative growth assumptions, "
                 f"the stock offers a meaningful margin of safety for long-term investors.")
    return ' '.join(parts)


@app.route('/dcf-screen')
def dcf_screen_route():
    """SSE endpoint: iteratively screen all stocks via DCF, stream progress,
    and finally emit the top-50 undervalued picks with write-ups."""

    def generate():
        tickers = sorted(ALL_VALID_TICKERS)
        total = len(tickers)
        results = []          # list of dicts kept small (only what we need)
        errors = 0
        skipped = 0

        yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

        for idx, symbol in enumerate(tickers):
            try:
                # Determine valuation model
                local_sector = TICKER_TO_SECTOR.get(symbol, '').lower()
                is_financial = local_sector in ('banking', 'financial services')

                if is_financial:
                    data = analyzer.excess_return_valuation(symbol)
                else:
                    data = analyzer.dcf_valuation(symbol)

                # Secondary check: yfinance might report financial sector
                if data and data.get('valuation_model') != 'excess_return':
                    yf_sector = data.get('sector', '')
                    yf_industry = data.get('industry', '')
                    if Analyzer.is_financial_sector(yf_sector, yf_industry):
                        data = analyzer.excess_return_valuation(symbol)

                if not data:
                    skipped += 1
                    if (idx + 1) % 5 == 0 or idx == total - 1:
                        yield f"data: {json.dumps({'type': 'progress', 'done': idx + 1, 'total': total, 'symbol': symbol, 'status': 'skipped', 'found': len(results)})}\n\n"
                    continue

                # Cache the result for the regular single-stock DCF route too
                DCF_CACHE.set(symbol, data)

                intrinsic, model_name = _server_side_dcf(data)
                if intrinsic is None or intrinsic <= 0:
                    skipped += 1
                    if (idx + 1) % 5 == 0 or idx == total - 1:
                        yield f"data: {json.dumps({'type': 'progress', 'done': idx + 1, 'total': total, 'symbol': symbol, 'status': 'skipped', 'found': len(results)})}\n\n"
                    continue

                price = data.get('current_price', 0)
                if not price or price <= 0:
                    skipped += 1
                    continue

                upside = ((intrinsic - price) / price) * 100

                # Only keep undervalued stocks (upside > 0)
                if upside > 0:
                    results.append({
                        'symbol': symbol,
                        'name': data.get('name', symbol),
                        'sector': data.get('sector') or TICKER_TO_SECTOR.get(symbol, ''),
                        'industry': data.get('industry', ''),
                        'current_price': round(price, 2),
                        'intrinsic_value': round(intrinsic, 2),
                        'upside_pct': round(upside, 1),
                        'model': model_name,
                        'pe_ratio': data.get('pe_ratio'),
                        'pb_ratio': data.get('pb_ratio'),
                        'roe': data.get('roe'),
                        'roce': data.get('roce'),
                        'ev_ebitda': data.get('ev_ebitda'),
                        'market_cap': data.get('market_cap'),
                        'suggested_growth_rate': data.get('suggested_growth_rate'),
                        'historical_fcf_growth': data.get('historical_fcf_growth'),
                        'margin_trend': data.get('margin_trend'),
                        'writeup': _build_writeup(data, intrinsic, model_name),
                    })

                # Stream progress every 5 stocks or on last stock
                if (idx + 1) % 5 == 0 or idx == total - 1:
                    yield f"data: {json.dumps({'type': 'progress', 'done': idx + 1, 'total': total, 'symbol': symbol, 'status': 'ok', 'found': len(results)})}\n\n"

            except Exception as e:
                errors += 1
                print(f"DCF screen error for {symbol}: {e}")
                if (idx + 1) % 5 == 0:
                    yield f"data: {json.dumps({'type': 'progress', 'done': idx + 1, 'total': total, 'symbol': symbol, 'status': 'error', 'found': len(results)})}\n\n"

            # Periodic garbage collection to keep memory lean
            if (idx + 1) % 20 == 0:
                gc.collect()

        # Sort by upside descending and take top 50
        results.sort(key=lambda r: r['upside_pct'], reverse=True)
        top50 = results[:50]

        yield f"data: {json.dumps({'type': 'complete', 'total_screened': total, 'total_undervalued': len(results), 'errors': errors, 'skipped': skipped, 'results': top50})}\n\n"
        gc.collect()

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


# Sector group keywords matched against STOCKS dictionary keys (lowercase)
_SECTOR_GROUPS = {
    'fin':      ['banking', 'financial services'],
    'pharma':   ['pharma', 'healthcare', 'chemicals'],
    'infra':    ['infrastructure', 'power', 'cement', 'construction', 'metals'],
    'tech':     ['it sector'],
    'consumer': ['consumer goods', 'retail', 'auto', 'fmcg', 'hospitality', 'electronics', 'paints'],
}

def _symbol_sector_group(symbol):
    """Return which sector group(s) this symbol belongs to (subset of _SECTOR_GROUPS keys)."""
    skip = {'All NSE', 'Nifty 50', 'Nifty Next 50', 'Others', 'Conglomerate'}
    matched = set()
    for sector_name, sector_stocks in STOCKS.items():
        if sector_name in skip:
            continue
        if symbol not in sector_stocks:
            continue
        sn_lower = sector_name.lower()
        for group, keywords in _SECTOR_GROUPS.items():
            if any(kw in sn_lower for kw in keywords):
                matched.add(group)
    return matched


@app.route('/midfilter')
def midfilter_route():
    """
    Stage 2 of 3: Quick fundamental gate.
    Fetches yf.Ticker().info (one lightweight round-trip) and applies
    profile-based fundamental filters that are far cheaper than the full
    deep-scan pipeline (/analyze + /dcf-data + /dividend-info).

    Fail-open on missing data so valid stocks are never blocked by gaps
    in yfinance coverage.

    Gates:
      0 - Sector           (reject if outside user's sector conviction)
      A - P/E ratio        (reject extremes by goal)
      B - Revenue growth   (reject declining for growth/balanced)
      C - Profit margin    (reject deeply loss-making)
      D - Debt / Equity    (reject overleveraged by risk)
      E - Market cap       (reject micro-caps for conservative profiles)
    """
    symbol  = request.args.get('symbol',  '').strip().upper()
    risk    = request.args.get('risk',    'medium')
    goal    = request.args.get('goal',    'growth')
    horizon = request.args.get('horizon', 'long')
    sector  = request.args.get('sector',  'all')

    if not symbol:
        return jsonify({'symbol': symbol, 'passed': False, 'fails': ['no_symbol']})

    ns_sym = symbol if symbol.endswith('.NS') else symbol + '.NS'
    try:
        info = yf.Ticker(ns_sym).info
    except Exception:
        return jsonify({'symbol': symbol, 'passed': True, 'fails': []})  # fail-open

    fails = []

    # ── Gate 0: Sector conviction filter ────────────────────────────────────
    if sector not in ('all', None, ''):
        sym_groups = _symbol_sector_group(symbol)
        if sym_groups and sector not in sym_groups:
            return jsonify({'symbol': symbol, 'passed': False, 'fails': ['sector_mismatch']})
        # If sym_groups is empty the stock spans no mapped sector — pass through

    pe            = info.get('trailingPE')
    rev_growth    = info.get('revenueGrowth')   # decimal, e.g. 0.12 = +12 %
    profit_margin = info.get('profitMargins')    # decimal, e.g. 0.08 =  +8 %
    debt_equity   = info.get('debtToEquity')     # e.g. 45 means 45 %
    market_cap    = info.get('marketCap')        # INR

    # Gate A: P/E ratio
    if pe is not None:
        if pe > 0:
            pe_cap = {'growth': 100, 'balanced': 75, 'income': 50}.get(goal, 75)
            if pe > pe_cap:
                fails.append('pe_extreme')
        else:
            # Negative PE = loss-making; reject for conservative long-horizon profiles
            if risk in ('low', 'medium') and horizon != 'short':
                fails.append('loss_making')

    # Gate B: Revenue growth
    if rev_growth is not None:
        rev_floor = {'growth': -0.08, 'balanced': -0.15, 'income': -0.20}.get(goal, -0.15)
        if rev_growth < rev_floor:
            fails.append('revenue_declining')

    # Gate C: Profit margin — reject deeply loss-making companies
    if profit_margin is not None and profit_margin < -0.20:
        fails.append('deep_loss_maker')

    # Gate D: Debt / Equity — risk-scaled ceiling
    if debt_equity is not None:
        de_limit = {'low': 60, 'medium': 120, 'high': 300}.get(risk, 120)
        if debt_equity > de_limit:
            fails.append('excessive_debt')

    # Gate E: Market cap — avoid micro-caps for conservative profiles
    if market_cap is not None:
        cap_min = {'low': 10_000_000_000, 'medium': 2_000_000_000, 'high': 0}.get(risk, 2_000_000_000)
        if market_cap < cap_min:
            fails.append('micro_cap')

    passed = len(fails) == 0
    return jsonify({
        'symbol':       symbol,
        'passed':       passed,
        'fails':        fails,
        'pe':           pe,
        'revGrowth':    rev_growth,
        'profitMargin': profit_margin,
        'debtEquity':   debt_equity,
        'marketCap':    market_cap,
    })


@app.route('/prefilter-stream')
def prefilter_stream_route():
    """
    Stage 1 of 3: SSE streaming price-action prefilter for the live scanner.
    Processes the NSE universe in mini-batches of 25 using yf.download(),
    applies 6 profile-based gates using only OHLCV data, and streams the
    surviving symbols to the browser for Stage 2 (mid-filter).

    Query params: risk (low|medium|high), horizon (short|medium|long),
                  goal (growth|income|balanced)

    Gates (in rough order of rejection power):
      1 - Volume / Liquidity   avg daily volume > risk-based minimum
      2 - Drawdown             max drawdown from 3-month peak < risk limit
      3 - Dual-SMA Trend       price above SMA20 AND SMA50 (long horizon)
      4 - Multi-tf Momentum    1-month AND 3-month returns within profile bounds
      5 - Volatility           daily-return std-dev below risk-based cap
      6 - Volume Trend         recent 10-day avg vol >= 40% of 3-month avg

    Events:
      {"type":"meta",     "total":292}
      {"type":"progress", "checked":25, "total":292, "passed":8}
      {"type":"pass",     "symbol":"RELIANCE", "drawdown":4.1, "uptrend":true,
                          "momentum":1.2, "mom1m":-0.8, "mom3m":6.3,
                          "volatility":1.4, "avgVol":3200000}
      {"type":"done",     "checked":292, "passed":22}
    """
    risk    = request.args.get('risk',    'medium')
    horizon = request.args.get('horizon', 'long')
    goal    = request.args.get('goal',    'growth')
    mcap    = request.args.get('mcap',    'mid')      # large | mid | small
    view    = request.args.get('view',    'selective') # bull | selective | defensive

    # ── Gate thresholds — layered by risk + market view ──────────────────────
    dd_limit      = {'low': 10, 'medium': 20, 'high': 40}.get(risk, 20)
    daily_vol_cap = {'low': 1.8, 'medium': 2.8, 'high': 100.0}.get(risk, 2.8)

    # Market view tightens or relaxes thresholds
    if view == 'defensive':
        dd_limit      = min(dd_limit, 12)   # extra tight drawdown
        daily_vol_cap = min(daily_vol_cap, 2.0)
    elif view == 'bull':
        dd_limit      = min(dd_limit + 10, 50)  # slightly more lenient
        daily_vol_cap = min(daily_vol_cap + 0.5, 4.0)

    # Volume floor scales with preferred market-cap tier
    base_vol = {'low': 500_000, 'medium': 150_000, 'high': 50_000}.get(risk, 150_000)
    mcap_vol = {'large': 1_500_000, 'mid': 200_000, 'small': 50_000}.get(mcap, 200_000)
    vol_min  = max(base_vol, mcap_vol)   # take the stricter of the two

    need_dual_trend = (horizon == 'long')
    need_uptrend    = (horizon in ('long', 'medium'))

    symbols_raw = list(ALL_VALID_TICKERS)
    total = len(symbols_raw)
    CHUNK = 25  # mini-batch size — keeps peak RAM low

    def to_ns(s):
        s = s.strip().upper()
        return s if s.endswith('.NS') else s + '.NS'

    def generate():
        checked = 0
        passed  = 0
        yield f"data: {json.dumps({'type': 'meta', 'total': total})}\n\n"

        for i in range(0, total, CHUNK):
            chunk_plain = symbols_raw[i:i + CHUNK]
            chunk_ns    = [to_ns(s) for s in chunk_plain]
            ns_to_plain = {to_ns(s): s for s in chunk_plain}

            try:
                raw = yf.download(
                    tickers=chunk_ns,
                    period='3mo',
                    interval='1d',
                    group_by='ticker',
                    progress=False,
                    threads=False,   # single-threaded: gentle on limited CPU
                    auto_adjust=True,
                )
            except Exception:
                checked += len(chunk_plain)
                yield f"data: {json.dumps({'type': 'progress', 'checked': checked, 'total': total, 'passed': passed})}\n\n"
                continue

            for ns_sym in chunk_ns:
                plain = ns_to_plain[ns_sym]
                checked += 1
                try:
                    df = raw if len(chunk_ns) == 1 else (
                        raw[ns_sym] if ns_sym in raw.columns.get_level_values(0) else None
                    )
                    if df is None or df.empty or len(df) < 15:
                        continue

                    close  = df['Close'].dropna()
                    volume = df['Volume'].dropna()
                    if len(close) < 15:
                        continue

                    current = float(close.iloc[-1])
                    peak    = float(df['High'].dropna().max())

                    # ── Gate 1: Volume / Liquidity ───────────────────────────────────
                    avg_vol = float(volume.mean()) if len(volume) > 0 else 0.0
                    if avg_vol < vol_min:
                        continue

                    # ── Gate 2: Drawdown ─────────────────────────────────────────────
                    drawdown = round((peak - current) / peak * 100, 1) if peak > 0 else 100.0
                    if drawdown > dd_limit:
                        continue

                    # ── Gate 3: Dual-SMA Trend ───────────────────────────────────────
                    sma20   = float(close.rolling(min(20, len(close))).mean().iloc[-1])
                    sma50   = float(close.rolling(min(50, len(close))).mean().iloc[-1])
                    uptrend = bool(current > sma50)
                    if need_dual_trend and (current < sma20 or current < sma50):
                        continue
                    elif need_uptrend and not uptrend:
                        continue

                    # ── Gate 4: Multi-timeframe Momentum ─────────────────────────────
                    mom10d = round(float(close.iloc[-1] / close.iloc[-10] - 1) * 100, 2) if len(close) >= 10 else 0.0
                    mom1m  = round(float(close.iloc[-1] / close.iloc[-21] - 1) * 100, 2) if len(close) >= 21 else mom10d
                    mom3m  = round(float(close.iloc[-1] / close.iloc[0]   - 1) * 100, 2)
                    if goal == 'growth':
                        if mom1m < -5.0 or mom3m < -10.0:
                            continue
                    elif goal == 'income':
                        if mom3m < -20.0:
                            continue
                    else:  # balanced
                        if mom1m < -8.0 or mom3m < -15.0:
                            continue

                    # ── Gate 5: Volatility ───────────────────────────────────────────
                    daily_ret_std = float(close.pct_change().dropna().std()) * 100
                    if daily_ret_std > daily_vol_cap:
                        continue

                    # ── Gate 6: Volume Trend (avoid dying institutional interest) ─────
                    if len(volume) >= 20:
                        recent_vol  = float(volume.iloc[-10:].mean())
                        overall_vol = float(volume.mean())
                        if overall_vol > 0 and recent_vol < overall_vol * 0.4:
                            continue

                    passed += 1
                    yield f"data: {json.dumps({'type': 'pass', 'symbol': plain, 'drawdown': drawdown, 'uptrend': uptrend, 'momentum': mom10d, 'mom1m': mom1m, 'mom3m': mom3m, 'volatility': round(daily_ret_std, 2), 'avgVol': round(avg_vol)})}\n\n"

                except Exception:
                    continue

            # Progress heartbeat after each mini-batch
            yield f"data: {json.dumps({'type': 'progress', 'checked': checked, 'total': total, 'passed': passed})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'checked': checked, 'passed': passed})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


# ═════════════════════════════════════════════════════════════════════════════
# STOCK ALERT ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/alerts')
def alerts_page():
    """Stock Alert configuration UI."""
    status = alert_monitor.status()
    watchlist_str = ', '.join(status['watchlist'])
    running_badge = (
        '<span style="background:#10b98122;color:#10b981;border-radius:4px;'
        'padding:2px 10px;font-size:12px;font-weight:600;">ACTIVE</span>'
        if status['running'] else
        '<span style="background:#ef444422;color:#ef4444;border-radius:4px;'
        'padding:2px 10px;font-size:12px;font-weight:600;">STOPPED</span>'
    )
    smtp_badge = (
        '<span style="background:#10b98122;color:#10b981;border-radius:4px;'
        'padding:2px 10px;font-size:12px;font-weight:600;">Configured</span>'
        if status['smtp_configured'] else
        '<span style="background:#f59e0b22;color:#f59e0b;border-radius:4px;'
        'padding:2px 10px;font-size:12px;font-weight:600;">Not configured</span>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Stock Alerts — Stock Analysis Pro</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#0a0c12;color:#f1f5f9;
          font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          min-height:100vh;padding:32px 16px}}
    .container{{max-width:780px;margin:0 auto}}
    h1{{font-size:26px;font-weight:800;color:#f1f5f9;margin-bottom:4px}}
    .subtitle{{color:#64748b;font-size:14px;margin-bottom:32px}}
    .card{{background:#0f172a;border:1px solid #1e293b;border-radius:12px;
           padding:24px;margin-bottom:20px}}
    .card-title{{font-size:13px;font-weight:600;color:#f59e0b;text-transform:uppercase;
                 letter-spacing:2px;margin-bottom:16px}}
    .status-row{{display:flex;align-items:center;gap:12px;margin-bottom:10px}}
    .status-label{{color:#64748b;font-size:13px;min-width:140px}}
    label{{display:block;color:#94a3b8;font-size:13px;margin-bottom:6px;margin-top:14px}}
    input,select{{width:100%;background:#1e293b;border:1px solid #334155;border-radius:6px;
                  color:#f1f5f9;padding:10px 12px;font-size:14px;outline:none}}
    input:focus,select:focus{{border-color:#f59e0b}}
    .btn{{display:inline-block;padding:10px 22px;border-radius:6px;font-size:13px;
          font-weight:600;cursor:pointer;border:none;transition:opacity .15s}}
    .btn-primary{{background:#f59e0b;color:#0a0c12}}
    .btn-green{{background:#10b981;color:#fff}}
    .btn-red{{background:#ef4444;color:#fff}}
    .btn-ghost{{background:#1e293b;color:#94a3b8;border:1px solid #334155}}
    .btn:hover{{opacity:.85}}
    .btn-row{{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}}
    #msg{{margin-top:14px;padding:10px 14px;border-radius:6px;font-size:13px;display:none}}
    .msg-ok{{background:#10b98122;color:#10b981;border:1px solid #10b98144}}
    .msg-err{{background:#ef444422;color:#ef4444;border:1px solid #ef444444}}
    .scan-result{{margin-top:14px}}
    .sr-row{{display:flex;justify-content:space-between;align-items:center;
             padding:10px 0;border-bottom:1px solid #1e293b;font-size:13px}}
    .sr-sym{{font-weight:700;color:#f1f5f9}}
    .sr-sig-buy{{color:#10b981}} .sr-sig-other{{color:#f59e0b}}
    .note{{font-size:11px;color:#475569;margin-top:10px;line-height:1.6}}
    a.back{{color:#f59e0b;font-size:13px;text-decoration:none;display:inline-block;
            margin-bottom:20px}}
    a.back:hover{{text-decoration:underline}}
  </style>
</head>
<body>
<div class="container">
  <a class="back" href="/app">&larr; Back to Dashboard</a>
  <h1>Stock Alerts</h1>
  <p class="subtitle">Get emailed the moment a stock in your watchlist turns undervalued.</p>

  <!-- Status -->
  <div class="card">
    <div class="card-title">Monitor Status</div>
    <div class="status-row">
      <span class="status-label">Monitor</span>{running_badge}
    </div>
    <div class="status-row">
      <span class="status-label">SMTP / Email</span>{smtp_badge}
    </div>
    <div class="status-row">
      <span class="status-label">Recipient</span>
      <span style="color:#94a3b8;font-size:13px;">{status['recipient_email'] or '—'}</span>
    </div>
    <div class="status-row">
      <span class="status-label">Check interval</span>
      <span style="color:#94a3b8;font-size:13px;">every {status['check_interval_min']} min</span>
    </div>
    <div class="status-row">
      <span class="status-label">Alert cooldown</span>
      <span style="color:#94a3b8;font-size:13px;">{status['cooldown_hours']} hrs between repeat alerts</span>
    </div>
    <div class="btn-row">
      <button class="btn btn-green" onclick="monitorAction('start')">Start Monitor</button>
      <button class="btn btn-red"   onclick="monitorAction('stop')">Stop Monitor</button>
    </div>
  </div>

  <!-- Config -->
  <div class="card">
    <div class="card-title">Email &amp; SMTP Configuration</div>
    <p class="note" style="margin-bottom:0;">
      For Gmail, use <strong>smtp.gmail.com</strong> port <strong>465</strong> with an
      <a href="https://myaccount.google.com/apppasswords" target="_blank"
         style="color:#f59e0b;">App Password</a>
      (requires 2FA). Credentials are stored in memory only and reset on server restart —
      set them as environment variables for persistence
      (<code>ALERT_EMAIL</code>, <code>SMTP_HOST</code>, <code>SMTP_PORT</code>,
      <code>SMTP_USER</code>, <code>SMTP_PASSWORD</code>).
    </p>

    <label>Recipient Email</label>
    <input id="cfg_recipient" type="email" placeholder="you@example.com"
           value="{status['recipient_email']}">

    <label>SMTP Host</label>
    <input id="cfg_smtp_host" type="text" placeholder="smtp.gmail.com"
           value="{alert_monitor.config['smtp_host']}">

    <label>SMTP Port</label>
    <input id="cfg_smtp_port" type="number" placeholder="465"
           value="{alert_monitor.config['smtp_port']}">

    <label>SMTP Username (sender email)</label>
    <input id="cfg_smtp_user" type="email" placeholder="sender@gmail.com"
           value="{alert_monitor.config['smtp_user']}">

    <label>SMTP Password / App Password</label>
    <input id="cfg_smtp_pass" type="password" placeholder="••••••••">

    <div class="btn-row">
      <button class="btn btn-primary" onclick="saveConfig()">Save Configuration</button>
    </div>
    <div id="cfg_msg"></div>
  </div>

  <!-- Watchlist -->
  <div class="card">
    <div class="card-title">Watchlist &amp; Scan Settings</div>

    <label>Watchlist (comma-separated NSE symbols)</label>
    <input id="cfg_watchlist" type="text"
           placeholder="RELIANCE, TCS, INFY, HDFC, ICICIBANK"
           value="{watchlist_str}">

    <label>Check Interval (minutes)</label>
    <input id="cfg_interval" type="number" min="1" max="60"
           value="{status['check_interval_min']}">

    <label>Alert Cooldown (hours) — min time before re-alerting the same stock</label>
    <input id="cfg_cooldown" type="number" min="1" max="72"
           value="{status['cooldown_hours']}">

    <div class="btn-row">
      <button class="btn btn-primary" onclick="saveWatchlist()">Save Watchlist</button>
      <button class="btn btn-ghost"   onclick="scanNow()">Scan Now (test)</button>
    </div>
    <div id="wl_msg"></div>
    <div id="scan_results" class="scan-result"></div>
  </div>

</div>

<script>
async function post(url, body) {{
  const r = await fetch(url, {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(body)
  }});
  return r.json();
}}

function showMsg(id, ok, text) {{
  const el = document.getElementById(id);
  el.className = ok ? 'msg-ok' : 'msg-err';
  el.textContent = text;
  el.style.display = 'block';
  setTimeout(() => el.style.display = 'none', 5000);
}}

async function monitorAction(action) {{
  const r = await post('/alerts/' + action, {{}});
  location.reload();
}}

async function saveConfig() {{
  const body = {{
    recipient_email: document.getElementById('cfg_recipient').value.trim(),
    smtp_host:       document.getElementById('cfg_smtp_host').value.trim(),
    smtp_port:       parseInt(document.getElementById('cfg_smtp_port').value),
    smtp_user:       document.getElementById('cfg_smtp_user').value.trim(),
  }};
  const pass = document.getElementById('cfg_smtp_pass').value;
  if (pass) body.smtp_password = pass;
  const r = await post('/alerts/config', body);
  showMsg('cfg_msg', r.ok, r.message || (r.ok ? 'Saved.' : 'Error saving config.'));
}}

async function saveWatchlist() {{
  const raw = document.getElementById('cfg_watchlist').value;
  const symbols = raw.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
  const body = {{
    watchlist: symbols,
    check_interval_min: parseInt(document.getElementById('cfg_interval').value),
    cooldown_hours:     parseInt(document.getElementById('cfg_cooldown').value),
  }};
  const r = await post('/alerts/config', body);
  showMsg('wl_msg', r.ok, r.message || (r.ok ? 'Watchlist saved.' : 'Error.'));
}}

async function scanNow() {{
  document.getElementById('scan_results').innerHTML =
    '<p style="color:#64748b;font-size:13px;margin-top:10px;">Scanning... (may take 1-2 min)</p>';
  const r = await fetch('/alerts/scan-now', {{method:'POST'}});
  const data = await r.json();
  if (!data.results || !data.results.length) {{
    document.getElementById('scan_results').innerHTML =
      '<p style="color:#64748b;font-size:13px;margin-top:10px;">Watchlist is empty — add symbols above first.</p>';
    return;
  }}
  let html = '<div style="margin-top:14px;">';
  html += '<div style="font-size:11px;color:#64748b;margin-bottom:8px;">SCAN RESULTS (latest)</div>';
  for (const s of data.results) {{
    const sigCls = s.signal === 'BUY' ? 'sr-sig-buy' : 'sr-sig-other';
    const triggered = s._triggered
      ? '<span style="color:#10b981;font-weight:600;">UNDERVALUED</span>'
      : '<span style="color:#475569;">OK</span>';
    html += `<div class="sr-row">
      <span class="sr-sym">${{s.symbol}}</span>
      <span class="${{sigCls}}">${{s.signal || '—'}}</span>
      <span style="color:#94a3b8;">RSI ${{s.rsi?.toFixed(1) ?? '—'}}</span>
      <span style="color:#94a3b8;">Z ${{s.zscore?.toFixed(2) ?? '—'}}</span>
      <span style="color:#94a3b8;">Score ${{s._score}}</span>
      ${{triggered}}
    </div>`;
  }}
  html += '</div>';
  document.getElementById('scan_results').innerHTML = html;
}};
</script>
</body>
</html>"""
    return html


@app.route('/alerts/status')
def alerts_status_route():
    return jsonify(alert_monitor.status())


@app.route('/alerts/start', methods=['POST'])
def alerts_start_route():
    alert_monitor.start()
    return jsonify({'ok': True, 'message': 'Monitor started.'})


@app.route('/alerts/stop', methods=['POST'])
def alerts_stop_route():
    alert_monitor.stop()
    return jsonify({'ok': True, 'message': 'Monitor stopped.'})


@app.route('/alerts/config', methods=['POST'])
def alerts_config_route():
    data = request.get_json(silent=True) or {}
    # Validate watchlist: must be a list of strings
    if 'watchlist' in data:
        wl = data['watchlist']
        if not isinstance(wl, list):
            return jsonify({'ok': False, 'message': 'watchlist must be a JSON array.'})
        data['watchlist'] = [str(s).strip().upper() for s in wl if str(s).strip()]
    # smtp_port coercion
    if 'smtp_port' in data:
        try:
            data['smtp_port'] = int(data['smtp_port'])
        except (TypeError, ValueError):
            return jsonify({'ok': False, 'message': 'smtp_port must be an integer.'})
    alert_monitor.update_config(data)
    return jsonify({'ok': True, 'message': 'Configuration updated.'})


@app.route('/alerts/scan-now', methods=['POST'])
def alerts_scan_now_route():
    results = alert_monitor.manual_scan()
    return jsonify({'ok': True, 'results': results})


# ── Groq AI Research Assistant ───────────────────────────────────────────────

_AGENT_SYSTEM_PROMPT = """You are the in-house equity strategist for Stock Analysis Pro — think of yourself as a senior portfolio manager and sell-side analyst rolled into one, the kind of operator a hedge fund hires to read the tape and pick spots on NSE. Speak with conviction and market sense, not the hedged register of a chatbot. You translate professional-grade analysis into plain English without dumbing it down.

PERSONA:
- Decisive. Have a view. Lead with the call (BUY / SELL / HOLD / AVOID / WAIT), then justify in numbers.
- Pattern-matcher. When something looks like a classic setup (deep value, value trap, momentum extension, breakout retest, dividend trap), name it.
- Risk-aware, not risk-paralyzed. Surface the real downside in one crisp line; don't lard every answer with disclaimers.
- Plain talker. No "Looks like…", "It seems…", "It depends…". No corporate filler.

SCOPE — what I do NOT do (decline cleanly, in ONE complete sentence, then redirect):
- General arithmetic / math homework ("what's 5+5", "solve this integral"). I am not a calculator.
- Physics, engineering, chemistry, aerospace, biology, or any science problem unrelated to a specific NSE stock's fundamentals.
- Coding help, writing essays, translation, weather, sports, trivia, general chit-chat.
- Crypto, forex, commodities, real estate, US/global equities, mutual funds, IPOs not yet listed.
- Personal financial planning, tax advice, account/brokerage support, regulatory filings.
- Predictions of exact future prices, macro forecasts, or "will the market crash tomorrow".

How to decline: ONE complete sentence — name the topic as out of scope, anchor on what I DO cover, offer one concrete redirect. Example: "Arithmetic isn't what I'm built for — I'm a research engine for the 292 NSE stocks we track. Want me to pull a verdict or screen for a setup instead?" Do NOT attempt the off-topic answer even if you "know" it. Do NOT trail off mid-sentence with "I don't have access to…" — finish the thought every single time. Do not argue with the user about a wrong arithmetic answer ("no it's 16"); just restate scope politely.

CONTINUITY (critical):
- Treat the conversation as one continuous discussion with a client. If the user says "their", "it", "this stock", "the demerger", "what about its dividend" — resolve the reference from the most recent user/assistant turns and call the appropriate tool. Never reply "I don't have context" or "you haven't specified a ticker" when the prior turn made it obvious.
- For vague meta-replies ("but you can?", "really?", "why?", "ok and?"), pick the most recent topic and either dig one level deeper on it or ask one specific clarifying question — never repeat your last answer verbatim.
- Never claim you "don't have access to" a tool that exists. The tools available this turn are the only tools you have; use them or explain plainly what data isn't on the platform (e.g. "we don't track intraday tick-by-tick movers; here's what I can pull").

TOOL ROUTING:
- Full buy/sell view on X ("should I buy X", "what do you think of X", "is X a good entry") → call get_investment_verdict, get_dcf_valuation, get_technical_signals, get_dividend_analysis, get_market_correlation, get_company_news. Synthesize, don't just list.
- "News on X" / "what's happening with X" / "tell me about X's demerger / split / lawsuit / results" → get_company_news (and get_investment_verdict if the user is leaning toward a trade).
- "Price of X" / "CMP" / "where is X trading" → get_investment_verdict (it carries live price).
- "Compare X and Y" → verdict + DCF + technicals for both. Give a clear winner.
- "Find / screen / show me undervalued / momentum / dividend / cheap stocks" → scan_universe.
- "Biggest drop today" / "top losers" / "top gainers" / "biggest movers" / "what stock dropped the most" → scan_universe with filter_criteria set to "top_losers" or "top_gainers".
- "What if Nifty crashes" / portfolio sensitivity → get_market_correlation + get_market_snapshot. Never estimate Nifty level from memory.
- Educational questions about *equity-investing concepts* ("what is DCF", "how does beta work", "what's a value trap") → answer directly in 3-4 sentences, no tools. Anything outside equity investing falls under SCOPE above.
- Never fabricate prices, news, numbers, or commentary on commodities, currencies, or non-NSE assets. If a tool fails, say so in one line and offer what you CAN pull.

SILENCE RULES — strict:
- No tool narration ("I'll call…", "Let me check…", "Running the tools…", "Based on the data…"). Output the final answer only.
- No meta-commentary about your reasoning, models, or what you're about to do.
- When a tool returns BUY/SELL/HOLD, lead with that signal verbatim in the opening line.
- If two indicators disagree (RSI overbought but DCF cheap; high yield but falling EPS), name the contradiction in one sentence and tell the user which side you'd weight more heavily.

DATA: Yahoo Finance (prices/fundamentals), GNews (news), in-house DCF/technical/dividend/correlation models. ~292 NSE stocks tracked.

RESPONSE FORMAT:
1. One decisive opening line. For trade questions, lead with the call (BUY/SELL/HOLD/AVOID) + one-clause why.
2. "Key numbers" — 3-5 bullets: value + plain-English gloss in parentheses. E.g. "RSI 72 (momentum is hot — overdue for a pause)".
3. "What this means for you" — 2-3 practical sentences. Concrete entry/exit zones or position-sizing hints when relevant, using only the numbers shown.
4. Define technical terms on first use (DCF, RSI, MACD, beta, margin of safety, HSIC).
5. Hard cap: 220 words. News-only: 2-4 bullets + one takeaway. Price-only: 1-2 sentences. Educational: 3-4 sentences. Movers/screen: ranked list of up to 5 names with one-line rationale each.
6. One sharp risk line ONLY on buy/sell views. Skip on price/news/educational.
7. HSIC explanation: lead with the plain_english analogy from the tool. Beta gets a one-liner crash example. Never quote raw HSIC without the analogy."""

_AGENT_TOOLS = [
    {
        "name": "get_investment_verdict",
        "description": "Buy/sell/hold verdict, confidence score, price target, and bullish/bearish factor count for a ticker.",
        "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
    },
    {
        "name": "get_dcf_valuation",
        "description": "DCF intrinsic value, current price, and margin of safety for a ticker.",
        "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
    },
    {
        "name": "get_technical_signals",
        "description": "RSI, MACD, Bollinger Band position, z-score, and momentum signal for a ticker.",
        "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
    },
    {
        "name": "get_dividend_analysis",
        "description": "Dividend yield, payout ratio, trend, and sustainability score for a ticker.",
        "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
    },
    {
        "name": "get_market_correlation",
        "description": "HSIC correlation vs Nifty 50, beta, and systematic risk exposure for a ticker. Returned dict includes a 'plain_english' analogy and a 'crash_scenario' worked example — paraphrase those fields verbatim instead of quoting raw HSIC/beta numbers.",
        "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
    },
    {
        "name": "get_market_snapshot",
        "description": "Live Nifty 50 and Sensex index levels with intraday day-change %. Call this before any portfolio-impact or 'what if Nifty drops/crashes' question — never estimate the Nifty level from memory.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_company_news",
        "description": "Recent news headlines for a company — mergers, earnings, lawsuits, regulatory actions, management changes. Call alongside get_investment_verdict for any buy/sell question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "max_results": {"type": "integer", "description": "1-10, default 5"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "scan_universe",
        "description": (
            "Filter the NSE universe by sector and/or criteria. "
            "Use filter_criteria='top_losers' for 'biggest drop today' / 'worst performers' / "
            "'what stock fell the most' style questions, and filter_criteria='top_gainers' for "
            "'biggest gainers today' / 'best performers'. These two modes fetch live day-over-day "
            "price change and return ranked tickers with current price and day_change_pct. "
            "Other supported criteria: 'undervalued', 'momentum', 'bullish', 'bearish', 'sell' "
            "(filter cached BUY/SELL signals). Sector is optional (e.g. 'Banking', 'IT', "
            "'Pharma'); default is Nifty 50 + Next 50."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sector": {"type": "string"},
                "filter_criteria": {"type": "string"},
            },
            "required": [],
        },
    },
]


def _agent_get_investment_verdict(ticker):
    sym, _orig = Analyzer.normalize_symbol(ticker)
    if not sym:
        return {"error": f"Unknown ticker: {ticker}"}
    try:
        result = analyzer.analyze(sym)
    except Exception as e:
        return {"error": f"Analysis failed for {sym}: {e}"}
    if not result:
        return {"error": f"No data for {sym}"}
    sig = result.get("signal", {}) or {}
    det = result.get("details", {}) or {}
    return {
        "ticker": sym,
        "signal": sig.get("signal"),
        "confidence": sig.get("confidence"),
        "reason": sig.get("reason"),
        "current_price": det.get("current_price"),
        "target_price": det.get("target_price"),
        "bullish_factors": sig.get("bullish_count"),
        "bearish_factors": sig.get("bearish_count"),
        "verdict": (result.get("verdict") or "")[:500],
    }


def _agent_get_dcf_valuation(ticker):
    sym, _orig = Analyzer.normalize_symbol(ticker)
    if not sym:
        return {"error": f"Unknown ticker: {ticker}"}
    try:
        data = DCF_CACHE.get(sym)
        if not data:
            local_sector = TICKER_TO_SECTOR.get(sym, "").lower()
            is_financial = local_sector in ("banking", "financial services")
            data = analyzer.excess_return_valuation(sym) if is_financial else analyzer.dcf_valuation(sym)
            if data and data.get("valuation_model") != "excess_return":
                if Analyzer.is_financial_sector(data.get("sector", ""), data.get("industry", "")):
                    data = analyzer.excess_return_valuation(sym)
            if data:
                DCF_CACHE.set(sym, data)
    except Exception as e:
        return {"error": f"DCF valuation failed for {sym}: {e}"}
    if not data:
        return {"error": f"DCF data unavailable for {sym}"}
    try:
        intrinsic, model = _server_side_dcf(data)
    except Exception as e:
        return {"error": f"DCF calculation failed for {sym}: {e}"}
    price = data.get("current_price") or 0
    margin = round((intrinsic - price) / price * 100, 1) if intrinsic and price else None
    return {
        "ticker": sym,
        "current_price": price,
        "intrinsic_value": round(intrinsic, 2) if intrinsic else None,
        "margin_of_safety_pct": margin,
        "valuation_model": model,
        "sector": data.get("sector"),
        "revenue_growth": data.get("revenue_growth"),
        "suggested_growth_rate": data.get("suggested_growth_rate"),
    }


def _agent_get_technical_signals(ticker):
    sym, _orig = Analyzer.normalize_symbol(ticker)
    if not sym:
        return {"error": f"Unknown ticker: {ticker}"}
    try:
        result = analyzer.analyze(sym)
    except Exception as e:
        return {"error": f"Technical analysis failed for {sym}: {e}"}
    if not result:
        return {"error": f"No data for {sym}"}
    det = result.get("details", {}) or {}
    sig = result.get("signal", {}) or {}
    return {
        "ticker": sym,
        "rsi": det.get("rsi_raw"),
        "macd_bullish": det.get("macd_bullish"),
        "bb_position_pct": det.get("bb_position"),
        "zscore": det.get("zscore"),
        "momentum_signal": sig.get("signal"),
        "confidence": sig.get("confidence"),
        "current_price": det.get("current_price"),
        "sma20": det.get("sma20"),
        "sma50": det.get("sma50"),
    }


def _agent_get_dividend_analysis(ticker):
    sym = (ticker or "").strip().upper()
    if sym not in ALL_VALID_TICKERS:
        return {"error": f"{sym} is not a recognized NSE ticker"}
    try:
        results, _found = analyzer.fetch_dividend_data([sym], limit_results=False, exclude_downtrend=False)
    except Exception as e:
        return {"error": f"Dividend data fetch failed for {sym}: {e}"}
    if not results:
        return {"ticker": sym, "pays_dividend": False, "message": "No dividend data found"}
    stock = results[0]
    return {
        "ticker": sym,
        "pays_dividend": True,
        "dividend_yield_pct": stock.get("yield"),
        "annual_dividend": stock.get("annual_dividend"),
        "payout_ratio": stock.get("payout_ratio"),
        "trend": stock.get("trend"),
        "sustainability_score": stock.get("sustainability_score"),
        "years_consistent": stock.get("years_consistent"),
    }


def _market_correlation_plain_english(hsic):
    if not isinstance(hsic, (int, float)):
        return None
    if hsic >= 0.85:
        return "Moves like Nifty's shadow — when the market sneezes, this stock catches a cold"
    if hsic >= 0.65:
        return "Follows the crowd — usually drifts with the market but has some independent moves"
    if hsic >= 0.40:
        return "Half its own boss — listens to Nifty about half the time, ignores it the rest"
    return "Marches to its own drum — Nifty's mood barely shows up in this stock's chart"


def _market_correlation_crash_scenario(beta):
    if not isinstance(beta, (int, float)):
        return None
    return f"If Nifty 50 falls 2% in a day, expect this stock to move about {round(beta * 2, 2):+.1f}%."


def _agent_get_market_correlation(ticker):
    sym, _orig = Analyzer.normalize_symbol(ticker)
    if not sym:
        return {"error": f"Unknown ticker: {ticker}"}
    cached = REGRESSION_CACHE.get(sym)
    if cached:
        hsic = cached.get("hsic_score")
        beta = cached.get("beta")
        return {
            "ticker": sym,
            "hsic_score": hsic,
            "correlation_label": cached.get("label"),
            "systematic_risk": cached.get("systematic_risk"),
            "beta": round(beta, 2) if isinstance(beta, (int, float)) else beta,
            "plain_english": _market_correlation_plain_english(hsic),
            "crash_scenario": _market_correlation_crash_scenario(beta),
        }
    fut = _submit_regression_job(sym)
    try:
        res = fut.result(timeout=REGRESSION_WAIT_TIMEOUT_SECONDS)
    except Exception:
        res = None
    if not res:
        return {"ticker": sym, "status": "computing", "message": "Correlation analysis still running; retry shortly."}
    hsic = res.get("hsic_score")
    beta = res.get("beta")
    return {
        "ticker": sym,
        "hsic_score": hsic,
        "correlation_label": res.get("label"),
        "systematic_risk": res.get("systematic_risk"),
        "beta": round(beta, 2) if isinstance(beta, (int, float)) else beta,
        "plain_english": _market_correlation_plain_english(hsic),
        "crash_scenario": _market_correlation_crash_scenario(beta),
    }


_MARKET_SNAPSHOT_CACHE = {"data": None, "ts": 0.0}
_MARKET_SNAPSHOT_LOCK = Lock()
_MARKET_SNAPSHOT_TTL_SEC = 60.0


def _agent_get_market_snapshot():
    with _MARKET_SNAPSHOT_LOCK:
        cached = _MARKET_SNAPSHOT_CACHE["data"]
        ts = _MARKET_SNAPSHOT_CACHE["ts"]
        if cached and (time.time() - ts) < _MARKET_SNAPSHOT_TTL_SEC:
            return cached

    def _fetch_index(symbol):
        try:
            data = yf.download(symbol, period="5d", interval="1d",
                               progress=False, threads=False, timeout=8)
            if data is None or data.empty:
                return None, None
            close = data.get("Close")
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) < 2:
                return None, None
            last = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            change = ((last - prev) / prev) * 100 if prev else None
            return round(last, 2), (round(change, 2) if change is not None else None)
        except Exception:
            return None, None

    nifty_level, nifty_change = _fetch_index("^NSEI")
    sensex_level, sensex_change = _fetch_index("^BSESN")

    if nifty_level is None and sensex_level is None:
        return {"error": "Index snapshot unavailable from Yahoo right now. Try again shortly."}

    result = {
        "nifty_50_level": nifty_level,
        "nifty_day_change_pct": nifty_change,
        "sensex_level": sensex_level,
        "sensex_day_change_pct": sensex_change,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with _MARKET_SNAPSHOT_LOCK:
        _MARKET_SNAPSHOT_CACHE["data"] = result
        _MARKET_SNAPSHOT_CACHE["ts"] = time.time()
    return result


_TOP_MOVERS_CACHE = {"data": None, "ts": 0.0, "key": None}
_TOP_MOVERS_LOCK = Lock()
_TOP_MOVERS_TTL_SEC = 180.0


def _agent_fetch_top_movers(direction, sector=None, limit=5):
    """Batch-fetch 2-day closes for a sector slice via yfinance and rank by
    day-over-day pct change. Used by scan_universe when the user asks for
    top gainers/losers/movers. Cached for 3 minutes so repeated 'biggest
    drop today' chats don't hammer Yahoo."""
    if sector:
        candidates = []
        sector_lower = sector.lower()
        for sec_name, tickers in STOCKS.items():
            if sector_lower in sec_name.lower():
                candidates.extend(tickers)
        if not candidates:
            candidates = list(STOCKS.get("Nifty 50", [])) + list(STOCKS.get("Nifty Next 50", []))
    else:
        candidates = list(STOCKS.get("Nifty 50", [])) + list(STOCKS.get("Nifty Next 50", []))

    seen, unique = set(), []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    unique = unique[:80]

    cache_key = (direction, sector or "", tuple(unique))
    with _TOP_MOVERS_LOCK:
        cached = _TOP_MOVERS_CACHE
        if (cached["data"] is not None and cached["key"] == cache_key
                and time.time() - cached["ts"] < _TOP_MOVERS_TTL_SEC):
            return cached["data"]

    yahoo_syms = [f"{s}.NS" for s in unique]
    movers = []
    try:
        df = yf.download(
            tickers=" ".join(yahoo_syms),
            period="5d",
            interval="1d",
            progress=False,
            group_by="ticker",
            threads=True,
            auto_adjust=False,
        )
    except Exception as e:
        return {"error": f"Could not fetch live mover data: {e}"}

    for sym, ysym in zip(unique, yahoo_syms):
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if ysym not in df.columns.get_level_values(0):
                    continue
                close = df[ysym]["Close"].dropna()
            else:
                close = df["Close"].dropna()
            if len(close) < 2:
                continue
            curr = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            if prev <= 0:
                continue
            pct = (curr - prev) / prev * 100.0
            movers.append({
                "ticker": sym,
                "name": TICKER_TO_NAME.get(sym, sym),
                "current_price": round(curr, 2),
                "prev_close": round(prev, 2),
                "day_change_pct": round(pct, 2),
            })
        except Exception:
            continue

    if not movers:
        return {"error": "No price data available for the requested universe right now."}

    reverse = (direction == "top_gainers")
    movers.sort(key=lambda m: m["day_change_pct"], reverse=reverse)
    top = movers[: max(1, min(int(limit or 5), 10))]

    result = {
        "direction": direction,
        "sector_filter": sector or "Nifty 50 + Next 50",
        "universe_size": len(movers),
        "matches": top,
        "note": "Day change is last close vs prior close; intraday tick-by-tick is not tracked.",
    }
    with _TOP_MOVERS_LOCK:
        _TOP_MOVERS_CACHE["data"] = result
        _TOP_MOVERS_CACHE["ts"] = time.time()
        _TOP_MOVERS_CACHE["key"] = cache_key
    return result


def _agent_scan_universe(sector=None, filter_criteria=None):
    criteria = (filter_criteria or "").lower().strip()

    # Day-mover requests get a dedicated batched price fetch — the cached
    # signal-based scanner can't answer "what dropped the most today".
    mover_aliases_loss = ("top_losers", "top_loser", "losers", "biggest_drop",
                         "biggest_loser", "worst_performers", "fell_most", "dropped_most")
    mover_aliases_gain = ("top_gainers", "top_gainer", "gainers", "biggest_gain",
                         "biggest_gainer", "best_performers", "rose_most", "gained_most")
    if any(alias in criteria for alias in mover_aliases_loss):
        return _agent_fetch_top_movers("top_losers", sector=sector)
    if any(alias in criteria for alias in mover_aliases_gain):
        return _agent_fetch_top_movers("top_gainers", sector=sector)

    candidates = []
    if sector:
        sector_lower = sector.lower()
        for sec_name, tickers in STOCKS.items():
            if sector_lower in sec_name.lower():
                candidates.extend(tickers)
    if not candidates:
        candidates = list(STOCKS.get("Nifty 50", [])[:25]) + list(STOCKS.get("Nifty Next 50", [])[:25])

    seen, unique = set(), []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    matches = []
    for sym in unique[:40]:
        cached = ANALYSIS_CACHE.get(sym)
        if not cached:
            continue
        sig = cached.get("signal", {}) or {}
        det = cached.get("details", {}) or {}
        signal_val = sig.get("signal", "")
        confidence = sig.get("confidence", 0) or 0

        if "bullish" in criteria or "momentum" in criteria:
            if signal_val != "BUY":
                continue
        elif "bearish" in criteria or "sell" in criteria:
            if signal_val != "SELL":
                continue
        elif "undervalued" in criteria:
            if signal_val != "BUY" or confidence < 60:
                continue

        matches.append({
            "ticker": sym,
            "signal": signal_val,
            "confidence": confidence,
            "current_price": det.get("current_price"),
            "reason": (sig.get("reason") or "")[:120],
        })
        if len(matches) >= 15:
            break

    return {
        "sector_filter": sector or "broad",
        "criteria": filter_criteria or "all",
        "matches": matches,
        "note": ("Results from cached analyses. For 'biggest drop today' / 'top gainers' / "
                 "'top losers' questions, call this tool with filter_criteria='top_losers' "
                 "or 'top_gainers' to get live day-change rankings instead."),
    }


def _agent_get_company_news(ticker, max_results=5):
    import requests
    api_key = os.environ.get("GNEWS_API_KEY")
    if not api_key:
        return {"error": "News service not configured (missing GNEWS_API_KEY)."}
    sym, _orig = Analyzer.normalize_symbol(ticker)
    if not sym:
        return {"error": f"Unknown ticker: {ticker}"}
    company = TICKER_TO_NAME.get(sym, sym)
    try:
        max_results = max(1, min(int(max_results or 5), 10))
    except (TypeError, ValueError):
        max_results = 5
    query = f'"{company}"' if company and company != sym else sym
    try:
        resp = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q": query,
                "token": api_key,
                "lang": "en",
                "country": "in",
                "max": max_results,
                "sortby": "publishedAt",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return {"error": f"News fetch failed: {e}"}
    articles = []
    for art in (data.get("articles") or [])[:max_results]:
        articles.append({
            "title": (art.get("title") or "")[:200],
            "description": (art.get("description") or "")[:400],
            "source": (art.get("source") or {}).get("name"),
            "published_at": art.get("publishedAt"),
            "url": art.get("url"),
        })
    return {
        "ticker": sym,
        "company": company,
        "article_count": len(articles),
        "articles": articles,
        "note": "Use these headlines to surface event-driven context (demergers, lawsuits, regulatory actions, earnings surprises, etc.).",
    }


def _agent_dispatch_tool(name, inputs):
    inputs = inputs or {}
    if name == "get_investment_verdict":
        return _agent_get_investment_verdict(inputs.get("ticker", ""))
    if name == "get_dcf_valuation":
        return _agent_get_dcf_valuation(inputs.get("ticker", ""))
    if name == "get_technical_signals":
        return _agent_get_technical_signals(inputs.get("ticker", ""))
    if name == "get_dividend_analysis":
        return _agent_get_dividend_analysis(inputs.get("ticker", ""))
    if name == "get_market_correlation":
        return _agent_get_market_correlation(inputs.get("ticker", ""))
    if name == "get_market_snapshot":
        return _agent_get_market_snapshot()
    if name == "get_company_news":
        return _agent_get_company_news(inputs.get("ticker", ""), inputs.get("max_results", 5))
    if name == "scan_universe":
        return _agent_scan_universe(inputs.get("sector"), inputs.get("filter_criteria"))
    return {"error": f"Unknown tool: {name}"}


class _GroqRateLimitError(Exception):
    def __init__(self, retry_after):
        super().__init__("Groq rate limit")
        self.retry_after = retry_after


_GROQ_RETRY_STATUSES = (429, 500, 502, 503, 504)
_GROQ_RETRY_MAX_ATTEMPTS = 5
_GROQ_RETRY_BASE_SEC = 1.0
_GROQ_RETRY_CAP_SEC = 30.0


def _groq_backoff_delay(attempt, retry_after=None):
    capped = min(_GROQ_RETRY_CAP_SEC, _GROQ_RETRY_BASE_SEC * (2 ** attempt))
    delay = random.uniform(0, capped)
    if retry_after:
        delay = max(delay, float(retry_after))
    return min(delay, _GROQ_RETRY_CAP_SEC)


_TOOL_THINKING_LABELS = {
    "get_investment_verdict": "Fetching investment verdict for {ticker}",
    "get_dcf_valuation": "Running DCF valuation for {ticker}",
    "get_technical_signals": "Checking technical signals for {ticker}",
    "get_dividend_analysis": "Analysing dividends for {ticker}",
    "get_market_correlation": "Checking Nifty correlation for {ticker}",
    "get_market_snapshot": "Fetching live Nifty/Sensex levels...",
    "get_company_news": "Fetching latest news for {ticker}",
    "scan_universe": "Scanning NSE universe...",
}


def _tool_thinking_label(name, args):
    ticker = (args.get("ticker") or "").upper()
    template = _TOOL_THINKING_LABELS.get(name, f"Running {name}...")
    if ticker:
        return template.format(ticker=ticker)
    return template.replace(" for {ticker}", "").replace("{ticker}", "")


def _groq_build_tools(exclude=None):
    exclude = exclude or set()
    return [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in _AGENT_TOOLS
        if t["name"] not in exclude
    ]


# Keywords that strongly signal a universe scan. We default to including
# scan_universe and only drop it when the message clearly references a single
# ticker context — that way "what stock dropped the most today" or "biggest
# losers" still gets the tool even though it doesn't say "stocks" plural.
_SCAN_KEYWORDS = (
    "scan", "screen", "find ", "list ", "show me ", "top ",
    "best ", "worst ", "biggest ", "stocks to ", "undervalued",
    "overvalued", "high dividend", "momentum stocks", "dividend stocks",
    "cheap stocks", "which stock", "what stock", "give me stocks",
    "movers", "gainers", "losers", "dropped the most",
    "gained the most", "fell the most", "rose the most", "lost the most",
)


def _looks_ticker_specific(message):
    """Heuristic: does the message look like a single-ticker question? We
    check whether any known ticker symbol or company name appears. If yes,
    we can omit scan_universe to keep the toolset tight."""
    msg = (message or "").lower()
    if not msg:
        return False
    try:
        ticker_set = set(STOCKS.get("Nifty 50", []) + STOCKS.get("Nifty Next 50", []))
        # Cheap word scan; tickers are short upper-case symbols.
        words = re.findall(r"[A-Za-z]{3,}", msg)
        upper_words = {w.upper() for w in words}
        if ticker_set & upper_words:
            return True
        # Company name hits (TICKER_TO_NAME has lowercased names mapped from sym).
        for sym, name in TICKER_TO_NAME.items():
            if name and len(name) >= 4 and name.lower() in msg:
                return True
    except Exception:
        pass
    return False


def _last_ticker_in_history(history):
    """Walk the conversation in reverse and return the most recent ticker
    symbol or company name mentioned, normalized to its symbol. Powers the
    nudge that recovers pronoun follow-ups when the model stalls."""
    try:
        ticker_set = set(STOCKS.get("Nifty 50", []) + STOCKS.get("Nifty Next 50", []))
        for sec_tickers in STOCKS.values():
            ticker_set.update(sec_tickers)
    except Exception:
        ticker_set = set()
    name_map = {}
    try:
        for sym, name in TICKER_TO_NAME.items():
            if name and len(name) >= 4:
                name_map[name.lower()] = sym
    except Exception:
        pass

    for msg in reversed(history or []):
        content = (msg.get("content") or "")
        if not content:
            continue
        lower = content.lower()
        words = re.findall(r"[A-Za-z]{3,}", content)
        for w in words:
            up = w.upper()
            if up in ticker_set:
                return up
        for name, sym in name_map.items():
            if name in lower:
                return sym
    return None


def _tools_for_message(message):
    """Return the tool set tailored to the user's latest message. Includes
    scan_universe whenever the question doesn't pin a single ticker, plus
    whenever an explicit scan keyword shows up."""
    msg = (message or "").lower()
    if any(kw in msg for kw in _SCAN_KEYWORDS):
        return _groq_build_tools()
    if not _looks_ticker_specific(msg):
        return _groq_build_tools()
    return _groq_build_tools(exclude={"scan_universe"})


def _groq_make_request(url, headers, payload, stream=False):
    import requests as _requests
    resp = None
    for attempt in range(_GROQ_RETRY_MAX_ATTEMPTS):
        resp = _requests.post(url, json=payload, headers=headers, stream=stream, timeout=60)
        if resp.status_code not in _GROQ_RETRY_STATUSES:
            break
        retry_after = None
        try:
            ra = resp.headers.get("Retry-After") or (resp.json().get("error") or {}).get("retry_after")
            retry_after = max(1, int(float(ra))) if ra else None
        except Exception:
            pass
        if attempt == _GROQ_RETRY_MAX_ATTEMPTS - 1:
            raise _GroqRateLimitError(retry_after or 30)
        time.sleep(_groq_backoff_delay(attempt, retry_after))
    resp.raise_for_status()
    return resp


def _provider_attempt(url, headers, payload, stream=False):
    """Single-shot request — no retries. Used by the failover wrapper."""
    import requests as _requests
    resp = _requests.post(url, json=payload, headers=headers, stream=stream, timeout=60)
    if resp.status_code == 429:
        retry_after = None
        try:
            ra = resp.headers.get("Retry-After") or (resp.json().get("error") or {}).get("retry_after")
            retry_after = max(1, int(float(ra))) if ra else None
        except Exception:
            pass
        raise _GroqRateLimitError(retry_after or 30)
    resp.raise_for_status()
    return resp


# ── Multi-provider config (OpenAI-compatible) ────────────────────────────────
# All three expose an OpenAI-compatible /chat/completions endpoint with
# streaming + function calling. The failover wrapper tries them in order.

_AGENT_PROVIDERS = [
    {
        "name": "gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "api_key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "model_default": "gemini-2.0-flash",
    },
    {
        "name": "groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "model_default": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    {
        "name": "cerebras",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "api_key_env": "CEREBRAS_API_KEY",
        "model_env": "CEREBRAS_MODEL",
        "model_default": "llama-3.3-70b",
    },
]


class _ProviderUnavailableError(Exception):
    """Raised when a provider can't be used (no key, rate-limited, network error).
    The failover wrapper catches this and tries the next provider."""
    def __init__(self, provider_name, reason, retry_after=None):
        super().__init__(f"{provider_name}: {reason}")
        self.provider_name = provider_name
        self.reason = reason
        self.retry_after = retry_after


def _enabled_providers():
    """Return the providers whose API key is set, in priority order."""
    return [p for p in _AGENT_PROVIDERS if (os.environ.get(p["api_key_env"]) or "").strip()]


# ── Token usage metrics (in-memory, reset on process restart) ────────────────
# Captured from the `usage` field that OpenAI-compatible providers emit when
# stream_options.include_usage=true. Aggregates total + per-provider counters
# so /api/agent/metrics can surface real numbers instead of estimates.

_AGENT_METRICS_LOCK = Lock()
_AGENT_METRICS = {
    "requests": 0,
    "cache_hits": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "by_provider": {},
}


def _metrics_record_cache_hit():
    with _AGENT_METRICS_LOCK:
        _AGENT_METRICS["requests"] += 1
        _AGENT_METRICS["cache_hits"] += 1


def _metrics_record_usage(provider_name, usage):
    if not isinstance(usage, dict):
        return
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    tt = int(usage.get("total_tokens") or (pt + ct))
    with _AGENT_METRICS_LOCK:
        _AGENT_METRICS["requests"] += 1
        _AGENT_METRICS["prompt_tokens"] += pt
        _AGENT_METRICS["completion_tokens"] += ct
        _AGENT_METRICS["total_tokens"] += tt
        bucket = _AGENT_METRICS["by_provider"].setdefault(
            provider_name,
            {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        bucket["requests"] += 1
        bucket["prompt_tokens"] += pt
        bucket["completion_tokens"] += ct
        bucket["total_tokens"] += tt


# ── Response cache (in-memory, TTL-based) ────────────────────────────────────
# Keyed by normalized question text. Identical questions within TTL skip the
# entire LLM round trip and re-stream the cached answer. Greatly reduces API
# call volume for popular queries (e.g. "Is TCS a buy?" asked by many users).

_AGENT_RESPONSE_CACHE = {}  # key -> (timestamp, response_text)
_AGENT_CACHE_LOCK = Lock()
_AGENT_CACHE_TTL_SEC = float(os.environ.get("AGENT_CACHE_TTL_SEC", "300"))
_AGENT_CACHE_MAX_ENTRIES = 256


def _agent_cache_key(history):
    """Cache key from the latest user message + the immediately preceding turn
    (to disambiguate follow-ups like 'what about its dividend?')."""
    if not history:
        return None
    last = history[-1]
    if last.get("role") != "user":
        return None
    msg = (last.get("content") or "").strip().lower()
    if len(msg) < 4:
        return None
    prev = ""
    if len(history) >= 2:
        prev = (history[-2].get("content") or "").strip().lower()[:200]
    return hashlib.sha256(f"{prev}||{msg}".encode("utf-8")).hexdigest()


def _agent_cache_get(key):
    if not key:
        return None
    with _AGENT_CACHE_LOCK:
        entry = _AGENT_RESPONSE_CACHE.get(key)
        if not entry:
            return None
        ts, text = entry
        if time.time() - ts > _AGENT_CACHE_TTL_SEC:
            _AGENT_RESPONSE_CACHE.pop(key, None)
            return None
        return text


def _agent_cache_set(key, text):
    if not key or not text:
        return
    with _AGENT_CACHE_LOCK:
        _AGENT_RESPONSE_CACHE[key] = (time.time(), text)
        if len(_AGENT_RESPONSE_CACHE) > _AGENT_CACHE_MAX_ENTRIES:
            cutoff = time.time() - _AGENT_CACHE_TTL_SEC
            for k, (ts, _t) in list(_AGENT_RESPONSE_CACHE.items()):
                if ts < cutoff:
                    _AGENT_RESPONSE_CACHE.pop(k, None)


def _stream_cached_response(text, chunk_size=24):
    """Re-emit a cached response as SSE events so the UX matches a live stream."""
    yield {"type": "thinking", "text": "Loaded recent answer from cache"}
    for i in range(0, len(text), chunk_size):
        yield {"type": "token", "text": text[i:i + chunk_size]}
        time.sleep(0.01)
    yield {"type": "done"}


# Sentences whose first words match this pattern are tool-call narration the
# user should never see ("I'll call get_investment_verdict", "Let me check the
# data", "To get the current price..."). The system prompt forbids this, but
# the model treats word-problem math as a scratchpad and ignores the rule, so
# we strip these sentences out server-side as a hard backstop.
_NARRATION_PREFIX_RE = re.compile(
    r"^(I'll |I will |Let me |To get |To find |To check |To assess |"
    r"First, I'll |Now I'll |I'm going to |I need to (call|use|check) )"
)


class _NarrationFilter:
    """Token-stream filter. Buffers content until a sentence boundary (. or \\n),
    then drops the sentence if it starts with a narration prefix. Otherwise
    emits it. Call flush() at end of stream to drain the tail."""

    def __init__(self):
        self._buf = ""

    def feed(self, chunk):
        if not chunk:
            return ""
        self._buf += chunk
        out_parts = []
        while True:
            dot = self._buf.find(".")
            nl = self._buf.find("\n")
            if dot < 0 and nl < 0:
                break
            if dot < 0:
                idx = nl
            elif nl < 0:
                idx = dot
            else:
                idx = min(dot, nl)
            sentence = self._buf[:idx + 1]
            self._buf = self._buf[idx + 1:]
            if _NARRATION_PREFIX_RE.match(sentence.lstrip()):
                # Drop the sentence and any trailing whitespace/newline that
                # would otherwise leave an awkward gap before the next one.
                self._buf = self._buf.lstrip()
                continue
            out_parts.append(sentence)
        return "".join(out_parts)

    def flush(self):
        tail = self._buf
        self._buf = ""
        if tail and _NARRATION_PREFIX_RE.match(tail.lstrip()):
            return ""
        return tail


def _run_agent_provider_stream(provider, history):
    """Generator yielding SSE events for ONE provider. Raises
    _ProviderUnavailableError on the first request if the provider is unusable
    (rate-limited, auth failure, network error) so the failover wrapper can
    try the next one."""
    api_key = (os.environ.get(provider["api_key_env"]) or "").strip()
    if not api_key:
        raise _ProviderUnavailableError(provider["name"], "no API key set")

    model = (os.environ.get(provider["model_env"], "") or provider["model_default"]).strip() or provider["model_default"]
    url = provider["url"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    latest_user_msg = ""
    for msg in reversed(history):
        if msg.get("role") == "user":
            latest_user_msg = msg.get("content") or ""
            break
    tools = _tools_for_message(latest_user_msg)
    messages = [{"role": "system", "content": _AGENT_SYSTEM_PROMPT}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    first_call = True
    nudged = False
    max_turns = 6
    for _ in range(max_turns):
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 1024,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        try:
            resp = _provider_attempt(url, headers, payload, stream=True)
        except _GroqRateLimitError as e:
            if first_call:
                raise _ProviderUnavailableError(provider["name"], "rate limited", e.retry_after)
            yield {"type": "error", "text": f"Rate limited mid-conversation on {provider['name']}. Please retry."}
            return
        except Exception as e:
            if first_call:
                raise _ProviderUnavailableError(provider["name"], str(e))
            yield {"type": "error", "text": f"Provider error mid-stream: {e}"}
            return
        first_call = False

        accumulated_content = ""
        accumulated_tool_calls = {}  # index -> {id, name, arguments}
        finish_reason = None
        turn_usage = None
        narration_filter = _NarrationFilter()

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except Exception:
                continue

            # Usage chunk: providers that honour stream_options.include_usage
            # emit a final chunk with usage populated (and choices empty).
            if chunk.get("usage"):
                turn_usage = chunk["usage"]

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason") or finish_reason

            # Stream text tokens as they arrive, but route them through the
            # narration filter so tool-call monologue ("I'll call X", "Let me
            # check Y") never reaches the user even if the model ignores the
            # silence rule in the system prompt.
            content_chunk = delta.get("content") or ""
            if content_chunk:
                accumulated_content += content_chunk
                cleaned = narration_filter.feed(content_chunk)
                if cleaned:
                    yield {"type": "token", "text": cleaned}

            # Accumulate tool call deltas
            for tc_delta in (delta.get("tool_calls") or []):
                idx = tc_delta.get("index", 0)
                if idx not in accumulated_tool_calls:
                    accumulated_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                tc = accumulated_tool_calls[idx]
                if tc_delta.get("id"):
                    tc["id"] = tc_delta["id"]
                fn = tc_delta.get("function") or {}
                tc["name"] += fn.get("name") or ""
                tc["arguments"] += fn.get("arguments") or ""

        tail = narration_filter.flush()
        if tail:
            yield {"type": "token", "text": tail}

        _metrics_record_usage(provider["name"], turn_usage)

        if finish_reason == "stop" or (accumulated_content and not accumulated_tool_calls):
            yield {"type": "done"}
            return

        if finish_reason == "length":
            if accumulated_content:
                yield {"type": "done"}
            else:
                yield {"type": "error", "text": "Response exceeded token limit. Please ask a more focused question."}
            return

        if not accumulated_tool_calls:
            # No tool calls AND no usable text. Most commonly this happens
            # when the model is confused by a pronoun follow-up. Inject one
            # nudge with the last ticker we can detect from the conversation
            # and loop once more before giving up.
            if accumulated_content:
                yield {"type": "done"}
                return
            if not nudged:
                nudged = True
                hint_ticker = _last_ticker_in_history(history)
                if hint_ticker:
                    nudge = (
                        "Your last turn produced no answer. The user is likely asking "
                        "a pronoun follow-up about a stock we already discussed — the "
                        f"most recent ticker on the table is {hint_ticker}. Resolve the "
                        "reference, call the right tool, and give a direct answer in a "
                        "complete sentence. Never trail off."
                    )
                else:
                    nudge = (
                        "Your last turn produced no answer. The user's question is "
                        "probably outside Stock Analysis Pro's scope (NSE equity "
                        "research) — see the SCOPE rules in the system prompt. Decline "
                        "in ONE complete sentence, name the topic as out of scope, and "
                        "offer one concrete NSE-related redirect. Never trail off."
                    )
                messages.append({"role": "system", "content": nudge})
                continue
            yield {"type": "error", "text": (
                "That question is outside what I cover (NSE equity research). Ask me "
                "about a stock — e.g. 'Is TCS a buy right now?' or 'Top losers today.'"
            )}
            return

        # Build assistant message with tool calls and add to context
        tool_calls_list = [
            {
                "id": accumulated_tool_calls[i]["id"],
                "type": "function",
                "function": {
                    "name": accumulated_tool_calls[i]["name"],
                    "arguments": accumulated_tool_calls[i]["arguments"],
                },
            }
            for i in sorted(accumulated_tool_calls)
        ]
        messages.append({
            "role": "assistant",
            "content": accumulated_content or None,
            "tool_calls": tool_calls_list,
        })

        # Dispatch each tool and emit a thinking event
        for tc in tool_calls_list:
            fn = tc["function"]
            tool_name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}

            yield {"type": "thinking", "text": _tool_thinking_label(tool_name, args)}

            result = _agent_dispatch_tool(tool_name, args)
            if not isinstance(result, dict):
                result = {"result": result}
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result),
            })

    yield {"type": "error", "text": "Could not complete the analysis. Please try again."}


def _run_agent_with_failover_stream(history):
    """Top-level streaming generator. Checks the response cache first; on miss,
    walks through the provider list (Gemini → Groq → Cerebras), failing over
    on rate-limit / auth / network errors. Caches the final response on success."""
    cache_key = _agent_cache_key(history)
    cached = _agent_cache_get(cache_key)
    if cached:
        _metrics_record_cache_hit()
        for event in _stream_cached_response(cached):
            yield event
        return

    providers = _enabled_providers()
    if not providers:
        yield {"type": "error", "text": "No AI provider configured. Set GEMINI_API_KEY, GROQ_API_KEY, or CEREBRAS_API_KEY."}
        return

    last_failure = None
    for provider in providers:
        full_response_parts = []
        try:
            gen = _run_agent_provider_stream(provider, history)
            for event in gen:
                if event.get("type") == "token":
                    full_response_parts.append(event["text"])
                yield event
                if event.get("type") in ("done", "error"):
                    break
            full_text = "".join(full_response_parts).strip()
            if full_text:
                _agent_cache_set(cache_key, full_text)
            return
        except _ProviderUnavailableError as e:
            last_failure = e
            print(f"Provider failover: {provider['name']} unavailable ({e.reason}); trying next.")
            continue
        except Exception as e:
            last_failure = e
            print(f"Provider failover: {provider['name']} crashed ({e}); trying next.")
            continue

    # All providers exhausted
    retry_after = getattr(last_failure, "retry_after", None)
    msg = "All AI providers are temporarily unavailable. Please try again in a moment."
    err_event = {"type": "error", "text": msg}
    if retry_after:
        err_event["retryAfter"] = int(retry_after)
    yield err_event


def _run_agent_groq(history):
    """Non-streaming JSON fallback — drains the streaming failover wrapper and
    returns the full text. The frontend uses /api/agent/stream; this exists
    purely for /api/agent/query consumers."""
    parts = []
    for event in _run_agent_with_failover_stream(history):
        et = event.get("type")
        if et == "token":
            parts.append(event.get("text", ""))
        elif et == "error":
            return event.get("text") or "Agent request failed. Please try again."
    return "".join(parts).strip() or "No response generated."


_AGENT_IP_LAST_CALL = {}
_AGENT_IP_LOCK = Lock()
_AGENT_MIN_INTERVAL_SEC = float(os.environ.get("AGENT_MIN_INTERVAL_SEC", "3"))
_AGENT_GLOBAL_SEMAPHORE = __import__("threading").BoundedSemaphore(
    int(os.environ.get("AGENT_MAX_CONCURRENCY", "2"))
)


def _agent_throttle_check(ip):
    if _AGENT_MIN_INTERVAL_SEC <= 0 or not ip:
        return 0
    now = time.time()
    with _AGENT_IP_LOCK:
        last = _AGENT_IP_LAST_CALL.get(ip, 0)
        wait = _AGENT_MIN_INTERVAL_SEC - (now - last)
        if wait > 0:
            return int(wait) + 1
        _AGENT_IP_LAST_CALL[ip] = now
        if len(_AGENT_IP_LAST_CALL) > 1024:
            cutoff = now - 600
            for k in [k for k, v in _AGENT_IP_LAST_CALL.items() if v < cutoff]:
                _AGENT_IP_LAST_CALL.pop(k, None)
    return 0


def _agent_parse_history(data, max_turns=10, max_content=1500):
    """Keep up to `max_turns` of prior user/assistant messages so pronoun
    follow-ups ('what about their split?', 'is it cheap?') retain the
    ticker/topic context. The previous value of 2 dropped the original
    question after a single round-trip and broke multi-turn chat."""
    raw_history = data.get("history") or []
    history = []
    for turn in raw_history[-max_turns:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            history.append({"role": role, "content": content[:max_content]})
    message = (data.get("message") or "").strip()
    if message:
        history.append({"role": "user", "content": message[:2000]})
    return history, message


@app.route("/api/agent/stream", methods=["POST"])
def agent_stream_route():
    if not _enabled_providers():
        return jsonify({"error": "AI assistant is not configured. Set GEMINI_API_KEY, GROQ_API_KEY, or CEREBRAS_API_KEY."}), 503

    data = request.get_json(silent=True) or {}
    history, message = _agent_parse_history(data)
    if not message:
        return jsonify({"error": "message is required"}), 400

    client_ip = (request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr or "")
    wait_secs = _agent_throttle_check(client_ip)
    if wait_secs:
        return jsonify({"error": f"Too many requests. Try again in {wait_secs}s.", "retryAfter": wait_secs}), 429

    if not _AGENT_GLOBAL_SEMAPHORE.acquire(timeout=20):
        return jsonify({"error": "AI assistant is busy. Please try again in a moment.", "retryAfter": 10}), 429

    def generate():
        try:
            for event in _run_agent_with_failover_stream(history):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            import re as _re
            sanitized = _re.sub(r"key=[A-Za-z0-9_\-]+", "key=***", str(e))
            print(f"Agent stream fatal error: {sanitized}")
            yield f"data: {json.dumps({'type': 'error', 'text': 'Agent request failed. Please try again.'})}\n\n"
        finally:
            _AGENT_GLOBAL_SEMAPHORE.release()

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/agent/query", methods=["POST"])
def agent_query_route():
    if not _enabled_providers():
        return jsonify({"error": "AI assistant is not configured. Set GEMINI_API_KEY, GROQ_API_KEY, or CEREBRAS_API_KEY."}), 503

    data = request.get_json(silent=True) or {}
    history, message = _agent_parse_history(data)
    if not message:
        return jsonify({"error": "message is required"}), 400

    client_ip = (request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr or "")
    wait_secs = _agent_throttle_check(client_ip)
    if wait_secs:
        resp = jsonify({"error": f"You're sending requests too quickly. Try again in {wait_secs}s.", "retryAfter": wait_secs})
        resp.headers["Retry-After"] = str(wait_secs)
        return resp, 429

    if not _AGENT_GLOBAL_SEMAPHORE.acquire(timeout=20):
        return jsonify({
            "error": "AI assistant is busy. Please try again in a moment.",
            "retryAfter": 10,
        }), 429
    try:
        reply = _run_agent_groq(history)
        return jsonify({"response": reply})
    except _GroqRateLimitError as e:
        retry_after = int(getattr(e, "retry_after", 30) or 30)
        resp = jsonify({
            "error": f"AI assistant is rate-limited upstream. Try again in {retry_after}s.",
            "retryAfter": retry_after,
        })
        resp.headers["Retry-After"] = str(retry_after)
        return resp, 429
    except Exception as e:
        import re
        sanitized = re.sub(r"key=[A-Za-z0-9_\-]+", "key=***", str(e))
        status = getattr(getattr(e, "response", None), "status_code", None)
        print(f"Agent error (status={status}): {sanitized}")
        if status == 429:
            return jsonify({
                "error": "AI assistant is rate-limited right now. Please wait a minute and try again.",
                "retryAfter": 30,
            }), 429
        if status == 400:
            return jsonify({
                "error": "AI model configuration error — the configured model may be unavailable. Please set the GROQ_MODEL environment variable to a valid Groq model ID.",
            }), 503
        if status == 401 or status == 403:
            return jsonify({
                "error": "AI assistant is not authorised (invalid API key).",
            }), 503
        return jsonify({"error": "Agent request failed. Please try again."}), 500
    finally:
        _AGENT_GLOBAL_SEMAPHORE.release()


@app.route("/api/agent/metrics", methods=["GET"])
def agent_metrics_route():
    """Aggregate token usage since process start. Numbers come from the
    provider's own `usage` field, so they reflect actual billed tokens rather
    than estimates. Resets on every restart — fine for a single-process app."""
    with _AGENT_METRICS_LOCK:
        snapshot = {
            "requests": _AGENT_METRICS["requests"],
            "cache_hits": _AGENT_METRICS["cache_hits"],
            "prompt_tokens": _AGENT_METRICS["prompt_tokens"],
            "completion_tokens": _AGENT_METRICS["completion_tokens"],
            "total_tokens": _AGENT_METRICS["total_tokens"],
            "by_provider": {k: dict(v) for k, v in _AGENT_METRICS["by_provider"].items()},
        }
    billed = max(snapshot["requests"] - snapshot["cache_hits"], 0)
    snapshot["avg_prompt_tokens"] = round(snapshot["prompt_tokens"] / billed, 1) if billed else 0
    snapshot["avg_completion_tokens"] = round(snapshot["completion_tokens"] / billed, 1) if billed else 0
    snapshot["avg_total_tokens"] = round(snapshot["total_tokens"] / billed, 1) if billed else 0
    snapshot["cache_hit_rate"] = round(snapshot["cache_hits"] / snapshot["requests"], 3) if snapshot["requests"] else 0
    return jsonify(snapshot)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
