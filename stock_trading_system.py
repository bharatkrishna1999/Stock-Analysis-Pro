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

from flask import Flask, jsonify, request, Response, stream_with_context
import json
import os
import pickle
import time
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
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

# Set non-interactive backend for Render server
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("yfinance").propagate = False

app = Flask(__name__)

DIVIDEND_CACHE_TTL = timedelta(hours=6)
DIVIDEND_CACHE = {}
DIVIDEND_MAX_RESULTS = 150
DIVIDEND_BATCH_SIZE = 50
DIVIDEND_MAX_WORKERS = 4

DEFAULT_ANALYSIS_PERIOD = '6mo'
DEFAULT_ANALYSIS_INTERVAL = '1d'
MAX_HISTORY_POINTS = 160
REGRESSION_WAIT_TIMEOUT_SECONDS = 2.0
REGRESSION_CACHE_TTL = timedelta(hours=8)
ANALYZE_CACHE_TTL = timedelta(minutes=20)
PRICE_HISTORY_CACHE_TTL = timedelta(minutes=30)

YAHOO_TICKER_ALIASES = {
    "ETERNAL": ["ZOMATO"],
}

# ===== EXPANDED STOCK LIST - ALL NSE STOCKS =====
# Organized by sector for better UX, but includes 500+ stocks

STOCKS = {
    'IT Sector': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM', 'LTIM', 'COFORGE', 'MPHASIS', 'PERSISTENT', 
                  'MINDTREE', 'L&TTS', 'SONATSOFTW', 'TATAELXSI', 'ROLTA', 'CYIENT', 'KPITTECH', 
                  'INTELLECT', 'MASTEK', 'ZENSAR', 'POLYCAB'],
    
    'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'BANKBARODA', 
                'PNB', 'FEDERALBNK', 'AUBANK', 'BANDHANBNK', 'IDFCFIRSTB', 'RBLBANK', 'CANBK', 
                'UNIONBANK', 'INDIANB', 'CENTRALBK', 'MAHABANK', 'JKBANK', 'KARNATBANK', 'DCBBANK'],
    
    'Financial Services': ['BAJFINANCE', 'BAJAJFINSV', 'SBILIFE', 'HDFCLIFE', 'ICICIGI', 'ICICIPRULI', 
                           'CHOLAFIN', 'PFC', 'RECLTD', 'MUTHOOTFIN', 'HDFCAMC', 'CDSL', 'CAMS', 
                           'LICHSGFIN', 'M&MFIN', 'SHRIRAMFIN', 'PNBHOUSING', 'IIFL', 'CREDITACC'],
    
    'Auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT', 'HEROMOTOCO', 'TVSMOTOR', 
             'ASHOKLEY', 'ESCORTS', 'FORCEMOT', 'MAHINDCIE', 'SONACOMS', 'TIINDIA'],
    
    'Auto Components': ['BOSCHLTD', 'MOTHERSON', 'BALKRISIND', 'MRF', 'APOLLOTYRE', 'EXIDEIND', 
                        'AMARAJABAT', 'BHARAT', 'CEATLTD', 'SCHAEFFLER', 'SUPRAJIT', 'ENDURANCE'],
    
    'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'LUPIN', 'BIOCON', 'AUROPHARMA', 
               'TORNTPHARM', 'ALKEM', 'CADILAHC', 'IPCALAB', 'GRANULES', 'GLENMARK', 'NATCOPHARMA',
               'JBCHEPHARM', 'LAURUSLABS', 'PFIZER', 'ABBOTINDIA', 'GLAXO', 'SANOFI'],
    
    'Healthcare': ['APOLLOHOSP', 'MAXHEALTH', 'FORTIS', 'LALPATHLAB', 'METROPOLIS', 'DRREDDY',
                   'THYROCARE', 'ASTER', 'RAINBOW'],
    
    'Consumer Goods': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO', 'GODREJCP', 
                       'COLPAL', 'TATACONSUM', 'EMAMILTD', 'VBL', 'RADICO', 'UBL', 'MCDOWELL-N',
                       'PGHH', 'GILLETTE', 'JYOTHYLAB', 'BAJAJCON', 'VINATIORGA'],
    
    'Retail': ['DMART', 'TRENT', 'TITAN', 'ABFRL', 'SHOPERSTOP', 'JUBLFOOD', 'WESTLIFE', 
               'DEVYANI', 'SPENCERS', 'VMART', 'BATA'],
    
    'Energy - Oil & Gas': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL', 'HINDPETRO', 'PETRONET', 
                           'OIL', 'MGL', 'IGL', 'GUJGASLTD', 'ATGL'],
    
    'Power': ['NTPC', 'POWERGRID', 'ADANIPOWER', 'TATAPOWER', 'TORNTPOWER', 'ADANIGREEN', 
              'NHPC', 'SJVN', 'JSW', 'CESC', 'PFC', 'RECLTD'],
    
    'Metals & Mining': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'COALINDIA', 'VEDL', 'NMDC', 'SAIL', 
                        'NATIONALUM', 'JINDALSTEL', 'HINDZINC', 'RATNAMANI', 'WELCORP', 'WELSPUNIND',
                        'MOIL', 'GMRINFRA'],
    
    'Cement': ['ULTRACEMCO', 'GRASIM', 'SHREECEM', 'AMBUJACEM', 'ACC', 'DALMIACEM', 'JKCEMENT',
               'RAMCOCEM', 'HEIDELBERG', 'ORIENTCEM', 'JKLAKSHMI', 'STARCEMENT'],
    
    'Real Estate': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'BRIGADE', 'PHOENIXLTD', 
                    'SOBHA', 'LODHA', 'MAHLIFE', 'SUNTECK'],
    
    'Infrastructure': ['LT', 'ADANIENT', 'ADANIPORTS', 'SIEMENS', 'ABB', 'CUMMINSIND', 'VOLTAS', 
                       'NCC', 'PNC', 'KNR', 'IRCTC', 'CONCOR', 'IRFC', 'GMR'],
    
    'Telecom': ['BHARTIARTL', 'IDEA', 'TATACOMM', 'ROUTE'],
    
    'Media': ['ZEEL', 'SUNTV', 'PVRINOX', 'SAREGAMA', 'TIPS', 'NAZARA', 'NETWORK18'],
    
    'Chemicals': ['UPL', 'PIDILITIND', 'AARTIIND', 'SRF', 'DEEPAKNTR', 'GNFC', 'CHAMBLFERT', 
                  'TATACHEM', 'BALRAMCHIN', 'ALKYLAMINE', 'CLEAN', 'NOCIL', 'TATAchemicals',
                  'ATUL', 'FINEORG', 'NAVINFLUOR'],
    
    'Paints': ['ASIANPAINT', 'BERGER', 'KANSAINER', 'INDIGO'],
    
    'Textiles': ['GRASIM', 'AIAENG', 'RAYMOND', 'ARVIND', 'WELSPUNIND', 'TRIDENT', 'KPR'],
    
    'Logistics': ['CONCOR', 'VRL', 'MAHLOG', 'BLUEDART', 'TCI', 'AEGISCHEM', 'GATI'],
    
    'Aviation': ['INDIGO', 'SPICEJET'],
    
    'Hospitality': ['INDHOTEL', 'LEMONTREE', 'CHOICEINT', 'EIH', 'CHALET', 'ITCHOTELS'],
    
    'Construction': ['LT', 'NCC', 'PNC', 'KNR', 'ASHOKA', 'SADBHAV', 'HG'],
    
    'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO', 'GODREJCP',
             'TATACONSUM', 'EMAMILTD', 'JYOTHYLAB', 'BAJAJCON', 'VBL'],
    
    'Electronics': ['DIXON', 'AMBER', 'ROUTE', 'POLYCAB', 'HAVELLS', 'CROMPTON', 'VGUARD',
                    'KEI', 'FINOLEX', 'KALYANKJIL'],
    
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


PRICE_HISTORY_CACHE = HybridTTLCache('price_history', PRICE_HISTORY_CACHE_TTL, max_entries=180)
ANALYSIS_CACHE = HybridTTLCache('analysis', ANALYZE_CACHE_TTL, max_entries=180)
REGRESSION_CACHE = HybridTTLCache('regression', REGRESSION_CACHE_TTL, max_entries=96)
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


@app.after_request
def _add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
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


def _compute_split_adjusted_dividend(dividends, splits):
    """Adjust historical dividends to current share count using split data."""
    if dividends is None or dividends.empty:
        return 0.0
    if splits is None or splits.empty:
        return float(dividends.sum())

    splits = splits[splits > 0].sort_index()
    if splits.empty:
        return float(dividends.sum())

    split_dates = splits.index
    split_values = splits.values
    cumulative_from_end = np.cumprod(split_values[::-1])[::-1]

    adjusted_total = 0.0
    for div_date, div_value in dividends.items():
        idx = split_dates.searchsorted(div_date, side="right")
        if idx < len(split_values):
            factor = cumulative_from_end[idx]
        else:
            factor = 1.0
        adjusted_total += float(div_value) / factor
    return float(adjusted_total)


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

# Enhanced company name mapping with MANY more variations
COMPANY_TO_TICKER = {
    # IT Sector
    'VEDANTA': 'VEDL', 'TATA CONSULTANCY': 'TCS', 'TATA CONSULTANCY SERVICES': 'TCS', 'INFOSYS': 'INFY',
    'HCL TECH': 'HCLTECH', 'HCL TECHNOLOGIES': 'HCLTECH', 'TECH MAHINDRA': 'TECHM', 'L&T INFOTECH': 'LTIM',
    'LTI': 'LTIM', 'MINDTREE': 'MINDTREE', 'COFORGE': 'COFORGE', 'MPHASIS': 'MPHASIS',
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
    'EXIDE': 'EXIDEIND', 'EXIDE INDUSTRIES': 'EXIDEIND', 'AMARA RAJA': 'AMARAJABAT',
    
    # Pharma
    'SUN PHARMA': 'SUNPHARMA', 'SUN PHARMACEUTICAL': 'SUNPHARMA', 'DR REDDY': 'DRREDDY',
    'DR REDDYS': 'DRREDDY', 'CIPLA': 'CIPLA', 'DIVIS': 'DIVISLAB', 'DIVIS LAB': 'DIVISLAB',
    'LUPIN': 'LUPIN', 'BIOCON': 'BIOCON', 'AUROBINDO': 'AUROPHARMA', 'TORRENT PHARMA': 'TORNTPHARM',
    'ALKEM': 'ALKEM', 'CADILA': 'CADILAHC', 'ZYDUS': 'CADILAHC', 'IPCA': 'IPCALAB',
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
            result = {
                'price': curr, 'sma9': sma9, 'sma5': sma5, 'sma20': sma20, 'sma50': sma50,
                'daily': daily_ret, 'hourly': hourly_ret,
                'rsi': rsi, 'macd_bullish': bool(macd_bullish), 'high': h, 'low': l,
                'pct_from_low': pct_from_low, 'zscore': zscore, 'pct_deviation': pct_deviation,
                'mean_price': mean_price, 'std_price': std_price, 'bb_upper': bb_upper,
                'bb_lower': bb_lower, 'bb_position': bb_position, 'volatility': volatility
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
            zscore_explain += f" → EXTREME OVEREXTENSION (+2σ). Price {abs(i['pct_deviation']):.1f}% above average - HIGH probability mean reversion DOWN expected."
        elif i['zscore'] > 1:
            zscore_explain += f" → MODERATELY OVERBOUGHT. Price {abs(i['pct_deviation']):.1f}% above mean - potential pullback zone."
        elif i['zscore'] < -2:
            zscore_explain += f" → EXTREME OVERSOLD (-2σ). Price {abs(i['pct_deviation']):.1f}% below average - HIGH probability bounce to mean."
        elif i['zscore'] < -1:
            zscore_explain += f" → MODERATELY OVERSOLD. Price {abs(i['pct_deviation']):.1f}% below mean - bounce opportunity."
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
            f"Definition: The percentage you could lose if the trade hits your stop loss - the invalidation level "
            f"below which the trade thesis no longer holds. "
            f"Inputs: Current price ({i['price']:.2f}), Stop loss ({stop_price:.2f}). "
            f"Formula: ((Current - Stop) / Current) x 100 = "
            f"(({i['price']:.2f} - {stop_price:.2f}) / {i['price']:.2f}) x 100 = {max_risk_pct:.1f}%. "
            f"The stop loss is placed 2% below the 18-day low (recent support - buffer), acting as the "
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
            confidence_oneliner = "Past similar setups moved upward more often than not." if sig == "BUY" else "Past similar setups moved downward more often than not." if sig == "SELL" else "Mixed signals - waiting for clearer direction is advisable."
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
                'bb_explain': bb_explain, 'macd_text': "BULLISH - momentum favors buyers" if i['macd_bullish'] else "BEARISH - momentum favors sellers"
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
                'macd_bullish': i['macd_bullish']
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
            fig.patch.set_facecolor('#131824')
            ax.set_facecolor('#0a0e1a')

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
            ax.axvline(x=hist_len - 1, color='#2d3748', linewidth=1, linestyle='-', alpha=0.5, zorder=1)
            mid_y = (max(hist_prices) + min(hist_prices)) / 2
            ax.text(hist_len + 1, ax.get_ylim()[1] * 0.99, 'FORECAST',
                    color='#718096', fontsize=7, fontstyle='italic', va='top', zorder=7)

            # Styling
            ax.tick_params(colors='#718096', labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#2d3748')
            ax.spines['left'].set_color('#2d3748')
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

            bg_color = '#131824'
            grid_color = '#2d3748'

            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)

            ax.scatter(X_win * 100, y_win * 100, alpha=0.6, c='#00d9ff',
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

    def fetch_dividend_data(self, symbols, limit_results=True):
        """Fetch dividend yield, current price, and annualized volatility for given symbols."""
        results = []
        dividend_found = 0

        if not symbols:
            return results, dividend_found

        def _batched(iterable, size):
            for idx in range(0, len(iterable), size):
                yield iterable[idx:idx + size]

        for batch in _batched(symbols, 75):
            tickers = [f"{symbol}.NS" for symbol in batch]
            try:
                data = yf.download(
                    tickers=tickers,
                    period='1y',
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
                        if 'Stock Splits' in data.columns.get_level_values(0):
                            splits = data['Stock Splits'][ticker_symbol].dropna()
                        elif 'Splits' in data.columns.get_level_values(0):
                            splits = data['Splits'][ticker_symbol].dropna()
                        else:
                            splits = pd.Series(dtype=float)
                    else:
                        close_series = data['Close'].dropna()
                        dividends = data['Dividends'].dropna() if 'Dividends' in data.columns else pd.Series(dtype=float)
                        if 'Stock Splits' in data.columns:
                            splits = data['Stock Splits'].dropna()
                        elif 'Splits' in data.columns:
                            splits = data['Splits'].dropna()
                        else:
                            splits = pd.Series(dtype=float)
                    if close_series.empty or len(close_series) < 10:
                        continue
                    current_price = float(close_series.iloc[-1])
                    if current_price <= 0:
                        continue
                    annual_dividend = _compute_split_adjusted_dividend(dividends, splits)
                    if annual_dividend <= 0:
                        continue
                    dividend_yield = (annual_dividend / current_price) * 100
                    returns = close_series.pct_change().dropna()
                    volatility = float(returns.std() * np.sqrt(252) * 100) if len(returns) > 5 else 0.0

                    dividend_found += 1

                    results.append({
                        'symbol': symbol,
                        'price': round(current_price, 2),
                        'annual_dividend': round(annual_dividend, 2),
                        'dividend_yield': round(dividend_yield, 2),
                        'volatility': round(volatility, 2)
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

        params = {
            'conservative': {'max_weight': 0.08, 'vol_penalty': 0.15, 'min_yield': 1.0},
            'moderate':     {'max_weight': 0.15, 'vol_penalty': 0.05, 'min_yield': 0.5},
            'aggressive':   {'max_weight': 0.30, 'vol_penalty': 0.01, 'min_yield': 0.0}
        }
        p = params.get(risk_appetite, params['moderate'])

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

analyzer = Analyzer()

# ===== HTML TEMPLATE (IDENTICAL TO WORKING VERSION) =====
# [Keeping your exact HTML - no changes needed]

@app.route('/')
def index():
    # Using your exact working HTML
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="NSE stock dashboard with technical analysis, dividend insights, and market dependency metrics.">
    <title>Stock Analysis Pro - All NSE Stocks</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root { --bg-dark: #0a0e1a; --bg-card: #131824; --bg-card-hover: #1a1f2e; --accent-cyan: #00d9ff; --accent-purple: #9d4edd; --accent-green: #06ffa5; --text-primary: #ffffff; --text-secondary: #a0aec0; --text-muted: #718096; --border-color: #2d3748; --success: #10b981; --warning: #f59e0b; --danger: #ef4444; }
        body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; background: var(--bg-dark); color: var(--text-primary); min-height: 100vh; line-height: 1.6; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header { text-align: center; padding: 40px 0; border-bottom: 1px solid var(--border-color); background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(157, 78, 221, 0.1)); }
        header h1 { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-size: 3em; font-weight: 700; margin-bottom: 10px; background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        header p { color: var(--text-secondary); font-size: 1.1em; }
        .stock-count { background: rgba(0, 217, 255, 0.1); color: var(--accent-cyan); padding: 8px 16px; border-radius: 20px; display: inline-block; margin-top: 10px; font-size: 0.9em; font-weight: 600; }
        .tabs { display: flex; gap: 0; margin: 30px 0; border-bottom: 2px solid var(--border-color); }
        .tab { flex: 1; min-width: 0; padding: 15px 10px; background: transparent; border: none; color: var(--text-secondary); font-size: 1em; font-weight: 600; cursor: pointer; transition: all 0.3s; border-bottom: 3px solid transparent; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; text-align: center; }
        .tab:hover { color: var(--accent-cyan); }
        .tab.active { color: var(--text-primary); border-bottom-color: var(--accent-cyan); }
        .tab-content { display: none; }
        .tab-content.active { display: block; min-height: 70vh; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px; }
        .card { background: var(--bg-card); border-radius: 12px; padding: 25px; border: 1px solid var(--border-color); transition: all 0.3s; min-height: 220px; }
        .card:hover { background: var(--bg-card-hover); border-color: var(--accent-cyan); }
        .card h2, .card h3 { color: var(--text-primary); margin-bottom: 15px; font-size: 1.3em; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 600; }
        #search, #regression-search { width: 100%; padding: 14px; border: 2px solid var(--border-color); border-radius: 8px; font-size: 1em; background: var(--bg-dark); color: var(--text-primary); transition: all 0.3s; }
        #search:focus, #regression-search:focus { outline: none; border-color: var(--accent-cyan); box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.1); }
        #dividend-search:focus { outline: none; border-color: var(--accent-cyan); box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.1); }
        .suggestions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 15px; max-height: 300px; overflow-y: auto; }
        .category { margin-bottom: 20px; }
        .category h4 { color: var(--accent-cyan); font-size: 0.85em; margin-bottom: 8px; text-transform: uppercase; font-weight: 600; letter-spacing: 1px; }
        .stocks { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
        button { padding: 10px 16px; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 6px; cursor: pointer; font-weight: 500; transition: all 0.2s; color: var(--text-secondary); font-size: 0.9em; }
        button:hover { background: var(--accent-cyan); color: var(--bg-dark); border-color: var(--accent-cyan); transform: translateY(-2px); }
        #result-view { display: none; }
        .result-card { background: var(--bg-card); border-radius: 12px; padding: 35px; border: 1px solid var(--border-color); }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 2px solid var(--border-color); padding-bottom: 20px; }
        .header h2 { color: var(--text-primary); font-size: 2.5em; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 700; }
        .signal-badge { font-size: 1.2em; font-weight: 700; padding: 12px 24px; border-radius: 8px; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; letter-spacing: 1px; }
        .signal-BUY { background: linear-gradient(135deg, #10b981, #059669); color: white; }
        .signal-SELL { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
        .signal-HOLD { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
        .action-banner { background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; font-weight: 600; font-size: 1.2em; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3); }
        .rec-box { background: var(--bg-card-hover); border-left: 4px solid var(--accent-green); color: var(--text-primary); padding: 20px; border-radius: 8px; margin-bottom: 25px; font-size: 1.05em; }
        .confidence-meter { margin: 25px 0; padding: 20px; background: var(--bg-card-hover); border-radius: 10px; border: 1px solid var(--border-color); }
        .confidence-label { font-size: 0.9em; color: var(--text-secondary); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
        .confidence-bar-container { background: var(--bg-dark); height: 30px; border-radius: 15px; overflow: hidden; margin-bottom: 10px; }
        .confidence-bar { height: 100%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.9em; transition: width 1s ease; }
        .confidence-text { color: var(--text-secondary); font-size: 0.95em; line-height: 1.6; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 15px; margin: 25px 0; }
        .metric { background: var(--bg-card-hover); padding: 18px; border-radius: 8px; border-left: 3px solid var(--accent-cyan); transition: all 0.3s; }
        .metric:hover { transform: translateY(-3px); border-left-color: var(--accent-green); }
        .metric-label { font-size: 0.75em; color: var(--text-muted); text-transform: uppercase; margin-bottom: 5px; font-weight: 600; letter-spacing: 0.5px; }
        .metric-value { font-size: 1.5em; font-weight: 700; color: var(--text-primary); font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
        .explanation-section { margin: 25px 0; }
        .explanation-section h3 { color: var(--accent-cyan); margin-bottom: 12px; font-size: 1em; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
        .explanation { background: var(--bg-card-hover); padding: 16px; border-radius: 8px; line-height: 1.7; color: var(--text-secondary); border-left: 3px solid var(--accent-cyan); }
        .trading-plan { background: var(--bg-card-hover); padding: 25px; border-radius: 10px; margin-top: 30px; border: 2px solid var(--accent-purple); }
        .trading-plan h3 { color: var(--accent-purple); margin-bottom: 20px; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-size: 1.3em; font-weight: 700; }
        .plan-item { display: grid; grid-template-columns: 140px 1fr; gap: 20px; margin-bottom: 15px; padding: 15px; background: var(--bg-dark); border-radius: 8px; transition: all 0.3s; }
        .plan-item:hover { background: var(--bg-card); transform: translateX(5px); }
        .plan-label { font-weight: 700; color: var(--accent-cyan); font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-size: 0.9em; }
        .plan-value { color: var(--text-primary); font-weight: 500; }
        .back-btn { background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); color: white; padding: 12px 28px; margin-bottom: 20px; border: none; font-weight: 600; font-size: 1em; }
        .back-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(0, 217, 255, 0.4); }
        .loading { text-align: center; color: var(--accent-cyan); font-size: 1.3em; padding: 40px; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
        .error { background: rgba(239, 68, 68, 0.1); color: var(--danger); padding: 20px; border-radius: 8px; border-left: 4px solid var(--danger); }
        .regression-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 25px 0; }
        .regression-metric { background: var(--bg-card-hover); padding: 20px; border-radius: 10px; border-left: 3px solid var(--accent-purple); }
        .regression-metric-label { font-size: 0.85em; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; font-weight: 600; letter-spacing: 0.5px; }
        .regression-metric-value { font-size: 2em; font-weight: 700; color: var(--text-primary); font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; margin-bottom: 8px; }
        .regression-metric-desc { font-size: 0.9em; color: var(--text-secondary); line-height: 1.5; }
        .plot-container { background: var(--bg-card-hover); padding: 20px; border-radius: 12px; margin-bottom: 30px; border: 1px solid var(--border-color); text-align: center; }
        .plot-img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .hsic-badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 0.85em; letter-spacing: 0.5px; color: #fff; }
        .hsic-hero { text-align: center; padding: 30px 20px; background: var(--bg-card-hover); border-radius: 14px; margin-bottom: 25px; border: 1px solid var(--border-color); position: relative; }
        .hsic-hero-score { font-size: 3.5em; font-weight: 700; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; line-height: 1.1; }
        .hsic-hero-label { font-size: 1.1em; margin-top: 6px; color: var(--text-secondary); }
        .hsic-hero-subtitle { font-size: 0.9em; margin-top: 12px; color: var(--text-muted); max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.6; }
        .hsic-tooltip { position: relative; cursor: help; border-bottom: 1px dashed var(--text-muted); }
        .hsic-tooltip .hsic-tooltip-text { visibility: hidden; opacity: 0; position: absolute; z-index: 10; bottom: 125%; left: 50%; transform: translateX(-50%); width: 300px; background: #1a1f2e; color: var(--text-secondary); padding: 14px; border-radius: 10px; font-size: 0.85em; line-height: 1.5; border: 1px solid var(--border-color); box-shadow: 0 8px 25px rgba(0,0,0,0.5); transition: opacity 0.2s; font-weight: 400; text-transform: none; letter-spacing: normal; }
        .hsic-tooltip .hsic-tooltip-text::after { content: ''; position: absolute; top: 100%; left: 50%; margin-left: -6px; border-width: 6px; border-style: solid; border-color: #1a1f2e transparent transparent transparent; }
        .hsic-tooltip:hover .hsic-tooltip-text { visibility: visible; opacity: 1; }
        .insight-card { background: var(--bg-card-hover); border-radius: 12px; padding: 22px; margin-bottom: 15px; border-left: 4px solid var(--accent-purple); }
        .insight-card-title { font-size: 0.8em; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; color: var(--text-muted); margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
        .insight-card-body { font-size: 0.95em; color: var(--text-secondary); line-height: 1.7; }
        .mirror-verdict { font-weight: 700; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-size: 1.3em; margin-bottom: 6px; }
        .tech-details-toggle { background: none; border: 1px solid var(--border-color); color: var(--text-muted); padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 0.85em; font-weight: 600; transition: all 0.2s; width: 100%; text-align: left; display: flex; justify-content: space-between; align-items: center; margin-top: 20px; }
        .tech-details-toggle:hover { border-color: var(--accent-cyan); color: var(--text-secondary); }
        .tech-details-content { max-height: 0; overflow: hidden; transition: max-height 0.3s ease-out; }
        .tech-details-content.open { max-height: 500px; }
        .tech-details-inner { padding: 20px 0 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; }
        .tech-detail-item { background: var(--bg-card); padding: 15px; border-radius: 8px; border: 1px solid var(--border-color); }
        .tech-detail-label { font-size: 0.75em; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px; }
        .tech-detail-value { font-size: 1.4em; font-weight: 700; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; color: var(--text-primary); }
        .tech-detail-note { font-size: 0.8em; color: var(--text-muted); margin-top: 4px; }

        /* ===== TRADING SIGNAL CARD STYLES ===== */
        .tsc { background: var(--bg-card); border-radius: 16px; padding: 0; border: 1px solid var(--border-color); overflow: hidden; animation: slideIn 0.5s ease; }
        .tsc-header { display: flex; justify-content: space-between; align-items: flex-start; padding: 24px 28px 16px; }
        .tsc-ticker { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-size: 1.8em; font-weight: 700; color: var(--text-primary); letter-spacing: -0.5px; }
        .tsc-price-row { display: flex; align-items: center; gap: 10px; margin-top: 4px; }
        .tsc-price { font-size: 1.05em; color: var(--text-secondary); font-weight: 500; }
        .tsc-change { font-size: 0.9em; font-weight: 600; padding: 2px 8px; border-radius: 4px; }
        .tsc-change.up { color: var(--accent-green); background: rgba(6, 255, 165, 0.1); }
        .tsc-change.down { color: var(--danger); background: rgba(239, 68, 68, 0.1); }
        .tsc-header-right { display: flex; align-items: center; gap: 10px; }
        .tsc-badge { font-size: 0.85em; font-weight: 700; padding: 8px 20px; border-radius: 8px; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; letter-spacing: 0.5px; }
        .tsc-badge-BUY { background: #10b981; color: white; }
        .tsc-badge-SELL { background: #ef4444; color: white; }
        .tsc-badge-HOLD { background: #f59e0b; color: white; }
        .tsc-menu-btn { background: rgba(255,255,255,0.06); border: 1px solid var(--border-color); border-radius: 8px; padding: 8px 12px; color: var(--text-muted); cursor: pointer; font-size: 1.1em; transition: all 0.2s; }
        .tsc-menu-btn:hover { background: rgba(255,255,255,0.1); color: var(--text-primary); }
        .tsc-body { padding: 0 28px 28px; }
        .tsc-setup-banner { background: linear-gradient(135deg, rgba(0, 217, 255, 0.15), rgba(157, 78, 221, 0.15)); border: 1px solid rgba(0, 217, 255, 0.25); color: var(--text-primary); padding: 12px 20px; border-radius: 10px; text-align: center; font-weight: 600; font-size: 0.95em; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; margin-bottom: 20px; }
        .tsc-confidence-card { background: var(--bg-card-hover); border-radius: 12px; padding: 22px 24px; margin-bottom: 20px; border: 1px solid var(--border-color); }
        .tsc-confidence-top { display: flex; align-items: center; gap: 16px; margin-bottom: 4px; }
        .tsc-signal-label { font-size: 1.1em; font-weight: 700; padding: 8px 20px; border-radius: 6px; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
        .tsc-signal-label-BUY { background: #10b981; color: white; }
        .tsc-signal-label-SELL { background: #ef4444; color: white; }
        .tsc-signal-label-HOLD { background: #f59e0b; color: white; }
        .tsc-confidence-info { flex: 1; }
        .tsc-confidence-pct { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-size: 1.1em; font-weight: 600; }
        .tsc-confidence-pct span { color: var(--text-secondary); font-weight: 400; font-size: 0.85em; }
        .tsc-confidence-hint { color: var(--text-muted); font-size: 0.85em; margin-top: 2px; }
        .tsc-rr-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0; margin-bottom: 20px; background: var(--bg-card-hover); border-radius: 12px; border: 1px solid var(--border-color); overflow: visible; }
        .tsc-rr-item { padding: 18px 16px; text-align: left; border-right: 1px solid var(--border-color); }
        .tsc-rr-item:first-child { border-radius: 12px 0 0 12px; }
        .tsc-rr-item:last-child { border-right: none; border-radius: 0 12px 12px 0; }
        .tsc-rr-label { font-size: 0.75em; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 6px; }
        .tsc-rr-value { font-size: 1.5em; font-weight: 700; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
        .tsc-rr-value.green { color: var(--accent-green); }
        .tsc-rr-value.red { color: var(--danger); }
        .tsc-rr-value.neutral { color: var(--text-primary); }
        .tsc-rr-bar { height: 4px; border-radius: 2px; margin-top: 10px; display: flex; overflow: hidden; background: var(--bg-dark); }
        .tsc-rr-bar-fill { height: 100%; border-radius: 2px; }
        .tsc-why { background: var(--bg-card-hover); border-radius: 12px; padding: 22px 24px; margin-bottom: 20px; border: 1px solid var(--border-color); }
        .tsc-why h4 { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 700; font-size: 1.05em; color: var(--text-primary); margin-bottom: 10px; }
        .tsc-why p { color: var(--text-secondary); line-height: 1.7; font-size: 0.95em; }
        .tsc-calc { background: var(--bg-card-hover); border-radius: 12px; padding: 24px; margin-bottom: 20px; border: 1px solid var(--border-color); }
        .tsc-calc-header { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
        .tsc-calc-check { width: 20px; height: 20px; background: var(--accent-green); border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.7em; font-weight: 700; }
        .tsc-calc-title { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 600; font-size: 1em; }
        .tsc-calc-row { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; }
        .tsc-calc-input-wrap { flex: 1; position: relative; }
        .tsc-calc-input { width: 100%; padding: 14px 14px 14px 8px; border: 1px solid var(--border-color); border-radius: 8px; background: var(--bg-dark); color: var(--text-primary); font-size: 1em; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 600; transition: all 0.2s; }
        .tsc-calc-input:focus { outline: none; border-color: var(--accent-green); box-shadow: 0 0 0 3px rgba(6, 255, 165, 0.1); }
        .tsc-calc-info-box { background: var(--bg-dark); border: 1px solid var(--border-color); border-radius: 8px; padding: 14px 16px; display: flex; justify-content: space-between; align-items: center; }
        .tsc-calc-info-label { color: var(--text-muted); font-size: 0.85em; }
        .tsc-calc-info-value { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 700; font-size: 1.1em; }
        .tsc-calc-result { background: linear-gradient(135deg, rgba(6, 255, 165, 0.1), rgba(6, 255, 165, 0.05)); border: 1px solid rgba(6, 255, 165, 0.25); border-radius: 10px; padding: 14px 18px; display: flex; justify-content: space-between; align-items: center; margin-top: 4px; }
        .tsc-calc-result-label { color: var(--text-secondary); font-size: 0.9em; font-weight: 500; }
        .tsc-calc-result-value { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 700; font-size: 1.3em; color: var(--accent-green); }
        .tsc-calc-details { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 14px; }
        .tsc-calc-detail-item { background: var(--bg-dark); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; }
        .tsc-calc-detail-label { font-size: 0.72em; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px; }
        .tsc-calc-detail-value { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 700; font-size: 1em; color: var(--text-primary); }
        .tsc-accordion { margin-bottom: 16px; }
        .tsc-accordion-toggle { background: var(--bg-card-hover); border: 1px solid var(--border-color); color: var(--text-secondary); padding: 14px 20px; border-radius: 10px; cursor: pointer; font-size: 0.9em; font-weight: 600; transition: all 0.2s; width: 100%; text-align: left; display: flex; justify-content: space-between; align-items: center; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
        .tsc-accordion-toggle:hover { border-color: var(--accent-cyan); color: var(--text-primary); background: var(--bg-card-hover); }
        .tsc-accordion-toggle .tsc-arrow { transition: transform 0.3s; font-size: 0.8em; }
        .tsc-accordion-toggle.open .tsc-arrow { transform: rotate(180deg); }
        .tsc-accordion-content { max-height: 0; overflow: hidden; transition: max-height 0.4s ease-out; }
        .tsc-accordion-content.open { max-height: 2000px; }
        .tsc-accordion-inner { padding: 16px 0 0; }
        .tsc-tech-item { background: var(--bg-card-hover); border: 1px solid var(--border-color); border-radius: 10px; padding: 18px 20px; margin-bottom: 12px; }
        .tsc-tech-item-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .tsc-tech-item-name { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 700; font-size: 0.95em; color: var(--text-primary); }
        .tsc-tech-item-value { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 700; font-size: 1.1em; }
        .tsc-tech-item-explain { color: var(--text-secondary); font-size: 0.85em; line-height: 1.7; margin-top: 6px; }
        .tsc-tech-item-example { color: var(--text-muted); font-size: 0.8em; line-height: 1.6; margin-top: 8px; padding: 10px 12px; background: var(--bg-dark); border-radius: 6px; border-left: 3px solid var(--accent-purple); }
        .tsc-capital-display { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 700; font-size: 1.5em; color: var(--text-primary); text-align: right; min-width: 120px; }
        .tsc-tip { position: relative; cursor: help; }
        .tsc-tip .tsc-tip-text { visibility: hidden; opacity: 0; position: absolute; z-index: 20; bottom: calc(100% + 10px); left: 50%; transform: translateX(-50%); width: 340px; background: #1a1f2e; color: var(--text-secondary); padding: 16px; border-radius: 10px; font-size: 0.82em; line-height: 1.65; border: 1px solid var(--border-color); box-shadow: 0 10px 30px rgba(0,0,0,0.55); transition: opacity 0.2s, visibility 0.2s; font-weight: 400; text-transform: none; letter-spacing: normal; pointer-events: none; }
        .tsc-tip .tsc-tip-text::after { content: ''; position: absolute; top: 100%; left: 50%; margin-left: -7px; border-width: 7px; border-style: solid; border-color: #1a1f2e transparent transparent transparent; }
        .tsc-tip:hover .tsc-tip-text { visibility: visible; opacity: 1; pointer-events: auto; }
        .tsc-tip .tsc-tip-text strong { color: var(--text-primary); }
        .tsc-tip .tsc-tip-formula { display: block; margin-top: 8px; padding: 8px 10px; background: var(--bg-dark); border-radius: 6px; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-size: 0.95em; color: var(--accent-cyan); word-break: break-word; }
        .tsc-rr-item .tsc-rr-label { display: inline-flex; align-items: center; gap: 5px; }
        .tsc-rr-item .tsc-rr-label .tsc-tip-icon { display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; border-radius: 50%; border: 1px solid var(--text-muted); font-size: 0.65em; color: var(--text-muted); flex-shrink: 0; transition: border-color 0.2s, color 0.2s; }
        .tsc-tip:hover .tsc-tip-icon { border-color: var(--accent-cyan); color: var(--accent-cyan); }
        .tsc-auto-risk { display: flex; align-items: center; gap: 12px; padding: 14px 16px; background: var(--bg-dark); border: 1px solid var(--border-color); border-radius: 10px; margin-bottom: 16px; }
        .tsc-auto-risk-badge { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 700; font-size: 1.5em; color: var(--accent-green); min-width: 60px; }
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
        .scope-btn.active, .risk-btn.active { background: var(--accent-cyan); color: var(--bg-dark); border-color: var(--accent-cyan); }
        .scope-btn:hover, .risk-btn:hover { border-color: var(--accent-cyan); color: var(--accent-cyan); }
        .scope-btn.active:hover, .risk-btn.active:hover { color: var(--bg-dark); }
        .btn-group { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }
        .sector-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; margin-top: 10px; max-height: 300px; overflow-y: auto; padding-right: 5px; }
        .sector-grid label { display: flex; align-items: center; gap: 6px; cursor: pointer; padding: 6px 8px; border-radius: 4px; color: var(--text-secondary); font-size: 0.85em; transition: background 0.2s; }
        .sector-grid label:hover { background: var(--bg-card-hover); }
        .dividend-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
        .dividend-table th { padding: 12px 10px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--accent-cyan); font-weight: 600; text-transform: uppercase; font-size: 0.8em; letter-spacing: 0.5px; position: sticky; top: 0; background: var(--bg-card); }
        .dividend-table td { padding: 10px; border-bottom: 1px solid var(--border-color); }
        .dividend-table tr:hover { background: var(--bg-card-hover); }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 25px 0; }
        .summary-card { background: var(--bg-card-hover); padding: 22px; border-radius: 10px; text-align: center; border: 1px solid var(--border-color); transition: all 0.3s; }
        .summary-card:hover { border-color: var(--accent-cyan); transform: translateY(-3px); }
        .summary-value { font-size: 1.8em; font-weight: 700; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
        .summary-label { font-size: 0.85em; color: var(--text-secondary); margin-top: 8px; }
        .optimize-btn { width: 100%; padding: 16px; background: linear-gradient(135deg, var(--accent-green), #059669); color: white; border: none; border-radius: 8px; font-size: 1.1em; font-weight: 700; cursor: pointer; transition: all 0.3s; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; letter-spacing: 0.5px; }
        .optimize-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(6, 255, 165, 0.3); }
        .risk-desc { margin-top: 10px; padding: 12px; background: var(--bg-dark); border-radius: 6px; color: var(--text-muted); font-size: 0.85em; line-height: 1.5; border-left: 3px solid var(--accent-purple); }
        #capital-input { width: 100%; padding: 14px; border: 2px solid var(--border-color); border-radius: 8px; font-size: 1.1em; background: var(--bg-dark); color: var(--accent-green); font-weight: 600; transition: all 0.3s; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
        #capital-input:focus { outline: none; border-color: var(--accent-green); box-shadow: 0 0 0 3px rgba(6, 255, 165, 0.1); }
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
            .tab { font-size: 0.85em; padding: 12px 6px; }
        }
        @media (max-width: 400px) {
            .tab { font-size: 0.78em; padding: 10px 4px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Stock Analysis Pro</h1>
            <p>Advanced Trading Insights with AI-Powered Analysis</p>
            <div class="stock-count">🚀 Now analyzing ''' + str(len(ALL_VALID_TICKERS)) + '''+ NSE stocks across ''' + str(len(STOCKS)) + ''' sectors</div>
        </header>
        <main>
        <div class="tabs">
            <button class="tab active" onclick="switchTab('analysis', event)">Technical Analysis</button>
            <button class="tab" onclick="switchTab('regression', event)">Market Connection</button>
            <button class="tab" onclick="switchTab('dividend', event)">Dividend Analyzer</button>
        </div>
        <div id="analysis-tab" class="tab-content active">
            <div id="search-view">
                <div class="grid">
                    <div class="card">
                        <h2>🔍 Search Any NSE Stock</h2>
                        <input type="text" id="search" placeholder="Search TCS, RELIANCE, INFY, or any NSE stock...">
                        <div class="suggestions" id="suggestions"></div>
                    </div>
                    <div class="card">
                        <h2>📊 Browse by Sector</h2>
                        <div id="categories" style="max-height: 500px; overflow-y: auto;"></div>
                    </div>
                </div>
            </div>
            <div id="result-view">
                <button class="back-btn" onclick="goBack()">← Back to Search</button>
                <div id="result"></div>
            </div>
        </div>
        <div id="regression-tab" class="tab-content">
            <div class="card">
                <h2>📈 Market Connection Analysis</h2>
                <p style="color: var(--text-secondary); margin-bottom: 20px;">Find out how closely any NSE stock is tied to the Nifty 50, including hidden connections that simple charts don't show</p>
                <input type="text" id="regression-search" placeholder="Enter stock symbol (e.g., TCS, INFY, RELIANCE)">
                <div class="suggestions" id="regression-suggestions"></div>
                <button onclick="analyzeRegression()" style="margin-top: 15px; width: 100%; background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); color: white; font-weight: 600; padding: 14px;">Analyze Connection</button>
            </div>
            <div class="card" style="margin-top: 20px; border-left: 3px solid var(--accent-purple); padding: 20px 25px;">
                <h3 style="color: var(--accent-purple); margin-bottom: 10px; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;">How to read your results</h3>
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
                    <p style="color: var(--text-secondary); margin-bottom: 15px; font-size: 0.9em;">Select which stocks to scan for dividend yields</p>
                    <div class="btn-group">
                        <button class="scope-btn active" onclick="setScope('all', this)">All Stocks</button>
                        <button class="scope-btn" onclick="setScope('nifty50', this)">Nifty 50</button>
                        <button class="scope-btn" onclick="setScope('custom', this)">Custom Sectors</button>
                    </div>
                    <div id="dividend-search-wrap" style="margin-top: 15px;">
                        <input type="text" id="dividend-search" placeholder="Search stocks (e.g., TCS, RELIANCE, INFY)..." style="width: 100%; padding: 12px 14px; border: 2px solid var(--border-color); border-radius: 8px; font-size: 0.95em; background: var(--bg-dark); color: var(--text-primary); transition: all 0.3s;">
                        <div class="suggestions" id="dividend-suggestions"></div>
                    </div>
                    <div id="nifty50-checkboxes" style="display:none; margin-top: 15px;">
                        <div style="margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border-color);">
                            <label style="cursor: pointer; color: var(--accent-cyan); font-weight: 600; font-size: 0.9em;">
                                <input type="checkbox" id="select-all-nifty" checked onchange="toggleAllNifty(this.checked)"> Select All Nifty 50
                            </label>
                        </div>
                        <div id="nifty50-grid" class="sector-grid"></div>
                    </div>
                    <div id="sector-checkboxes" style="display:none; margin-top: 15px;">
                        <div style="margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border-color);">
                            <label style="cursor: pointer; color: var(--accent-cyan); font-weight: 600; font-size: 0.9em;">
                                <input type="checkbox" id="select-all-sectors" onchange="toggleAllSectors(this.checked)"> Select All Sectors
                            </label>
                        </div>
                        <div id="sector-grid" class="sector-grid"></div>
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
                    <button class="optimize-btn" onclick="analyzeDividends()">Scan Dividends & Optimize Portfolio</button>
                </div>
            </div>
            <div id="dividend-results"></div>
        </div>
        </main>
    </div>
    <script>
        const stocks = ''' + str(STOCKS).replace("'", '"') + ''';
        const nifty50List = ''' + json.dumps(NIFTY_50_STOCKS) + ''';
        const tickerNames = ''' + json.dumps(TICKER_TO_NAME) + ''';
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
        let currentTab = 'analysis';
        let loadedTabs = new Set();
        function ensureTabLoaded(tab) {
            if (loadedTabs.has(tab)) return;
            if (tab === 'analysis') {
                const cat = document.getElementById('categories');
                Object.entries(stocks).forEach(([name, list]) => {
                    let html = `<div class="category"><h4>${name} (${list.length})</h4><div class="stocks">`;
                    list.slice(0, 30).forEach(s => html += `<button onclick="analyze('${s}')">${s}</button>`);
                    html += '</div></div>';
                    cat.innerHTML += html;
                });
                setupAutocomplete('search', 'suggestions', 'analyze');
            } else if (tab === 'regression') {
                setupAutocomplete('regression-search', 'regression-suggestions', 'analyzeRegression');
                document.getElementById('regression-search').addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') analyzeRegression();
                });
            } else if (tab === 'dividend') {
                setupDividendSearch();
            }
            loadedTabs.add(tab);
        }
        function switchTab(tab, event) {
            currentTab = tab;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tab + '-tab').classList.add('active');
            ensureTabLoaded(tab);
        }
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
                    else return `<button onclick="analyze('${s}')">${label}</button>`;
                }).join('');
            });
        }
        function init() {
            ensureTabLoaded('analysis');
        }
        function setupDividendSearch() {
            const input = document.getElementById('dividend-search');
            if (!input) return;
            const sug = document.getElementById('dividend-suggestions');
            input.addEventListener('input', () => {
                const raw = input.value.trim();
                const q = raw.toUpperCase();
                if (q.length === 0) { sug.innerHTML = ''; return; }
                const all = [...new Set(Object.values(stocks).flat())];
                const filtered = all.filter(s => {
                    const name = getStockName(s).toUpperCase();
                    return s.includes(q) || name.includes(q);
                }).slice(0, 12);
                sug.innerHTML = filtered.map(s =>
                    `<button onclick="scanStockSector('${s}')">${s} <span style='font-size:0.8em;color:var(--text-muted);'>${getStockName(s)}</span></button>`
                ).join('');
            });
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    const q = input.value.toUpperCase().trim();
                    if (q) scanStockSector(q);
                }
            });
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
                            <div class="tsc-ticker">${symbol}</div>
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
                            <h4>Why This Makes Sense</h4>
                            <p>${s.why_makes_sense || s.rec}</p>
                        </div>

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
                                            <strong>What is MACD?</strong> Imagine two runners - one fast and one slow. MACD tracks the gap between them. When the fast runner pulls ahead (Bullish), it means momentum is building upward - like a car accelerating. When the slow runner catches up (Bearish), the stock is losing steam. It's one of the most reliable ways to spot when a trend is gaining or losing strength.
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
                                            <strong>What are Moving Averages?</strong> A moving average smooths out daily price noise to show the real trend. The 20-day SMA shows the short-term trend (like last month's direction), while the 50-day SMA shows the bigger picture. When the price is above both, it's like a boat sailing with the current - the trend is your friend. Below both means you're swimming against the tide.
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
                                    </div>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            `;
            document.getElementById('result').innerHTML = html;
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
            const marketInfo = data.market_source ? `<div style="background: rgba(0, 217, 255, 0.1); padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 3px solid var(--accent-cyan);"><strong>📊 Market Benchmark:</strong> ${data.market_source} ${data.market_source !== 'Nifty 50 Index' ? '<br><small style="color: var(--text-muted);">Note: Using alternative benchmark due to Nifty 50 data availability.</small>' : ''}</div>` : '';
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
                                <div style="font-size:1.4em; font-weight:700; font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif; color:var(--text-primary);">${scorePercent}%</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="font-size:0.7em; text-transform:uppercase; color:var(--text-muted); letter-spacing:0.5px;">
                                    <span class="hsic-tooltip">Correlation<span class="hsic-tooltip-text">How closely the stock's daily returns move with the market. +1 = perfect match, 0 = no pattern, -1 = opposite.</span></span>
                                </div>
                                <div style="font-size:1.4em; font-weight:700; font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif; color:var(--text-primary);">${data.correlation >= 0 ? '+' : ''}${data.correlation.toFixed(2)}</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="font-size:0.7em; text-transform:uppercase; color:var(--text-muted); letter-spacing:0.5px;">
                                    <span class="hsic-tooltip">Beta<span class="hsic-tooltip-text">If Nifty 50 moves 1%, this stock historically moves about ${Math.abs(data.beta).toFixed(1)}% in the ${data.beta >= 0 ? 'same' : 'opposite'} direction. Beta > 1 means it amplifies market moves.</span></span>
                                </div>
                                <div style="font-size:1.4em; font-weight:700; font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif; color:var(--text-primary);">${data.beta.toFixed(2)}</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="font-size:0.7em; text-transform:uppercase; color:var(--text-muted); letter-spacing:0.5px;">
                                    <span class="hsic-tooltip">Downside Beta<span class="hsic-tooltip-text">Same as beta, but measured only on days when the market fell. A high downside beta means the stock tends to fall harder than the market during sell-offs. This is the most relevant measure for crash protection.</span></span>
                                </div>
                                <div style="font-size:1.4em; font-weight:700; font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif; color:${data.downside_beta > 1.2 ? '#ff6b6b' : data.downside_beta > 0.8 ? '#ffa94d' : 'var(--text-primary)'};">${data.downside_beta.toFixed(2)}</div>
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
                            <div style="font-weight: 700; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-size: 1.3em; margin-bottom: 6px; color: ${protColor};">
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
            document.getElementById('regression-result').innerHTML = html;
        }
        function goBack() {
            document.getElementById('search-view').style.display = 'block';
            document.getElementById('result-view').style.display = 'none';
            document.getElementById('search').value = '';
            document.getElementById('suggestions').innerHTML = '';
        }
        let dividendScope = 'all';
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
        function setScope(scope, btn) {
            dividendScope = scope;
            document.querySelectorAll('.scope-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('sector-checkboxes').style.display = scope === 'custom' ? 'block' : 'none';
            document.getElementById('nifty50-checkboxes').style.display = scope === 'nifty50' ? 'block' : 'none';
            document.getElementById('dividend-search-wrap').style.display = scope === 'all' ? 'block' : 'none';
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
            tbody.innerHTML = liveDividendEntries.map((s, idx) => `
                <tr>
                    <td>${idx + 1}. ${s.symbol}<br><span style="font-size:0.8em; color: var(--text-muted);">${getStockName(s.symbol)}</span></td>
                    <td style="font-size:0.8em; color: var(--text-muted);">${getStockSector(s.symbol)}</td>
                    <td style="text-align: right;">${fmt(s.price)}</td>
                    <td style="text-align: right;">${fmt(s.annual_dividend)}</td>
                    <td style="color: var(--accent-green); font-weight: 600;">${s.dividend_yield}%</td>
                    <td>${s.volatility}%</td>
                </tr>`).join('');
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
            if (dividendScope === 'all') { sectors = 'all'; dividendSectorLabel = 'All NSE'; }
            else if (dividendScope === 'nifty50') {
                const checked = document.querySelectorAll('.nifty-cb:checked');
                if (checked.length === 0) { alert('Please select at least one Nifty 50 stock'); return; }
                symbolsParam = Array.from(checked).map(c => c.value).join(',');
                dividendSectorLabel = 'Nifty 50';
            }
            else if (dividendScope === 'custom') {
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
                    <h3 style="color: var(--accent-purple); margin: 20px 0 10px; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 700;">Top Dividend Payers (Live)</h3>
                    <div style="overflow-x: auto; max-height: 400px; border: 1px solid var(--border-color); border-radius: 8px;">
                        <table class="dividend-table">
                            <thead><tr>
                                <th>Stock</th><th>Sector</th><th style="text-align:right;">Price (INR)</th><th style="text-align:right;">Annual Div (INR)</th><th>Div Yield</th><th>Volatility</th>
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
                return `<tr style="${isHL ? 'background: rgba(0,217,255,0.1); border-left: 3px solid var(--accent-cyan);' : ''}">
                    <td>${idx + 1}. ${s.symbol}<br><span style="font-size:0.8em; color: var(--text-muted);">${getStockName(s.symbol)}</span></td>
                    <td style="font-size:0.85em; color: var(--text-muted);">${getStockSector(s.symbol)}</td>
                    <td style="text-align: right;">${fmt(s.price)}</td>
                    <td style="text-align: right;">${fmt(s.annual_dividend)}</td>
                    <td style="color: var(--accent-green); font-weight: 600;">${s.dividend_yield}%</td>
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
                    <h3 style="color: var(--accent-cyan); margin: 30px 0 15px; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 700;">Optimized Allocation</h3>
                    <div style="overflow-x: auto;">
                        <table class="dividend-table">
                            <thead><tr>
                                <th>Stock</th><th>Sector</th><th>Weight</th><th style="text-align:right;">Shares</th><th style="text-align:right;">Price (INR)</th><th style="text-align:right;">Investment (INR)</th><th>Div Yield</th><th style="text-align:right;">Expected Div (INR)</th><th>Volatility</th>
                            </tr></thead>
                            <tbody>${allocRows}</tbody>
                        </table>
                    </div>
                    <h3 style="color: var(--accent-purple); margin: 35px 0 15px; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 700;">All Dividend-Paying Stocks (${data.all_dividend_stocks.length} shown)</h3>
                    ${data.dividend_results_truncated ? `<div style="margin-bottom: 10px; color: var(--warning); font-size: 0.85em;">Showing top ${data.all_dividend_stocks.length} dividend payers to reduce memory usage. ${data.dividend_stocks_found} total dividend-paying stocks found.</div>` : ''}
                    <div style="overflow-x: auto; max-height: 400px; border: 1px solid var(--border-color); border-radius: 8px;">
                        <table class="dividend-table">
                            <thead><tr>
                                <th>Stock</th><th>Sector</th><th style="text-align:right;">Price (INR)</th><th style="text-align:right;">Annual Div (INR)</th><th>Div Yield</th><th>Volatility</th>
                            </tr></thead>
                            <tbody>${allStockRows}</tbody>
                        </table>
                    </div>
                </div>`;
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
                    <h3 style="color: var(--accent-cyan); margin: 20px 0 10px; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 700;">Top Allocation (Live)</h3>
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
            Object.keys(stocks).forEach(sector => {
                const label = document.createElement('label');
                label.innerHTML = '<input type="checkbox" class="sector-cb" value="' + sector + '"> ' + sector + ' (' + stocks[sector].length + ')';
                grid.appendChild(label);
            });
        }
        window.addEventListener('DOMContentLoaded', () => { init(); initDividendSectors(); initNifty50Checkboxes(); setupCapitalInput(); });
    </script>
</body>
</html>'''
    return html

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
                    data = yf.download(
                        tickers=tickers,
                        period='1y',
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
                            if 'Stock Splits' in data.columns.get_level_values(0):
                                splits = data['Stock Splits'][ticker_symbol].dropna()
                            elif 'Splits' in data.columns.get_level_values(0):
                                splits = data['Splits'][ticker_symbol].dropna()
                            else:
                                splits = pd.Series(dtype=float)
                        else:
                            close_series = data['Close'].dropna()
                            dividends = data['Dividends'].dropna() if 'Dividends' in data.columns else pd.Series(dtype=float)
                            if 'Stock Splits' in data.columns:
                                splits = data['Stock Splits'].dropna()
                            elif 'Splits' in data.columns:
                                splits = data['Splits'].dropna()
                            else:
                                splits = pd.Series(dtype=float)
                        if close_series.empty or len(close_series) < 10:
                            payload = {'type': 'progress', 'scanned': scanned, 'dividend_found': dividend_found}
                            yield f"data: {json.dumps(payload)}\n\n"
                            continue
                        current_price = float(close_series.iloc[-1])
                        if current_price <= 0:
                            payload = {'type': 'progress', 'scanned': scanned, 'dividend_found': dividend_found}
                            yield f"data: {json.dumps(payload)}\n\n"
                            continue
                        annual_dividend = _compute_split_adjusted_dividend(dividends, splits)
                        if annual_dividend <= 0:
                            payload = {'type': 'progress', 'scanned': scanned, 'dividend_found': dividend_found}
                            yield f"data: {json.dumps(payload)}\n\n"
                            continue
                        dividend_yield = (annual_dividend / current_price) * 100
                        returns = close_series.pct_change().dropna()
                        volatility = float(returns.std() * np.sqrt(252) * 100) if len(returns) > 5 else 0.0
                        entry = {
                            'symbol': symbol,
                            'price': round(current_price, 2),
                            'annual_dividend': round(annual_dividend, 2),
                            'dividend_yield': round(dividend_yield, 2),
                            'volatility': round(volatility, 2)
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

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
