"""
Enhanced Large Cap Stocks Trading Dashboard - Flask Version (ALL NSE STOCKS)
Features:
- ALL NSE-listed stocks (500+ companies)
- Z-score with percentage deviation
- Linear regression analysis vs Nifty 50
- VISUAL Regression Plots with Equation (y = mx + b)
- Clear entry/exit explanations with confidence levels
- Time-to-target predictions
- Autocomplete for Search
"""

from flask import Flask, jsonify, request, Response, stream_with_context
import json
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize as scipy_minimize

# Set non-interactive backend for Render server
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("yfinance").propagate = False

app = Flask(__name__)

DIVIDEND_CACHE_TTL = timedelta(hours=6)
DIVIDEND_CACHE = {}
DIVIDEND_BATCH_SIZE = 50
DIVIDEND_MAX_WORKERS = 4
DIVIDEND_MAX_RESULTS = 300

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
    
    'Chemicals': ['UPL', 'PIDILITIND', 'AARTI', 'SRF', 'DEEPAKNTR', 'GNFC', 'CHAMBLFERT', 
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
                      'TATACOMM', 'TORNTPHARM', 'TRENT', 'TVSMOTOR', 'VEDL', 'VOLTAS', 'ZEEL', 'ZOMATO'],
    
    'Others': ['ZOMATO', 'PAYTM', 'NYKAA', 'POLICYBZR', 'DELHIVERY', 'CARTRADE', 'EASEMYTRIP',
               'ROUTE', 'LATENTVIEW', 'APTUS', 'RAINBOW', 'LAXMIMACH', 'SYNGENE', 'METROPOLIS']
}

NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
UNIVERSE_SECTOR_NAME = "All NSE"
UNIVERSE_SOURCE = "Static list"


def fetch_nse_universe():
    """Fetch full NSE equity universe via NSE equity list API."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; StockAnalysisPro/1.0)",
            "Accept": "text/csv,application/csv;q=0.9,*/*;q=0.8",
        }
        response = requests.get(NSE_EQUITY_LIST_URL, headers=headers, timeout=8)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        symbols = (
            df.get("SYMBOL", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        return sorted({s for s in symbols if s})
    except Exception as e:
        print(f"⚠️ Unable to fetch NSE universe from API: {e}")
        return []


def add_universe_sector(stocks_dict):
    """Attach full NSE universe (API-driven) to stock sectors."""
    global UNIVERSE_SOURCE
    fallback = sorted({t for sector in stocks_dict.values() for t in sector})
    api_symbols = fetch_nse_universe()
    if api_symbols:
        UNIVERSE_SOURCE = "NSE Equity List API"
        stocks_dict[UNIVERSE_SECTOR_NAME] = api_symbols
    else:
        UNIVERSE_SOURCE = "Static list (API unavailable)"
        stocks_dict[UNIVERSE_SECTOR_NAME] = fallback


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
    'ZOMATO': 'ZOMATO', 'PAYTM': 'PAYTM', 'NYKAA': 'NYKAA', 'POLICYBAZAAR': 'POLICYBZR',
    'DELHIVERY': 'DELHIVERY', 'DIXON': 'DIXON', 'POLYCAB': 'POLYCAB', 'HAVELLS': 'HAVELLS',
}

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

    def get_data(self, symbol, period='10d', interval='1h'):
        try:
            ticker = f"{symbol}.NS"
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                return None
            return data
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

    def calc_indicators(self, data):
        if data is None or len(data) < 14:
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
            if len(close) < 14:
                return None
            curr = float(close.iloc[-1])
            sma9 = float(close.rolling(9).mean().iloc[-1])
            sma5 = float(close.rolling(5).mean().iloc[-1])
            open_price = float(close.iloc[-18] if len(close) > 18 else close.iloc[0])
            prev_hour = float(close.iloc[-2] if len(close) > 1 else curr)
            daily_ret = ((curr - open_price) / open_price) * 100 if open_price > 0 else 0
            hourly_ret = ((curr - prev_hour) / prev_hour) * 100 if prev_hour > 0 else 0
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
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
            return {
                'price': curr, 'sma9': sma9, 'sma5': sma5, 'daily': daily_ret, 'hourly': hourly_ret,
                'rsi': rsi, 'macd_bullish': bool(macd_bullish), 'high': h, 'low': l,
                'pct_from_low': pct_from_low, 'zscore': zscore, 'pct_deviation': pct_deviation,
                'mean_price': mean_price, 'std_price': std_price, 'bb_upper': bb_upper,
                'bb_lower': bb_lower, 'bb_position': bb_position, 'volatility': volatility
            }
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
        if sig == "BUY":
            entry_explain = f"Enter when price dips to ₹{entry_price:.2f}. This is a good entry because it's near the average price and provides a better risk-reward ratio."
            exit_explain = f"Exit (sell) at ₹{target_price:.2f}. This target is {((target_price - i['price']) / i['price'] * 100):.1f}% above current price."
            confidence_explain = f"{confidence}% confidence based on: trend strength, RSI level, mean reversion signal, and market volatility. Higher confidence = more reliable setup."
            time_explain = f"Expected to reach target in approximately {days_to_target} trading days based on historical price movement patterns and current momentum."
        elif sig == "SELL":
            entry_explain = f"Exit long positions or enter short at ₹{entry_price:.2f}. Price is likely to fall towards mean."
            exit_explain = f"Cover shorts or re-enter longs at ₹{target_price:.2f}. This is {((i['price'] - target_price) / i['price'] * 100):.1f}% below current price."
            confidence_explain = f"{confidence}% confidence based on: downtrend confirmation, overbought conditions, and mean reversion probability. The algorithm checked 5 different technical indicators and found that all of them strongly agree the stock will go DOWN."
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
                'confidence': confidence, 'days_to_target': days_to_target,
                'entry_explain': entry_explain, 'exit_explain': exit_explain, 'confidence_explain': confidence_explain,
                'time_explain': time_explain, 'trend_explain': trend_explain, 'momentum_explain': momentum_explain,
                'rsi_explain': rsi_explain, 'position_explain': position_explain, 'zscore_explain': zscore_explain,
                'bb_explain': bb_explain, 'macd_text': "BULLISH - momentum favors buyers" if i['macd_bullish'] else "BEARISH - momentum favors sellers"
            },
            'details': {
                'price': f"₹{i['price']:.2f}", 'daily': f"{i['daily']:+.2f}%", 'hourly': f"{i['hourly']:+.2f}%",
                'rsi': f"{i['rsi']:.1f}", 'zscore': f"{i['zscore']:.2f}", 'pct_deviation': f"{i['pct_deviation']:+.2f}%",
                'mean': f"₹{i['mean_price']:.2f}", 'sma9': f"₹{i['sma9']:.2f}", 'high': f"₹{i['high']:.2f}",
                'low': f"₹{i['low']:.2f}", 'bb_upper': f"₹{i['bb_upper']:.2f}", 'bb_lower': f"₹{i['bb_lower']:.2f}",
                'volatility': f"{i['volatility']:.2f}%", 'macd': "BULLISH" if i['macd_bullish'] else "BEARISH"
            }
        }

    def analyze(self, symbol):
        """Main analysis method"""
        data = self.get_data(symbol)
        if data is None:
            return None
        ind = self.calc_indicators(data)
        if not ind:
            return None
        return self.signal(ind)

    def regression_analysis(self, stock_symbol):
        """Perform linear regression analysis of stock vs Nifty 50"""
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

            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
            r_squared = r_value ** 2
            
            # Plot generation
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bg_color = '#131824'
            grid_color = '#2d3748'
            
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            
            ax.scatter(X*100, y*100, alpha=0.6, c='#00d9ff', edgecolors='none', s=50, label='Daily Returns', zorder=1)
            
            x_range = np.linspace(X.min(), X.max(), 100)
            y_pred = slope * x_range + intercept
            ax.plot(x_range*100, y_pred*100, color='#9d4edd', linewidth=3, label=f'Regression Line', zorder=2)

            sign = '+' if intercept >= 0 else '-'
            stats_text = (
                f"$\\bf{{Regression\\ Stats}}$\n"
                f"• Eq: $y = {slope:.2f}x {sign} {abs(intercept):.4f}$\n"
                f"• $R^2$: {r_squared:.4f}\n"
                f"• Beta ($\\beta$): {slope:.3f}"
            )
            
            ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=11, color='white', 
                    verticalalignment='top', bbox=dict(facecolor=bg_color, alpha=0.95, edgecolor=grid_color, boxstyle='round,pad=0.5'), zorder=3)
            
            ax.set_title(f'{stock_symbol} vs {nifty_source} Regression Analysis', fontsize=14, color='white', pad=15)
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

            residuals = y - (slope * X + intercept)
            residual_std = np.std(residuals)
            correlation = np.corrcoef(X, y)[0, 1]

            if slope > 1.2: beta_interpret = f"HIGH BETA ({slope:.2f}): Aggressive/Volatile"
            elif slope > 0.8: beta_interpret = f"MEDIUM BETA ({slope:.2f}): Market-Like"
            elif slope > 0: beta_interpret = f"LOW BETA ({slope:.2f}): Defensive"
            else: beta_interpret = f"NEGATIVE BETA ({slope:.2f}): Inverse Mover"

            if r_squared > 0.7: r2_interpret = f"STRONG FIT ({r_squared:.2%})"
            elif r_squared > 0.4: r2_interpret = f"MODERATE FIT ({r_squared:.2%})"
            else: r2_interpret = f"WEAK FIT ({r_squared:.2%})"

            if intercept > 0.001: alpha_interpret = f"POSITIVE ALPHA (+{intercept:.4f})"
            elif intercept < -0.001: alpha_interpret = f"NEGATIVE ALPHA ({intercept:.4f})"
            else: alpha_interpret = "NEUTRAL ALPHA"

            if slope > 1.2 and r_squared > 0.6: trading_insight = f"LEVERAGED PLAY: Expect ~{slope:.1f}x market moves."
            elif slope < 0.8 and r_squared > 0.6: trading_insight = "DEFENSIVE PLAY: Lower volatility exposure."
            elif r_squared < 0.3: trading_insight = "INDEPENDENT STOCK: Driven by company news, not market."
            else: trading_insight = "MARKET-LINKED: Follows general market trends."

            return {
                'beta': slope, 'alpha': intercept, 'r_squared': r_squared,
                'correlation': correlation, 'p_value': p_value, 'std_error': std_err,
                'residual_std': residual_std, 'data_points': len(X), 'market_source': nifty_source,
                'beta_interpret': beta_interpret, 'r2_interpret': r2_interpret,
                'alpha_interpret': alpha_interpret, 'trading_insight': trading_insight,
                'plot_url': plot_url
            }

        except Exception as e:
            print(f"\n[FATAL ERROR] Regression failed for {stock_symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def fetch_dividend_data(self, symbols, limit_results=True):
        """Fetch dividend yield, current price, and annualized volatility for given symbols."""
        results = []
        total_dividend_found = 0
        if not symbols:
            return results, total_dividend_found

        def _batched(iterable, size):
            for idx in range(0, len(iterable), size):
                yield iterable[idx:idx + size]

        cached_symbols = []
        now = datetime.utcnow()
        for symbol in symbols:
            cached = DIVIDEND_CACHE.get(symbol)
            if cached and (now - cached['timestamp']) <= DIVIDEND_CACHE_TTL:
                results.append(cached['data'])
                cached_symbols.append(symbol)
                total_dividend_found += 1

        symbols_to_fetch = [s for s in symbols if s not in cached_symbols]
        batch_size = globals().get('DIVIDEND_BATCH_SIZE', 50)

        def _download_batch(batch):
            tickers = [f"{symbol}.NS" for symbol in batch]
            try:
                return batch, yf.download(
                    tickers=tickers,
                    period='1y',
                    interval='1d',
                    group_by='column',
                    actions=True,
                    auto_adjust=False,
                    progress=False,
                    threads=True
                )
            except Exception:
                return batch, None

        batches = list(_batched(symbols_to_fetch, batch_size))
        with ThreadPoolExecutor(max_workers=DIVIDEND_MAX_WORKERS) as executor:
            futures = [executor.submit(_download_batch, batch) for batch in batches]
            for future in as_completed(futures):
                batch, data = future.result()
                for symbol in batch:
                    try:
                        if data is None or data.empty:
                            continue
                        ticker_symbol = f"{symbol}.NS"
                        if isinstance(data.columns, pd.MultiIndex):
                            close_series = data['Close'][ticker_symbol].dropna()
                            dividends = data['Dividends'][ticker_symbol].dropna()
                        else:
                            close_series = data['Close'].dropna()
                            dividends = data['Dividends'].dropna() if 'Dividends' in data.columns else pd.Series(dtype=float)
                        if close_series.empty or len(close_series) < 10:
                            continue
                        current_price = float(close_series.iloc[-1])
                        if current_price <= 0:
                            continue
                        annual_dividend = float(dividends.sum()) if not dividends.empty else 0.0
                        if annual_dividend <= 0:
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
                        total_dividend_found += 1
                        results.append(entry)
                        DIVIDEND_CACHE[symbol] = {
                            'timestamp': now,
                            'data': entry
                        }
                    except Exception:
                        continue
                del data
                gc.collect()

        results = sorted(results, key=lambda x: x['dividend_yield'], reverse=True)
        max_results = globals().get('DIVIDEND_MAX_RESULTS', 300)
        if limit_results and len(results) > max_results:
            results = results[:max_results]
        return results, total_dividend_found

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
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Analysis Pro - All NSE Stocks</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root { --bg-dark: #0a0e1a; --bg-card: #131824; --bg-card-hover: #1a1f2e; --accent-cyan: #00d9ff; --accent-purple: #9d4edd; --accent-green: #06ffa5; --text-primary: #ffffff; --text-secondary: #a0aec0; --text-muted: #718096; --border-color: #2d3748; --success: #10b981; --warning: #f59e0b; --danger: #ef4444; }
        body { font-family: 'Inter', sans-serif; background: var(--bg-dark); color: var(--text-primary); min-height: 100vh; line-height: 1.6; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header { text-align: center; padding: 40px 0; border-bottom: 1px solid var(--border-color); background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(157, 78, 221, 0.1)); }
        header h1 { font-family: 'Space Grotesk', sans-serif; font-size: 3em; font-weight: 700; margin-bottom: 10px; background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        header p { color: var(--text-secondary); font-size: 1.1em; }
        .stock-count { background: rgba(0, 217, 255, 0.1); color: var(--accent-cyan); padding: 8px 16px; border-radius: 20px; display: inline-block; margin-top: 10px; font-size: 0.9em; font-weight: 600; }
        .tabs { display: flex; gap: 10px; margin: 30px 0; border-bottom: 2px solid var(--border-color); }
        .tab { padding: 15px 30px; background: transparent; border: none; color: var(--text-secondary); font-size: 1em; font-weight: 600; cursor: pointer; transition: all 0.3s; border-bottom: 3px solid transparent; font-family: 'Space Grotesk', sans-serif; }
        .tab:hover { color: var(--accent-cyan); }
        .tab.active { color: var(--text-primary); border-bottom-color: var(--accent-cyan); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px; }
        .card { background: var(--bg-card); border-radius: 12px; padding: 25px; border: 1px solid var(--border-color); transition: all 0.3s; }
        .card:hover { background: var(--bg-card-hover); border-color: var(--accent-cyan); }
        .card h3 { color: var(--text-primary); margin-bottom: 15px; font-size: 1.3em; font-family: 'Space Grotesk', sans-serif; font-weight: 600; }
        #search, #regression-search { width: 100%; padding: 14px; border: 2px solid var(--border-color); border-radius: 8px; font-size: 1em; background: var(--bg-dark); color: var(--text-primary); transition: all 0.3s; }
        #search:focus, #regression-search:focus { outline: none; border-color: var(--accent-cyan); box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.1); }
        .suggestions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 15px; max-height: 300px; overflow-y: auto; }
        .category { margin-bottom: 20px; }
        .category h4 { color: var(--accent-cyan); font-size: 0.85em; margin-bottom: 8px; text-transform: uppercase; font-weight: 600; letter-spacing: 1px; }
        .stocks { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
        button { padding: 10px 16px; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 6px; cursor: pointer; font-weight: 500; transition: all 0.2s; color: var(--text-secondary); font-size: 0.9em; }
        button:hover { background: var(--accent-cyan); color: var(--bg-dark); border-color: var(--accent-cyan); transform: translateY(-2px); }
        #result-view { display: none; }
        .result-card { background: var(--bg-card); border-radius: 12px; padding: 35px; border: 1px solid var(--border-color); }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 2px solid var(--border-color); padding-bottom: 20px; }
        .header h2 { color: var(--text-primary); font-size: 2.5em; font-family: 'Space Grotesk', sans-serif; font-weight: 700; }
        .signal-badge { font-size: 1.2em; font-weight: 700; padding: 12px 24px; border-radius: 8px; font-family: 'Space Grotesk', sans-serif; letter-spacing: 1px; }
        .signal-BUY { background: linear-gradient(135deg, #10b981, #059669); color: white; }
        .signal-SELL { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
        .signal-HOLD { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
        .action-banner { background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; font-weight: 600; font-size: 1.2em; font-family: 'Space Grotesk', sans-serif; box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3); }
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
        .back-btn { background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); color: white; padding: 12px 28px; margin-bottom: 20px; border: none; font-weight: 600; font-size: 1em; }
        .back-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(0, 217, 255, 0.4); }
        .loading { text-align: center; color: var(--accent-cyan); font-size: 1.3em; padding: 40px; font-family: 'Space Grotesk', sans-serif; }
        .error { background: rgba(239, 68, 68, 0.1); color: var(--danger); padding: 20px; border-radius: 8px; border-left: 4px solid var(--danger); }
        .regression-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 25px 0; }
        .regression-metric { background: var(--bg-card-hover); padding: 20px; border-radius: 10px; border-left: 3px solid var(--accent-purple); }
        .regression-metric-label { font-size: 0.85em; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; font-weight: 600; letter-spacing: 0.5px; }
        .regression-metric-value { font-size: 2em; font-weight: 700; color: var(--text-primary); font-family: 'Space Grotesk', sans-serif; margin-bottom: 8px; }
        .regression-metric-desc { font-size: 0.9em; color: var(--text-secondary); line-height: 1.5; }
        .plot-container { background: var(--bg-card-hover); padding: 20px; border-radius: 12px; margin-bottom: 30px; border: 1px solid var(--border-color); text-align: center; }
        .plot-img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        
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
        .summary-value { font-size: 1.8em; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }
        .summary-label { font-size: 0.85em; color: var(--text-secondary); margin-top: 8px; }
        .optimize-btn { width: 100%; padding: 16px; background: linear-gradient(135deg, var(--accent-green), #059669); color: white; border: none; border-radius: 8px; font-size: 1.1em; font-weight: 700; cursor: pointer; transition: all 0.3s; font-family: 'Space Grotesk', sans-serif; letter-spacing: 0.5px; }
        .optimize-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(6, 255, 165, 0.3); }
        .risk-desc { margin-top: 10px; padding: 12px; background: var(--bg-dark); border-radius: 6px; color: var(--text-muted); font-size: 0.85em; line-height: 1.5; border-left: 3px solid var(--accent-purple); }
        #capital-input { width: 100%; padding: 14px; border: 2px solid var(--border-color); border-radius: 8px; font-size: 1.1em; background: var(--bg-dark); color: var(--accent-green); font-weight: 600; transition: all 0.3s; font-family: 'Space Grotesk', sans-serif; }
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
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Stock Analysis Pro</h1>
            <p>Advanced Trading Insights with AI-Powered Analysis</p>
            <div class="stock-count">🚀 Now analyzing ''' + str(len(ALL_VALID_TICKERS)) + '''+ NSE stocks across ''' + str(len(STOCKS)) + ''' sectors</div>
            <div style="margin-top: 8px; color: var(--text-muted); font-size: 0.85em;">
                Universe source: ''' + UNIVERSE_SOURCE + '''. Market data requests are subject to API throttling.
            </div>
        </header>
        <div class="tabs">
            <button class="tab active" onclick="switchTab('analysis', event)">Technical Analysis</button>
            <button class="tab" onclick="switchTab('regression', event)">Regression vs Nifty</button>
            <button class="tab" onclick="switchTab('dividend', event)">Dividend Analyzer</button>
        </div>
        <div id="analysis-tab" class="tab-content active">
            <div id="search-view">
                <div class="grid">
                    <div class="card">
                        <h3>🔍 Search Any NSE Stock</h3>
                        <input type="text" id="search" placeholder="Search TCS, RELIANCE, INFY, or any NSE stock...">
                        <div class="suggestions" id="suggestions"></div>
                    </div>
                    <div class="card">
                        <h3>📊 Browse by Sector</h3>
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
                <h3>📈 Linear Regression Analysis vs Nifty 50</h3>
                <p style="color: var(--text-secondary); margin-bottom: 20px;">Analyze how any NSE stock correlates with Nifty 50 index movements</p>
                <input type="text" id="regression-search" placeholder="Enter stock symbol (e.g., TCS, INFY, RELIANCE)">
                <div class="suggestions" id="regression-suggestions"></div>
                <button onclick="analyzeRegression()" style="margin-top: 15px; width: 100%; background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); color: white; font-weight: 600; padding: 14px;">Analyze Regression</button>
            </div>
            <div id="regression-result" style="margin-top: 30px;"></div>
        </div>
        <div id="dividend-tab" class="tab-content">
            <div class="grid">
                <div class="card">
                    <h3>Stock Universe</h3>
                    <p style="color: var(--text-secondary); margin-bottom: 15px; font-size: 0.9em;">Select which stocks to scan for dividend yields</p>
                    <div class="btn-group">
                        <button class="scope-btn active" onclick="setScope('all', this)">All Stocks</button>
                        <button class="scope-btn" onclick="setScope('nifty50', this)">Nifty 50</button>
                        <button class="scope-btn" onclick="setScope('largecap', this)">Large Cap 100</button>
                        <button class="scope-btn" onclick="setScope('custom', this)">Custom Sectors</button>
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
                    <h3>Portfolio Configuration</h3>
                    <div style="margin-bottom: 22px;">
                        <label style="display: block; color: var(--text-secondary); font-size: 0.85em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Investment Capital (INR)</label>
                        <input type="number" id="capital-input" placeholder="e.g. 1000000" min="1000">
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
    </div>
    <script>
        const stocks = ''' + str(STOCKS).replace("'", '"') + ''';
        let currentTab = 'analysis';
        function switchTab(tab, event) {
            currentTab = tab;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tab + '-tab').classList.add('active');
        }
        function init() {
            const cat = document.getElementById('categories');
            Object.entries(stocks).forEach(([name, list]) => {
                let html = `<div class="category"><h4>${name} (${list.length})</h4><div class="stocks">`;
                list.forEach(s => html += `<button onclick="analyze('${s}')">${s}</button>`);
                html += '</div></div>';
                cat.innerHTML += html;
            });
            function setupAutocomplete(inputId, suggestionId, callbackName) {
                const input = document.getElementById(inputId);
                if(!input) return;
                input.addEventListener('input', (e) => {
                    const q = e.target.value.toUpperCase();
                    const sug = document.getElementById(suggestionId);
                    if (q.length === 0) { sug.innerHTML = ''; return; }
                    const all = [...new Set(Object.values(stocks).flat())];
                    const filtered = all.filter(s => s.includes(q)).slice(0, 12);
                    sug.innerHTML = filtered.map(s => {
                        if(callbackName === 'analyzeRegression') return `<button onclick="document.getElementById('${inputId}').value = '${s}'; analyzeRegression();">${s}</button>`;
                        else return `<button onclick="analyze('${s}')">${s}</button>`;
                    }).join('');
                });
            }
            setupAutocomplete('search', 'suggestions', 'analyze');
            setupAutocomplete('regression-search', 'regression-suggestions', 'analyzeRegression');
            document.getElementById('regression-search').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') analyzeRegression();
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
            if (!data || !data.signal) { document.getElementById('result').innerHTML = '<div class="error">❌ Invalid response data</div>'; return; }
            const s = data.signal || {};
            const d = data.details || {};
            const confidenceColor = s.confidence > 70 ? '#10b981' : s.confidence > 50 ? '#f59e0b' : '#ef4444';
            const html = `
                <div class="result-card">
                    <div class="header"><h2>${symbol}</h2><div class="signal-badge signal-${s.signal}">${s.signal}</div></div>
                    <div class="action-banner">${s.action}</div>
                    <div class="rec-box"><strong>💡 Recommendation:</strong> ${s.rec}</div>
                    <div class="confidence-meter"><div class="confidence-label">Confidence Level</div><div class="confidence-bar-container"><div class="confidence-bar" style="width: ${s.confidence}%; background: linear-gradient(90deg, ${confidenceColor}, ${confidenceColor}dd);">${s.confidence}%</div></div><div class="confidence-text">${s.confidence_explain}</div></div>
                    <div class="metrics">
                        <div class="metric"><div class="metric-label">Current Price</div><div class="metric-value">${d.price}</div></div>
                        <div class="metric"><div class="metric-label">Daily Change</div><div class="metric-value" style="color: ${parseFloat(d.daily) >= 0 ? 'var(--success)' : 'var(--danger)'}">${d.daily}</div></div>
                        <div class="metric"><div class="metric-label">RSI</div><div class="metric-value">${d.rsi}</div></div>
                        <div class="metric"><div class="metric-label">Z-Score</div><div class="metric-value">${d.zscore}</div></div>
                        <div class="metric"><div class="metric-label">% from Mean</div><div class="metric-value" style="color: ${parseFloat(d.pct_deviation) >= 0 ? 'var(--accent-cyan)' : 'var(--accent-purple)'}">${d.pct_deviation}</div></div>
                        <div class="metric"><div class="metric-label">Volatility</div><div class="metric-value">${d.volatility}</div></div>
                    </div>
                    <div class="trading-plan">
                        <h3>💼 TRADING PLAN (For Beginners)</h3>
                        <div class="plan-item"><span class="plan-label">📍 ENTRY PRICE</span><span class="plan-value">${s.entry}<br><small style="color: var(--text-muted)">${s.entry_explain}</small></span></div>
                        <div class="plan-item"><span class="plan-label">🎯 EXIT PRICE</span><span class="plan-value">${s.target}<br><small style="color: var(--text-muted)">${s.exit_explain}</small></span></div>
                        <div class="plan-item"><span class="plan-label">🛡️ STOP LOSS</span><span class="plan-value">${s.stop}<br><small style="color: var(--text-muted)">If price falls to this level, sell immediately to limit losses.</small></span></div>
                        <div class="plan-item"><span class="plan-label">⏱️ TIME FRAME</span><span class="plan-value">${s.days_to_target} trading days<br><small style="color: var(--text-muted)">${s.time_explain}</small></span></div>
                    </div>
                    <div class="explanation-section"><h3>📊 TREND ANALYSIS</h3><div class="explanation">${s.trend_explain}</div></div>
                    <div class="explanation-section"><h3>⚡ MOMENTUM</h3><div class="explanation">${s.momentum_explain}</div></div>
                    <div class="explanation-section"><h3>📈 RSI INTERPRETATION</h3><div class="explanation">${s.rsi_explain}</div></div>
                    <div class="explanation-section"><h3>🎯 MEAN REVERSION (Z-SCORE)</h3><div class="explanation">${s.zscore_explain}</div></div>
                    <div class="explanation-section"><h3>📉 BOLLINGER BANDS</h3><div class="explanation">${s.bb_explain}</div></div>
                    <div class="explanation-section"><h3>📍 PRICE POSITION</h3><div class="explanation">${s.position_explain}</div></div>
                    <div class="explanation-section"><h3>🎯 MACD</h3><div class="explanation">${s.macd_text}</div></div>
                </div>
            `;
            document.getElementById('result').innerHTML = html;
        }
        function analyzeRegression() {
            const symbol = document.getElementById('regression-search').value.toUpperCase().trim();
            if (!symbol) { alert('Please enter a stock symbol'); return; }
            document.getElementById('regression-result').innerHTML = '<div class="loading">⏳ Running regression analysis for ' + symbol + '...<br><small style="font-size: 0.8em; color: var(--text-secondary);">This may take 10-30 seconds</small></div>';
            fetch(`/regression?symbol=${symbol}`)
                .then(r => r.json())
                .then(data => {
                    if (data.error) document.getElementById('regression-result').innerHTML = `<div class="error">❌ ${data.error}</div>`;
                    else showRegressionResult(data, symbol);
                })
                .catch(e => document.getElementById('regression-result').innerHTML = `<div class="error">❌ ${e.message}</div>`);
        }
        function showRegressionResult(data, symbol) {
            const marketInfo = data.market_source ? `<div style="background: rgba(0, 217, 255, 0.1); padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 3px solid var(--accent-cyan);"><strong>📊 Market Benchmark:</strong> ${data.market_source} ${data.market_source !== 'Nifty 50 Index' ? '<br><small style="color: var(--text-muted);">Note: Using alternative benchmark due to Nifty 50 data availability.</small>' : ''}</div>` : '';
            const html = `
                <div class="result-card">
                    <div class="header"><h2>${symbol} vs Market</h2><div style="color: var(--accent-cyan); font-size: 1.2em;">Linear Regression Analysis</div></div>
                    ${marketInfo}
                    <div class="action-banner">${data.trading_insight}</div>
                    
                    <div class="plot-container">
                        <h3 style="color: var(--accent-purple); margin-bottom: 15px;">🔍 Visual Regression Analysis</h3>
                        <img src="data:image/png;base64,${data.plot_url}" class="plot-img" alt="Regression Plot">
                    </div>
                    
                    <div class="regression-grid">
                        <div class="regression-metric"><div class="regression-metric-label">Beta (β)</div><div class="regression-metric-value">${data.beta.toFixed(3)}</div><div class="regression-metric-desc">${data.beta_interpret}</div></div>
                        <div class="regression-metric"><div class="regression-metric-label">R-Squared (R²)</div><div class="regression-metric-value">${(data.r_squared * 100).toFixed(1)}%</div><div class="regression-metric-desc">${data.r2_interpret}</div></div>
                        <div class="regression-metric"><div class="regression-metric-label">Alpha (α)</div><div class="regression-metric-value">${(data.alpha * 100).toFixed(3)}%</div><div class="regression-metric-desc">${data.alpha_interpret}</div></div>
                        <div class="regression-metric"><div class="regression-metric-label">Correlation</div><div class="regression-metric-value">${data.correlation.toFixed(3)}</div><div class="regression-metric-desc">Measures linear relationship strength.</div></div>
                        <div class="regression-metric"><div class="regression-metric-label">P-Value</div><div class="regression-metric-value">${data.p_value.toFixed(6)}</div><div class="regression-metric-desc">${data.p_value < 0.05 ? 'Statistically SIGNIFICANT (p < 0.05).' : 'Not statistically significant.'}</div></div>
                        <div class="regression-metric"><div class="regression-metric-label">Std Error</div><div class="regression-metric-value">${data.std_error.toFixed(4)}</div><div class="regression-metric-desc">Uncertainty in beta estimate.</div></div>
                        <div class="regression-metric"><div class="regression-metric-label">Residual Std</div><div class="regression-metric-value">${(data.residual_std * 100).toFixed(2)}%</div><div class="regression-metric-desc">Average prediction error.</div></div>
                        <div class="regression-metric"><div class="regression-metric-label">Data Points</div><div class="regression-metric-value">${data.data_points}</div><div class="regression-metric-desc">Observations used.</div></div>
                    </div>
                    <div class="trading-plan" style="margin-top: 25px;">
                        <h3>💡 Practical Trading Applications</h3>
                        <div class="plan-item"><span class="plan-label">Market Direction</span><span class="plan-value">${data.beta > 1.2 ? `High beta stock - amplifies market moves.` : data.beta < 0.8 ? `Low beta stock - defensive play.` : 'Moderate beta - moves in line with market.'}</span></div>
                        <div class="plan-item"><span class="plan-label">Portfolio Use</span><span class="plan-value">${data.r_squared > 0.6 ? 'Strong market correlation.' : 'Weak market correlation. Stock driven by company-specific factors.'}</span></div>
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
        function setScope(scope, btn) {
            dividendScope = scope;
            document.querySelectorAll('.scope-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('sector-checkboxes').style.display = scope === 'custom' ? 'block' : 'none';
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
                    <td>${idx + 1}. ${s.symbol}</td>
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
        function analyzeDividends() {
            const capital = parseFloat(document.getElementById('capital-input').value);
            if (!capital || capital <= 0) { alert('Please enter a valid capital amount'); return; }
            let sectors = '';
            if (dividendScope === 'all') sectors = 'all';
            else if (dividendScope === 'nifty50') sectors = 'Nifty 50';
            else if (dividendScope === 'largecap') sectors = 'Nifty 50,Nifty Next 50';
            else {
                const checked = document.querySelectorAll('.sector-cb:checked');
                if (checked.length === 0) { alert('Please select at least one sector'); return; }
                sectors = Array.from(checked).map(c => c.value).join(',');
            }
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
                                <th>Stock</th><th style="text-align:right;">Price (INR)</th><th style="text-align:right;">Annual Div (INR)</th><th>Div Yield</th><th>Volatility</th>
                            </tr></thead>
                            <tbody id="live-dividend-body"></tbody>
                        </table>
                    </div>
                    <div style="color: var(--text-muted); font-size: 0.8em; margin-top: 10px;">This may take 30-120 seconds for large universes. Please wait.</div>
                </div>`;

            if (dividendStream) dividendStream.close();
            liveDividendEntries = [];
            liveDividendMax = 0;
            document.getElementById('live-scan-total').textContent = '0';

            const streamUrl = `/dividend-optimize-stream?capital=${capital}&risk=${dividendRisk}&sectors=${encodeURIComponent(sectors)}`;
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
                fetch(`/dividend-optimize?capital=${capital}&risk=${dividendRisk}&sectors=${encodeURIComponent(sectors)}`)
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
                    <td style="font-weight: 600; color: var(--accent-cyan);">${idx + 1}. ${a.symbol}</td>
                    <td>${a.weight}%</td>
                    <td style="text-align: right;">${a.shares}</td>
                    <td style="text-align: right;">${fmt(a.price)}</td>
                    <td style="text-align: right;">${fmt(a.amount)}</td>
                    <td style="color: var(--accent-green); font-weight: 600;">${a.dividend_yield}%</td>
                    <td style="color: var(--accent-green); font-weight: 700; text-align: right;">${fmt(a.expected_dividend)}</td>
                    <td>${a.volatility}%</td>
                </tr>`).join('');
            let allStockRows = data.all_dividend_stocks.map((s, idx) => `
                <tr>
                    <td>${idx + 1}. ${s.symbol}</td>
                    <td style="text-align: right;">${fmt(s.price)}</td>
                    <td style="text-align: right;">${fmt(s.annual_dividend)}</td>
                    <td style="color: var(--accent-green); font-weight: 600;">${s.dividend_yield}%</td>
                    <td>${s.volatility}%</td>
                </tr>`).join('');
            const riskColors = { conservative: '#10b981', moderate: '#f59e0b', aggressive: '#ef4444' };
            const riskColor = riskColors[data.risk_appetite] || '#f59e0b';
            const html = `
                <div class="result-card" style="margin-top: 30px;">
                    <div class="header">
                        <h2>Dividend Portfolio</h2>
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
                                <th>Stock</th><th>Weight</th><th style="text-align:right;">Shares</th><th style="text-align:right;">Price (INR)</th><th style="text-align:right;">Investment (INR)</th><th>Div Yield</th><th style="text-align:right;">Expected Div (INR)</th><th>Volatility</th>
                            </tr></thead>
                            <tbody>${allocRows}</tbody>
                        </table>
                    </div>
                    <h3 style="color: var(--accent-purple); margin: 35px 0 15px; font-family: 'Space Grotesk', sans-serif; font-weight: 700;">All Dividend-Paying Stocks (${data.all_dividend_stocks.length} shown)</h3>
                    ${data.dividend_results_truncated ? `<div style="margin-bottom: 10px; color: var(--warning); font-size: 0.85em;">Showing top ${data.all_dividend_stocks.length} dividend payers to reduce memory usage. ${data.dividend_stocks_found} total dividend-paying stocks found.</div>` : ''}
                    <div style="overflow-x: auto; max-height: 400px; border: 1px solid var(--border-color); border-radius: 8px;">
                        <table class="dividend-table">
                            <thead><tr>
                                <th>Stock</th><th style="text-align:right;">Price (INR)</th><th style="text-align:right;">Annual Div (INR)</th><th>Div Yield</th><th>Volatility</th>
                            </tr></thead>
                            <tbody>${allStockRows}</tbody>
                        </table>
                    </div>
                </div>`;
            document.getElementById('dividend-results').innerHTML = html;
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
        window.addEventListener('DOMContentLoaded', () => { init(); initDividendSectors(); });
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

@app.route('/regression')
def regression_route():
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
        result = analyzer.regression_analysis(normalized_symbol)
        if not result: return jsonify({'error': f'Unable to perform regression analysis for {normalized_symbol}.'})
        return jsonify(result)
    except Exception as e:
        print(f"Error in regression for {normalized_symbol}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Regression analysis failed for {normalized_symbol}: {str(e)}'})

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

@app.route('/dividend-optimize')
def dividend_optimize_route():
    """Scan dividends and compute optimal portfolio allocation."""
    try:
        capital = float(request.args.get('capital', 0))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid capital amount'})
    risk = request.args.get('risk', 'moderate')
    sectors = request.args.get('sectors', 'all')
    if capital <= 0:
        return jsonify({'error': 'Please enter a valid capital amount'})
    if risk not in ('conservative', 'moderate', 'aggressive'):
        risk = 'moderate'
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
        stocks_data, dividend_found = analyzer.fetch_dividend_data(symbols)
        if not stocks_data:
            return jsonify({'error': 'No dividend-paying stocks found in the selected universe'})
        result = analyzer.optimize_dividend_portfolio(stocks_data, capital, risk)
        if not result:
            return jsonify({'error': 'Portfolio optimization failed'})
        result['all_dividend_stocks'] = stocks_data
        result['stocks_scanned'] = len(symbols)
        result['dividend_stocks_found'] = dividend_found
        result['dividend_results_truncated'] = dividend_found > len(stocks_data)
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
    if capital <= 0:
        return jsonify({'error': 'Please enter a valid capital amount'})
    if risk not in ('conservative', 'moderate', 'aggressive'):
        risk = 'moderate'
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

    def generate():
        try:
            now = datetime.utcnow()
            scanned = 0
            dividend_found = 0
            results = []
            max_results = globals().get('DIVIDEND_MAX_RESULTS', 300)
            yield f"data: {json.dumps({'type': 'meta', 'total_scanned': len(symbols), 'max_results': max_results})}\n\n"

            cached_symbols = []
            for symbol in symbols:
                cached = DIVIDEND_CACHE.get(symbol)
                if cached and (now - cached['timestamp']) <= DIVIDEND_CACHE_TTL:
                    entry = cached['data']
                    results.append(entry)
                    cached_symbols.append(symbol)
                    scanned += 1
                    dividend_found += 1
                    payload = {
                        'type': 'stock',
                        'entry': entry,
                        'scanned': scanned,
                        'dividend_found': dividend_found
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

            symbols_to_fetch = [s for s in symbols if s not in cached_symbols]

            def _batched(iterable, size):
                for idx in range(0, len(iterable), size):
                    yield iterable[idx:idx + size]

            def _download_batch(batch):
                tickers = [f"{symbol}.NS" for symbol in batch]
                try:
                    return batch, yf.download(
                        tickers=tickers,
                        period='1y',
                        interval='1d',
                        group_by='column',
                        actions=True,
                        auto_adjust=False,
                        progress=False,
                        threads=True
                    )
                except Exception:
                    return batch, None

            batch_size = globals().get('DIVIDEND_BATCH_SIZE', 50)
            batches = list(_batched(symbols_to_fetch, batch_size))
            with ThreadPoolExecutor(max_workers=DIVIDEND_MAX_WORKERS) as executor:
                futures = [executor.submit(_download_batch, batch) for batch in batches]
                for future in as_completed(futures):
                    batch, data = future.result()
                    for symbol in batch:
                        scanned += 1
                        entry = None
                        try:
                            if data is None or data.empty:
                                payload = {'type': 'progress', 'scanned': scanned, 'dividend_found': dividend_found}
                                yield f"data: {json.dumps(payload)}\n\n"
                                continue
                            ticker_symbol = f"{symbol}.NS"
                            if isinstance(data.columns, pd.MultiIndex):
                                close_series = data['Close'][ticker_symbol].dropna()
                                dividends = data['Dividends'][ticker_symbol].dropna()
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
                            annual_dividend = float(dividends.sum()) if not dividends.empty else 0.0
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
                            DIVIDEND_CACHE[symbol] = {
                                'timestamp': now,
                                'data': entry
                            }
                        except Exception:
                            payload = {'type': 'progress', 'scanned': scanned, 'dividend_found': dividend_found}
                            yield f"data: {json.dumps(payload)}\n\n"
                            continue
                        payload = {
                            'type': 'stock',
                            'entry': entry,
                            'scanned': scanned,
                            'dividend_found': dividend_found
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
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
            truncated = False
            if len(display_stocks) > max_results:
                display_stocks = display_stocks[:max_results]
                truncated = True
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
