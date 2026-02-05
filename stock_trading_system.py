"""
Enhanced Large Cap Stocks Trading Dashboard - Flask Version (DYNAMIC NSE STOCKS)
Features:
- DYNAMIC fetching of ALL NSE-listed stocks from official sources
- 24-hour caching to minimize memory usage (512MB Render-friendly)
- Z-score with percentage deviation
- Linear regression analysis vs Nifty 50
- VISUAL Regression Plots with Equation (y = mx + b)
- Clear entry/exit explanations with confidence levels
- Time-to-target predictions
- Autocomplete for Search
"""

from flask import Flask, jsonify, request
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
import json
import os
import time

# Set non-interactive backend for Render server
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ===== ARMORED DATA SESSION =====
# This fixes the "Expecting value: line 1 column 1" error from your logs
# by making the server appear as a browser.
custom_session = requests.Session()
custom_session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Origin': 'https://finance.yahoo.com',
    'Referer': 'https://finance.yahoo.com'
})

# ===== DYNAMIC MARKET DATA ENGINE =====
class MarketData:
    """
    Dynamic Market Engine:
    1. Scrapes official NSE archives for all active equities.
    2. Fetches Nifty 500 for sector categorization.
    3. Caches data to 'stock_master_cache.json' for 24h to ensure instant startup.
    4. Memory-optimized for 512MB RAM environments.
    """
    CACHE_FILE = "stock_master_cache.json"
    CACHE_DURATION = 86400  # 24 hours (in seconds)

    def __init__(self):
        self.stocks_by_sector = {}  # Stores sector map (e.g. "Auto" -> ["TATAMOTORS", ...])
        self.all_tickers = []       # Stores ALL raw symbols (e.g. ["RELIANCE", "TCS", ...])
        self.company_map = {}       # Maps Company Name -> Symbol (e.g. "ONE 97" -> "PAYTM")
        self.load_data()

    def load_data(self):
        """Check cache first; if invalid or missing, download fresh data."""
        if self._is_cache_valid():
            try:
                print("✅ Loading stock list from local cache...")
                with open(self.CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    self.stocks_by_sector = data.get('sectors', {})
                    self.all_tickers = data.get('all_tickers', [])
                    self.company_map = data.get('company_map', {})
                print(f"✅ Loaded {len(self.all_tickers)} stocks across {len(self.stocks_by_sector)} sectors from cache")
                return
            except Exception as e:
                print(f"⚠️ Cache corrupted ({e}), fetching fresh data...")

        print("⏳ Downloading fresh stock list from NSE...")
        self._fetch_fresh_data()

    def _is_cache_valid(self):
        """Returns True if cache exists and is less than 24 hours old."""
        if not os.path.exists(self.CACHE_FILE): 
            return False
        return (time.time() - os.path.getmtime(self.CACHE_FILE)) < self.CACHE_DURATION

    def _fetch_fresh_data(self):
        """
        Fetches data from:
        1. NiftyIndices (for Sectors)
        2. NSE Archives (for full market list)
        Memory-optimized: Only stores essential data, limits sector stocks to top 30
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        # 1. Fetch Nifty 500 for Sector Categorization
        print("⏳ Fetching Nifty 500 for sectors...")
        sector_success = False
        try:
            url_500 = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
            r = requests.get(url_500, headers=headers, timeout=30, verify=True)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                # Store top 30 stocks per sector to keep memory usage low
                for ind, group in df.groupby('Industry'):
                    self.stocks_by_sector[ind] = group['Symbol'].tolist()[:30]
                print(f"✅ Fetched {len(self.stocks_by_sector)} sectors from Nifty 500")
                sector_success = True
            else:
                print(f"⚠️ Nifty 500 fetch returned status {r.status_code}")
        except Exception as e:
            print(f"⚠️ Sector fetch failed: {e}")
        
        if not sector_success:
            self._use_fallback_sectors()

        # 2. Fetch ALL Active NSE Stocks
        print("⏳ Fetching full NSE stock list...")
        stock_success = False
        try:
            # This CSV contains every single active equity on NSE
            url_all = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
            r = requests.get(url_all, headers=headers, timeout=30, verify=True)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                self.all_tickers = df['SYMBOL'].unique().tolist()
                
                # Build smart map: "RELIANCE INDUSTRIES" -> "RELIANCE"
                # Only store essential mappings to save memory
                for _, row in df.iterrows():
                    try:
                        clean_name = str(row['NAME OF COMPANY']).upper().strip()
                        symbol = str(row['SYMBOL']).upper().strip()
                        
                        # Store full name
                        self.company_map[clean_name] = symbol
                        
                        # Store common shortened versions to improve search
                        words = clean_name.split()
                        if len(words) > 1:
                            # Store first word if meaningful (length > 3)
                            if len(words[0]) > 3:
                                self.company_map[words[0]] = symbol
                    except Exception:
                        continue
                
                print(f"✅ Fetched {len(self.all_tickers)} stocks from NSE archives")
                stock_success = True
            else:
                print(f"⚠️ NSE archives returned status {r.status_code}")
        except Exception as e:
            print(f"⚠️ Full market fetch failed: {e}")
        
        if not stock_success:
            self._use_fallback_tickers()

        # Ensure we have at least some data
        if not self.all_tickers:
            print("⚠️ Using fallback tickers")
            self._use_fallback_tickers()
        
        if not self.stocks_by_sector:
            print("⚠️ Using fallback sectors")
            self._use_fallback_sectors()

        # Save to cache
        try:
            with open(self.CACHE_FILE, 'w') as f:
                json.dump({
                    'sectors': self.stocks_by_sector,
                    'all_tickers': self.all_tickers,
                    'company_map': self.company_map,
                    'timestamp': time.time()
                }, f)
            print(f"✅ Fresh data cached ({len(self.all_tickers)} stocks, {len(self.stocks_by_sector)} sectors)")
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")

    def _use_fallback_sectors(self):
        """Fallback sector data if API fails"""
        self.stocks_by_sector = {
            'Top Stocks': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'HINDUNILVR', 'LT'],
            'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'BANKBARODA', 'PNB'],
            'IT': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM', 'LTIM', 'COFORGE', 'PERSISTENT'],
            'Auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT', 'HEROMOTOCO', 'TVSMOTOR'],
        }

    def _use_fallback_tickers(self):
        """Fallback ticker list if API fails - includes popular stocks"""
        self.all_tickers = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 
            'ITC', 'HINDUNILVR', 'LT', 'KOTAKBANK', 'AXISBANK', 'WIPRO', 'TECHM', 
            'TATAMOTORS', 'MARUTI', 'BAJFINANCE', 'SUNPHARMA', 'ONGC', 'NTPC',
            'SUZLON', 'ADANIPOWER', 'TATASTEEL', 'JSWSTEEL', 'COALINDIA', 'VEDL',
            'ZOMATO', 'PAYTM', 'NYKAA', 'DIXON', 'POLYCAB', 'HAVELLS'
        ]

    def normalize_symbol(self, search_term):
        """
        Converts user input (e.g., 'paytm', 'zomato', 'reliance industries') 
        into a valid ticker symbol (e.g., 'PAYTM', 'ZOMATO', 'RELIANCE').
        """
        if not search_term: 
            return None, ""
        
        original = search_term.strip()
        s_upper = original.upper()

        # 1. Exact Match (e.g. "RELIANCE")
        if s_upper in self.all_tickers: 
            return s_upper, original

        # 2. Exact Company Name Match (e.g. "ONE 97 COMMUNICATIONS LTD")
        if s_upper in self.company_map: 
            return self.company_map[s_upper], original
        
        # 3. Fuzzy/Partial Match (e.g. "PAYTM" found inside company name)
        for name, ticker in self.company_map.items():
            if s_upper in name or name in s_upper: 
                return ticker, original
        
        # 4. Partial ticker match
        matches = [t for t in self.all_tickers if s_upper in t or t in s_upper]
        if len(matches) == 1:
            return matches[0], original
            
        return None, original

# Initialize market data (loads from cache or fetches fresh)
print("🚀 Initializing Market Data Engine...")
market = MarketData()
ALL_VALID_TICKERS = set(market.all_tickers)
STOCKS = market.stocks_by_sector

print(f"✅ Market Engine Ready: {len(ALL_VALID_TICKERS)} stocks loaded")

# ===== ANALYZER CLASS =====
class Analyzer:
    @staticmethod
    def normalize_symbol(symbol):
        """Wrapper to use MarketData's normalize function"""
        return market.normalize_symbol(symbol)

    def get_data(self, symbol, period='10d', interval='1h'):
        try:
            ticker = f"{symbol}.NS"
            # FIX: Use armored custom_session and threads=False for Render
            data = yf.download(ticker, period=period, interval=interval, 
                               session=custom_session, progress=False, threads=False)
            
            # Fallback for delisted or differently indexed tickers
            if data is None or data.empty:
                data = yf.download(symbol, period=period, interval=interval, 
                                   session=custom_session, progress=False, threads=False)
                
            if data is None or data.empty:
                return None

            # CRITICAL FIX: Flatten MultiIndex columns (The 'TCS.NS' header issue)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            return data
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

    def calc_indicators(self, data):
        if data is None or len(data) < 14:
            return None
        try:
            # FIX: Ensure flat Series using squeeze() and handle potential MultiIndex leftovers
            close = data['Close'].squeeze().astype(float).dropna()
            high = data['High'].squeeze().astype(float).dropna()
            low = data['Low'].squeeze().astype(float).dropna()

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
            ema5 = close.ewm(span=5).mean()
            ema13 = close.ewm(span=13).mean()
            macd = ema5 - ema13
            signal = macd.ewm(span=5).mean()
            macd_hist = macd - signal
            macd_val = float(macd_hist.iloc[-1])
            macd_bullish = macd_val > 0 if not pd.isna(macd_val) else True
            h = float(high.iloc[-18:].max() if len(high) > 18 else high.iloc[-1])
            l = float(low.iloc[-18:].min() if len(low) > 18 else low.iloc[-1])
            pct_from_low = ((curr - l) / l) * 100 if l > 0 else 50
            lookback = 20 if len(close) >= 20 else len(close)
            mean_price = float(close.iloc[-lookback:].mean())
            std_price = float(close.iloc[-lookback:].std())
            
            # FIXED: DivisionByZero guard for dead/flat stocks
            if std_price > 0:
                zscore = (curr - mean_price) / std_price
                pct_deviation = ((curr - mean_price) / mean_price) * 100
            else:
                zscore, pct_deviation = 0.0, 0.0

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
                    # FIX: MultiIndex flattening for regression datasets
                    stock_data = yf.download(f"{stock_symbol}.NS", period=period, 
                                             session=custom_session, interval='1d', 
                                             progress=False, threads=False)
                    if isinstance(stock_data.columns, pd.MultiIndex):
                        stock_data.columns = stock_data.columns.get_level_values(0)

                    if stock_data is None or stock_data.empty: continue
                    nifty_data = yf.download("^NSEI", period=period, session=custom_session, 
                                             interval='1d', progress=False, threads=False)
                    if isinstance(nifty_data.columns, pd.MultiIndex):
                        nifty_data.columns = nifty_data.columns.get_level_values(0)

                    if nifty_data is None or nifty_data.empty:
                        nifty_data = yf.download("NIFTYBEES.NS", period=period, 
                                                 session=custom_session, interval='1d', 
                                                 progress=False, threads=False)
                        if isinstance(nifty_data.columns, pd.MultiIndex):
                            nifty_data.columns = nifty_data.columns.get_level_values(0)
                        if nifty_data is None or nifty_data.empty:
                            nifty_data = yf.download("RELIANCE.NS", period=period, 
                                                     session=custom_session, interval='1d', 
                                                     progress=False, threads=False)
                            if isinstance(nifty_data.columns, pd.MultiIndex):
                                nifty_data.columns = nifty_data.columns.get_level_values(0)
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
            return None

analyzer = Analyzer()

# ===== ROUTES =====
@app.route('/')
def index():
    stock_count = len(ALL_VALID_TICKERS)
    sector_count = len(STOCKS)
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
        .back-btn { background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); color: white; padding: 12px 28px; margin-bottom: 20px; border: none; font-weight: 600; font-size: 1em; cursor: pointer;}
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
            <div class="stock-count">🚀 Now analyzing ''' + str(stock_count) + '''+ NSE stocks across ''' + str(sector_count) + ''' sectors</div>
        </header>
        <div class="tabs">
            <button class="tab active" onclick="switchTab('analysis', event)">Technical Analysis</button>
            <button class="tab" onclick="switchTab('regression', event)">Regression vs Nifty</button>
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
                <button onclick="analyzeRegression()" style="margin-top: 15px; width: 100%; background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); color: white; font-weight: 600; padding: 14px; cursor: pointer; border: none;">Analyze Regression</button>
            </div>
            <div id="regression-result" style="margin-top: 30px;"></div>
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
                input.addEventListener('input', async (e) => {
                    const q = e.target.value.trim();
                    const sug = document.getElementById(suggestionId);
                    if (q.length === 0) { sug.innerHTML = ''; return; }
                    const all = Object.values(stocks).flat();
                    const qUpper = q.toUpperCase();
                    const filtered = all.filter(s => s.includes(qUpper)).slice(0, 12);
                    if (filtered.length > 0) {
                        sug.innerHTML = filtered.map(s => {
                            if(callbackName === 'analyzeRegression') return `<button onclick="document.getElementById('${inputId}').value = '${s}'; analyzeRegression();">${s}</button>`;
                            else return `<button onclick="analyze('${s}')">${s}</button>`;
                        }).join('');
                    } else {
                        sug.innerHTML = '<div style="color: var(--text-muted); padding: 10px;">Searching...</div>';
                        try {
                            const response = await fetch(`/search?q=${encodeURIComponent(q)}`);
                            const data = await response.json();
                            if (data.results && data.results.length > 0) {
                                sug.innerHTML = data.results.map(s => {
                                    if(callbackName === 'analyzeRegression') return `<button onclick="document.getElementById('${inputId}').value = '${s}'; analyzeRegression();">${s}</button>`;
                                    else return `<button onclick="analyze('${s}')">${s}</button>`;
                                }).join('');
                            } else {
                                sug.innerHTML = '<div style="color: var(--danger); padding: 10px;">No matches found</div>';
                            }
                        } catch(e) {
                            sug.innerHTML = '<div style="color: var(--danger); padding: 10px;">Search failed</div>';
                        }
                    }
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
        window.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>'''
    return html

@app.route('/search')
def search_route():
    query = request.args.get('q', '').strip().upper()
    if not query or len(query) < 2:
        return jsonify({'results': []})
    matches = []
    for ticker in sorted(ALL_VALID_TICKERS):
        if query in ticker:
            matches.append(ticker)
            if len(matches) >= 20: break
    if len(matches) == 0:
        for company_name, ticker in market.company_map.items():
            if query in company_name:
                matches.append(ticker)
                if len(matches) >= 20: break
    return jsonify({'results': list(set(matches))[:20]})

@app.route('/analyze')
def analyze_route():
    symbol = request.args.get('symbol', '').strip()
    norm_s, original = analyzer.normalize_symbol(symbol)
    if not norm_s: return jsonify({'error': f'Invalid symbol {symbol}'})
    try:
        result = analyzer.analyze(norm_s)
        if not result: return jsonify({'error': 'Insufficient data for analysis.'})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

@app.route('/regression')
def regression_route():
    symbol = request.args.get('symbol', '').strip()
    norm_s, _ = analyzer.normalize_symbol(symbol)
    try:
        result = analyzer.regression_analysis(norm_s)
        if not result: return jsonify({'error': 'Regression failed.'})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
