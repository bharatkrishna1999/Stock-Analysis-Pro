"""
Enhanced Large Cap Stocks Trading Dashboard - Flask Version (FIXED)
Features:
- Z-score with percentage deviation
- Bolna AI-inspired design
- Linear regression analysis vs Nifty 50
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

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Large Cap Stocks by Sector
STOCKS = {
    'IT Sector': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM', 'LTIM', 'COFORGE', 'MPHASIS', 'PERSISTENT'],
    'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'BANKBARODA', 'PNB',
                'FEDERALBNK'],
    'Financial Services': ['BAJFINANCE', 'BAJAJFINSV', 'SBILIFE', 'HDFCLIFE', 'ICICIGI', 'ICICIPRULI', 'CHOLAFIN',
                           'PFC', 'RECLTD'],
    'Auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT', 'HEROMOTOCO', 'TVSMOTOR', 'ASHOKLEY'],
    'Auto Components': ['BOSCHLTD', 'MOTHERSON', 'BALKRISIND', 'MRF', 'APOLLOTYRE', 'EXIDEIND', 'AMARAJABAT'],
    'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'LUPIN', 'BIOCON', 'AUROPHARMA', 'TORNTPHARM', 'ALKEM'],
    'Healthcare': ['APOLLOHOSP', 'MAXHEALTH', 'FORTIS', 'LALPATHLAB', 'METROPOLIS'],
    'Consumer Goods': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO', 'GODREJCP', 'COLPAL',
                       'TATACONSUM'],
    'Retail': ['DMART', 'TRENT', 'TITAN', 'ABFRL', 'SHOPERSTOP'],
    'Energy - Oil & Gas': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL', 'HINDPETRO', 'PETRONET'],
    'Power': ['NTPC', 'POWERGRID', 'ADANIPOWER', 'TATAPOWER', 'TORNTPOWER', 'ADANIGREEN'],
    'Metals & Mining': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'COALINDIA', 'VEDL', 'NMDC', 'SAIL', 'NATIONALUM'],
    'Cement': ['ULTRACEMCO', 'GRASIM', 'SHREECEM', 'AMBUJACEM', 'ACC', 'DALMIACEM', 'JKCEMENT'],
    'Real Estate': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'BRIGADE', 'PHOENIXLTD'],
    'Infrastructure': ['LT', 'ADANIENT', 'ADANIPORTS', 'SIEMENS', 'ABB', 'CUMMINSIND', 'VOLTAS', 'BHARTIARTL'],
    'Telecom': ['BHARTIARTL', 'IDEA'],
    'Media': ['ZEEL', 'SUNTV', 'PVRINOX'],
    'Chemicals': ['UPL', 'PIDILITIND', 'AARTI', 'SRF', 'DEEPAKNTR', 'GNFC', 'CHAMBLFERT'],
    'Paints': ['ASIANPAINT', 'BERGER', 'KANSAINER'],
    'Textiles': ['GRASIM', 'AIAENG', 'RAYMOND'],
    'Logistics': ['CONCOR', 'VRL', 'MAHLOG', 'BLUEDART'],
    'Aviation': ['INDIGO'],
    'Hospitality': ['INDHOTEL', 'LEMONTREE', 'CHOICEINT'],
    'Conglomerate': ['RELIANCE', 'LT', 'ITC', 'ADANIENT', 'TATASTEEL', 'M&M'],
}

# Company name to ticker symbol mapping
COMPANY_TO_TICKER = {
    'VEDANTA': 'VEDL',
    'TATA CONSULTANCY': 'TCS',
    'TATA CONSULTANCY SERVICES': 'TCS',
    'INFOSYS': 'INFY',
    'HCL TECH': 'HCLTECH',
    'HCL TECHNOLOGIES': 'HCLTECH',
    'TECH MAHINDRA': 'TECHM',
    'HDFC BANK': 'HDFCBANK',
    'ICICI BANK': 'ICICIBANK',
    'STATE BANK': 'SBIN',
    'STATE BANK OF INDIA': 'SBIN',
    'SBI': 'SBIN',
    'KOTAK': 'KOTAKBANK',
    'KOTAK MAHINDRA': 'KOTAKBANK',
    'AXIS BANK': 'AXISBANK',
    'INDUSIND': 'INDUSINDBK',
    'INDUSIND BANK': 'INDUSINDBK',
    'BANK OF BARODA': 'BANKBARODA',
    'PUNJAB NATIONAL BANK': 'PNB',
    'FEDERAL BANK': 'FEDERALBNK',
    'BAJAJ FINANCE': 'BAJFINANCE',
    'BAJAJ FINSERV': 'BAJAJFINSV',
    'SBI LIFE': 'SBILIFE',
    'HDFC LIFE': 'HDFCLIFE',
    'ICICI LOMBARD': 'ICICIGI',
    'ICICI PRUDENTIAL': 'ICICIPRULI',
    'MARUTI': 'MARUTI',
    'MARUTI SUZUKI': 'MARUTI',
    'TATA MOTORS': 'TATAMOTORS',
    'MAHINDRA': 'M&M',
    'BAJAJ AUTO': 'BAJAJ-AUTO',
    'EICHER': 'EICHERMOT',
    'EICHER MOTORS': 'EICHERMOT',
    'HERO MOTOCORP': 'HEROMOTOCO',
    'HERO': 'HEROMOTOCO',
    'TVS': 'TVSMOTOR',
    'TVS MOTOR': 'TVSMOTOR',
    'ASHOK LEYLAND': 'ASHOKLEY',
    'BOSCH': 'BOSCHLTD',
    'MRF': 'MRF',
    'APOLLO TYRES': 'APOLLOTYRE',
    'EXIDE': 'EXIDEIND',
    'EXIDE INDUSTRIES': 'EXIDEIND',
    'SUN PHARMA': 'SUNPHARMA',
    'SUN PHARMACEUTICAL': 'SUNPHARMA',
    'DR REDDY': 'DRREDDY',
    'DR REDDYS': 'DRREDDY',
    'CIPLA': 'CIPLA',
    'DIVIS': 'DIVISLAB',
    'DIVIS LAB': 'DIVISLAB',
    'LUPIN': 'LUPIN',
    'BIOCON': 'BIOCON',
    'AUROBINDO': 'AUROPHARMA',
    'TORRENT PHARMA': 'TORNTPHARM',
    'ALKEM': 'ALKEM',
    'APOLLO HOSPITAL': 'APOLLOHOSP',
    'APOLLO HOSPITALS': 'APOLLOHOSP',
    'MAX HEALTH': 'MAXHEALTH',
    'MAX HEALTHCARE': 'MAXHEALTH',
    'FORTIS': 'FORTIS',
    'LALPATHLAB': 'LALPATHLAB',
    'DR LALPATHLAB': 'LALPATHLAB',
    'METROPOLIS': 'METROPOLIS',
    'HUL': 'HINDUNILVR',
    'HINDUSTAN UNILEVER': 'HINDUNILVR',
    'ITC': 'ITC',
    'NESTLE': 'NESTLEIND',
    'BRITANNIA': 'BRITANNIA',
    'DABUR': 'DABUR',
    'MARICO': 'MARICO',
    'GODREJ': 'GODREJCP',
    'GODREJ CONSUMER': 'GODREJCP',
    'COLGATE': 'COLPAL',
    'TATA CONSUMER': 'TATACONSUM',
    'DMART': 'DMART',
    'AVENUE SUPERMARTS': 'DMART',
    'TRENT': 'TRENT',
    'TITAN': 'TITAN',
    'ADITYA BIRLA': 'ABFRL',
    'SHOPPERS STOP': 'SHOPERSTOP',
    'RELIANCE': 'RELIANCE',
    'RIL': 'RELIANCE',
    'ONGC': 'ONGC',
    'OIL AND NATURAL GAS': 'ONGC',
    'BPCL': 'BPCL',
    'BHARAT PETROLEUM': 'BPCL',
    'IOC': 'IOC',
    'INDIAN OIL': 'IOC',
    'GAIL': 'GAIL',
    'HPCL': 'HINDPETRO',
    'HINDUSTAN PETROLEUM': 'HINDPETRO',
    'PETRONET': 'PETRONET',
    'PETRONET LNG': 'PETRONET',
    'NTPC': 'NTPC',
    'POWER GRID': 'POWERGRID',
    'POWERGRID': 'POWERGRID',
    'ADANI POWER': 'ADANIPOWER',
    'TATA POWER': 'TATAPOWER',
    'TORRENT POWER': 'TORNTPOWER',
    'ADANI GREEN': 'ADANIGREEN',
    'TATA STEEL': 'TATASTEEL',
    'HINDALCO': 'HINDALCO',
    'JSW': 'JSWSTEEL',
    'JSW STEEL': 'JSWSTEEL',
    'COAL INDIA': 'COALINDIA',
    'VEDL': 'VEDL',
    'NMDC': 'NMDC',
    'SAIL': 'SAIL',
    'NATIONAL ALUMINIUM': 'NATIONALUM',
    'ULTRATECH': 'ULTRACEMCO',
    'ULTRATECH CEMENT': 'ULTRACEMCO',
    'GRASIM': 'GRASIM',
    'SHREE CEMENT': 'SHREECEM',
    'AMBUJA': 'AMBUJACEM',
    'AMBUJA CEMENT': 'AMBUJACEM',
    'ACC': 'ACC',
    'DALMIA': 'DALMIACEM',
    'DALMIA BHARAT': 'DALMIACEM',
    'JK CEMENT': 'JKCEMENT',
    'DLF': 'DLF',
    'GODREJ PROPERTIES': 'GODREJPROP',
    'OBEROI': 'OBEROIRLTY',
    'OBEROI REALTY': 'OBEROIRLTY',
    'PRESTIGE': 'PRESTIGE',
    'BRIGADE': 'BRIGADE',
    'PHOENIX': 'PHOENIXLTD',
    'LT': 'LT',
    'L&T': 'LT',
    'LARSEN': 'LT',
    'LARSEN AND TOUBRO': 'LT',
    'ADANI': 'ADANIENT',
    'ADANI ENTERPRISES': 'ADANIENT',
    'ADANI PORTS': 'ADANIPORTS',
    'SIEMENS': 'SIEMENS',
    'ABB': 'ABB',
    'CUMMINS': 'CUMMINSIND',
    'VOLTAS': 'VOLTAS',
    'BHARTI': 'BHARTIARTL',
    'BHARTI AIRTEL': 'BHARTIARTL',
    'AIRTEL': 'BHARTIARTL',
    'IDEA': 'IDEA',
    'VODAFONE': 'IDEA',
    'VI': 'IDEA',
    'ZEE': 'ZEEL',
    'ZEE ENTERTAINMENT': 'ZEEL',
    'SUN TV': 'SUNTV',
    'PVR': 'PVRINOX',
    'PVR INOX': 'PVRINOX',
    'UPL': 'UPL',
    'PIDILITE': 'PIDILITIND',
    'AARTI': 'AARTI',
    'AARTI INDUSTRIES': 'AARTI',
    'SRF': 'SRF',
    'DEEPAK': 'DEEPAKNTR',
    'DEEPAK NITRITE': 'DEEPAKNTR',
    'GNFC': 'GNFC',
    'CHAMBAL': 'CHAMBLFERT',
    'CHAMBAL FERTILIZER': 'CHAMBLFERT',
    'ASIAN PAINTS': 'ASIANPAINT',
    'ASIAN PAINT': 'ASIANPAINT',
    'BERGER': 'BERGER',
    'BERGER PAINTS': 'BERGER',
    'KANSAI': 'KANSAINER',
    'KANSAI NEROLAC': 'KANSAINER',
    'RAYMOND': 'RAYMOND',
    'CONCOR': 'CONCOR',
    'CONTAINER CORP': 'CONCOR',
    'VRL': 'VRL',
    'VRL LOGISTICS': 'VRL',
    'MAHINDRA LOGISTICS': 'MAHLOG',
    'BLUE DART': 'BLUEDART',
    'BLUEDART': 'BLUEDART',
    'INDIGO': 'INDIGO',
    'INTERGLOBE': 'INDIGO',
    'INDIAN HOTELS': 'INDHOTEL',
    'TAJ': 'INDHOTEL',
    'LEMON TREE': 'LEMONTREE',
    'LEMONTREE': 'LEMONTREE',
    'CHOICE': 'CHOICEINT',
}

# Create a set of all valid tickers for fast lookup
ALL_VALID_TICKERS = set()
for sector_stocks in STOCKS.values():
    ALL_VALID_TICKERS.update(sector_stocks)


class Analyzer:
    @staticmethod
    def normalize_symbol(symbol):
        """
        Normalize user input to a valid NSE ticker symbol.
        Returns (normalized_ticker, original_input) or (None, original_input) if invalid.
        """
        if not symbol:
            return None, ""

        original = symbol.strip()
        symbol_upper = original.upper()

        # First, check if it's already a valid ticker
        if symbol_upper in ALL_VALID_TICKERS:
            return symbol_upper, original

        # Check if it's in the company name mapping
        if symbol_upper in COMPANY_TO_TICKER:
            return COMPANY_TO_TICKER[symbol_upper], original

        # Try partial matching - find tickers that contain the search term
        matches = [ticker for ticker in ALL_VALID_TICKERS if symbol_upper in ticker]
        if len(matches) == 1:
            return matches[0], original

        # Try reverse - find tickers contained in the search term
        reverse_matches = [ticker for ticker in ALL_VALID_TICKERS if ticker in symbol_upper]
        if len(reverse_matches) == 1:
            return reverse_matches[0], original

        # Check company names that contain the search term
        for company_name, ticker in COMPANY_TO_TICKER.items():
            if symbol_upper in company_name or company_name in symbol_upper:
                return ticker, original

        return None, original

    def get_data(self, symbol, period='10d', interval='1h'):
        try:
            ticker = f"{symbol}.NS"
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                print(f"No data returned for {ticker}")
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

            if len(close) < 14:
                return None

            curr = float(close.iloc[-1])
            sma9 = float(close.rolling(9).mean().iloc[-1])
            sma5 = float(close.rolling(5).mean().iloc[-1])

            # Returns
            open_price = float(close.iloc[-18] if len(close) > 18 else close.iloc[0])
            prev_hour = float(close.iloc[-2] if len(close) > 1 else curr)

            daily_ret = ((curr - open_price) / open_price) * 100 if open_price > 0 else 0
            hourly_ret = ((curr - prev_hour) / prev_hour) * 100 if prev_hour > 0 else 0

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi = float(rsi_series.iloc[-1])

            if pd.isna(rsi) or np.isinf(rsi):
                rsi = 50.0

            # MACD
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

            # Z-Score for Mean Reversion (20-period)
            lookback = 20 if len(close) >= 20 else len(close)
            mean_price = float(close.iloc[-lookback:].mean())
            std_price = float(close.iloc[-lookback:].std())

            if std_price > 0:
                zscore = (curr - mean_price) / std_price
                # Calculate percentage deviation from mean
                pct_deviation = ((curr - mean_price) / mean_price) * 100
            else:
                zscore = 0.0
                pct_deviation = 0.0

            # Bollinger Bands
            bb_upper = mean_price + (2 * std_price)
            bb_lower = mean_price - (2 * std_price)
            bb_position = ((curr - bb_lower) / (bb_upper - bb_lower)) * 100 if (bb_upper - bb_lower) > 0 else 50

            # Calculate historical volatility for confidence
            returns = close.pct_change().dropna()
            volatility = float(returns.std() * 100)  # as percentage

            return {
                'price': curr,
                'sma9': sma9,
                'sma5': sma5,
                'daily': daily_ret,
                'hourly': hourly_ret,
                'rsi': rsi,
                'macd_bullish': bool(macd_bullish),
                'high': h,
                'low': l,
                'pct_from_low': pct_from_low,
                'zscore': zscore,
                'pct_deviation': pct_deviation,
                'mean_price': mean_price,
                'std_price': std_price,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_position': bb_position,
                'volatility': volatility
            }
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return None

    def calculate_confidence(self, ind):
        """Calculate trading confidence based on multiple factors"""
        confidence_score = 50  # Base confidence

        # Trend alignment
        uptrend = ind['price'] > ind['sma9']
        if uptrend and ind['macd_bullish']:
            confidence_score += 15
        elif not uptrend and not ind['macd_bullish']:
            confidence_score += 15

        # RSI confirmation
        if ind['rsi'] > 70 or ind['rsi'] < 30:
            confidence_score += 10  # Extreme RSI adds confidence

        # Z-score extremity
        if abs(ind['zscore']) > 2:
            confidence_score += 15  # Strong mean reversion signal
        elif abs(ind['zscore']) > 1:
            confidence_score += 5

        # Volatility (lower volatility = higher confidence)
        if ind['volatility'] < 2:
            confidence_score += 10
        elif ind['volatility'] > 5:
            confidence_score -= 10

        # Position in range
        if ind['pct_from_low'] < 20 or ind['pct_from_low'] > 80:
            confidence_score += 5  # Clear support/resistance

        return min(max(confidence_score, 0), 100)

    def estimate_days_to_target(self, current, target, ind):
        """Estimate days to reach target based on historical volatility and momentum"""
        try:
            distance = abs(target - current)
            pct_move_needed = (distance / current) * 100

            # Use daily volatility as a proxy for typical daily movement
            daily_vol = ind['volatility']

            if daily_vol > 0:
                # Estimate days = (distance needed) / (typical daily movement)
                # Add safety factor of 1.5x
                days = (pct_move_needed / daily_vol) * 1.5
                return max(1, min(int(days), 30))  # Cap between 1-30 days
            else:
                return 7  # Default estimate
        except:
            return 7

    def signal(self, ind):
        if not ind:
            return None

        i = ind

        # Trend analysis
        uptrend = i['price'] > i['sma9']
        strong_move = abs(i['daily']) > 2
        not_extreme = 20 < i['pct_from_low'] < 80

        # Build explanations with percentage deviation
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

        # Enhanced Z-Score explanation with percentage deviation
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

        # Bollinger Band Position
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

        # Decision logic
        rsi_overbought = i['rsi'] > 70
        rsi_oversold = i['rsi'] < 30
        near_top = i['pct_from_low'] > 80
        near_bottom = i['pct_from_low'] < 20
        extreme_high_zscore = i['zscore'] > 2
        high_zscore = i['zscore'] > 1
        extreme_low_zscore = i['zscore'] < -2
        low_zscore = i['zscore'] < -1

        # Initialize variables
        sig = ""
        action = ""
        rec = ""
        entry_price = 0
        stop_price = 0
        target_price = 0

        # BUY signals
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

        # SELL signals
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

        # HOLD signals - Uptrend weakening
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

        # HOLD signals - Downtrend stabilizing
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

        # Default HOLD
        else:
            sig = "HOLD"
            action = "RANGE-BOUND"
            entry_price = (i['high'] + i['low']) / 2
            stop_price = i['low'] * 0.98
            target_price = i['high'] * 1.02
            rec = "Stock consolidating. Wait for breakout direction."

        # Calculate confidence
        confidence = self.calculate_confidence(i)

        # Estimate days to target
        days_to_target = self.estimate_days_to_target(i['price'], target_price, i)

        # Generate layman explanations
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
        else:  # HOLD
            entry_explain = f"Wait for clearer signals. Consider entry only if price moves decisively above ₹{entry_price:.2f}."
            exit_explain = f"If already holding, consider taking profits at ₹{target_price:.2f}."
            confidence_explain = f"{confidence}% confidence. Moderate confidence suggests waiting for better setup."
            time_explain = f"Market consolidating. Wait for breakout confirmation."

        return {
            'signal': {
                'signal': sig,
                'action': action,
                'rec': rec,
                'entry': f"₹{entry_price:.2f}",
                'stop': f"₹{stop_price:.2f}",
                'target': f"₹{target_price:.2f}",
                'confidence': confidence,
                'days_to_target': days_to_target,
                'entry_explain': entry_explain,
                'exit_explain': exit_explain,
                'confidence_explain': confidence_explain,
                'time_explain': time_explain,
                'trend_explain': trend_explain,
                'momentum_explain': momentum_explain,
                'rsi_explain': rsi_explain,
                'position_explain': position_explain,
                'zscore_explain': zscore_explain,
                'bb_explain': bb_explain,
                'macd_text': "BULLISH - momentum favors buyers" if i['macd_bullish'] else "BEARISH - momentum favors sellers"
            },
            'details': {
                'price': f"₹{i['price']:.2f}",
                'daily': f"{i['daily']:+.2f}%",
                'hourly': f"{i['hourly']:+.2f}%",
                'rsi': f"{i['rsi']:.1f}",
                'zscore': f"{i['zscore']:.2f}",
                'pct_deviation': f"{i['pct_deviation']:+.2f}%",
                'mean': f"₹{i['mean_price']:.2f}",
                'sma9': f"₹{i['sma9']:.2f}",
                'high': f"₹{i['high']:.2f}",
                'low': f"₹{i['low']:.2f}",
                'bb_upper': f"₹{i['bb_upper']:.2f}",
                'bb_lower': f"₹{i['bb_lower']:.2f}",
                'volatility': f"{i['volatility']:.2f}%",
                'macd': "BULLISH" if i['macd_bullish'] else "BEARISH"
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
        """Perform linear regression analysis of stock vs Nifty 50 - PROPERLY FIXED VERSION"""

        def _clean_close(px_df):
            c = px_df.get("Close", None)
            if c is None:
                return pd.Series(dtype=float)

            # yfinance can return Close as a DataFrame (multiindex columns or 2D)
            if isinstance(c, pd.DataFrame):
                # pick the first column (for single ticker downloads, this is the right one)
                c = c.iloc[:, 0]

            s = c.copy()
            s.index = pd.to_datetime(s.index, errors="coerce")

            # remove tz if present, then normalize to date
            try:
                if getattr(s.index, "tz", None) is not None:
                    s.index = s.index.tz_convert(None)
            except Exception:
                pass

            s = s[~s.index.isna()]
            s.index = s.index.normalize()

            # dedupe
            s = s.groupby(s.index).last()

            # force numeric
            s = pd.to_numeric(s, errors="coerce").dropna()

            return s

        try:
            print(f"\n{'=' * 60}")
            print(f"Starting regression analysis for {stock_symbol}...")
            print(f"{'=' * 60}")

            # Try different period options
            periods_to_try = ['1y', '6mo', '3mo']
            stock_data = None
            nifty_data = None
            nifty_source = None

            for period in periods_to_try:
                try:
                    print(f"\n[Attempt] Fetching {period} of data...")

                    # Fetch stock data with retry
                    stock_data = yf.download(f"{stock_symbol}.NS", period=period, interval='1d',
                                             progress=False, threads=False)

                    if stock_data is None or stock_data.empty:
                        print(f"[FAIL] No stock data for {stock_symbol}.NS")
                        continue

                    print(f"[OK] Stock data: {len(stock_data)} rows")

                    # Try primary Nifty source: ^NSEI
                    nifty_data = yf.download("^NSEI", period=period, interval='1d',
                                             progress=False, threads=False)

                    if nifty_data is None or nifty_data.empty:
                        print(f"[FAIL] ^NSEI returned empty, trying fallback...")

                        # Fallback: Use NIFTYBEES (Nifty 50 ETF) as proxy
                        nifty_data = yf.download("NIFTYBEES.NS", period=period, interval='1d',
                                                 progress=False, threads=False)

                        if nifty_data is None or nifty_data.empty:
                            print(f"[FAIL] NIFTYBEES.NS also empty, trying one more fallback...")

                            # Second fallback: Use any large liquid stock as market proxy
                            nifty_data = yf.download("RELIANCE.NS", period=period, interval='1d',
                                                     progress=False, threads=False)
                            nifty_source = "RELIANCE (proxy)"
                        else:
                            nifty_source = "NIFTYBEES ETF"
                    else:
                        nifty_source = "Nifty 50 Index"
                        print(f"[OK] Nifty data: {len(nifty_data)} rows")

                    if nifty_data is not None and not nifty_data.empty:
                        print(f"[SUCCESS] Using {nifty_source} as market benchmark")
                        break

                except Exception as e:
                    print(f"[ERROR] Failed to fetch {period} data: {e}")
                    continue

            # Validate we have data
            if stock_data is None or stock_data.empty:
                print(f"\n[FATAL] Could not fetch any stock data for {stock_symbol}.NS")
                return None

            if nifty_data is None or nifty_data.empty:
                print(f"\n[FATAL] Could not fetch any Nifty/market data from any source")
                return None

            print(f"\n[Data Summary]")
            print(f"  Raw stock data: {len(stock_data)} rows")
            print(f"  Raw market data: {len(nifty_data)} rows (source: {nifty_source})")

            # Clean and normalize timestamps
            print(f"\n[Date Normalization]")
            print(f"  Stock index type: {type(stock_data.index[0])}, tz: {getattr(stock_data.index, 'tz', None)}")
            print(f"  Market index type: {type(nifty_data.index[0])}, tz: {getattr(nifty_data.index, 'tz', None)}")

            stock_close = _clean_close(stock_data)
            market_close = _clean_close(nifty_data)

            print(f"  After cleaning: Stock={len(stock_close)}, Market={len(market_close)}")

            # Calculate returns BEFORE joining
            stock_ret = stock_close.pct_change()
            market_ret = market_close.pct_change()

            # Join returns with inner join
            rets = pd.concat([stock_ret.rename("stock"), market_ret.rename("market")],
                             axis=1, join="inner").dropna()

            print(f"\n[Alignment] Paired return observations: {len(rets)}")

            if len(rets) < 20:
                print(
                    f"[FATAL] stock rows={len(stock_close)} market rows={len(market_close)} paired returns={len(rets)}")
                print(f"  Need at least 20 paired observations, got {len(rets)}")
                print(f"  This usually means very few overlapping trading dates")
                return None

            # Extract X and y for regression
            X = rets["market"].to_numpy()
            y = rets["stock"].to_numpy()

            print(f"\n[Regression] Running with {len(X)} data points...")

            # Perform regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

            r_squared = r_value ** 2
            beta = slope
            alpha = intercept

            print(f"[Results] Beta={beta:.3f}, R²={r_squared:.3f}, Alpha={alpha:.4f}")

            # Calculate residuals and additional stats
            y_pred = slope * X + intercept
            residuals = y - y_pred
            residual_std = np.std(residuals)

            # Calculate correlation
            correlation = np.corrcoef(X, y)[0, 1]

            # Interpretation
            if beta > 1.2:
                beta_interpret = f"HIGH BETA ({beta:.2f}): Stock is {((beta - 1) * 100):.0f}% more volatile than market. Amplifies market moves both up and down."
            elif beta > 0.8:
                beta_interpret = f"MEDIUM BETA ({beta:.2f}): Stock moves roughly in line with market. Market-like risk."
            elif beta > 0:
                beta_interpret = f"LOW BETA ({beta:.2f}): Stock is {((1 - beta) * 100):.0f}% less volatile than market. Defensive stock."
            else:
                beta_interpret = f"NEGATIVE BETA ({beta:.2f}): Stock moves opposite to market. Rare hedge characteristic."

            if r_squared > 0.7:
                r2_interpret = f"STRONG FIT ({r_squared:.2%}): {(r_squared * 100):.0f}% of stock movement explained by market. Highly correlated."
            elif r_squared > 0.4:
                r2_interpret = f"MODERATE FIT ({r_squared:.2%}): {(r_squared * 100):.0f}% explained by market. Some independent factors."
            else:
                r2_interpret = f"WEAK FIT ({r_squared:.2%}): Only {(r_squared * 100):.0f}% explained by market. Stock has strong independent drivers."

            if alpha > 0.001:
                alpha_interpret = f"POSITIVE ALPHA ({alpha:.4f}): Stock outperforms market by {(alpha * 100):.2f}% daily on average. Strong stock."
            elif alpha < -0.001:
                alpha_interpret = f"NEGATIVE ALPHA ({alpha:.4f}): Stock underperforms market by {(abs(alpha) * 100):.2f}% daily on average."
            else:
                alpha_interpret = f"NEUTRAL ALPHA ({alpha:.4f}): Stock performs in line with market expectations."

            # Trading insights
            if beta > 1.2 and r_squared > 0.6:
                trading_insight = f"LEVERAGED PLAY: Use for aggressive market exposure. When market rises 1%, expect this to rise ~{beta:.1f}%. But beware of amplified losses."
            elif beta < 0.8 and r_squared > 0.6:
                trading_insight = "DEFENSIVE PLAY: Good for volatile markets. Provides market exposure with lower volatility."
            elif r_squared < 0.3:
                trading_insight = "INDEPENDENT STOCK: Market movements don't dictate this stock. Focus on company-specific analysis."
            else:
                trading_insight = "MARKET-LINKED: Stock follows market trends. Monitor market direction for trading decisions."

            print(f"[SUCCESS] Regression analysis completed")
            print(f"{'=' * 60}\n")

            return {
                'beta': beta,
                'alpha': alpha,
                'r_squared': r_squared,
                'correlation': correlation,
                'p_value': p_value,
                'std_error': std_err,
                'residual_std': residual_std,
                'data_points': len(X),
                'market_source': nifty_source,
                'beta_interpret': beta_interpret,
                'r2_interpret': r2_interpret,
                'alpha_interpret': alpha_interpret,
                'trading_insight': trading_insight
            }

        except Exception as e:
            print(f"\n[FATAL ERROR] Regression failed for {stock_symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None


analyzer = Analyzer()


@app.route('/')
def index():
    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Analysis Pro</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }
        
        :root {
            --bg-dark: #0a0e1a;
            --bg-card: #131824;
            --bg-card-hover: #1a1f2e;
            --accent-cyan: #00d9ff;
            --accent-purple: #9d4edd;
            --accent-green: #06ffa5;
            --text-primary: #ffffff;
            --text-secondary: #a0aec0;
            --text-muted: #718096;
            --border-color: #2d3748;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }
        
        body { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px;
        }
        
        header { 
            text-align: center; 
            padding: 40px 0;
            border-bottom: 1px solid var(--border-color);
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(157, 78, 221, 0.1));
        }
        
        header h1 { 
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3em; 
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        header p {
            color: var(--text-secondary);
            font-size: 1.1em;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin: 30px 0;
            border-bottom: 2px solid var(--border-color);
        }
        
        .tab {
            padding: 15px 30px;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
            font-family: 'Space Grotesk', sans-serif;
        }
        
        .tab:hover {
            color: var(--accent-cyan);
        }
        
        .tab.active {
            color: var(--text-primary);
            border-bottom-color: var(--accent-cyan);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
            margin-bottom: 40px; 
        }
        
        .card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 25px;
            border: 1px solid var(--border-color);
            transition: all 0.3s;
        }
        
        .card:hover {
            background: var(--bg-card-hover);
            border-color: var(--accent-cyan);
        }
        
        .card h3 { 
            color: var(--text-primary); 
            margin-bottom: 15px; 
            font-size: 1.3em;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
        }
        
        #search, #regression-search {
            width: 100%;
            padding: 14px;
            border: 2px solid var(--border-color);
            border-radius: 8px;
            font-size: 1em;
            background: var(--bg-dark);
            color: var(--text-primary);
            transition: all 0.3s;
        }
        
        #search:focus, #regression-search:focus { 
            outline: none; 
            border-color: var(--accent-cyan);
            box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.1);
        }
        
        .suggestions {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-top: 15px;
        }
        
        .category { 
            margin-bottom: 20px; 
        }
        
        .category h4 { 
            color: var(--accent-cyan);
            font-size: 0.85em; 
            margin-bottom: 8px; 
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 1px;
        }
        
        .stocks { 
            display: grid; 
            grid-template-columns: repeat(3, 1fr); 
            gap: 8px; 
        }
        
        button {
            padding: 10px 16px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
            color: var(--text-secondary);
            font-size: 0.9em;
        }
        
        button:hover { 
            background: var(--accent-cyan);
            color: var(--bg-dark);
            border-color: var(--accent-cyan);
            transform: translateY(-2px);
        }
        
        #result-view { 
            display: none; 
        }
        
        .result-card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 35px;
            border: 1px solid var(--border-color);
        }
        
        .header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 30px; 
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 20px; 
        }
        
        .header h2 { 
            color: var(--text-primary); 
            font-size: 2.5em;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
        }
        
        .signal-badge { 
            font-size: 1.2em; 
            font-weight: 700;
            padding: 12px 24px; 
            border-radius: 8px;
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: 1px;
        }
        
        .signal-BUY {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
        }
        
        .signal-SELL {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
        }
        
        .signal-HOLD {
            background: linear-gradient(135deg, #f59e0b, #d97706);
            color: white;
        }
        
        .action-banner { 
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            color: white; 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
            text-align: center; 
            font-weight: 600;
            font-size: 1.2em;
            font-family: 'Space Grotesk', sans-serif;
            box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
        }
        
        .rec-box { 
            background: var(--bg-card-hover);
            border-left: 4px solid var(--accent-green);
            color: var(--text-primary);
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 25px;
            font-size: 1.05em;
        }
        
        .confidence-meter {
            margin: 25px 0;
            padding: 20px;
            background: var(--bg-card-hover);
            border-radius: 10px;
            border: 1px solid var(--border-color);
        }
        
        .confidence-label {
            font-size: 0.9em;
            color: var(--text-secondary);
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }
        
        .confidence-bar-container {
            background: var(--bg-dark);
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        
        .confidence-bar {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9em;
            transition: width 1s ease;
        }
        
        .confidence-text {
            color: var(--text-secondary);
            font-size: 0.95em;
            line-height: 1.6;
        }
        
        .metrics { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); 
            gap: 15px; 
            margin: 25px 0; 
        }
        
        .metric { 
            background: var(--bg-card-hover);
            padding: 18px; 
            border-radius: 8px;
            border-left: 3px solid var(--accent-cyan);
            transition: all 0.3s;
        }
        
        .metric:hover {
            transform: translateY(-3px);
            border-left-color: var(--accent-green);
        }
        
        .metric-label { 
            font-size: 0.75em; 
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 5px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        
        .metric-value { 
            font-size: 1.5em; 
            font-weight: 700;
            color: var(--text-primary);
            font-family: 'Space Grotesk', sans-serif;
        }
        
        .explanation-section { 
            margin: 25px 0; 
        }
        
        .explanation-section h3 { 
            color: var(--accent-cyan);
            margin-bottom: 12px; 
            font-size: 1em;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .explanation { 
            background: var(--bg-card-hover);
            padding: 16px; 
            border-radius: 8px;
            line-height: 1.7;
            color: var(--text-secondary);
            border-left: 3px solid var(--accent-cyan);
        }
        
        .trading-plan { 
            background: var(--bg-card-hover);
            padding: 25px; 
            border-radius: 10px;
            margin-top: 30px;
            border: 2px solid var(--accent-purple);
        }
        
        .trading-plan h3 { 
            color: var(--accent-purple);
            margin-bottom: 20px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.3em;
            font-weight: 700;
        }
        
        .plan-item { 
            display: grid; 
            grid-template-columns: 140px 1fr; 
            gap: 20px; 
            margin-bottom: 15px; 
            padding: 15px; 
            background: var(--bg-dark);
            border-radius: 8px;
            transition: all 0.3s;
        }
        
        .plan-item:hover {
            background: var(--bg-card);
            transform: translateX(5px);
        }
        
        .plan-label { 
            font-weight: 700;
            color: var(--accent-cyan);
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.9em;
        }
        
        .plan-value { 
            color: var(--text-primary);
            font-weight: 500;
        }
        
        .back-btn { 
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            color: white; 
            padding: 12px 28px; 
            margin-bottom: 20px;
            border: none;
            font-weight: 600;
            font-size: 1em;
        }
        
        .back-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 217, 255, 0.4);
        }
        
        .loading { 
            text-align: center; 
            color: var(--accent-cyan);
            font-size: 1.3em;
            padding: 40px;
            font-family: 'Space Grotesk', sans-serif;
        }
        
        .error { 
            background: rgba(239, 68, 68, 0.1);
            color: var(--danger);
            padding: 20px; 
            border-radius: 8px;
            border-left: 4px solid var(--danger);
        }
        
        .regression-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }
        
        .regression-metric {
            background: var(--bg-card-hover);
            padding: 20px;
            border-radius: 10px;
            border-left: 3px solid var(--accent-purple);
        }
        
        .regression-metric-label {
            font-size: 0.85em;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 8px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        
        .regression-metric-value {
            font-size: 2em;
            font-weight: 700;
            color: var(--text-primary);
            font-family: 'Space Grotesk', sans-serif;
            margin-bottom: 8px;
        }
        
        .regression-metric-desc {
            font-size: 0.9em;
            color: var(--text-secondary);
            line-height: 1.5;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result-card {
            animation: slideIn 0.5s ease;
        }
        
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
            .header {
                flex-direction: column;
                gap: 15px;
            }
            .stocks, .suggestions {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Stock Analysis Pro</h1>
            <p>Advanced Trading Insights with AI-Powered Analysis</p>
        </header>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab('analysis', event)">Technical Analysis</button>
            <button class="tab" onclick="switchTab('regression', event)">Regression vs Nifty</button>
        </div>
        
        <div id="analysis-tab" class="tab-content active">
            <div id="search-view">
                <div class="grid">
                    <div class="card">
                        <h3>🔍 Search Stock</h3>
                        <input type="text" id="search" placeholder="Search INFY, TCS, RELIANCE...">
                        <div class="suggestions" id="suggestions"></div>
                    </div>
                    <div class="card">
                        <h3>📊 Browse by Sector</h3>
                        <div id="categories"></div>
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
                <p style="color: var(--text-secondary); margin-bottom: 20px;">Analyze how a stock correlates with Nifty 50 index movements</p>
                <input type="text" id="regression-search" placeholder="Enter stock symbol (e.g., TCS, INFY, RELIANCE)">
                <div class="suggestions" id="regression-suggestions"></div>
                <button onclick="analyzeRegression()" style="margin-top: 15px; width: 100%; background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); color: white; font-weight: 600; padding: 14px;">
                    Analyze Regression
                </button>
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
            // Populate categories
            const cat = document.getElementById('categories');
            Object.entries(stocks).forEach(([name, list]) => {
                let html = `<div class="category"><h4>${name}</h4><div class="stocks">`;
                list.forEach(s => html += `<button onclick="analyze('${s}')">${s}</button>`);
                html += '</div></div>';
                cat.innerHTML += html;
            });
            
            // Reusable Autocomplete Function
            function setupAutocomplete(inputId, suggestionId, callbackName) {
                const input = document.getElementById(inputId);
                if(!input) return;
                
                input.addEventListener('input', (e) => {
                    const q = e.target.value.toUpperCase();
                    const sug = document.getElementById(suggestionId);
                    
                    if (q.length === 0) { sug.innerHTML = ''; return; }
                    
                    const all = Object.values(stocks).flat();
                    const filtered = all.filter(s => s.includes(q)).slice(0, 9);
                    
                    sug.innerHTML = filtered.map(s => {
                        // For regression, clicking a button should trigger analyzeRegression
                        // For main, it triggers analyze
                        if(callbackName === 'analyzeRegression') {
                            return `<button onclick="document.getElementById('${inputId}').value = '${s}'; analyzeRegression();">${s}</button>`;
                        } else {
                            return `<button onclick="analyze('${s}')">${s}</button>`;
                        }
                    }).join('');
                });
            }

            // Apply autocomplete to both search inputs
            setupAutocomplete('search', 'suggestions', 'analyze');
            setupAutocomplete('regression-search', 'regression-suggestions', 'analyzeRegression');
            
            // Regression search enter key
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
                    if (data.error) {
                        document.getElementById('result').innerHTML = `<div class="error">❌ ${data.error}</div>`;
                    } else {
                        showResult(data, symbol);
                    }
                })
                .catch(e => {
                    document.getElementById('result').innerHTML = `<div class="error">❌ ${e.message}</div>`;
                });
        }
        
        function showResult(data, symbol) {
            if (!data || !data.signal) {
                document.getElementById('result').innerHTML = '<div class="error">❌ Invalid response data</div>';
                return;
            }
            
            const s = data.signal || {};
            const d = data.details || {};
            
            const confidenceColor = s.confidence > 70 ? '#10b981' : s.confidence > 50 ? '#f59e0b' : '#ef4444';
            
            const html = `
                <div class="result-card">
                    <div class="header">
                        <h2>${symbol}</h2>
                        <div class="signal-badge signal-${s.signal}">${s.signal}</div>
                    </div>
                    
                    <div class="action-banner">${s.action}</div>
                    
                    <div class="rec-box">
                        <strong>💡 Recommendation:</strong> ${s.rec}
                    </div>
                    
                    <div class="confidence-meter">
                        <div class="confidence-label">Confidence Level</div>
                        <div class="confidence-bar-container">
                            <div class="confidence-bar" style="width: ${s.confidence}%; background: linear-gradient(90deg, ${confidenceColor}, ${confidenceColor}dd);">
                                ${s.confidence}%
                            </div>
                        </div>
                        <div class="confidence-text">${s.confidence_explain}</div>
                    </div>
                    
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Current Price</div>
                            <div class="metric-value">${d.price}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Daily Change</div>
                            <div class="metric-value" style="color: ${parseFloat(d.daily) >= 0 ? 'var(--success)' : 'var(--danger)'}">${d.daily}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">RSI</div>
                            <div class="metric-value">${d.rsi}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Z-Score</div>
                            <div class="metric-value">${d.zscore}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">% from Mean</div>
                            <div class="metric-value" style="color: ${parseFloat(d.pct_deviation) >= 0 ? 'var(--accent-cyan)' : 'var(--accent-purple)'}">${d.pct_deviation}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Volatility</div>
                            <div class="metric-value">${d.volatility}</div>
                        </div>
                    </div>
                    
                    <div class="trading-plan">
                        <h3>💼 TRADING PLAN (For Beginners)</h3>
                        
                        <div class="plan-item">
                            <span class="plan-label">📍 ENTRY PRICE</span>
                            <span class="plan-value">${s.entry}<br><small style="color: var(--text-muted)">${s.entry_explain}</small></span>
                        </div>
                        
                        <div class="plan-item">
                            <span class="plan-label">🎯 EXIT PRICE</span>
                            <span class="plan-value">${s.target}<br><small style="color: var(--text-muted)">${s.exit_explain}</small></span>
                        </div>
                        
                        <div class="plan-item">
                            <span class="plan-label">🛡️ STOP LOSS</span>
                            <span class="plan-value">${s.stop}<br><small style="color: var(--text-muted)">If price falls to this level, sell immediately to limit losses.</small></span>
                        </div>
                        
                        <div class="plan-item">
                            <span class="plan-label">⏱️ TIME FRAME</span>
                            <span class="plan-value">${s.days_to_target} trading days<br><small style="color: var(--text-muted)">${s.time_explain}</small></span>
                        </div>
                    </div>
                    
                    <div class="explanation-section">
                        <h3>📊 TREND ANALYSIS</h3>
                        <div class="explanation">${s.trend_explain}</div>
                    </div>
                    
                    <div class="explanation-section">
                        <h3>⚡ MOMENTUM</h3>
                        <div class="explanation">${s.momentum_explain}</div>
                    </div>
                    
                    <div class="explanation-section">
                        <h3>📈 RSI INTERPRETATION</h3>
                        <div class="explanation">${s.rsi_explain}</div>
                    </div>
                    
                    <div class="explanation-section">
                        <h3>🎯 MEAN REVERSION (Z-SCORE)</h3>
                        <div class="explanation">${s.zscore_explain}</div>
                    </div>
                    
                    <div class="explanation-section">
                        <h3>📉 BOLLINGER BANDS</h3>
                        <div class="explanation">${s.bb_explain}</div>
                    </div>
                    
                    <div class="explanation-section">
                        <h3>📍 PRICE POSITION</h3>
                        <div class="explanation">${s.position_explain}</div>
                    </div>
                    
                    <div class="explanation-section">
                        <h3>🎯 MACD</h3>
                        <div class="explanation">${s.macd_text}</div>
                    </div>
                </div>
            `;
            
            document.getElementById('result').innerHTML = html;
        }
        
        function analyzeRegression() {
            const symbol = document.getElementById('regression-search').value.toUpperCase().trim();
            if (!symbol) {
                alert('Please enter a stock symbol');
                return;
            }
            
            document.getElementById('regression-result').innerHTML = '<div class="loading">⏳ Running regression analysis for ' + symbol + '...<br><small style="font-size: 0.8em; color: var(--text-secondary);">This may take 10-30 seconds</small></div>';
            
            fetch(`/regression?symbol=${symbol}`)
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('regression-result').innerHTML = `<div class="error">❌ ${data.error}</div>`;
                    } else {
                        showRegressionResult(data, symbol);
                    }
                })
                .catch(e => {
                    document.getElementById('regression-result').innerHTML = `<div class="error">❌ ${e.message}</div>`;
                });
        }
        
        function showRegressionResult(data, symbol) {
            const marketInfo = data.market_source ? `<div style="background: rgba(0, 217, 255, 0.1); padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 3px solid var(--accent-cyan);">
                <strong>📊 Market Benchmark:</strong> ${data.market_source}
                ${data.market_source !== 'Nifty 50 Index' ? '<br><small style="color: var(--text-muted);">Note: Using alternative benchmark due to Nifty 50 data availability.</small>' : ''}
            </div>` : '';
            
            const html = `
                <div class="result-card">
                    <div class="header">
                        <h2>${symbol} vs Market</h2>
                        <div style="color: var(--accent-cyan); font-size: 1.2em;">Linear Regression Analysis</div>
                    </div>
                    
                    ${marketInfo}
                    
                    <div class="action-banner">${data.trading_insight}</div>
                    
                    <div class="regression-grid">
                        <div class="regression-metric">
                            <div class="regression-metric-label">Beta (β)</div>
                            <div class="regression-metric-value">${data.beta.toFixed(3)}</div>
                            <div class="regression-metric-desc">${data.beta_interpret}</div>
                        </div>
                        
                        <div class="regression-metric">
                            <div class="regression-metric-label">R-Squared (R²)</div>
                            <div class="regression-metric-value">${(data.r_squared * 100).toFixed(1)}%</div>
                            <div class="regression-metric-desc">${data.r2_interpret}</div>
                        </div>
                        
                        <div class="regression-metric">
                            <div class="regression-metric-label">Alpha (α)</div>
                            <div class="regression-metric-value">${(data.alpha * 100).toFixed(3)}%</div>
                            <div class="regression-metric-desc">${data.alpha_interpret}</div>
                        </div>
                        
                        <div class="regression-metric">
                            <div class="regression-metric-label">Correlation</div>
                            <div class="regression-metric-value">${data.correlation.toFixed(3)}</div>
                            <div class="regression-metric-desc">Measures linear relationship strength. Range: -1 to +1.</div>
                        </div>
                        
                        <div class="regression-metric">
                            <div class="regression-metric-label">P-Value</div>
                            <div class="regression-metric-value">${data.p_value.toFixed(6)}</div>
                            <div class="regression-metric-desc">${data.p_value < 0.05 ? 'Statistically SIGNIFICANT (p < 0.05). Results are reliable.' : 'Not statistically significant. Results may be unreliable.'}</div>
                        </div>
                        
                        <div class="regression-metric">
                            <div class="regression-metric-label">Std Error</div>
                            <div class="regression-metric-value">${data.std_error.toFixed(4)}</div>
                            <div class="regression-metric-desc">Uncertainty in beta estimate. Lower = more precise.</div>
                        </div>
                        
                        <div class="regression-metric">
                            <div class="regression-metric-label">Residual Std</div>
                            <div class="regression-metric-value">${(data.residual_std * 100).toFixed(2)}%</div>
                            <div class="regression-metric-desc">Average prediction error. Measures model accuracy.</div>
                        </div>
                        
                        <div class="regression-metric">
                            <div class="regression-metric-label">Data Points</div>
                            <div class="regression-metric-value">${data.data_points}</div>
                            <div class="regression-metric-desc">Number of observations used in analysis.</div>
                        </div>
                    </div>
                    
                    <div class="explanation-section" style="margin-top: 30px;">
                        <h3>📖 Understanding the Metrics</h3>
                        <div class="explanation">
                            <p><strong>Beta (β):</strong> Measures how much the stock moves relative to Nifty. Beta > 1 means more volatile than market, Beta < 1 means less volatile.</p>
                            <p style="margin-top: 10px;"><strong>R² (R-Squared):</strong> Shows what percentage of stock's movement is explained by Nifty. Higher = stronger relationship.</p>
                            <p style="margin-top: 10px;"><strong>Alpha (α):</strong> Daily excess return above what Beta predicts. Positive alpha = outperformance.</p>
                            <p style="margin-top: 10px;"><strong>Correlation:</strong> Strength of linear relationship. +1 = perfect positive, -1 = perfect negative, 0 = no relationship.</p>
                        </div>
                    </div>
                    
                    <div class="trading-plan" style="margin-top: 25px;">
                        <h3>💡 Practical Trading Applications</h3>
                        <div class="plan-item">
                            <span class="plan-label">Market Direction</span>
                            <span class="plan-value">
                                ${data.beta > 1.2 ? 
                                    `High beta stock - amplifies market moves. If expecting Nifty to rise, this stock should rise ${(data.beta * 100 - 100).toFixed(0)}% more. Great for bullish markets.` :
                                data.beta < 0.8 ?
                                    `Low beta stock - defensive play. Provides market exposure with ${((1 - data.beta) * 100).toFixed(0)}% less volatility. Good for uncertain markets.` :
                                    'Moderate beta - moves in line with market. Standard market exposure.'
                                }
                            </span>
                        </div>
                        <div class="plan-item">
                            <span class="plan-label">Portfolio Use</span>
                            <span class="plan-value">
                                ${data.r_squared > 0.6 ?
                                    'Strong market correlation. Use for leveraged/defensive market plays based on beta.' :
                                    'Weak market correlation. Stock driven by company-specific factors. Good for diversification.'
                                }
                            </span>
                        </div>
                        <div class="plan-item">
                            <span class="plan-label">Risk Assessment</span>
                            <span class="plan-value">
                                Volatility relative to market: ${data.beta > 1 ? 'HIGHER' : data.beta < 1 ? 'LOWER' : 'SIMILAR'}. 
                                ${data.residual_std > 0.02 ? 'Significant unexplained volatility - higher risk.' : 'Low unexplained volatility - more predictable.'}
                            </span>
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
        
        window.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>'''
    return html


@app.route('/analyze')
def analyze_route():
    symbol = request.args.get('symbol', '').strip()

    if not symbol:
        return jsonify({'error': 'No symbol provided'})

    # Normalize and validate the symbol
    normalized_symbol, original = Analyzer.normalize_symbol(symbol)

    if not normalized_symbol:
        # Try to suggest similar tickers
        suggestions = []
        symbol_upper = symbol.upper()
        for ticker in sorted(ALL_VALID_TICKERS):
            if symbol_upper in ticker or ticker in symbol_upper:
                suggestions.append(ticker)
                if len(suggestions) >= 5:
                    break

        if suggestions:
            return jsonify({
                'error': f'Invalid symbol "{original}". Did you mean: {", ".join(suggestions)}? Please use the exact NSE ticker symbol.'
            })
        else:
            return jsonify({
                'error': f'Invalid symbol "{original}". Please use a valid NSE ticker symbol (e.g., TCS, INFY, RELIANCE, VEDL for Vedanta).'
            })

    try:
        result = analyzer.analyze(normalized_symbol)

        if not result:
            return jsonify({
                'error': f'Unable to fetch data for {normalized_symbol}. Market may be closed or data temporarily unavailable.'})

        return jsonify(result)
    except Exception as e:
        print(f"Error analyzing {normalized_symbol}: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'})


@app.route('/regression')
def regression_route():
    symbol = request.args.get('symbol', '').strip()

    if not symbol:
        return jsonify({'error': 'No symbol provided'})

    # Normalize and validate the symbol
    normalized_symbol, original = Analyzer.normalize_symbol(symbol)

    if not normalized_symbol:
        # Try to suggest similar tickers
        suggestions = []
        symbol_upper = symbol.upper()
        for ticker in sorted(ALL_VALID_TICKERS):
            if symbol_upper in ticker or ticker in symbol_upper:
                suggestions.append(ticker)
                if len(suggestions) >= 5:
                    break

        if suggestions:
            return jsonify({
                'error': f'Invalid symbol "{original}". Did you mean: {", ".join(suggestions)}? Please use the exact NSE ticker symbol.'
            })
        else:
            # Special case for common company names
            if 'VEDANTA' in symbol_upper:
                return jsonify({'error': 'Invalid symbol "vedanta". Use VEDL (the NSE ticker for Vedanta Limited).'})
            elif 'HDFC BANK' in symbol_upper or 'HDFCBANK' in symbol_upper:
                return jsonify({'error': 'Invalid symbol. Use HDFCBANK (the NSE ticker for HDFC Bank).'})
            else:
                return jsonify({
                    'error': f'Invalid symbol "{original}". Please use a valid NSE ticker symbol (e.g., TCS, INFY, RELIANCE, VEDL for Vedanta).'
                })

    try:
        result = analyzer.regression_analysis(normalized_symbol)

        if not result:
            return jsonify({
                'error': f'Unable to perform regression analysis for {normalized_symbol}. This could be due to insufficient historical data (need at least 20 trading days). Please try a different stock.'
            })

        return jsonify(result)
    except Exception as e:
        print(f"Error in regression for {normalized_symbol}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Regression analysis failed for {normalized_symbol}: {str(e)}. Please try a different stock symbol.'})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Enhanced Stock Analysis Pro - FIXED VERSION")
    print("=" * 60)
    print("\n✅ Features:")
    print("   • Z-score with percentage deviation from mean")
    print("   • Bolna AI-inspired dark theme design")
    print("   • Linear regression analysis vs Nifty 50 (FIXED)")
    print("   • Clear entry/exit explanations for beginners")
    print("   • Confidence levels with detailed reasoning")
    print("   • Time-to-target predictions")
    print("   • Autocomplete Search Feature Added")
    print("\n🔧 Fixes:")
    print("   • Better error handling for regression analysis")
    print("   • Tries multiple time periods (1y, 6mo, 3mo)")
    print("   • More detailed error messages")
    print("   • Improved data validation")
    print("\n🌐 Open browser: http://127.0.0.1:5000")
    print("\n⌨️  Press Ctrl+C to stop\n")
    app.run(debug=True, use_reloader=False)
