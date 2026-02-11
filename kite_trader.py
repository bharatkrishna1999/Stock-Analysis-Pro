"""
Kite Connect Automated Trader - Long Only
==========================================
Integrates with Zerodha's Kite Connect API to automate long-only trading
based on signals from the stock analysis system.

Flow:
  1. Authenticate via Kite Connect login flow (API key + request token)
  2. Configure watchlist, capital, and risk parameters
  3. Start the automation loop which periodically:
     - Scans watchlist stocks for BUY/SELL signals
     - Places buy orders when a BUY signal fires (with SL and target)
     - Exits positions when a SELL signal fires or stop/target is hit
  4. All orders are NSE equity, CNC (delivery) product type
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from collections import OrderedDict

try:
    from kiteconnect import KiteConnect, KiteTicker
except ImportError:
    KiteConnect = None
    KiteTicker = None

logger = logging.getLogger("kite_trader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

# ── Persistence file for state across restarts ──────────────────────────
STATE_FILE = os.path.join(os.path.dirname(__file__), ".kite_state.json")


def _load_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


class KiteTrader:
    """
    Long-only automated trader using Kite Connect API.

    Usage:
        trader = KiteTrader(api_key="xxx")
        trader.set_access_token(request_token, api_secret)
        trader.configure(capital=100000, risk_per_trade_pct=1.0, watchlist=["RELIANCE", "TCS"])
        trader.start()
    """

    # ── Exchange and product constants ──
    EXCHANGE = "NSE"
    PRODUCT = "CNC"         # delivery (long-term holding)
    ORDER_TYPE_MARKET = "MARKET"
    ORDER_TYPE_LIMIT = "LIMIT"
    ORDER_TYPE_SL = "SL"
    ORDER_TYPE_SLM = "SL-M"
    VARIETY_REGULAR = "regular"
    TRANSACTION_BUY = "BUY"
    TRANSACTION_SELL = "SELL"

    def __init__(self, api_key=None):
        if KiteConnect is None:
            raise ImportError(
                "kiteconnect package not installed. Run: pip install kiteconnect"
            )
        self.api_key = api_key or os.environ.get("KITE_API_KEY", "")
        self.api_secret = os.environ.get("KITE_API_SECRET", "")
        self.kite = KiteConnect(api_key=self.api_key) if self.api_key else None

        # ── Config ──
        self.capital = 100000           # total capital for trading (₹)
        self.risk_per_trade_pct = 1.0   # % of capital risked per trade
        self.max_open_positions = 5     # max concurrent positions
        self.min_confidence = 60        # minimum signal confidence to act
        self.min_risk_reward = 1.5      # minimum R:R ratio to enter
        self.scan_interval_sec = 300    # seconds between scans (5 min)
        self.watchlist = []             # list of NSE ticker symbols

        # ── State ──
        self.positions = OrderedDict()  # symbol -> position dict
        self.order_log = []             # list of all order records
        self.is_running = False
        self._thread = None
        self._stop_event = threading.Event()
        self.access_token = None
        self.last_scan_time = None
        self.last_error = None

        # ── Restore persisted state ──
        self._restore_state()

    # ════════════════════════════════════════════════════════════════════
    # Authentication
    # ════════════════════════════════════════════════════════════════════

    def get_login_url(self):
        """Return the Kite Connect login URL the user must visit."""
        if not self.kite:
            return None
        return self.kite.login_url()

    def set_access_token(self, request_token, api_secret=None):
        """
        Exchange the request_token for an access_token.
        Called after the user completes the Kite login redirect.
        """
        secret = api_secret or self.api_secret
        if not secret:
            raise ValueError("api_secret is required")
        data = self.kite.generate_session(request_token, api_secret=secret)
        self.access_token = data["access_token"]
        self.kite.set_access_token(self.access_token)
        logger.info("Kite access token set successfully")
        self._persist_state()
        return self.access_token

    def set_access_token_direct(self, access_token):
        """Set an already-obtained access token directly."""
        self.access_token = access_token
        self.kite.set_access_token(access_token)
        logger.info("Kite access token set directly")
        self._persist_state()

    def is_authenticated(self):
        """Check if we have a valid session."""
        if not self.access_token:
            return False
        try:
            profile = self.kite.profile()
            return bool(profile.get("user_id"))
        except Exception:
            return False

    # ════════════════════════════════════════════════════════════════════
    # Configuration
    # ════════════════════════════════════════════════════════════════════

    def configure(self, capital=None, risk_per_trade_pct=None, max_open_positions=None,
                  min_confidence=None, min_risk_reward=None, scan_interval_sec=None,
                  watchlist=None):
        """Update trading parameters. Only provided values are changed."""
        if capital is not None:
            self.capital = float(capital)
        if risk_per_trade_pct is not None:
            self.risk_per_trade_pct = float(risk_per_trade_pct)
        if max_open_positions is not None:
            self.max_open_positions = int(max_open_positions)
        if min_confidence is not None:
            self.min_confidence = int(min_confidence)
        if min_risk_reward is not None:
            self.min_risk_reward = float(min_risk_reward)
        if scan_interval_sec is not None:
            self.scan_interval_sec = int(scan_interval_sec)
        if watchlist is not None:
            self.watchlist = [s.upper().strip() for s in watchlist if s.strip()]
        self._persist_state()
        logger.info(f"Config updated: capital={self.capital}, risk={self.risk_per_trade_pct}%, "
                     f"max_pos={self.max_open_positions}, watchlist={len(self.watchlist)} stocks")

    def get_config(self):
        return {
            "capital": self.capital,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "max_open_positions": self.max_open_positions,
            "min_confidence": self.min_confidence,
            "min_risk_reward": self.min_risk_reward,
            "scan_interval_sec": self.scan_interval_sec,
            "watchlist": self.watchlist,
        }

    # ════════════════════════════════════════════════════════════════════
    # Position sizing
    # ════════════════════════════════════════════════════════════════════

    def _calculate_quantity(self, price, stop_loss):
        """
        Position sizing using fixed-fractional risk model.
        risk_amount = capital * risk_per_trade_pct / 100
        quantity = risk_amount / (price - stop_loss)
        """
        risk_amount = self.capital * (self.risk_per_trade_pct / 100.0)
        risk_per_share = abs(price - stop_loss)
        if risk_per_share <= 0:
            return 0
        qty = int(risk_amount / risk_per_share)
        # Ensure we can afford the position
        max_affordable = int(self.capital / price) if price > 0 else 0
        qty = min(qty, max_affordable)
        return max(qty, 0)

    # ════════════════════════════════════════════════════════════════════
    # Order Execution
    # ════════════════════════════════════════════════════════════════════

    def _place_buy_order(self, symbol, quantity, price=None):
        """Place a buy order. Returns order_id or None."""
        try:
            params = {
                "tradingsymbol": symbol,
                "exchange": self.EXCHANGE,
                "transaction_type": self.TRANSACTION_BUY,
                "quantity": quantity,
                "product": self.PRODUCT,
                "variety": self.VARIETY_REGULAR,
            }
            if price:
                params["order_type"] = self.ORDER_TYPE_LIMIT
                params["price"] = round(price, 1)
            else:
                params["order_type"] = self.ORDER_TYPE_MARKET

            order_id = self.kite.place_order(**params)
            logger.info(f"BUY order placed: {symbol} qty={quantity} price={price or 'MARKET'} order_id={order_id}")
            return order_id
        except Exception as e:
            logger.error(f"BUY order failed for {symbol}: {e}")
            self.last_error = str(e)
            return None

    def _place_sell_order(self, symbol, quantity, price=None):
        """Place a sell order to exit a long position. Returns order_id or None."""
        try:
            params = {
                "tradingsymbol": symbol,
                "exchange": self.EXCHANGE,
                "transaction_type": self.TRANSACTION_SELL,
                "quantity": quantity,
                "product": self.PRODUCT,
                "variety": self.VARIETY_REGULAR,
            }
            if price:
                params["order_type"] = self.ORDER_TYPE_LIMIT
                params["price"] = round(price, 1)
            else:
                params["order_type"] = self.ORDER_TYPE_MARKET

            order_id = self.kite.place_order(**params)
            logger.info(f"SELL order placed: {symbol} qty={quantity} price={price or 'MARKET'} order_id={order_id}")
            return order_id
        except Exception as e:
            logger.error(f"SELL order failed for {symbol}: {e}")
            self.last_error = str(e)
            return None

    def _place_stoploss_order(self, symbol, quantity, trigger_price):
        """Place a stop-loss market order to protect a long position."""
        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=self.EXCHANGE,
                transaction_type=self.TRANSACTION_SELL,
                quantity=quantity,
                product=self.PRODUCT,
                order_type=self.ORDER_TYPE_SLM,
                trigger_price=round(trigger_price, 1),
                variety=self.VARIETY_REGULAR,
            )
            logger.info(f"SL order placed: {symbol} qty={quantity} trigger={trigger_price:.2f} order_id={order_id}")
            return order_id
        except Exception as e:
            logger.error(f"SL order failed for {symbol}: {e}")
            self.last_error = str(e)
            return None

    def _cancel_order(self, order_id):
        """Cancel an open order."""
        try:
            self.kite.cancel_order(variety=self.VARIETY_REGULAR, order_id=order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel failed for order {order_id}: {e}")
            return False

    # ════════════════════════════════════════════════════════════════════
    # Signal-to-Trade Logic (long only)
    # ════════════════════════════════════════════════════════════════════

    def process_signal(self, symbol, analysis_result, analyzer):
        """
        Core decision engine. Given an analysis result from the Analyzer,
        decide whether to buy, sell, or hold.

        Long-only rules:
          BUY  → If no position exists, confidence >= threshold, R:R >= threshold → enter long
          SELL → If position exists → exit long (market sell)
          HOLD → Do nothing
        """
        if not analysis_result:
            return None

        sig_data = analysis_result.get("signal", {})
        signal = sig_data.get("signal", "HOLD")
        confidence = sig_data.get("confidence", 0)
        risk_reward = sig_data.get("risk_reward", 0)
        entry_raw = sig_data.get("entry_raw", 0)
        stop_raw = sig_data.get("stop_raw", 0)
        target_raw = sig_data.get("target_raw", 0)
        price_raw = analysis_result.get("details", {}).get("price_raw", 0)

        has_position = symbol in self.positions
        action_taken = None

        if signal == "BUY" and not has_position:
            # ── Validate entry criteria ──
            if confidence < self.min_confidence:
                logger.info(f"SKIP {symbol}: confidence {confidence}% < {self.min_confidence}% threshold")
                return {"action": "SKIP", "reason": f"confidence {confidence}% below threshold"}

            if risk_reward < self.min_risk_reward:
                logger.info(f"SKIP {symbol}: R:R {risk_reward}x < {self.min_risk_reward}x threshold")
                return {"action": "SKIP", "reason": f"R:R {risk_reward}x below threshold"}

            if len(self.positions) >= self.max_open_positions:
                logger.info(f"SKIP {symbol}: max {self.max_open_positions} positions reached")
                return {"action": "SKIP", "reason": "max open positions reached"}

            # ── Calculate position size ──
            buy_price = price_raw  # use current market price for market orders
            qty = self._calculate_quantity(buy_price, stop_raw)
            if qty <= 0:
                logger.info(f"SKIP {symbol}: calculated quantity is 0")
                return {"action": "SKIP", "reason": "position size too small"}

            # ── Place buy order (market) ──
            order_id = self._place_buy_order(symbol, qty)
            if not order_id:
                return {"action": "ERROR", "reason": self.last_error}

            # ── Place protective stop-loss ──
            sl_order_id = self._place_stoploss_order(symbol, qty, stop_raw)

            # ── Record position ──
            position = {
                "symbol": symbol,
                "quantity": qty,
                "entry_price": buy_price,
                "stop_loss": stop_raw,
                "target": target_raw,
                "buy_order_id": order_id,
                "sl_order_id": sl_order_id,
                "entry_time": datetime.now().isoformat(),
                "confidence": confidence,
                "risk_reward": risk_reward,
                "status": "OPEN",
            }
            self.positions[symbol] = position
            self._log_order("BUY", symbol, qty, buy_price, order_id, confidence, risk_reward)
            self._persist_state()

            action_taken = {
                "action": "BUY",
                "symbol": symbol,
                "quantity": qty,
                "price": buy_price,
                "stop_loss": stop_raw,
                "target": target_raw,
                "order_id": order_id,
                "confidence": confidence,
                "risk_reward": risk_reward,
            }
            logger.info(f"ENTERED LONG: {symbol} qty={qty} price={buy_price:.2f} "
                        f"SL={stop_raw:.2f} target={target_raw:.2f} conf={confidence}%")

        elif signal == "SELL" and has_position:
            # ── Exit the long position ──
            pos = self.positions[symbol]
            qty = pos["quantity"]

            # Cancel existing SL order before selling
            if pos.get("sl_order_id"):
                self._cancel_order(pos["sl_order_id"])

            # Place market sell
            order_id = self._place_sell_order(symbol, qty)
            if not order_id:
                return {"action": "ERROR", "reason": self.last_error}

            # Calculate P&L
            exit_price = price_raw
            pnl = (exit_price - pos["entry_price"]) * qty
            pnl_pct = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100

            pos["exit_price"] = exit_price
            pos["exit_time"] = datetime.now().isoformat()
            pos["sell_order_id"] = order_id
            pos["pnl"] = round(pnl, 2)
            pos["pnl_pct"] = round(pnl_pct, 2)
            pos["status"] = "CLOSED"

            self._log_order("SELL", symbol, qty, exit_price, order_id, confidence, risk_reward, pnl=pnl)
            del self.positions[symbol]
            self._persist_state()

            action_taken = {
                "action": "SELL",
                "symbol": symbol,
                "quantity": qty,
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "order_id": order_id,
            }
            logger.info(f"EXITED LONG: {symbol} qty={qty} exit={exit_price:.2f} "
                        f"P&L=₹{pnl:.2f} ({pnl_pct:+.2f}%)")

        else:
            action_taken = {
                "action": "HOLD",
                "symbol": symbol,
                "signal": signal,
                "has_position": has_position,
            }

        return action_taken

    # ════════════════════════════════════════════════════════════════════
    # Automated Scan Loop
    # ════════════════════════════════════════════════════════════════════

    def _scan_once(self, analyzer):
        """Run one full scan of the watchlist and process signals."""
        results = []
        self.last_scan_time = datetime.now().isoformat()

        # First check positions for exit signals (even if not in watchlist)
        symbols_to_scan = list(set(self.watchlist + list(self.positions.keys())))

        for symbol in symbols_to_scan:
            try:
                analysis = analyzer.analyze(symbol)
                if not analysis:
                    logger.warning(f"No analysis data for {symbol}")
                    continue
                action = self.process_signal(symbol, analysis, analyzer)
                if action:
                    results.append(action)
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                results.append({"action": "ERROR", "symbol": symbol, "reason": str(e)})

        return results

    def _run_loop(self, analyzer):
        """Background loop that scans periodically."""
        logger.info(f"Automation loop started. Interval: {self.scan_interval_sec}s, "
                     f"Watchlist: {len(self.watchlist)} stocks")
        while not self._stop_event.is_set():
            try:
                # Only trade during market hours (9:15 AM - 3:30 PM IST)
                now = datetime.now()
                market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

                if now.weekday() >= 5:  # Saturday or Sunday
                    logger.info("Weekend - market closed. Sleeping 1 hour.")
                    self._stop_event.wait(3600)
                    continue

                if now < market_open or now > market_close:
                    logger.info(f"Outside market hours ({now.strftime('%H:%M')}). Sleeping 5 min.")
                    self._stop_event.wait(300)
                    continue

                results = self._scan_once(analyzer)
                actions = [r for r in results if r.get("action") not in ("HOLD", None)]
                if actions:
                    logger.info(f"Scan complete: {len(actions)} actions taken")
                    for a in actions:
                        logger.info(f"  {a}")
                else:
                    logger.info(f"Scan complete: no actions. {len(self.positions)} open positions.")

            except Exception as e:
                logger.error(f"Scan loop error: {e}")
                self.last_error = str(e)

            self._stop_event.wait(self.scan_interval_sec)

        logger.info("Automation loop stopped")

    def start(self, analyzer):
        """Start the automated trading loop in a background thread."""
        if self.is_running:
            logger.warning("Already running")
            return False

        if not self.access_token:
            raise ValueError("Not authenticated. Call set_access_token() first.")

        if not self.watchlist:
            raise ValueError("Watchlist is empty. Call configure(watchlist=[...]) first.")

        self._stop_event.clear()
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, args=(analyzer,), daemon=True)
        self._thread.start()
        logger.info("Kite Trader automation started")
        return True

    def stop(self):
        """Stop the automated trading loop."""
        if not self.is_running:
            return False
        self._stop_event.set()
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("Kite Trader automation stopped")
        return True

    # ════════════════════════════════════════════════════════════════════
    # Manual trade operations
    # ════════════════════════════════════════════════════════════════════

    def manual_buy(self, symbol, quantity=None, analyzer=None):
        """Manually buy a stock. If no quantity, auto-calculate from analysis."""
        symbol = symbol.upper().strip()
        if symbol in self.positions:
            return {"error": f"Already holding {symbol}"}

        if not quantity and analyzer:
            analysis = analyzer.analyze(symbol)
            if analysis:
                sig_data = analysis.get("signal", {})
                price = analysis.get("details", {}).get("price_raw", 0)
                stop = sig_data.get("stop_raw", 0)
                quantity = self._calculate_quantity(price, stop)

        if not quantity or quantity <= 0:
            return {"error": "Could not determine quantity"}

        order_id = self._place_buy_order(symbol, quantity)
        if order_id:
            self.positions[symbol] = {
                "symbol": symbol,
                "quantity": quantity,
                "buy_order_id": order_id,
                "entry_time": datetime.now().isoformat(),
                "status": "OPEN",
            }
            self._log_order("BUY", symbol, quantity, 0, order_id, 0, 0)
            self._persist_state()
            return {"success": True, "order_id": order_id, "quantity": quantity}
        return {"error": self.last_error}

    def manual_sell(self, symbol):
        """Manually sell an existing position."""
        symbol = symbol.upper().strip()
        if symbol not in self.positions:
            return {"error": f"No position in {symbol}"}

        pos = self.positions[symbol]
        qty = pos["quantity"]

        if pos.get("sl_order_id"):
            self._cancel_order(pos["sl_order_id"])

        order_id = self._place_sell_order(symbol, qty)
        if order_id:
            pos["sell_order_id"] = order_id
            pos["exit_time"] = datetime.now().isoformat()
            pos["status"] = "CLOSED"
            self._log_order("SELL", symbol, qty, 0, order_id, 0, 0)
            del self.positions[symbol]
            self._persist_state()
            return {"success": True, "order_id": order_id, "quantity": qty}
        return {"error": self.last_error}

    def exit_all_positions(self):
        """Emergency exit: sell all open positions at market."""
        results = []
        for symbol in list(self.positions.keys()):
            result = self.manual_sell(symbol)
            results.append({"symbol": symbol, **result})
        return results

    # ════════════════════════════════════════════════════════════════════
    # Portfolio & Order Status
    # ════════════════════════════════════════════════════════════════════

    def get_positions(self):
        """Return current open positions."""
        return dict(self.positions)

    def get_kite_positions(self):
        """Fetch live positions from Kite."""
        try:
            return self.kite.positions()
        except Exception as e:
            logger.error(f"Failed to fetch Kite positions: {e}")
            return None

    def get_kite_orders(self):
        """Fetch today's orders from Kite."""
        try:
            return self.kite.orders()
        except Exception as e:
            logger.error(f"Failed to fetch Kite orders: {e}")
            return None

    def get_kite_holdings(self):
        """Fetch holdings (CNC positions) from Kite."""
        try:
            return self.kite.holdings()
        except Exception as e:
            logger.error(f"Failed to fetch Kite holdings: {e}")
            return None

    def get_margins(self):
        """Fetch account margins/funds."""
        try:
            return self.kite.margins()
        except Exception as e:
            logger.error(f"Failed to fetch margins: {e}")
            return None

    def get_order_log(self, limit=50):
        """Return recent trade log entries."""
        return self.order_log[-limit:]

    def get_status(self):
        """Return full trader status."""
        return {
            "authenticated": self.is_authenticated() if self.access_token else False,
            "is_running": self.is_running,
            "access_token_set": bool(self.access_token),
            "open_positions": len(self.positions),
            "positions": dict(self.positions),
            "watchlist_count": len(self.watchlist),
            "watchlist": self.watchlist,
            "config": self.get_config(),
            "last_scan_time": self.last_scan_time,
            "last_error": self.last_error,
            "total_trades": len(self.order_log),
            "recent_trades": self.order_log[-10:],
        }

    # ════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ════════════════════════════════════════════════════════════════════

    def _log_order(self, side, symbol, qty, price, order_id, confidence, risk_reward, pnl=None):
        entry = {
            "time": datetime.now().isoformat(),
            "side": side,
            "symbol": symbol,
            "quantity": qty,
            "price": price,
            "order_id": order_id,
            "confidence": confidence,
            "risk_reward": risk_reward,
        }
        if pnl is not None:
            entry["pnl"] = round(pnl, 2)
        self.order_log.append(entry)

    def _persist_state(self):
        state = {
            "positions": dict(self.positions),
            "order_log": self.order_log[-200:],  # keep last 200
            "config": self.get_config(),
            "access_token": self.access_token,
            "last_scan_time": self.last_scan_time,
        }
        _save_state(state)

    def _restore_state(self):
        state = _load_state()
        if not state:
            return
        if "positions" in state:
            self.positions = OrderedDict(state["positions"])
        if "order_log" in state:
            self.order_log = state["order_log"]
        if "config" in state:
            cfg = state["config"]
            self.capital = cfg.get("capital", self.capital)
            self.risk_per_trade_pct = cfg.get("risk_per_trade_pct", self.risk_per_trade_pct)
            self.max_open_positions = cfg.get("max_open_positions", self.max_open_positions)
            self.min_confidence = cfg.get("min_confidence", self.min_confidence)
            self.min_risk_reward = cfg.get("min_risk_reward", self.min_risk_reward)
            self.scan_interval_sec = cfg.get("scan_interval_sec", self.scan_interval_sec)
            self.watchlist = cfg.get("watchlist", self.watchlist)
        if "access_token" in state and state["access_token"]:
            self.access_token = state["access_token"]
            if self.kite:
                self.kite.set_access_token(self.access_token)
        if "last_scan_time" in state:
            self.last_scan_time = state["last_scan_time"]
        logger.info(f"Restored state: {len(self.positions)} positions, {len(self.order_log)} log entries")
