"""
Kite Trading Dashboard
======================
Standalone web app for automated trading with Zerodha Kite Connect.
Uses Stock Analysis Pro's technical analysis engine for signals.

Deploy on Render free tier:
    gunicorn app:app --timeout 300
"""

from flask import Flask, jsonify, request
import json
import os
import logging
from datetime import datetime
from threading import Lock

# ── Import the analysis engine ──────────────────────────────────────────
from stock_trading_system import Analyzer, STOCKS, ALL_VALID_TICKERS

try:
    from kite_trader import KiteTrader
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kite_dashboard")

app = Flask(__name__)

# ── Global instances ────────────────────────────────────────────────────
analyzer = Analyzer()
kite_trader = None

if KITE_AVAILABLE:
    _api_key = os.environ.get("KITE_API_KEY", "")
    if _api_key:
        kite_trader = KiteTrader(api_key=_api_key)

# ── Decision Log (persists every signal evaluation) ─────────────────────
DECISION_LOG_FILE = os.path.join(os.path.dirname(__file__), ".decisions.json")
MAX_DECISIONS = 500
_decision_lock = Lock()


def _load_decisions():
    try:
        with open(DECISION_LOG_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_decisions(decisions):
    try:
        with open(DECISION_LOG_FILE, "w") as f:
            json.dump(decisions, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save decisions: {e}")


def log_decision(symbol, analysis, action):
    """Callback invoked by KiteTrader on every signal evaluation."""
    sig = analysis.get("signal", {}) if analysis else {}
    det = analysis.get("details", {}) if analysis else {}
    decision = {
        "time": datetime.now().isoformat(),
        "symbol": symbol,
        "signal": sig.get("signal", ""),
        "confidence": sig.get("confidence", 0),
        "risk_reward": sig.get("risk_reward", 0),
        "price": det.get("price_raw", 0),
        "entry": sig.get("entry_raw", 0),
        "stop": sig.get("stop_raw", 0),
        "target": sig.get("target_raw", 0),
        "action": action.get("action", ""),
        "reason": action.get("reason", ""),
        "rec": sig.get("rec", ""),
        "rsi": det.get("rsi_raw", 0),
        "volatility": sig.get("volatility_raw", 0),
    }
    with _decision_lock:
        decisions = _load_decisions()
        decisions.append(decision)
        if len(decisions) > MAX_DECISIONS:
            decisions = decisions[-MAX_DECISIONS:]
        _save_decisions(decisions)


# Wire up the callback
if kite_trader:
    kite_trader.on_decision = log_decision


# ── Helper ──────────────────────────────────────────────────────────────
def _require_kite():
    global kite_trader
    if not KITE_AVAILABLE:
        return None, (jsonify({"error": "kiteconnect package not installed"}), 500)
    if kite_trader is None:
        return None, (jsonify({"error": "Kite trader not initialized. POST /api/init with api_key."}), 400)
    return kite_trader, None


# ════════════════════════════════════════════════════════════════════════
# API ROUTES
# ════════════════════════════════════════════════════════════════════════

# ── Initialization & Auth ───────────────────────────────────────────────

@app.route('/api/init', methods=['POST'])
def api_init():
    global kite_trader
    if not KITE_AVAILABLE:
        return jsonify({"error": "kiteconnect package not installed"}), 500
    data = request.get_json(silent=True) or {}
    api_key = data.get("api_key", os.environ.get("KITE_API_KEY", ""))
    if not api_key:
        return jsonify({"error": "api_key is required"}), 400
    kite_trader = KiteTrader(api_key=api_key)
    kite_trader.on_decision = log_decision
    if data.get("api_secret"):
        kite_trader.api_secret = data["api_secret"]
    login_url = kite_trader.get_login_url()
    return jsonify({"status": "initialized", "login_url": login_url})


@app.route('/api/login-url')
def api_login_url():
    trader, err = _require_kite()
    if err:
        return err
    return jsonify({"login_url": trader.get_login_url()})


@app.route('/api/auth', methods=['POST'])
def api_auth():
    trader, err = _require_kite()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    request_token = data.get("request_token", "")
    api_secret = data.get("api_secret", trader.api_secret)
    if not request_token:
        return jsonify({"error": "request_token is required"}), 400
    if not api_secret:
        return jsonify({"error": "api_secret is required"}), 400
    try:
        token = trader.set_access_token(request_token, api_secret)
        return jsonify({"status": "authenticated", "access_token": token})
    except Exception as e:
        return jsonify({"error": f"Authentication failed: {e}"}), 401


@app.route('/api/auth/token', methods=['POST'])
def api_auth_token():
    trader, err = _require_kite()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    token = data.get("access_token", "")
    if not token:
        return jsonify({"error": "access_token is required"}), 400
    trader.set_access_token_direct(token)
    return jsonify({"status": "token set"})


# ── Configuration ───────────────────────────────────────────────────────

@app.route('/api/configure', methods=['POST'])
def api_configure():
    trader, err = _require_kite()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    trader.configure(
        capital=data.get("capital"),
        risk_per_trade_pct=data.get("risk_per_trade_pct"),
        max_open_positions=data.get("max_open_positions"),
        min_confidence=data.get("min_confidence"),
        min_risk_reward=data.get("min_risk_reward"),
        scan_interval_sec=data.get("scan_interval_sec"),
        watchlist=data.get("watchlist"),
    )
    return jsonify({"status": "configured", "config": trader.get_config()})


# ── Trading Control ─────────────────────────────────────────────────────

@app.route('/api/start', methods=['POST'])
def api_start():
    trader, err = _require_kite()
    if err:
        return err
    try:
        started = trader.start(analyzer)
        return jsonify({"status": "started" if started else "already_running"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/stop', methods=['POST'])
def api_stop():
    trader, err = _require_kite()
    if err:
        return err
    stopped = trader.stop()
    return jsonify({"status": "stopped" if stopped else "was_not_running"})


@app.route('/api/scan', methods=['POST'])
def api_scan():
    """Run a single scan of the watchlist. Works even without Kite auth (analysis only)."""
    trader, err = _require_kite()
    if err:
        return err
    results = trader._scan_once(analyzer)
    return jsonify({"scan_time": trader.last_scan_time, "results": results})


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Analyze a single stock (no Kite required). Returns full analysis + what the trader would do."""
    data = request.get_json(silent=True) or {}
    symbol = data.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    normalized, _ = Analyzer.normalize_symbol(symbol)
    if not normalized:
        return jsonify({"error": f"Unknown symbol: {symbol}"}), 400
    analysis = analyzer.analyze(normalized)
    if not analysis:
        return jsonify({"error": f"Analysis failed for {normalized}"}), 500

    # Simulate what the trader would decide
    sig = analysis.get("signal", {})
    decision_preview = {
        "signal": sig.get("signal", "HOLD"),
        "confidence": sig.get("confidence", 0),
        "risk_reward": sig.get("risk_reward", 0),
        "entry": sig.get("entry", ""),
        "stop": sig.get("stop", ""),
        "target": sig.get("target", ""),
        "rec": sig.get("rec", ""),
        "action": sig.get("action", ""),
    }
    # Would it pass the thresholds?
    if kite_trader:
        if sig.get("signal") == "BUY":
            if sig.get("confidence", 0) < kite_trader.min_confidence:
                decision_preview["would_trade"] = False
                decision_preview["skip_reason"] = f"confidence {sig.get('confidence')}% < {kite_trader.min_confidence}% threshold"
            elif sig.get("risk_reward", 0) < kite_trader.min_risk_reward:
                decision_preview["would_trade"] = False
                decision_preview["skip_reason"] = f"R:R {sig.get('risk_reward')}x < {kite_trader.min_risk_reward}x threshold"
            else:
                decision_preview["would_trade"] = True
        else:
            decision_preview["would_trade"] = False
            decision_preview["skip_reason"] = f"Signal is {sig.get('signal', 'HOLD')}, not BUY"

    return jsonify({"symbol": normalized, "decision": decision_preview, "analysis": analysis})


# ── Manual Trading ──────────────────────────────────────────────────────

@app.route('/api/buy', methods=['POST'])
def api_buy():
    trader, err = _require_kite()
    if err:
        return err
    if not trader.access_token:
        return jsonify({"error": "Not authenticated"}), 401
    data = request.get_json(silent=True) or {}
    symbol = data.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    normalized, _ = Analyzer.normalize_symbol(symbol)
    if not normalized:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
    result = trader.manual_buy(normalized, quantity=data.get("quantity"), analyzer=analyzer)
    if result.get("error"):
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/sell', methods=['POST'])
def api_sell():
    trader, err = _require_kite()
    if err:
        return err
    if not trader.access_token:
        return jsonify({"error": "Not authenticated"}), 401
    data = request.get_json(silent=True) or {}
    symbol = data.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    result = trader.manual_sell(symbol)
    if result.get("error"):
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/exit-all', methods=['POST'])
def api_exit_all():
    trader, err = _require_kite()
    if err:
        return err
    if not trader.access_token:
        return jsonify({"error": "Not authenticated"}), 401
    return jsonify({"results": trader.exit_all_positions()})


# ── Data Endpoints ──────────────────────────────────────────────────────

@app.route('/api/decisions')
def api_decisions():
    limit = request.args.get("limit", 100, type=int)
    decisions = _load_decisions()
    return jsonify({"decisions": decisions[-limit:]})


@app.route('/api/status')
def api_status():
    status = {
        "kite_available": KITE_AVAILABLE,
        "trader_initialized": kite_trader is not None,
        "authenticated": False,
        "is_running": False,
        "open_positions": 0,
        "config": {},
        "last_scan_time": None,
        "total_decisions": len(_load_decisions()),
    }
    if kite_trader:
        status.update({
            "authenticated": kite_trader.is_authenticated() if kite_trader.access_token else False,
            "access_token_set": bool(kite_trader.access_token),
            "is_running": kite_trader.is_running,
            "open_positions": len(kite_trader.positions),
            "config": kite_trader.get_config(),
            "last_scan_time": kite_trader.last_scan_time,
            "last_error": kite_trader.last_error,
        })
    return jsonify(status)


@app.route('/api/positions')
def api_positions():
    if not kite_trader:
        return jsonify({"positions": {}})
    return jsonify({"positions": kite_trader.get_positions()})


@app.route('/api/positions/live')
def api_positions_live():
    trader, err = _require_kite()
    if err:
        return err
    if not trader.access_token:
        return jsonify({"error": "Not authenticated"}), 401
    positions = trader.get_kite_positions()
    if positions is None:
        return jsonify({"error": "Failed to fetch positions"}), 500
    return jsonify(positions)


@app.route('/api/orders')
def api_orders():
    trader, err = _require_kite()
    if err:
        return err
    if not trader.access_token:
        return jsonify({"error": "Not authenticated"}), 401
    orders = trader.get_kite_orders()
    if orders is None:
        return jsonify({"error": "Failed to fetch orders"}), 500
    return jsonify({"orders": orders})


@app.route('/api/holdings')
def api_holdings():
    trader, err = _require_kite()
    if err:
        return err
    if not trader.access_token:
        return jsonify({"error": "Not authenticated"}), 401
    holdings = trader.get_kite_holdings()
    if holdings is None:
        return jsonify({"error": "Failed to fetch holdings"}), 500
    return jsonify({"holdings": holdings})


@app.route('/api/margins')
def api_margins():
    trader, err = _require_kite()
    if err:
        return err
    if not trader.access_token:
        return jsonify({"error": "Not authenticated"}), 401
    margins = trader.get_margins()
    if margins is None:
        return jsonify({"error": "Failed to fetch margins"}), 500
    return jsonify(margins)


@app.route('/api/log')
def api_trade_log():
    if not kite_trader:
        return jsonify({"trades": []})
    limit = request.args.get("limit", 50, type=int)
    return jsonify({"trades": kite_trader.get_order_log(limit)})


@app.route('/api/stocks')
def api_stocks():
    """Return sector-wise stock list for the watchlist picker."""
    return jsonify({"sectors": {k: v for k, v in STOCKS.items()}, "total": len(ALL_VALID_TICKERS)})


# ════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kite Trading Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
    --bg: #0b0f19;
    --bg-card: #131a2b;
    --bg-card-hover: #182035;
    --border: #1e2a42;
    --accent: #00d9ff;
    --accent-dim: rgba(0,217,255,0.15);
    --green: #06ffa5;
    --green-dim: rgba(6,255,165,0.12);
    --red: #ff4757;
    --red-dim: rgba(255,71,87,0.12);
    --yellow: #ffc048;
    --yellow-dim: rgba(255,192,72,0.12);
    --text: #e8ecf4;
    --text-dim: #7a8ba8;
    --text-muted: #4a5872;
    --purple: #9d4edd;
    --mono: 'JetBrains Mono', monospace;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
}
.container { max-width: 1280px; margin: 0 auto; padding: 20px; }

/* ── Header ──────────────────────────────── */
header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
    flex-wrap: wrap;
    gap: 12px;
}
.logo {
    font-size: 20px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.5px;
}
.logo span { color: var(--text-dim); font-weight: 400; }
.status-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    border: 1px solid var(--border);
    background: var(--bg-card);
}
.pill .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--text-muted);
}
.pill.on .dot { background: var(--green); box-shadow: 0 0 6px var(--green); }
.pill.off .dot { background: var(--red); }
.pill.warn .dot { background: var(--yellow); }

/* ── Tabs ────────────────────────────────── */
.tabs {
    display: flex;
    gap: 2px;
    background: var(--bg-card);
    border-radius: 10px;
    padding: 3px;
    margin-bottom: 24px;
    border: 1px solid var(--border);
}
.tab-btn {
    flex: 1;
    padding: 10px 16px;
    border: none;
    background: transparent;
    color: var(--text-dim);
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border-radius: 8px;
    transition: all 0.2s;
}
.tab-btn:hover { color: var(--text); background: rgba(255,255,255,0.04); }
.tab-btn.active {
    color: var(--accent);
    background: var(--accent-dim);
}
.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* ── Cards ───────────────────────────────── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.card h3 {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 16px;
}
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}
.stat-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
}
.stat-card .label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.stat-card .value {
    font-size: 24px;
    font-weight: 700;
    font-family: var(--mono);
}
.stat-card .value.green { color: var(--green); }
.stat-card .value.red { color: var(--red); }
.stat-card .value.accent { color: var(--accent); }

/* ── Table ───────────────────────────────── */
.table-wrap {
    overflow-x: auto;
    border-radius: 10px;
    border: 1px solid var(--border);
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
th {
    text-align: left;
    padding: 10px 14px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}
td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(255,255,255,0.02); }

/* ── Badges ──────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    font-family: var(--mono);
}
.badge-buy { background: var(--green-dim); color: var(--green); }
.badge-sell { background: var(--red-dim); color: var(--red); }
.badge-hold { background: rgba(255,255,255,0.06); color: var(--text-dim); }
.badge-skip { background: var(--yellow-dim); color: var(--yellow); }
.badge-error { background: var(--red-dim); color: var(--red); }

/* ── Buttons ─────────────────────────────── */
.btn {
    padding: 8px 18px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-card);
    color: var(--text);
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
}
.btn:hover { border-color: var(--accent); color: var(--accent); }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-primary {
    background: var(--accent-dim);
    border-color: var(--accent);
    color: var(--accent);
}
.btn-primary:hover { background: rgba(0,217,255,0.25); }
.btn-danger {
    border-color: var(--red);
    color: var(--red);
}
.btn-danger:hover { background: var(--red-dim); }
.btn-success {
    border-color: var(--green);
    color: var(--green);
}
.btn-success:hover { background: var(--green-dim); }
.btn-group { display: flex; gap: 8px; flex-wrap: wrap; }

/* ── Forms ────────────────────────────────── */
.form-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}
.form-group { display: flex; flex-direction: column; gap: 4px; }
.form-group label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.form-group input, .form-group select {
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
}
.form-group input:focus, .form-group select:focus {
    outline: none;
    border-color: var(--accent);
}
textarea {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    resize: vertical;
}
textarea:focus { outline: none; border-color: var(--accent); }

/* ── Notice ──────────────────────────────── */
.notice {
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 13px;
    border: 1px solid;
    margin-bottom: 16px;
}
.notice-info { background: var(--accent-dim); border-color: rgba(0,217,255,0.3); color: var(--accent); }
.notice-warn { background: var(--yellow-dim); border-color: rgba(255,192,72,0.3); color: var(--yellow); }
.notice-ok { background: var(--green-dim); border-color: rgba(6,255,165,0.3); color: var(--green); }
.notice-err { background: var(--red-dim); border-color: rgba(255,71,87,0.3); color: var(--red); }

/* ── Empty state ─────────────────────────── */
.empty {
    text-align: center;
    padding: 48px 20px;
    color: var(--text-muted);
}
.empty .icon { font-size: 40px; margin-bottom: 12px; }
.empty p { font-size: 14px; }

/* ── Confidence Bar ──────────────────────── */
.conf-bar-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
}
.conf-bar {
    width: 60px;
    height: 6px;
    border-radius: 3px;
    background: var(--border);
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
}
.conf-val {
    font-size: 12px;
    font-family: var(--mono);
    font-weight: 600;
    min-width: 32px;
}

/* ── Spinner ─────────────────────────────── */
.spinner {
    display: inline-block;
    width: 16px; height: 16px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Responsive ──────────────────────────── */
@media (max-width: 640px) {
    .container { padding: 12px; }
    header { flex-direction: column; align-items: flex-start; }
    .tabs { flex-wrap: wrap; }
    .tab-btn { font-size: 12px; padding: 8px 12px; }
    .stats-grid { grid-template-columns: 1fr 1fr; }
    .form-row { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="container">

<!-- ── Header ───────────────────────────────── -->
<header>
    <div class="logo">Kite Trader <span>Dashboard</span></div>
    <div class="status-pills" id="header-pills">
        <span class="pill off" id="pill-auth"><span class="dot"></span>Not Connected</span>
        <span class="pill off" id="pill-auto"><span class="dot"></span>Automation Off</span>
    </div>
</header>

<!-- ── Tabs ──────────────────────────────────── -->
<div class="tabs">
    <button class="tab-btn active" onclick="switchTab('overview',this)">Overview</button>
    <button class="tab-btn" onclick="switchTab('decisions',this)">Decisions</button>
    <button class="tab-btn" onclick="switchTab('positions',this)">Positions</button>
    <button class="tab-btn" onclick="switchTab('config',this)">Config</button>
</div>

<!-- ════════════ OVERVIEW TAB ════════════ -->
<div id="tab-overview" class="tab-panel active">

    <div class="stats-grid" id="overview-stats">
        <div class="stat-card"><div class="label">Open Positions</div><div class="value accent" id="stat-positions">--</div></div>
        <div class="stat-card"><div class="label">Total Decisions</div><div class="value" id="stat-decisions">--</div></div>
        <div class="stat-card"><div class="label">Last Scan</div><div class="value" id="stat-lastscan" style="font-size:14px;">--</div></div>
        <div class="stat-card"><div class="label">Automation</div><div class="value" id="stat-auto">--</div></div>
    </div>

    <div class="card">
        <h3>Quick Actions</h3>
        <div class="btn-group">
            <button class="btn btn-primary" onclick="doScan()">Scan Now</button>
            <button class="btn btn-success" id="btn-start" onclick="doStart()">Start Automation</button>
            <button class="btn btn-danger" id="btn-stop" onclick="doStop()" disabled>Stop Automation</button>
            <button class="btn" onclick="refreshAll()">Refresh</button>
        </div>
        <div id="scan-result" style="margin-top:12px;"></div>
    </div>

    <div class="card">
        <h3>Analyze a Stock</h3>
        <div style="display:flex;gap:8px;">
            <input type="text" id="analyze-input" placeholder="Enter symbol (e.g. RELIANCE)" style="flex:1;padding:8px 12px;border:1px solid var(--border);border-radius:8px;background:var(--bg);color:var(--text);font-family:var(--mono);font-size:13px;">
            <button class="btn btn-primary" onclick="doAnalyze()">Analyze</button>
        </div>
        <div id="analyze-result" style="margin-top:12px;"></div>
    </div>

    <div class="card">
        <h3>Recent Decisions</h3>
        <div id="recent-decisions"></div>
    </div>
</div>

<!-- ════════════ DECISIONS TAB ════════════ -->
<div id="tab-decisions" class="tab-panel">
    <div class="card">
        <h3>Decision Log</h3>
        <div style="display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;">
            <button class="btn btn-primary" onclick="loadDecisions()">Refresh</button>
            <select id="decision-filter" onchange="filterDecisions()" style="padding:6px 12px;border:1px solid var(--border);border-radius:8px;background:var(--bg);color:var(--text);font-size:13px;">
                <option value="all">All Actions</option>
                <option value="BUY">BUY</option>
                <option value="SELL">SELL</option>
                <option value="SKIP">SKIP</option>
                <option value="HOLD">HOLD</option>
            </select>
        </div>
        <p style="color:var(--text-muted);font-size:12px;margin-bottom:12px;">Every signal evaluation by your trading logic is recorded here.</p>
        <div id="decisions-table"></div>
    </div>
</div>

<!-- ════════════ POSITIONS TAB ════════════ -->
<div id="tab-positions" class="tab-panel">
    <div class="card">
        <h3>Open Positions</h3>
        <div class="btn-group" style="margin-bottom:16px;">
            <button class="btn btn-primary" onclick="loadPositions()">Refresh</button>
            <button class="btn btn-danger" onclick="doExitAll()">Exit All</button>
        </div>
        <div id="positions-table"></div>
    </div>

    <div class="card">
        <h3>Trade Log</h3>
        <div id="trade-log-table"></div>
    </div>
</div>

<!-- ════════════ CONFIG TAB ════════════ -->
<div id="tab-config" class="tab-panel">
    <div class="card">
        <h3>Kite Connect Authentication</h3>
        <div id="auth-section">
            <div class="form-row">
                <div class="form-group">
                    <label>API Key</label>
                    <input type="text" id="cfg-api-key" placeholder="Your Kite API key">
                </div>
                <div class="form-group">
                    <label>API Secret</label>
                    <input type="password" id="cfg-api-secret" placeholder="Your Kite API secret">
                </div>
            </div>
            <div class="btn-group" style="margin-bottom:12px;">
                <button class="btn btn-primary" onclick="doInit()">Initialize</button>
                <button class="btn" onclick="doGetLoginUrl()">Get Login URL</button>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>Request Token (from redirect)</label>
                    <input type="text" id="cfg-request-token" placeholder="Paste request token">
                </div>
            </div>
            <div class="btn-group" style="margin-bottom:12px;">
                <button class="btn btn-primary" onclick="doAuth()">Authenticate</button>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>Or set Access Token directly</label>
                    <input type="text" id="cfg-access-token" placeholder="Paste access token">
                </div>
            </div>
            <button class="btn" onclick="doSetToken()">Set Token</button>
            <div id="auth-msg" style="margin-top:12px;"></div>
        </div>
    </div>

    <div class="card">
        <h3>Trading Parameters</h3>
        <div class="form-row">
            <div class="form-group">
                <label>Capital (INR)</label>
                <input type="number" id="cfg-capital" value="100000">
            </div>
            <div class="form-group">
                <label>Risk per Trade %</label>
                <input type="number" id="cfg-risk" value="1.0" step="0.1">
            </div>
            <div class="form-group">
                <label>Max Open Positions</label>
                <input type="number" id="cfg-max-pos" value="5">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Min Confidence %</label>
                <input type="number" id="cfg-min-conf" value="60">
            </div>
            <div class="form-group">
                <label>Min Risk:Reward</label>
                <input type="number" id="cfg-min-rr" value="1.5" step="0.1">
            </div>
            <div class="form-group">
                <label>Scan Interval (sec)</label>
                <input type="number" id="cfg-interval" value="300">
            </div>
        </div>
        <div class="form-group" style="margin-bottom:12px;">
            <label>Watchlist (comma-separated symbols)</label>
            <textarea id="cfg-watchlist" rows="3" placeholder="RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK"></textarea>
        </div>
        <button class="btn btn-primary" onclick="doSaveConfig()">Save Configuration</button>
        <div id="config-msg" style="margin-top:12px;"></div>
    </div>

    <div class="card">
        <h3>Stock Universe</h3>
        <p style="color:var(--text-dim);font-size:13px;margin-bottom:12px;">Click a sector to add all its stocks to the watchlist.</p>
        <div id="sector-grid" style="display:flex;flex-wrap:wrap;gap:6px;"></div>
    </div>

    <div class="notice notice-info" style="margin-top:16px;">
        <strong>Render Free Tier Note:</strong> The free web service sleeps after 15 min of inactivity.
        Automation will stop when the service sleeps. Use manual scans or keep the tab open to stay awake.
    </div>
</div>

</div><!-- /container -->

<script>
// ── State ──────────────────────────────────
let allDecisions = [];
let sectors = {};

// ── Tab switching ──────────────────────────
function switchTab(name, btn) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
    if (name === 'decisions') loadDecisions();
    if (name === 'positions') loadPositions();
}

// ── API helpers ────────────────────────────
async function api(url, opts) {
    try {
        const res = await fetch(url, opts);
        return await res.json();
    } catch (e) {
        return { error: e.message };
    }
}
function post(url, body) {
    return api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
}

// ── Format helpers ─────────────────────────
function fmtTime(iso) {
    if (!iso) return '--';
    const d = new Date(iso);
    return d.toLocaleString('en-IN', { day:'2-digit', month:'short', hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false });
}
function fmtPrice(v) {
    if (!v && v !== 0) return '--';
    return '\\u20B9' + Number(v).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function actionBadge(action) {
    const cls = { BUY:'badge-buy', SELL:'badge-sell', HOLD:'badge-hold', SKIP:'badge-skip', ERROR:'badge-error' }[action] || 'badge-hold';
    return '<span class="badge ' + cls + '">' + (action||'--') + '</span>';
}
function signalBadge(sig) {
    return actionBadge(sig);
}
function confBar(val) {
    const v = Number(val) || 0;
    const color = v >= 70 ? 'var(--green)' : v >= 50 ? 'var(--yellow)' : 'var(--red)';
    return '<div class="conf-bar-wrap"><div class="conf-bar"><div class="conf-bar-fill" style="width:' + v + '%;background:' + color + '"></div></div><span class="conf-val" style="color:' + color + '">' + v + '%</span></div>';
}

// ── Load status ────────────────────────────
async function loadStatus() {
    const s = await api('/api/status');
    if (s.error) return;
    // Pills
    const pa = document.getElementById('pill-auth');
    if (s.authenticated) { pa.className = 'pill on'; pa.innerHTML = '<span class="dot"></span>Connected'; }
    else if (s.access_token_set) { pa.className = 'pill warn'; pa.innerHTML = '<span class="dot"></span>Token Set'; }
    else { pa.className = 'pill off'; pa.innerHTML = '<span class="dot"></span>Not Connected'; }

    const pb = document.getElementById('pill-auto');
    if (s.is_running) { pb.className = 'pill on'; pb.innerHTML = '<span class="dot"></span>Automation On'; }
    else { pb.className = 'pill off'; pb.innerHTML = '<span class="dot"></span>Automation Off'; }

    // Stats
    document.getElementById('stat-positions').textContent = s.open_positions;
    document.getElementById('stat-decisions').textContent = s.total_decisions;
    document.getElementById('stat-lastscan').textContent = fmtTime(s.last_scan_time);
    const autoEl = document.getElementById('stat-auto');
    autoEl.textContent = s.is_running ? 'RUNNING' : 'STOPPED';
    autoEl.className = 'value ' + (s.is_running ? 'green' : '');

    document.getElementById('btn-start').disabled = s.is_running;
    document.getElementById('btn-stop').disabled = !s.is_running;

    // Fill config form
    if (s.config) {
        document.getElementById('cfg-capital').value = s.config.capital || 100000;
        document.getElementById('cfg-risk').value = s.config.risk_per_trade_pct || 1.0;
        document.getElementById('cfg-max-pos').value = s.config.max_open_positions || 5;
        document.getElementById('cfg-min-conf').value = s.config.min_confidence || 60;
        document.getElementById('cfg-min-rr').value = s.config.min_risk_reward || 1.5;
        document.getElementById('cfg-interval').value = s.config.scan_interval_sec || 300;
        if (s.config.watchlist && s.config.watchlist.length) {
            document.getElementById('cfg-watchlist').value = s.config.watchlist.join(', ');
        }
    }
}

// ── Load recent decisions (overview) ───────
async function loadRecentDecisions() {
    const d = await api('/api/decisions?limit=10');
    if (d.error || !d.decisions) return;
    const el = document.getElementById('recent-decisions');
    if (!d.decisions.length) {
        el.innerHTML = '<div class="empty"><div class="icon">&#128202;</div><p>No decisions yet. Run a scan to see your trading logic in action.</p></div>';
        return;
    }
    const rows = d.decisions.reverse().map(r =>
        '<tr><td>' + fmtTime(r.time) + '</td><td><strong>' + r.symbol + '</strong></td>' +
        '<td>' + signalBadge(r.signal) + '</td><td>' + confBar(r.confidence) + '</td>' +
        '<td>' + actionBadge(r.action) + '</td><td style="color:var(--text-dim);max-width:200px;overflow:hidden;text-overflow:ellipsis;">' + (r.reason || r.rec || '--') + '</td></tr>'
    ).join('');
    el.innerHTML = '<div class="table-wrap"><table><thead><tr><th>Time</th><th>Symbol</th><th>Signal</th><th>Confidence</th><th>Action</th><th>Reason</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
}

// ── Load full decisions ────────────────────
async function loadDecisions() {
    const d = await api('/api/decisions?limit=500');
    if (d.error || !d.decisions) return;
    allDecisions = d.decisions.reverse();
    filterDecisions();
}
function filterDecisions() {
    const filter = document.getElementById('decision-filter').value;
    const data = filter === 'all' ? allDecisions : allDecisions.filter(r => r.action === filter);
    const el = document.getElementById('decisions-table');
    if (!data.length) {
        el.innerHTML = '<div class="empty"><p>No decisions matching filter.</p></div>';
        return;
    }
    const rows = data.map(r =>
        '<tr><td>' + fmtTime(r.time) + '</td><td><strong>' + r.symbol + '</strong></td>' +
        '<td>' + signalBadge(r.signal) + '</td><td>' + confBar(r.confidence) + '</td>' +
        '<td style="font-family:var(--mono)">' + (r.risk_reward ? r.risk_reward.toFixed(1) + 'x' : '--') + '</td>' +
        '<td>' + fmtPrice(r.price) + '</td>' +
        '<td>' + fmtPrice(r.entry) + '</td><td>' + fmtPrice(r.stop) + '</td><td>' + fmtPrice(r.target) + '</td>' +
        '<td>' + actionBadge(r.action) + '</td>' +
        '<td style="color:var(--text-dim);max-width:250px;overflow:hidden;text-overflow:ellipsis;" title="' + (r.reason||r.rec||'').replace(/"/g,'&quot;') + '">' + (r.reason || r.rec || '--') + '</td></tr>'
    ).join('');
    el.innerHTML = '<div class="table-wrap"><table><thead><tr><th>Time</th><th>Symbol</th><th>Signal</th><th>Confidence</th><th>R:R</th><th>Price</th><th>Entry</th><th>Stop</th><th>Target</th><th>Action</th><th>Reason</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
}

// ── Load positions ─────────────────────────
async function loadPositions() {
    const d = await api('/api/positions');
    const el = document.getElementById('positions-table');
    if (d.error || !d.positions) {
        el.innerHTML = '<div class="empty"><p>No position data.</p></div>';
        return;
    }
    const positions = Object.values(d.positions);
    if (!positions.length) {
        el.innerHTML = '<div class="empty"><div class="icon">&#128230;</div><p>No open positions.</p></div>';
    } else {
        const rows = positions.map(p =>
            '<tr><td><strong>' + p.symbol + '</strong></td><td>' + p.quantity + '</td>' +
            '<td>' + fmtPrice(p.entry_price) + '</td><td>' + fmtPrice(p.stop_loss) + '</td>' +
            '<td>' + fmtPrice(p.target) + '</td><td>' + confBar(p.confidence) + '</td>' +
            '<td>' + fmtTime(p.entry_time) + '</td><td>' + (p.status || 'OPEN') + '</td></tr>'
        ).join('');
        el.innerHTML = '<div class="table-wrap"><table><thead><tr><th>Symbol</th><th>Qty</th><th>Entry</th><th>Stop</th><th>Target</th><th>Confidence</th><th>Time</th><th>Status</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
    }
    // Trade log
    const tl = await api('/api/log?limit=50');
    const tlEl = document.getElementById('trade-log-table');
    if (!tl.trades || !tl.trades.length) {
        tlEl.innerHTML = '<div class="empty"><p>No trades executed yet.</p></div>';
    } else {
        const rows = tl.trades.reverse().map(t =>
            '<tr><td>' + fmtTime(t.time) + '</td><td>' + actionBadge(t.side) + '</td>' +
            '<td><strong>' + t.symbol + '</strong></td><td>' + t.quantity + '</td>' +
            '<td>' + fmtPrice(t.price) + '</td>' +
            '<td style="color:' + (t.pnl > 0 ? 'var(--green)' : t.pnl < 0 ? 'var(--red)' : 'var(--text-dim)') + ';font-family:var(--mono)">' + (t.pnl != null ? fmtPrice(t.pnl) : '--') + '</td>' +
            '<td>' + (t.order_id || '--') + '</td></tr>'
        ).join('');
        tlEl.innerHTML = '<div class="table-wrap"><table><thead><tr><th>Time</th><th>Side</th><th>Symbol</th><th>Qty</th><th>Price</th><th>P&L</th><th>Order ID</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
    }
}

// ── Actions ────────────────────────────────
async function doScan() {
    const el = document.getElementById('scan-result');
    el.innerHTML = '<span class="spinner"></span> Scanning watchlist...';
    const res = await post('/api/scan');
    if (res.error) {
        el.innerHTML = '<div class="notice notice-err">' + res.error + '</div>';
    } else {
        const n = (res.results || []).length;
        const actions = (res.results || []).filter(r => r.action !== 'HOLD');
        el.innerHTML = '<div class="notice notice-ok">Scan complete: ' + n + ' stocks evaluated, ' + actions.length + ' actions.</div>';
        refreshAll();
    }
}

async function doStart() {
    const res = await post('/api/start');
    if (res.error) alert(res.error);
    loadStatus();
}

async function doStop() {
    const res = await post('/api/stop');
    if (res.error) alert(res.error);
    loadStatus();
}

async function doExitAll() {
    if (!confirm('Exit ALL open positions at market price?')) return;
    await post('/api/exit-all');
    loadPositions();
    loadStatus();
}

async function doAnalyze() {
    const symbol = document.getElementById('analyze-input').value.trim();
    if (!symbol) return;
    const el = document.getElementById('analyze-result');
    el.innerHTML = '<span class="spinner"></span> Analyzing ' + symbol + '...';
    const res = await post('/api/analyze', { symbol });
    if (res.error) {
        el.innerHTML = '<div class="notice notice-err">' + res.error + '</div>';
        return;
    }
    const d = res.decision || {};
    const a = res.analysis || {};
    const sig = a.signal || {};
    const det = a.details || {};
    let html = '<div style="margin-top:12px;padding:16px;background:var(--bg);border:1px solid var(--border);border-radius:10px;">';
    html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">';
    html += '<strong style="font-size:16px;">' + res.symbol + '</strong> ' + signalBadge(d.signal);
    html += '</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;font-size:13px;">';
    html += '<div><span style="color:var(--text-muted)">Price</span><br><strong>' + (det.price || '--') + '</strong></div>';
    html += '<div><span style="color:var(--text-muted)">Entry</span><br><strong>' + (d.entry || sig.entry || '--') + '</strong></div>';
    html += '<div><span style="color:var(--text-muted)">Stop</span><br><strong>' + (d.stop || sig.stop || '--') + '</strong></div>';
    html += '<div><span style="color:var(--text-muted)">Target</span><br><strong>' + (d.target || sig.target || '--') + '</strong></div>';
    html += '<div><span style="color:var(--text-muted)">Confidence</span><br>' + confBar(d.confidence) + '</div>';
    html += '<div><span style="color:var(--text-muted)">R:R</span><br><strong style="font-family:var(--mono)">' + (d.risk_reward ? d.risk_reward.toFixed(1) + 'x' : '--') + '</strong></div>';
    html += '</div>';
    if (d.rec) html += '<p style="margin-top:12px;color:var(--text-dim);font-size:13px;">' + d.rec + '</p>';
    if (d.would_trade === true) {
        html += '<div class="notice notice-ok" style="margin-top:12px;">Would place a BUY order with current config.</div>';
    } else if (d.skip_reason) {
        html += '<div class="notice notice-warn" style="margin-top:12px;">Would SKIP: ' + d.skip_reason + '</div>';
    }
    // Technical details
    if (det.rsi || det.zscore || det.volatility) {
        html += '<div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;font-size:12px;color:var(--text-dim);">';
        if (det.rsi) html += '<div>RSI: <strong style="color:var(--text)">' + det.rsi + '</strong></div>';
        if (det.zscore) html += '<div>Z-Score: <strong style="color:var(--text)">' + det.zscore + '</strong></div>';
        if (det.volatility) html += '<div>Volatility: <strong style="color:var(--text)">' + det.volatility + '</strong></div>';
        if (det.daily) html += '<div>Daily: <strong style="color:var(--text)">' + det.daily + '</strong></div>';
        html += '</div>';
    }
    html += '</div>';
    el.innerHTML = html;
}

// ── Config actions ─────────────────────────
async function doInit() {
    const key = document.getElementById('cfg-api-key').value.trim();
    const secret = document.getElementById('cfg-api-secret').value.trim();
    if (!key) { alert('API key required'); return; }
    const res = await post('/api/init', { api_key: key, api_secret: secret });
    const el = document.getElementById('auth-msg');
    if (res.error) {
        el.innerHTML = '<div class="notice notice-err">' + res.error + '</div>';
    } else {
        el.innerHTML = '<div class="notice notice-ok">Initialized. <a href="' + res.login_url + '" target="_blank" style="color:var(--accent)">Click here to log in to Kite</a></div>';
    }
    loadStatus();
}

async function doGetLoginUrl() {
    const res = await api('/api/login-url');
    const el = document.getElementById('auth-msg');
    if (res.error) {
        el.innerHTML = '<div class="notice notice-err">' + res.error + '</div>';
    } else {
        el.innerHTML = '<div class="notice notice-info"><a href="' + res.login_url + '" target="_blank" style="color:var(--accent)">' + res.login_url + '</a></div>';
    }
}

async function doAuth() {
    const token = document.getElementById('cfg-request-token').value.trim();
    const secret = document.getElementById('cfg-api-secret').value.trim();
    if (!token) { alert('Request token required'); return; }
    const body = { request_token: token };
    if (secret) body.api_secret = secret;
    const res = await post('/api/auth', body);
    const el = document.getElementById('auth-msg');
    if (res.error) {
        el.innerHTML = '<div class="notice notice-err">' + res.error + '</div>';
    } else {
        el.innerHTML = '<div class="notice notice-ok">Authenticated successfully!</div>';
    }
    loadStatus();
}

async function doSetToken() {
    const token = document.getElementById('cfg-access-token').value.trim();
    if (!token) { alert('Access token required'); return; }
    const res = await post('/api/auth/token', { access_token: token });
    const el = document.getElementById('auth-msg');
    if (res.error) {
        el.innerHTML = '<div class="notice notice-err">' + res.error + '</div>';
    } else {
        el.innerHTML = '<div class="notice notice-ok">Token set.</div>';
    }
    loadStatus();
}

async function doSaveConfig() {
    const watchlistRaw = document.getElementById('cfg-watchlist').value;
    const watchlist = watchlistRaw.split(',').map(s => s.trim()).filter(Boolean);
    const body = {
        capital: parseFloat(document.getElementById('cfg-capital').value),
        risk_per_trade_pct: parseFloat(document.getElementById('cfg-risk').value),
        max_open_positions: parseInt(document.getElementById('cfg-max-pos').value),
        min_confidence: parseInt(document.getElementById('cfg-min-conf').value),
        min_risk_reward: parseFloat(document.getElementById('cfg-min-rr').value),
        scan_interval_sec: parseInt(document.getElementById('cfg-interval').value),
        watchlist: watchlist,
    };
    const res = await post('/api/configure', body);
    const el = document.getElementById('config-msg');
    if (res.error) {
        el.innerHTML = '<div class="notice notice-err">' + res.error + '</div>';
    } else {
        el.innerHTML = '<div class="notice notice-ok">Configuration saved.</div>';
        setTimeout(() => { el.innerHTML = ''; }, 3000);
    }
}

// ── Sector grid ────────────────────────────
async function loadSectors() {
    const d = await api('/api/stocks');
    if (d.error || !d.sectors) return;
    sectors = d.sectors;
    const el = document.getElementById('sector-grid');
    el.innerHTML = Object.keys(sectors).map(s =>
        '<button class="btn" onclick="addSector(\'' + s.replace(/'/g,"\\'") + '\')" style="font-size:12px;padding:5px 10px;">' + s + ' (' + sectors[s].length + ')</button>'
    ).join('');
}
function addSector(name) {
    const existing = document.getElementById('cfg-watchlist').value;
    const current = existing ? existing.split(',').map(s=>s.trim()).filter(Boolean) : [];
    const merged = [...new Set([...current, ...sectors[name]])];
    document.getElementById('cfg-watchlist').value = merged.join(', ');
}

// ── Refresh all ────────────────────────────
function refreshAll() {
    loadStatus();
    loadRecentDecisions();
}

// ── Init ───────────────────────────────────
refreshAll();
loadSectors();
// Auto-refresh every 30s
setInterval(refreshAll, 30000);
</script>
</body>
</html>'''
    return html


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)
