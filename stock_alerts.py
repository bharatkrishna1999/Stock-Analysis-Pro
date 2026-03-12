"""
Stock Alert System
==================
Monitors a watchlist of NSE stocks and sends an email alert the moment
undervalued technical conditions are detected.

Undervaluation scoring (score >= threshold to trigger):
  BUY signal detected        → +40 pts
  RSI < 35 (oversold)        → +25 pts  |  RSI < 40 → +10 pts
  Z-score < -2.0 (extreme)   → +25 pts  |  Z-score < -1.0 → +15 pts
  Bollinger %B < 5%          → +20 pts  |  BB < 15% → +12 pts

Default threshold: 40 pts  (BUY signal alone is enough to fire)

Configuration via environment variables:
  ALERT_EMAIL        – recipient email address
  SMTP_HOST          – SMTP server host  (default: smtp.gmail.com)
  SMTP_PORT          – SMTP port         (default: 465, SSL)
  SMTP_USER          – sender email / login
  SMTP_PASSWORD      – SMTP password or app-password

Optional:
  ALERT_WATCHLIST    – comma-separated stock symbols to monitor
                       (e.g. "RELIANCE,TCS,INFY")
  ALERT_INTERVAL_MIN – polling interval in minutes  (default: 5)
  ALERT_COOLDOWN_HRS – min hours between repeated alerts (default: 4)
"""

import logging
import os
import smtplib
import threading
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


class StockAlertMonitor:
    """
    Background daemon that polls the existing Analyzer for undervalued
    conditions and fires email alerts the moment a stock qualifies.
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

        # symbol → datetime of last alert  (for cooldown enforcement)
        self._alerted: dict[str, datetime] = {}

        # Resolve watchlist from env
        env_watchlist = os.environ.get("ALERT_WATCHLIST", "")
        default_watchlist = (
            [s.strip().upper() for s in env_watchlist.split(",") if s.strip()]
            if env_watchlist
            else []
        )

        self.config: dict = {
            "recipient_email": os.environ.get("ALERT_EMAIL", ""),
            "smtp_host": os.environ.get("SMTP_HOST", "smtp.gmail.com"),
            "smtp_port": int(os.environ.get("SMTP_PORT", "465")),
            "smtp_user": os.environ.get("SMTP_USER", ""),
            "smtp_password": os.environ.get("SMTP_PASSWORD", ""),
            "smtp_use_ssl": True,
            "watchlist": default_watchlist,
            "check_interval_min": int(os.environ.get("ALERT_INTERVAL_MIN", "5")),
            "cooldown_hours": int(os.environ.get("ALERT_COOLDOWN_HRS", "4")),
            "score_threshold": 40,  # BUY signal alone (40 pts) is enough
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Undervaluation scoring
    # ─────────────────────────────────────────────────────────────────────────

    def score_undervaluation(self, symbol: str) -> tuple[int, dict]:
        """
        Run the existing technical analysis on *symbol* and return
        (score, details_dict).  score >= threshold means undervalued.
        """
        try:
            result = self.analyzer.analyze(symbol)
        except Exception as exc:
            logger.warning("[Alerts] analyze(%s) raised: %s", symbol, exc)
            return 0, {}

        if not result:
            return 0, {}

        sig = result.get("signal", {})
        det = result.get("details", {})

        signal_str = sig.get("signal", "")
        confidence = sig.get("confidence", 0)
        rsi = det.get("rsi_raw", 50.0)
        zscore = det.get("zscore_raw", 0.0)
        bb_pos = det.get("bb_position", 50.0)
        price = det.get("price_raw", 0.0)
        target = sig.get("target_raw", 0.0)
        stop = sig.get("stop_raw", 0.0)
        entry = sig.get("entry_raw", 0.0)

        score = 0
        reasons: list[str] = []

        # ── Signal ────────────────────────────────────────────────────────────
        if signal_str == "BUY":
            score += 40
            reasons.append(f"BUY signal ({confidence}% confidence)")

        # ── RSI ───────────────────────────────────────────────────────────────
        if rsi < 35:
            score += 25
            reasons.append(f"RSI oversold at {rsi:.1f} (threshold < 35)")
        elif rsi < 40:
            score += 10
            reasons.append(f"RSI approaching oversold at {rsi:.1f}")

        # ── Z-score ───────────────────────────────────────────────────────────
        if zscore < -2.0:
            score += 25
            reasons.append(
                f"Z-score {zscore:.2f} — price extremely below mean (−2σ)"
            )
        elif zscore < -1.0:
            score += 15
            reasons.append(f"Z-score {zscore:.2f} — price below statistical mean")

        # ── Bollinger Band position ───────────────────────────────────────────
        if bb_pos < 5:
            score += 20
            reasons.append(
                f"Bollinger Band position {bb_pos:.0f}% — outside lower band"
            )
        elif bb_pos < 15:
            score += 12
            reasons.append(
                f"Bollinger Band position {bb_pos:.0f}% — near lower band"
            )

        upside_pct = (
            (target - price) / price * 100
            if price > 0 and target > price
            else 0.0
        )

        details = {
            "symbol": symbol,
            "signal": signal_str,
            "confidence": confidence,
            "price": price,
            "entry": entry,
            "stop": stop,
            "target": target,
            "upside_pct": round(upside_pct, 1),
            "rsi": rsi,
            "zscore": zscore,
            "bb_position": bb_pos,
            "score": score,
            "reasons": reasons,
            "rec": sig.get("rec", ""),
            "expected_move_pct": sig.get("expected_move_pct", 0),
            "days_to_target": sig.get("days_to_target", 0),
            "risk_reward": sig.get("risk_reward", 0),
            "scanned_at": datetime.now().isoformat(),
        }
        return score, details

    # ─────────────────────────────────────────────────────────────────────────
    # Email
    # ─────────────────────────────────────────────────────────────────────────

    def send_alert_email(self, stocks: list[dict]) -> bool:
        """Build and dispatch the HTML alert email."""
        cfg = self.config
        if not (cfg["recipient_email"] and cfg["smtp_user"] and cfg["smtp_password"]):
            logger.error(
                "[Alerts] Email credentials not configured — alert skipped."
            )
            return False

        if len(stocks) == 1:
            subject = (
                f"STOCK ALERT: {stocks[0]['symbol']} is undervalued now"
            )
        else:
            symbols = ", ".join(s["symbol"] for s in stocks)
            subject = f"STOCK ALERT: {len(stocks)} undervalued stocks — {symbols}"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = cfg["smtp_user"]
        msg["To"] = cfg["recipient_email"]
        msg.attach(MIMEText(self._build_email_html(stocks), "html"))

        try:
            if cfg["smtp_use_ssl"]:
                server = smtplib.SMTP_SSL(cfg["smtp_host"], cfg["smtp_port"])
            else:
                server = smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"])
                server.ehlo()
                server.starttls()

            server.login(cfg["smtp_user"], cfg["smtp_password"])
            server.sendmail(
                cfg["smtp_user"], cfg["recipient_email"], msg.as_string()
            )
            server.quit()
            logger.info(
                "[Alerts] Email sent for: %s", [s["symbol"] for s in stocks]
            )
            return True
        except Exception as exc:
            logger.error("[Alerts] Failed to send email: %s", exc)
            return False

    def _build_email_html(self, stocks: list[dict]) -> str:
        now_str = datetime.now().strftime("%d %b %Y, %I:%M %p IST")

        rows = ""
        for s in stocks:
            signal_color = "#10b981" if s["signal"] == "BUY" else "#f59e0b"
            reasons_html = "<br>".join(f"&bull; {r}" for r in s["reasons"])
            upside = s.get("upside_pct", 0)

            rows += f"""
              <tr style="border-bottom:1px solid #1e293b;">
                <td style="padding:16px 12px;">
                  <div style="font-size:18px;font-weight:700;color:#f1f5f9;">{s['symbol']}</div>
                  <div style="margin-top:4px;">
                    <span style="background:{signal_color}22;color:{signal_color};
                          border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600;">
                      {s['signal']} &bull; {s['confidence']}% confidence
                    </span>
                  </div>
                </td>
                <td style="padding:16px 12px;text-align:right;">
                  <div style="font-size:20px;font-weight:700;color:#f1f5f9;">&#8377;{s['price']:,.2f}</div>
                  <div style="color:#64748b;font-size:11px;">Current Price</div>
                </td>
                <td style="padding:16px 12px;text-align:center;">
                  <div style="font-size:15px;font-weight:600;color:#10b981;">&#8377;{s['target']:,.2f}</div>
                  <div style="color:#64748b;font-size:11px;">Target (+{upside:.1f}%)</div>
                  <div style="color:#64748b;font-size:10px;">{s['days_to_target']}d &bull; {s['risk_reward']:.1f}x R:R</div>
                </td>
                <td style="padding:16px 12px;text-align:center;">
                  <div style="font-size:15px;font-weight:600;color:#ef4444;">&#8377;{s['stop']:,.2f}</div>
                  <div style="color:#64748b;font-size:11px;">Stop Loss</div>
                </td>
                <td style="padding:16px 12px;">
                  <div style="font-size:12px;color:#94a3b8;line-height:1.6;">{reasons_html}</div>
                  <div style="margin-top:6px;font-size:11px;color:#64748b;">
                    RSI {s['rsi']:.1f} &bull; Z-Score {s['zscore']:.2f} &bull; BB {s['bb_position']:.0f}%
                  </div>
                </td>
              </tr>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Stock Alert</title>
</head>
<body style="margin:0;padding:0;background:#0a0c12;
             font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0"
         style="background:#0a0c12;padding:32px 0;">
    <tr><td align="center">
      <table width="700" cellpadding="0" cellspacing="0"
             style="max-width:700px;width:100%;">

        <!-- ── Header ─────────────────────────────────────────────────── -->
        <tr>
          <td style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
                     border:1px solid #334155;border-radius:12px 12px 0 0;
                     padding:28px 32px;">
            <table width="100%"><tr>
              <td>
                <div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;
                            color:#f59e0b;font-weight:600;margin-bottom:8px;">
                  Stock Analysis Pro
                </div>
                <div style="font-size:26px;font-weight:800;color:#f1f5f9;line-height:1.2;">
                  Undervalued Stock Alert
                </div>
                <div style="font-size:13px;color:#64748b;margin-top:6px;">{now_str}</div>
              </td>
              <td align="right">
                <div style="background:#f59e0b22;border:1px solid #f59e0b44;
                            border-radius:50px;padding:8px 16px;
                            color:#f59e0b;font-size:12px;font-weight:600;white-space:nowrap;">
                  {len(stocks)} Stock{"s" if len(stocks) != 1 else ""} Flagged
                </div>
              </td>
            </tr></table>
          </td>
        </tr>

        <!-- ── Sub-banner ─────────────────────────────────────────────── -->
        <tr>
          <td style="background:#10b98111;
                     border-left:1px solid #334155;border-right:1px solid #334155;
                     padding:14px 32px;">
            <span style="color:#10b981;font-size:13px;font-weight:600;">
              Real-time scan detected undervalued condition(s) based on
              RSI, Bollinger Bands, Z-Score, and technical signal confluence.
            </span>
          </td>
        </tr>

        <!-- ── Stock table ────────────────────────────────────────────── -->
        <tr>
          <td style="background:#0f172a;
                     border-left:1px solid #334155;border-right:1px solid #334155;">
            <table width="100%" cellpadding="0" cellspacing="0">
              <thead>
                <tr style="background:#1e293b;">
                  <th style="padding:10px 12px;text-align:left;font-size:11px;
                             color:#64748b;font-weight:600;text-transform:uppercase;
                             letter-spacing:1px;">Symbol</th>
                  <th style="padding:10px 12px;text-align:right;font-size:11px;
                             color:#64748b;font-weight:600;text-transform:uppercase;
                             letter-spacing:1px;">Price</th>
                  <th style="padding:10px 12px;text-align:center;font-size:11px;
                             color:#64748b;font-weight:600;text-transform:uppercase;
                             letter-spacing:1px;">Target</th>
                  <th style="padding:10px 12px;text-align:center;font-size:11px;
                             color:#64748b;font-weight:600;text-transform:uppercase;
                             letter-spacing:1px;">Stop</th>
                  <th style="padding:10px 12px;text-align:left;font-size:11px;
                             color:#64748b;font-weight:600;text-transform:uppercase;
                             letter-spacing:1px;">Why Undervalued</th>
                </tr>
              </thead>
              <tbody>{rows}</tbody>
            </table>
          </td>
        </tr>

        <!-- ── Footer ─────────────────────────────────────────────────── -->
        <tr>
          <td style="background:#0f172a;
                     border:1px solid #334155;border-top:none;
                     border-radius:0 0 12px 12px;padding:20px 32px;">
            <p style="margin:0;font-size:11px;color:#475569;line-height:1.7;">
              This alert was generated automatically by Stock Analysis Pro based
              on real-time technical indicators.
              <strong style="color:#64748b;">This is not financial advice.</strong>
              Always conduct your own research before investing.<br>
              To manage your watchlist or stop alerts, visit the
              <em>Stock Alerts</em> section in your app.
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""

    # ─────────────────────────────────────────────────────────────────────────
    # Background monitor loop
    # ─────────────────────────────────────────────────────────────────────────

    def _in_cooldown(self, symbol: str) -> bool:
        last = self._alerted.get(symbol)
        if not last:
            return False
        return datetime.now() - last < timedelta(hours=self.config["cooldown_hours"])

    def _scan_once(self):
        with self._lock:
            watchlist = list(self.config.get("watchlist", []))
        if not watchlist:
            return

        threshold = self.config["score_threshold"]
        triggered: list[dict] = []

        for symbol in watchlist:
            if not self._running:
                break
            if self._in_cooldown(symbol):
                logger.debug("[Alerts] %s in cooldown — skipped.", symbol)
                continue

            score, details = self.score_undervaluation(symbol)

            if score >= threshold and details.get("signal") == "BUY":
                triggered.append(details)
                with self._lock:
                    self._alerted[symbol] = datetime.now()
                logger.info(
                    "[Alerts] %s flagged as undervalued (score=%d)", symbol, score
                )
            else:
                logger.debug(
                    "[Alerts] %s — score=%d signal=%s (not triggered)",
                    symbol,
                    score,
                    details.get("signal", "N/A"),
                )

        if triggered:
            self.send_alert_email(triggered)

    def _monitor_loop(self):
        logger.info(
            "[Alerts] Monitor started. Watchlist: %s | Interval: %d min",
            self.config.get("watchlist"),
            self.config["check_interval_min"],
        )
        while self._running:
            try:
                self._scan_once()
            except Exception as exc:
                logger.error("[Alerts] Unexpected error in scan: %s", exc)

            # Sleep in 1-second ticks so stop() is responsive
            interval_sec = self.config["check_interval_min"] * 60
            for _ in range(interval_sec):
                if not self._running:
                    break
                time.sleep(1)

        logger.info("[Alerts] Monitor stopped.")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        """Start the background monitoring thread (idempotent)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="StockAlertMonitor",
        )
        self._thread.start()
        logger.info("[Alerts] Thread started.")

    def stop(self):
        """Signal the monitor to stop (non-blocking)."""
        self._running = False
        logger.info("[Alerts] Stop requested.")

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "watchlist": list(self.config.get("watchlist", [])),
                "check_interval_min": self.config["check_interval_min"],
                "cooldown_hours": self.config["cooldown_hours"],
                "score_threshold": self.config["score_threshold"],
                "recipient_email": self.config["recipient_email"],
                "smtp_configured": bool(
                    self.config["smtp_user"] and self.config["smtp_password"]
                ),
                "alerted": {
                    k: v.isoformat() for k, v in self._alerted.items()
                },
            }

    def update_config(self, new_config: dict):
        """Merge *new_config* keys into self.config (only known keys)."""
        allowed = set(self.config.keys())
        with self._lock:
            for k, v in new_config.items():
                if k in allowed:
                    self.config[k] = v

    def manual_scan(self) -> list[dict]:
        """
        Run one scan immediately (blocking) on the current watchlist.
        Returns all results (triggered and not) with score and details.
        Useful for testing from the UI without waiting for the interval.
        """
        with self._lock:
            watchlist = list(self.config.get("watchlist", []))
        threshold = self.config["score_threshold"]
        results = []
        for symbol in watchlist:
            score, details = self.score_undervaluation(symbol)
            details["_score"] = score
            details["_triggered"] = (
                score >= threshold and details.get("signal") == "BUY"
            )
            results.append(details)
        return results
