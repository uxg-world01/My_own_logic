"""Telegram VIP signal delivery for Forex Elite bots."""
import html
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Optional


TELEGRAM_BOT_TOKEN = os.getenv("8783386088:AAHuLLa6rgwH1YKyaiPeRFpA-luNPP2hyhk", "")
TELEGRAM_VIP_CHAT_ID = os.getenv("8783386088", "")
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
TELEGRAM_TIMEOUT = float(os.getenv("TELEGRAM_TIMEOUT", "8"))


def _fmt(value: object, precision: int = 5) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return str(value)


def _escape(value: object) -> str:
    return html.escape(_fmt(value), quote=False)


def telegram_is_ready() -> bool:
    return TELEGRAM_ENABLED and bool(TELEGRAM_BOT_TOKEN and TELEGRAM_VIP_CHAT_ID)


def send_telegram_message(message: str, chat_id: Optional[str] = None) -> bool:
    """Send one HTML-formatted message to Telegram."""
    if not telegram_is_ready():
        print("Telegram VIP delivery skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_VIP_CHAT_ID is missing.")
        return False

    payload = urllib.parse.urlencode({
        "chat_id": chat_id or TELEGRAM_VIP_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode("utf-8")

    request = urllib.request.Request(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data=payload,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TELEGRAM_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
            if not body.get("ok"):
                print(f"Telegram VIP delivery failed: {body}")
                return False
            return True
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        print(f"Telegram VIP delivery failed: {exc}")
        return False


def format_vip_signal(
    *,
    symbol: str,
    side: str,
    entry: object,
    sl: object,
    tp1: object,
    tp2: object,
    source: str,
    timeframe: str,
    status: str = "LIVE",
    order_id: Optional[object] = None,
    confidence: Optional[object] = None,
) -> str:
    sent_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "<b>FOREX ELITE VIP SIGNAL</b>",
        f"<b>Status:</b> {_escape(status)}",
        f"<b>Source:</b> {_escape(source)}",
        f"<b>Symbol:</b> {_escape(symbol)}",
        f"<b>Action:</b> {_escape(str(side).upper())}",
        f"<b>Entry:</b> {_escape(entry)}",
        f"<b>Stop Loss:</b> {_escape(sl)}",
        f"<b>Take Profit 1:</b> {_escape(tp1)}",
        f"<b>Take Profit 2:</b> {_escape(tp2)}",
        f"<b>Timeframe:</b> {_escape(timeframe)}",
        f"<b>Sent:</b> {_escape(sent_at)}",
    ]

    if confidence is not None:
        lines.insert(5, f"<b>Confidence:</b> {_escape(confidence)}")
    if order_id is not None:
        lines.append(f"<b>Order ID:</b> {_escape(order_id)}")

    lines.append("")
    lines.append("Manage risk. Use your own confirmation before copying.")
    return "\n".join(lines)


def send_vip_signal(**signal_fields: object) -> bool:
    return send_telegram_message(format_vip_signal(**signal_fields))
