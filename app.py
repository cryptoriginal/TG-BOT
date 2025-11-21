import os
import asyncio
import json
from datetime import datetime, timezone

import requests
from google import genai
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

# ===============================
# ENVIRONMENT VARIABLES
# ===============================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0"))

GEMINI_MODEL = "gemini-2.5-flash"

# Scanner settings
SCAN_INTERVAL_SECONDS = 300        # every 5 minutes
MIN_VOLUME = 35_000_000            # min 24h quote volume in USDT
MAX_SCAN_SYMBOLS = 25
MIN_PROBABILITY_FOR_TRADE = 65

DEFAULT_TIMEFRAMES = [
    "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"
]
SCAN_TIMEFRAMES = ["15m", "1h", "4h"]

BINANCE_FAPI = "https://fapi.binance.com"

# To avoid spamming same signal repeatedly
last_signal_time: dict[tuple[str, str], datetime] = {}

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN missing (set it in Render env vars)")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing (set it in Render env vars)")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ===============================
# BINANCE API HELPERS
# ===============================
def get_klines(symbol: str, interval: str, limit: int = 120):
    """Fetch OHLCV from Binance USDT-M Futures."""
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def get_top_symbols() -> list[str]:
    """Get USDT-M futures symbols sorted by 24h quote volume and filtered by MIN_VOLUME."""
    url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    pairs = [
        s for s in data
        if s.get("symbol", "").endswith("USDT")
        and float(s.get("quoteVolume", 0.0)) >= MIN_VOLUME
    ]
    pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [p["symbol"] for p in pairs[:MAX_SCAN_SYMBOLS]]


def build_snapshot(symbol: str, timeframes: list[str]):
    """Build OHLCV snapshot per timeframe + current price."""
    snapshot: dict[str, list[dict]] = {}
    current_price: float | None = None

    for tf in timeframes:
        kl = get_klines(symbol, tf, 100)
        candles = []
        for c in kl:
            candles.append(
                {
                    "open_time": c[0],
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                }
            )
        snapshot[tf] = candles
        if candles:
            current_price = candles[-1]["close"]

    return snapshot, current_price


# ===============================
# GEMINI PROMPTS
# ===============================
def prompt_for_pair(symbol: str, timeframe: str | None, snapshot: dict, price: float) -> str:
    return f"""
You are an elite crypto futures analyst and trader.

Symbol: {symbol}
Current price: {price}
Timeframe focus: {timeframe if timeframe else "Multi-timeframe"}

OHLCV JSON:
{json.dumps(snapshot)[:80000]}

TASK:
1. Determine probabilities (0-100):
   - upside
   - downside
   - flat

2. Decide best_direction.

3. If best_direction is upside/downside AND probability >= {MIN_PROBABILITY_FOR_TRADE}, generate:
   - direction (long/short)
   - entry
   - stop_loss
   - take_profit_1
   - take_profit_2
   - rr_ratio
   - leverage_hint
   - confidence
   - reasoning

Return ONLY valid JSON with this schema:

{{
  "symbol": "{symbol}",
  "probabilities": {{
    "upside": 0,
    "downside": 0,
    "flat": 0
  }},
  "best_direction": "upside | downside | flat",
  "overall_view": "text",
  "trade_plan": {{
    "direction": "long | short | none",
    "entry": 0,
    "stop_loss": 0,
    "take_profit_1": 0,
    "take_profit_2": 0,
    "rr_ratio": 0,
    "leverage_hint": "",
    "confidence": 0,
    "reasoning": "text"
  }}
}}
"""


def prompt_for_scan(symbol: str, snapshot: dict, price: float) -> str:
    return f"""
Quick scan of crypto futures pair:

Symbol: {symbol}
Price: {price}

OHLCV JSON:
{json.dumps(snapshot)[:60000]}

TASK:
1. Determine probabilities (0-100) of upside/downside/flat.
2. Decide best_direction.
3. If best_direction is not flat and probability >= {MIN_PROBABILITY_FOR_TRADE}, produce a compact trade plan.

Return ONLY JSON:

{{
 "symbol": "{symbol}",
 "probabilities": {{
   "upside": 0,
   "downside": 0,
   "flat": 0
 }},
 "best_direction": "",
 "trade_plan": {{
   "direction": "",
   "entry": 0,
   "stop_loss": 0,
   "take_profit_1": 0,
   "take_profit_2": 0,
   "confidence": 0
 }}
}}
"""


# ===============================
# GEMINI CALL + JSON PARSE
# ===============================
def call_gemini(prompt: str) -> str:
    r = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return r.text


def extract_json(text: str):
    try:
        start = text.index("{")
        end = text.rindex("}")
        return json.loads(text[start : end + 1])
    except Exception:
        return None


# ===============================
# ANALYSIS FUNCTIONS
# ===============================
def analyze_command(symbol: str, timeframe: str | None) -> str:
    tfs = [timeframe] if timeframe else DEFAULT_TIMEFRAMES
    snapshot, price = build_snapshot(symbol, tfs)

    if price is None:
        return f"❌ Could not fetch data for *{symbol}* (wrong symbol or API issue)."

    prompt = prompt_for_pair(symbol, timeframe, snapshot, price)
    raw = call_gemini(prompt)
    data = extract_json(raw)

    if not data:
        return "⚠️ Gemini JSON parse error. Raw response:\n\n" + raw[:3000]

    p = data.get("probabilities", {})
    tp = data.get("trade_plan", {})
    best = data.get("best_direction", "flat")
    view = data.get("overall_view", "")

    up = p.get("upside", 0)
    down = p.get("downside", 0)
    flat = p.get("flat", 0)

    msg_lines = []
    tf_label = timeframe if timeframe else "Multi-timeframe"
    msg_lines.append(f"📊 *{symbol}* — *{tf_label}*")
    msg_lines.append(f"Price: `{price}`\n")
    msg_lines.append("*Probabilities:*")
    msg_lines.append(f"⬆️ Upside: `{up}%`")
    msg_lines.append(f"⬇️ Downside: `{down}%`")
    msg_lines.append(f"➖ Flat: `{flat}%`")
    msg_lines.append(f"🎯 Best direction: *{best.upper()}*\n")
    if view:
        msg_lines.append(f"*View:* {view}\n")

    if tp.get("direction", "none") != "none":
        msg_lines.append("*🔥 Trade Setup:*")
        msg_lines.append(f"Direction: *{tp['direction'].upper()}*")
        msg_lines.append(f"Entry: `{tp['entry']}`")
        msg_lines.append(f"SL: `{tp['stop_loss']}`")
        msg_lines.append(f"TP1: `{tp['take_profit_1']}`")
        msg_lines.append(f"TP2: `{tp['take_profit_2']}`")
        msg_lines.append(f"RR: `{tp.get('rr_ratio', 0)}`")
        msg_lines.append(f"Confidence: `{tp.get('confidence', 0)}%`")
        msg_lines.append(f"Reason: {tp.get('reasoning', '')}")
    else:
        msg_lines.append("🚫 No strong setup (probability < 65% or market too choppy).")

    msg_lines.append("\n_Not financial advice. Manage your own risk._")
    return "\n".join(msg_lines)


def analyze_scan(symbol: str):
    snapshot, price = build_snapshot(symbol, SCAN_TIMEFRAMES)
    if price is None:
        return None

    prompt = prompt_for_scan(symbol, snapshot, price)
    raw = call_gemini(prompt)
    data = extract_json(raw)
    if not data:
        return None

    probs = data.get("probabilities", {})
    up = probs.get("upside", 0)
    down = probs.get("downside", 0)
    flat = probs.get("flat", 0)
    best = data.get("best_direction", "flat")

    tp = data.get("trade_plan", {})
    direction = tp.get("direction", "none")

    if direction == "none" or best == "flat":
        return None

    prob = up if best == "upside" else down
    if prob < MIN_PROBABILITY_FOR_TRADE:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "probability": prob,
        "entry": tp.get("entry", 0),
        "sl": tp.get("stop_loss", 0),
        "tp1": tp.get("take_profit_1", 0),
        "tp2": tp.get("take_profit_2", 0),
        "confidence": tp.get("confidence", prob),
    }


# ===============================
# TELEGRAM HANDLERS
# ===============================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 *Gemini 2.5 Futures Bot*\n\n"
        "Usage examples:\n"
        "`/suiusdt` → Multi-timeframe AI analysis\n"
        "`/suiusdt 4h` → 4h-focused AI analysis\n\n"
        "Scanner runs every 5 minutes on high-volume Binance USDT-M futures "
        "and sends high-probability setups directly to the owner."
    )
    await update.message.reply_markdown(text)


async def handle_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    parts = text.split()

    cmd = parts[0].lstrip("/").upper()
    timeframe = parts[1].lower() if len(parts) > 1 else None

    await update.message.reply_text(
        f"⏳ Analysing {cmd}"
        + (f" on {timeframe}" if timeframe else "")
        + " using Gemini 2.5..."
    )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, analyze_command, cmd, timeframe)

    await update.message.reply_markdown(result)


# ===============================
# SCANNER JOB
# ===============================
async def scanner(context: ContextTypes.DEFAULT_TYPE):
    if OWNER_CHAT_ID == 0:
        return

    loop = asyncio.get_running_loop()
    symbols = await loop.run_in_executor(None, get_top_symbols)

    for sym in symbols:
        sig = await loop.run_in_executor(None, analyze_scan, sym)
        if not sig:
            continue

        key = (sym, sig["direction"])
        now = datetime.now(timezone.utc)
        last = last_signal_time.get(key)
        if last and (now - last).total_seconds() < 1800:  # 30 min cooldown
            continue

        last_signal_time[key] = now

        msg = (
            "🚨 *AI Scanner Signal*\n"
            f"Pair: *{sym}*\n"
            f"Direction: *{sig['direction'].upper()}*\n"
            f"Probability: `{sig['probability']}%`\n"
            f"Entry: `{sig['entry']}`\n"
            f"SL: `{sig['sl']}`\n"
            f"TP1: `{sig['tp1']}`\n"
            f"TP2: `{sig['tp2']}`\n"
            f"Confidence: `{sig['confidence']}%`\n\n"
            "_Use your own position sizing & risk control._"
        )

        await context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=msg,
            parse_mode="Markdown",
        )


# ===============================
# MAIN
# ===============================
def main():
    # IMPORTANT: JobQueue works only if PTB installed with [job-queue] extra
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)
        .build()
    )

    # If job_queue is None, requirements not installed with [job-queue]
    if app.job_queue is None:
        raise RuntimeError(
            "JobQueue is None. Install dependencies with "
            '"python-telegram-bot[job-queue]" in requirements.txt'
        )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.COMMAND, handle_pair))

    jq = app.job_queue
    jq.run_repeating(scanner, interval=SCAN_INTERVAL_SECONDS, first=30)

    print("✅ Bot running with polling + scanner...")
    app.run_polling()


if __name__ == "__main__":
    main()
