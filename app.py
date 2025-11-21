import os
import json
from datetime import datetime, timezone

import requests
from google import genai
from telegram import Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext,
)

# ===============================
# ENVIRONMENT VARIABLES
# ===============================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0"))

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN missing (set it in Render env vars)")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing (set it in Render env vars)")

GEMINI_MODEL = "gemini-2.5-flash"

# ===============================
# STRATEGY / SCANNER SETTINGS
# ===============================

# Auto scanner toggle (controlled by /start and /stop)
SCAN_ENABLED = True

# Scanner settings
SCAN_INTERVAL_SECONDS = 300          # every 5 minutes
MIN_VOLUME = 50_000_000              # min 24h quote volume in USDT (changed to 50M)
MAX_SCAN_SYMBOLS = 25
MIN_PROBABILITY_FOR_TRADE = 75       # min probability for upside/downside (changed to 75%)
MIN_RR = 1.8                         # minimum risk:reward 1:1.8

DEFAULT_TIMEFRAMES = [
    "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"
]
SCAN_TIMEFRAMES = ["15m", "1h", "4h"]

BINANCE_FAPI = "https://fapi.binance.com"

# avoid spamming identical signals
last_signal_time = {}   # (symbol, direction) -> datetime

# Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ===============================
# BINANCE HELPERS
# ===============================

def get_klines(symbol, interval, limit=120):
    """Fetch OHLCV data from Binance USDT-M Futures."""
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def get_top_symbols():
    """Return high-volume USDT-M futures symbols filtered by MIN_VOLUME."""
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


def build_snapshot(symbol, timeframes):
    """Build OHLCV snapshot dict + current price for given timeframes."""
    snapshot = {}
    current_price = None

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

def prompt_for_pair(symbol, timeframe, snapshot, price):
    """
    Prompt for manual /pair analysis.
    SL must be at a strong key level, using structure + fixed range volume profile logic,
    and RR must be at least MIN_RR (1:1.8).
    """
    return f"""
You are an elite crypto futures analyst and trader.

Symbol: {symbol}
Current price: {price}
Timeframe focus: {timeframe if timeframe else "Multi-timeframe"}

OHLCV JSON:
{json.dumps(snapshot)[:80000]}

MAIN ANALYSIS RULES:
- Base your view mainly on PRICE ACTION and MARKET STRUCTURE:
  - Trend (uptrend / downtrend / range) on each timeframe.
  - Important support/resistance zones.
  - Trendlines, channels, breakouts, retests, bounces.
  - Reversal candles (hammer, shooting star, engulfing, pin bars) at key levels.
  - Chart patterns (triangles, flags, head & shoulders, double top/bottom, etc.).

- Use "fixed range volume profile" in your reasoning:
  - Identify high-volume nodes (HVN) and low-volume nodes (LVN) in this range.
  - Stop loss must be placed around a logical structural invalidation level,
    ideally just beyond a key HVN/LVN or clear swing high/low, NOT random
    and NOT extremely tight.

- Mentally also use EMAs (20, 50, 200), RSI, MACD, and volume as confirmation,
  but price action + key levels are primary.

TASK:
1. Determine probabilities (0-100):
   - upside
   - downside
   - flat

2. Decide best_direction.

3. If best_direction is upside or downside AND its probability >= {MIN_PROBABILITY_FOR_TRADE},
   generate a detailed trade plan:
   - direction (long/short)
   - entry (a single reasonable entry price)
   - stop_loss (at a STRUCTURAL invalidation/key level as described above)
   - take_profit_1
   - take_profit_2
   - rr_ratio (approx risk:reward, must be >= {MIN_RR})
   - leverage_hint (safe leverage range like 2x-5x)
   - confidence (0-100)
   - reasoning (short explanation referencing key levels, structure, volume profile zones)

IMPORTANT:
- Do NOT propose a trade if you cannot find a clean structure with rr_ratio >= {MIN_RR}.
- If the conditions are not met, set direction="none" and leave rr_ratio at 0.

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


def prompt_for_scan(symbol, snapshot, price):
    """
    Prompt for fast scanner.
    Must only return trades with prob >= MIN_PROBABILITY_FOR_TRADE and RR >= MIN_RR.
    SL should still be at strong structural / volume-profile level.
    """
    return f"""
Quick scan of crypto futures pair:

Symbol: {symbol}
Price: {price}

OHLCV JSON:
{json.dumps(snapshot)[:60000]}

RULES:
- Use price action and structure (trend, key levels, breakouts, reversals).
- Use a fixed range volume profile mental model to place STOP LOSS at a clear invalidation
  zone beyond a key HVN/LVN or swing high/low, not random.
- Only accept setups where risk:reward ratio is at least {MIN_RR} (1:{MIN_RR}).

TASK:
1. Determine probabilities (0-100) of:
   - upside
   - downside
   - flat

2. Decide best_direction.

3. If best_direction is upside or downside AND its probability >= {MIN_PROBABILITY_FOR_TRADE}
   AND a clean trade with rr_ratio >= {MIN_RR} exists, produce a compact trade plan:
   - direction
   - entry
   - stop_loss
   - take_profit_1
   - take_profit_2
   - rr_ratio (>= {MIN_RR})
   - confidence

4. If conditions are not met, set direction="none".

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
   "rr_ratio": 0,
   "confidence": 0
 }}
}}
"""


# ===============================
# GEMINI CALL + JSON PARSE
# ===============================

def call_gemini(prompt):
    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return resp.text


def extract_json(text):
    try:
        start = text.index("{")
        end = text.rindex("}")
        return json.loads(text[start:end + 1])
    except Exception:
        return None


# ===============================
# ANALYSIS FUNCTIONS
# ===============================

def analyze_command(symbol, timeframe):
    """Analysis for manual /pair and /pair timeframe commands."""
    tfs = [timeframe] if timeframe else DEFAULT_TIMEFRAMES
    snapshot, price = build_snapshot(symbol, tfs)

    if price is None:
        return f"❌ Could not fetch market data for *{symbol}*."

    prompt = prompt_for_pair(symbol, timeframe, snapshot, price)
    raw = call_gemini(prompt)
    data = extract_json(raw)

    if not data:
        return "⚠️ Gemini JSON parse error. Raw response:\n\n" + raw[:3000]

    probs = data.get("probabilities", {})
    up = probs.get("upside", 0)
    down = probs.get("downside", 0)
    flat = probs.get("flat", 0)
    best = data.get("best_direction", "flat")
    view = data.get("overall_view", "")

    tp = data.get("trade_plan", {}) or {}
    direction = tp.get("direction", "none")
    rr_ratio = float(tp.get("rr_ratio", 0) or 0.0)

    # If RR is below threshold, treat as no valid setup
    if rr_ratio < MIN_RR:
        direction = "none"

    tf_label = timeframe if timeframe else "Multi-timeframe"
    lines = []
    lines.append(f"📊 *{symbol}* — *{tf_label}*")
    lines.append(f"Price: `{price}`\n")
    lines.append("*Probabilities (next move):*")
    lines.append(f"⬆️ Upside: `{up}%`")
    lines.append(f"⬇️ Downside: `{down}%`")
    lines.append(f"➖ Flat: `{flat}%`")
    lines.append(f"🎯 Best direction: *{best.upper()}*\n")

    if view:
        lines.append(f"*View:* {view}\n")

    if direction != "none":
        lines.append("*🔥 Trade Setup (AI):*")
        lines.append(f"Direction: *{direction.upper()}*")
        lines.append(f"Entry: `{tp.get('entry', 0)}`")
        lines.append(f"SL (key level): `{tp.get('stop_loss', 0)}`")
        lines.append(f"TP1: `{tp.get('take_profit_1', 0)}`")
        lines.append(f"TP2: `{tp.get('take_profit_2', 0)}`")
        lines.append(f"RR: `{rr_ratio}`  (>= {MIN_RR})")
        lines.append(f"Leverage: `{tp.get('leverage_hint', '')}`")
        lines.append(f"Confidence: `{tp.get('confidence', 0)}%`")
        if tp.get("reasoning"):
            lines.append(f"Reason: {tp['reasoning']}")
    else:
        lines.append(
            "🚫 No strong setup (probability < "
            f"{MIN_PROBABILITY_FOR_TRADE}% or RR < {MIN_RR})."
        )

    lines.append("\n_Not financial advice. Manage your own risk._")
    return "\n".join(lines)


def analyze_scan(symbol):
    """Return a compact signal dict for scanner, or None if no setup."""
    snapshot, price = build_snapshot(symbol, SCAN_TIMEFRAMES)
    if price is None:
        return None

    prompt = prompt_for_scan(symbol, snapshot, price)
    raw = call_gemini(prompt)
    data = extract_json(raw)
    if not data:
        return None

    probs = data.get("probabilities", {})
    up = float(probs.get("upside", 0) or 0.0)
    down = float(probs.get("downside", 0) or 0.0)
    flat = float(probs.get("flat", 0) or 0.0)
    best = data.get("best_direction", "flat")

    tp = data.get("trade_plan", {}) or {}
    direction = tp.get("direction", "none")
    rr_ratio = float(tp.get("rr_ratio", 0) or 0.0)

    if direction == "none" or best == "flat":
        return None

    # Use upside or downside probability, ignore flat
    prob = up if best == "upside" else down
    if prob < MIN_PROBABILITY_FOR_TRADE:
        return None

    # Enforce RR threshold
    if rr_ratio < MIN_RR:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "probability": prob,
        "entry": tp.get("entry", 0),
        "sl": tp.get("stop_loss", 0),
        "tp1": tp.get("take_profit_1", 0),
        "tp2": tp.get("take_profit_2", 0),
        "rr": rr_ratio,
        "confidence": tp.get("confidence", prob),
    }


# ===============================
# TELEGRAM HANDLERS
# ===============================

def start(update: Update, context: CallbackContext):
    """Enable scanner and show help."""
    global SCAN_ENABLED
    SCAN_ENABLED = True

    text = (
        "🤖 *Gemini 2.5 Futures Bot*\n\n"
        "Scanner is now: *ON* ✅\n\n"
        "Commands:\n"
        "• `/coin` → Multi-timeframe AI analysis (e.g. `/suiusdt`)\n"
        "• `/coin timeframe` → Single TF analysis (e.g. `/suiusdt 4h`)\n"
        "• `/stop` → Stop auto scanner (manual analysis still works)\n"
        "• `/start` → Turn auto scanner ON again\n\n"
        f"Scanner rules:\n"
        f"• Scans Binance USDT-M futures with 24h vol ≥ {MIN_VOLUME:,} USDT\n"
        f"• Signals only if upside/downside prob ≥ {MIN_PROBABILITY_FOR_TRADE}%\n"
        f"• Trade RR must be ≥ 1:{MIN_RR} and SL at key structural / volume-profile level."
    )
    update.message.reply_markdown(text)


def stop(update: Update, context: CallbackContext):
    """Disable scanner only; manual commands still work."""
    global SCAN_ENABLED
    SCAN_ENABLED = False
    update.message.reply_text(
        "⏹ Auto scanner is now *OFF*.\n\n"
        "You can still use manual analysis like `/btcusdt` or `/ethusdt 4h`.\n"
        "Send `/start` to turn auto scanner ON again."
    )


def handle_pair(update: Update, context: CallbackContext):
    """Handle any /coin or /coin timeframe command."""
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    parts = text.split()

    symbol = parts[0].lstrip("/").upper()
    timeframe = parts[1].lower() if len(parts) > 1 else None

    update.message.reply_text(
        f"⏳ Analysing {symbol}"
        + (f" on {timeframe}" if timeframe else "")
        + " using Gemini 2.5..."
    )

    try:
        result = analyze_command(symbol, timeframe)
    except Exception as e:
        result = f"❌ Error while analysing {symbol}: {e}"

    update.message.reply_markdown(result)


# ===============================
# SCANNER JOB
# ===============================

def scanner_job(context: CallbackContext):
    """Runs every 5 minutes via JobQueue, if SCAN_ENABLED."""
    if OWNER_CHAT_ID == 0:
        return

    # If user stopped scanner via /stop, do nothing
    if not SCAN_ENABLED:
        return

    try:
        symbols = get_top_symbols()
    except Exception as e:
        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=f"❌ Scanner: failed to fetch symbols: {e}",
        )
        return

    now = datetime.now(timezone.utc)

    for sym in symbols:
        try:
            sig = analyze_scan(sym)
        except Exception:
            continue

        if not sig:
            continue

        key = (sig["symbol"], sig["direction"])
        last = last_signal_time.get(key)
        if last and (now - last).total_seconds() < 1800:  # 30 min cooldown
            continue

        last_signal_time[key] = now

        msg = (
            "🚨 *AI Scanner Signal*\n"
            f"Pair: *{sig['symbol']}*\n"
            f"Direction: *{sig['direction'].upper()}*\n"
            f"Probability: `{sig['probability']}%`\n"
            f"RR: `{sig['rr']}` (>= {MIN_RR})\n"
            f"Entry: `{sig['entry']}`\n"
            f"SL (key level): `{sig['sl']}`\n"
            f"TP1: `{sig['tp1']}`\n"
            f"TP2: `{sig['tp2']}`\n"
            f"Confidence: `{sig['confidence']}%`\n\n"
            "_Use your own position sizing & risk control._"
        )

        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=msg,
            parse_mode="Markdown",
        )


# ===============================
# MAIN
# ===============================

def main():
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    # start/stop control scanner + help
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", start))
    dp.add_handler(CommandHandler("stop", stop))

    # catch all /pair commands like /suiusdt, /btcusdt 4h etc.
    dp.add_handler(MessageHandler(Filters.command, handle_pair))

    # JobQueue is built-in in v13
    jq = updater.job_queue
    jq.run_repeating(scanner_job, interval=SCAN_INTERVAL_SECONDS, first=30)

    print("✅ Bot running with polling + scanner (PTB v13.15)...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
