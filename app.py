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
    raise RuntimeError("TELEGRAM_BOT_TOKEN missing")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing")

GEMINI_MODEL = "gemini-2.5-flash"

# ===============================
# STRATEGY SETTINGS
# ===============================

# Auto scanner toggle (controlled by /start and /stop)
SCAN_ENABLED = True

# Scanner & filter settings
SCAN_INTERVAL_SECONDS = 300       # 5 minutes
MIN_VOLUME = 50_000_000           # 24h quote volume filter (>= 50M)
MAX_SCAN_SYMBOLS = 25
MIN_PROBABILITY_FOR_TRADE = 75    # min upside/downside probability (in %)

# NEW: required RR for both manual and auto signals
MIN_RR = 2.1                      # minimum RR 1 : 2.1

DEFAULT_TIMEFRAMES = [
    "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"
]
SCAN_TIMEFRAMES = ["15m", "1h", "4h"]

BINANCE_FAPI = "https://fapi.binance.com"

# Throttle repeated signals
last_signal_time = {}  # (symbol, direction) -> datetime

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
    RR must be >= MIN_RR (2.1) and TP targets should be realistic in relation
    to the upside/downside probability.
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
  - Overall trend (uptrend / downtrend / range) on each timeframe.
  - Important support/resistance zones.
  - Trendlines and channels, breakouts or retests.
  - Reversal candles (hammer, shooting star, engulfing, pin bars) at key levels.
  - Chart patterns (triangles, flags, head & shoulders, double top/bottom, etc.).

- Use a FIXED RANGE VOLUME PROFILE mental model:
  - Identify high-volume nodes (HVN) and low-volume nodes (LVN) within this range.
  - STOP LOSS must sit at a logical structural invalidation level:
    just beyond a key HVN/LVN or clear swing high/low.
  - SL must NOT be random and NOT extremely tight.

- Use EMAs (20, 50, 200), RSI, MACD and volume as confirmation only.

- TAKE PROFIT LOGIC:
  - RR (risk:reward) for the main idea must be at least {MIN_RR} (1:{MIN_RR}).
  - TP1 should be a realistic, high-probability target consistent with the
    upside/downside probability you give.
  - TP2 can be more ambitious but still logically reachable based on structure,
    trend and volume profile (not crazy far away).

TASK:
1. Determine probabilities (0-100) of the next meaningful move:
   - upside
   - downside
   - flat

2. Decide best_direction.

3. ONLY IF:
   - best_direction is upside or downside, AND
   - its probability >= {MIN_PROBABILITY_FOR_TRADE}, AND
   - a clean setup with rr_ratio >= {MIN_RR} exists,
   THEN generate a detailed trade plan:
   - direction (long/short)
   - entry (single entry price)
   - stop_loss (key level invalidation as above)
   - take_profit_1 (realistic target matching the probability)
   - take_profit_2 (more ambitious but still logical)
   - rr_ratio (>= {MIN_RR})
   - leverage_hint (safe range like 2x-4x)
   - confidence (0-100)
   - reasoning (short explanation referencing structure & volume profile)

4. If conditions are NOT met, set direction="none" and rr_ratio=0.

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
    RR must be >= MIN_RR and probability >= MIN_PROBABILITY_FOR_TRADE.
    """
    return f"""
Quick scan of crypto futures pair:

Symbol: {symbol}
Price: {price}

OHLCV JSON:
{json.dumps(snapshot)[:60000]}

RULES:
- Use price action + structure (trend, key levels, breakouts, reversals).
- Use a fixed range volume profile mental model for SL positioning at a
  structural invalidation level (beyond key HVN/LVN or swing high/low).
- Only accept setups where:
  - best_direction is upside or downside,
  - the probability of that move >= {MIN_PROBABILITY_FOR_TRADE}%, and
  - risk:reward rr_ratio >= {MIN_RR} (1:{MIN_RR} or better).

- TP1 must be a realistic, high-probability target consistent with the
  given move probability. TP2 can be more ambitious but still logical.

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

def call_gemini(prompt: str) -> str:
    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return resp.text


def extract_json(text: str):
    try:
        start = text.index("{")
        end = text.rindex("}")
        return json.loads(text[start:end + 1])
    except Exception:
        return None


# ===============================
# ANALYSIS FUNCTIONS
# ===============================

def analyze_command(symbol: str, timeframe: str | None):
    """Analysis for manual /pair and /pair timeframe commands."""
    tfs = [timeframe] if timeframe else DEFAULT_TIMEFRAMES
    snapshot, price = build_snapshot(symbol, tfs)

    if price is None:
        return f"❌ Could not fetch market data for *{symbol}*."

    prompt = prompt_for_pair(symbol, timeframe, snapshot, price)
    raw = call_gemini(prompt)
    data = extract_json(raw)

    if not data:
        return "⚠️ Gemini JSON parse error. Raw:\n\n" + raw[:2500]

    probs = data.get("probabilities", {})
    up = probs.get("upside", 0)
    down = probs.get("downside", 0)
    flat = probs.get("flat", 0)
    best = data.get("best_direction", "flat")
    view = data.get("overall_view", "")

    tp = data.get("trade_plan", {}) or {}
    direction = tp.get("direction", "none")
    rr_ratio = float(tp.get("rr_ratio", 0) or 0.0)

    # Enforce RR threshold for manual analysis too
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
        lines.append(f"RR: `{rr_ratio}` (≥ {MIN_RR})")
        lines.append(f"Leverage: `{tp.get('leverage_hint', '')}`")
        lines.append(f"Confidence: `{tp.get('confidence', 0)}%`")
        if tp.get("reasoning"):
            lines.append(f"Reason: {tp['reasoning']}")
    else:
        lines.append(
            f"🚫 No strong setup (probability < {MIN_PROBABILITY_FOR_TRADE}% "
            f"or RR < {MIN_RR})."
        )

    lines.append("\n_Not financial advice. Manage your own risk._")
    return "\n".join(lines)


def analyze_scan(symbol: str):
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
    flat = float(probs.get("flat", 0) or 0.0)  # not directly used
    best = data.get("best_direction", "flat")

    tp = data.get("trade_plan", {}) or {}
    direction = tp.get("direction", "none")
    rr_ratio = float(tp.get("rr_ratio", 0) or 0.0)

    if direction == "none" or best == "flat":
        return None

    prob = up if best == "upside" else down
    if prob < MIN_PROBABILITY_FOR_TRADE:
        return None
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
    """Enable scanner and show basic help."""
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
        "Scanner rules:\n"
        f"• Scans Binance USDT-M futures with 24h vol ≥ {MIN_VOLUME:,} USDT\n"
        f"• Signals only if upside/downside prob ≥ {MIN_PROBABILITY_FOR_TRADE}%\n"
        f"• Trade RR must be ≥ 1:{MIN_RR} and SL at strong structural/volume-profile levels."
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

    if not SCAN_ENABLED:
        return

    try:
        symbols = get_top_symbols()
    except Exception:
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
            f"Pair: *{sym}*\n"
            f"Direction: *{sig['direction'].upper()}*\n"
            f"Probability: `{sig['probability']}%`\n"
            f"RR: `{sig['rr']}` (≥ {MIN_RR})\n"
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

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", start))
    dp.add_handler(CommandHandler("stop", stop))

    # Catch all /coin commands such as /suiusdt or /btcusdt 4h
    dp.add_handler(MessageHandler(Filters.command, handle_pair))

    jq = updater.job_queue
    jq.run_repeating(scanner_job, interval=SCAN_INTERVAL_SECONDS, first=30)

    print("✅ Bot running with polling + scanner (PTB v13.15)...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
