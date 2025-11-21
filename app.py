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
    MessageHandler,
    CommandHandler,
    filters,
)

# =========================
# CONFIG
# =========================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Your own Telegram numeric user ID (where scanner signals will be sent)
# Set this in Render environment as OWNER_CHAT_ID.
OWNER_CHAT_ID_ENV = os.getenv("OWNER_CHAT_ID")
OWNER_CHAT_ID = int(OWNER_CHAT_ID_ENV) if OWNER_CHAT_ID_ENV else None

GEMINI_MODEL = "gemini-2.5-flash"

# Scan every 5 minutes
SCAN_INTERVAL_SECONDS = 300
# Minimum 24h quote volume (USDT) to consider in scanner
MIN_QUOTE_VOLUME = 35_000_000
# Max number of symbols to scan on each run (to control cost/speed)
MAX_SCAN_SYMBOLS = 25

# Default timeframes for /pair analysis
DEFAULT_TFS_FOR_MULTI = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]
# Timeframes used by scanner (shorter to keep it lighter)
SCAN_TIMEFRAMES = ["15m", "1h", "4h"]

# Probability threshold to consider a trade “good”
MIN_PROBABILITY_FOR_TRADE = 65.0

# Track last signals so we don't spam same thing every 5 minutes
last_signal_time = {}  # (symbol, direction) -> datetime


# =========================
# GEMINI CLIENT
# =========================

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY env var is not set!")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# =========================
# MARKET DATA HELPERS (BINANCE FUTURES)
# =========================

BINANCE_FAPI_BASE = "https://fapi.binance.com"


def get_klines(symbol: str, interval: str, limit: int = 150):
    """
    Get OHLCV candles from Binance USDT-M Futures.
    """
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data


def get_top_futures_symbols(
    min_quote_volume: float = MIN_QUOTE_VOLUME,
    max_symbols: int = MAX_SCAN_SYMBOLS,
):
    """
    Get all USDT-M futures tickers, filter by 24h quoteVolume and return top symbols.
    """
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/ticker/24hr"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    stats = resp.json()

    filtered = [
        s
        for s in stats
        if s.get("symbol", "").endswith("USDT")
        and float(s.get("quoteVolume", 0.0)) >= min_quote_volume
    ]

    # Sort by quote volume desc
    filtered.sort(key=lambda s: float(s.get("quoteVolume", 0.0)), reverse=True)
    symbols = [s["symbol"] for s in filtered[:max_symbols]]
    return symbols


def build_market_snapshot(symbol: str, timeframes: list[str], limit: int = 120):
    """
    Build a dict of OHLCV data per timeframe for Gemini.
    Each timeframe: list of candle dicts with timestamp, O, H, L, C, V.
    """
    snapshot = {}
    current_price = None

    for tf in timeframes:
        klines = get_klines(symbol, tf, limit=limit)
        candles = []
        for k in klines:
            # Binance kline fields:
            # 0: Open time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume, ...
            candles.append(
                {
                    "open_time": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )
        snapshot[tf] = candles
        if candles:
            current_price = candles[-1]["close"]

    return snapshot, current_price


# =========================
# GEMINI PROMPTS
# =========================

def build_command_prompt(symbol: str, timeframe: str | None, market_snapshot: dict, current_price: float):
    """
    Prompt for user command analysis (/SUIUSDT or /SUIUSDT 4h).
    """
    if timeframe:
        tf_text = f"Focus mainly on timeframe: {timeframe}."
        tfs_used = [timeframe]
    else:
        tf_text = (
            "Analyse the market using ALL these timeframes together: "
            + ", ".join(sorted(market_snapshot.keys()))
        )
        tfs_used = sorted(market_snapshot.keys())

    prompt = f"""
You are an extremely skilled financial analyst and crypto futures trader with GOD-TIER skills.
You analyse crypto futures pairs as if your life depends on it.

Symbol: {symbol}
Current price: {current_price}
Timeframes included: {", ".join(tfs_used)}
{tf_text}

Input data:
The following is OHLCV data per timeframe in JSON format.
Use it to understand trend, key levels, breakout/bounce points, strong reversals, etc.

OHLCV_JSON:
{json.dumps(market_snapshot)[:80000]}

TASK:
1. First, analyse pure PRICE ACTION:
   - Trend direction (uptrend, downtrend, ranging) on each timeframe
   - Important support/resistance zones
   - Trendlines and channels, breakouts or bounces
   - Reversal candles (hammer, shooting star, engulfing, pin bars) at key levels
   - Chart patterns if any (triangles, flags, head & shoulders, double top/bottom, etc.)

2. Then, imagine you also have access to top indicators
   (EMA/SMA 20/50/200, RSI, MACD, volume profile) and **mentally** use them
   as additional confirmation for your price action view.

3. Based on everything, estimate PROBABILITY (0–100%) of the next significant move for the pair over the next
   4–24 hours (for the focused timeframe if given, otherwise overall):
   - upside_prob  = probability of a meaningful bullish move
   - downside_prob = probability of a meaningful bearish move
   - flat_prob = probability it stays choppy / sideways, not worth trading

4. Decide the best_direction = "upside", "downside", or "flat" based on which probability is highest.

5. If best_direction is "upside" OR "downside" and its probability is >= {MIN_PROBABILITY_FOR_TRADE},
   you MUST propose a detailed trade plan (for a futures trade):
   - direction: "long" if upside, "short" if downside
   - entry: good entry zone (single price number)
   - stop_loss: safe SL where the idea is clearly invalidated
   - take_profit_1: first realistic target
   - take_profit_2: more ambitious target
   - rr_ratio: approximate risk:reward ratio of the main setup
   - leverage_hint: safe max leverage suggestion (e.g., 3x-5x)
   - confidence: confidence in this trade idea (0–100)
   - reasoning: short explanation based mainly on price action from key levels,
     trendlines, chart patterns, and confirmation from indicators.

6. If best_direction is "flat" OR probability < {MIN_PROBABILITY_FOR_TRADE},
   clearly say that it is better to avoid this pair for now and set direction="none"
   in trade plan (entry/SL/TP can be null).

IMPORTANT:
- Answer ONLY in **VALID JSON**, no extra text, no comments.
- Use this exact schema:

{{
  "symbol": "{symbol}",
  "timeframe_focus": "{timeframe if timeframe else 'multi'}",
  "probabilities": {{
    "upside": 0,
    "downside": 0,
    "flat": 0
  }},
  "best_direction": "upside | downside | flat",
  "overall_view": "short natural language summary of your view",
  "trade_plan": {{
    "direction": "long | short | none",
    "entry": 0,
    "stop_loss": 0,
    "take_profit_1": 0,
    "take_profit_2": 0,
    "rr_ratio": 0,
    "leverage_hint": "string",
    "confidence": 0,
    "reasoning": "short explanation"
  }}
}}
"""
    return prompt


def build_scan_prompt(symbol: str, market_snapshot: dict, current_price: float):
    """
    Shorter prompt for scanner, focused on: should we trade or not?
    """
    prompt = f"""
You are an elite crypto futures trader.
Your job is to quickly scan this symbol and ONLY tell me if it is a HIGH-PROBABILITY trade setup now.

Symbol: {symbol}
Current price: {current_price}
Timeframes included: {", ".join(sorted(market_snapshot.keys()))}

OHLCV_JSON:
{json.dumps(market_snapshot)[:60000]}

Consider mainly:
- Price action around key support/resistance
- Trend direction
- Reversal candles or strong breakout/bounce from key levels
- Momentum and trend confirmation using imagined EMAs/RSI/MACD

Your target horizon is the next 4–12 hours.

TASK:
1. Estimate probabilities (0–100) for:
   - upside_prob: meaningful bullish move
   - downside_prob: meaningful bearish move
   - flat_prob: choppy / no clean move

2. Decide best_direction = "upside", "downside", or "flat".

3. If best_direction is "upside" or "downside" AND its probability >= {MIN_PROBABILITY_FOR_TRADE},
   propose a simple trade plan:
   - direction: "long" if upside, "short" if downside
   - entry, stop_loss, take_profit_1, take_profit_2
   - confidence: 0–100
   Keep it concise.

4. If best_direction is "flat" OR probability < {MIN_PROBABILITY_FOR_TRADE},
   set direction="none" and you may leave entry/SL/TP as 0.

Respond **ONLY** with JSON using this schema:

{{
  "symbol": "{symbol}",
  "probabilities": {{
    "upside": 0,
    "downside": 0,
    "flat": 0
  }},
  "best_direction": "upside | downside | flat",
  "trade_plan": {{
    "direction": "long | short | none",
    "entry": 0,
    "stop_loss": 0,
    "take_profit_1": 0,
    "take_profit_2": 0,
    "confidence": 0
  }}
}}
"""
    return prompt


# =========================
# GEMINI CALL HELPERS
# =========================

def call_gemini(prompt_text: str) -> str:
    """
    Call Gemini 2.5 Flash and return response text.
    """
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt_text,
    )
    # google-genai SDK exposes .text convenience property :contentReference[oaicite:2]{index=2}
    return response.text


def extract_json(text: str) -> dict | None:
    """
    Try to extract JSON block from response and parse it.
    """
    try:
        first = text.index("{")
        last = text.rindex("}")
        json_str = text[first : last + 1]
        return json.loads(json_str)
    except Exception:
        return None


# =========================
# ANALYSIS FUNCTIONS (SYNC, RUN IN EXECUTOR)
# =========================

def analyze_pair_for_command(symbol: str, timeframe: str | None = None) -> str:
    """
    Full analysis for user commands.
    Returns formatted string message for Telegram.
    """
    tf_list = [timeframe] if timeframe else DEFAULT_TFS_FOR_MULTI

    snapshot, current_price = build_market_snapshot(symbol, tf_list)
    if not snapshot or current_price is None:
        return f"❌ Could not fetch market data for {symbol}. Maybe symbol is invalid?"

    prompt = build_command_prompt(symbol, timeframe, snapshot, current_price)
    raw_text = call_gemini(prompt)
    data = extract_json(raw_text)

    if not data:
        # Fallback: just return raw model text
        return (
            f"⚠️ Could not parse JSON from Gemini.\n\n"
            f"Raw response:\n{raw_text[:3500]}"
        )

    probs = data.get("probabilities", {})
    up = probs.get("upside", 0)
    down = probs.get("downside", 0)
    flat = probs.get("flat", 0)
    best = data.get("best_direction", "flat")
    view = data.get("overall_view", "")

    tp = data.get("trade_plan", {}) or {}
    direction = tp.get("direction", "none")
    entry = tp.get("entry", 0)
    sl = tp.get("stop_loss", 0)
    tp1 = tp.get("take_profit_1", 0)
    tp2 = tp.get("take_profit_2", 0)
    rr = tp.get("rr_ratio", 0)
    lev = tp.get("leverage_hint", "")
    conf = tp.get("confidence", 0)
    reasoning = tp.get("reasoning", "")

    # Build nice Telegram message
    lines = []
    title_tf = timeframe if timeframe else "Multi-timeframe view"
    lines.append(f"📊 *{symbol}* analysis — *{title_tf}*")
    lines.append("")
    lines.append(f"Price: `{current_price}`")
    lines.append("")
    lines.append("*Probabilities (next move)*")
    lines.append(f"⬆️ Upside: `{up:.1f}%`")
    lines.append(f"⬇️ Downside: `{down:.1f}%`")
    lines.append(f"➖ Flat/Choppy: `{flat:.1f}%`")
    lines.append(f"🎯 Best direction: *{best.upper()}*")
    if view:
        lines.append("")
        lines.append(f"*View:* {view}")

    # Trade plan
    if direction != "none" and (best in ["upside", "downside"]) and (
        (best == "upside" and up >= MIN_PROBABILITY_FOR_TRADE)
        or (best == "downside" and down >= MIN_PROBABILITY_FOR_TRADE)
        or conf >= MIN_PROBABILITY_FOR_TRADE
    ):
        lines.append("")
        lines.append("*⚔️ Trade Setup (AI)*")
        lines.append(f"Direction: *{direction.upper()}*")
        lines.append(f"Entry: `{entry}`")
        lines.append(f"Stop Loss: `{sl}`")
        lines.append(f"TP1: `{tp1}`")
        lines.append(f"TP2: `{tp2}`")
        lines.append(f"Approx RR: `{rr}`")
        if lev:
            lines.append(f"Leverage (hint): `{lev}`")
        lines.append(f"AI Confidence: `{conf:.1f}%`")
        if reasoning:
            lines.append("")
            lines.append(f"*Reasoning:* {reasoning}")
    else:
        lines.append("")
        lines.append(
            "🚫 No high-probability trade setup (>= "
            f"{MIN_PROBABILITY_FOR_TRADE:.0f}% ) right now. Better to avoid or wait."
        )

    lines.append("")
    lines.append("_Note: This is NOT financial advice. Use your own risk management._")

    return "\n".join(lines)


def analyze_pair_for_scan(symbol: str) -> dict | None:
    """
    Short analysis for scanner.
    Returns dict with symbol, direction, probability, entry, sl, tp1, tp2
    or None if no good setup.
    """
    snapshot, current_price = build_market_snapshot(symbol, SCAN_TIMEFRAMES, limit=80)
    if not snapshot or current_price is None:
        return None

    prompt = build_scan_prompt(symbol, snapshot, current_price)
    raw_text = call_gemini(prompt)
    data = extract_json(raw_text)
    if not data:
        return None

    probs = data.get("probabilities", {})
    up = probs.get("upside", 0)
    down = probs.get("downside", 0)
    flat = probs.get("flat", 0)
    best = data.get("best_direction", "flat")

    tp = data.get("trade_plan", {}) or {}
    direction = tp.get("direction", "none")

    # Decide if it's signal-worthy
    if best == "upside":
        prob = float(up)
    elif best == "downside":
        prob = float(down)
    else:
        prob = float(flat)

    if direction == "none" or best == "flat" or prob < MIN_PROBABILITY_FOR_TRADE:
        return None

    # Build signal dict
    signal = {
        "symbol": symbol,
        "best_direction": best,
        "direction": direction,
        "probability": prob,
        "entry": tp.get("entry", 0),
        "stop_loss": tp.get("stop_loss", 0),
        "take_profit_1": tp.get("take_profit_1", 0),
        "take_profit_2": tp.get("take_profit_2", 0),
        "confidence": tp.get("confidence", prob),
    }
    return signal


# =========================
# TELEGRAM HANDLERS
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 Hey! I'm your *Gemini 2.5 AI Futures Bot*.\n\n"
        "Usage:\n"
        "• `/suiusdt` → Multi-timeframe AI analysis for SUIUSDT futures\n"
        "• `/suiusdt 4h` → AI analysis focused on 4h timeframe\n\n"
        "The bot estimates probability of *Upside / Downside / Flat* and\n"
        "if probability ≥ 65%, it suggests entry, SL & TP targets based on\n"
        "price action + key levels + trendlines + indicators.\n\n"
        "_Scanner:_ Every 5 min it scans Binance USDT-M futures with\n"
        "24h volume ≥ 35M and pushes high-probability setups to my owner."
    )
    await update.message.reply_markdown(text)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)


async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Catch all /commands so /suiusdt and /suiusdt 4h work without defining each one.
    """
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    # text like "/suiusdt" or "/suiusdt 4h"
    parts = text.split()
    cmd_raw = parts[0]  # "/suiusdt"
    symbol = cmd_raw.lstrip("/").upper()

    timeframe = None
    if len(parts) > 1:
        timeframe = parts[1].lower()

    await update.message.reply_text(f"⏳ Analysing {symbol} "
                                    f"{'on ' + timeframe if timeframe else ''} with Gemini 2.5...")

    loop = asyncio.get_running_loop()
    try:
        result_text = await loop.run_in_executor(
            None, analyze_pair_for_command, symbol, timeframe
        )
    except Exception as e:
        result_text = f"❌ Error while analysing {symbol}: {e}"

    await update.message.reply_markdown(result_text)


# =========================
# SCANNER (JOB QUEUE)
# =========================

def format_signal_message(signal: dict) -> str:
    symbol = signal["symbol"]
    direction = signal["direction"].upper()
    best_dir = signal["best_direction"].upper()
    prob = signal["probability"]
    conf = signal.get("confidence", prob)
    entry = signal["entry"]
    sl = signal["stop_loss"]
    tp1 = signal["take_profit_1"]
    tp2 = signal["take_profit_2"]

    lines = []
    lines.append("🚨 *AI Scanner Signal*")
    lines.append(f"Pair: *{symbol}*")
    lines.append(f"Move bias: *{best_dir}*")
    lines.append(f"Trade direction: *{direction}*")
    lines.append(f"Probability: `{prob:.1f}%`")
    lines.append(f"AI Confidence: `{conf:.1f}%`")
    lines.append("")
    lines.append(f"Entry: `{entry}`")
    lines.append(f"Stop Loss: `{sl}`")
    lines.append(f"TP1: `{tp1}`")
    lines.append(f"TP2: `{tp2}`")
    lines.append("")
    lines.append("_Use your own position sizing & risk management._")
    return "\n".join(lines)


def scan_market_sync() -> list[dict]:
    """
    Sync function that:
    - fetches top futures symbols by volume
    - runs short AI scan on each
    - returns list of signal dicts
    """
    symbols = get_top_futures_symbols()
    signals: list[dict] = []

    for symbol in symbols:
        try:
            sig = analyze_pair_for_scan(symbol)
            if sig:
                signals.append(sig)
        except Exception:
            # Ignore failures for individual symbols
            continue

    return signals


async def scan_market_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Async job that calls scan_market_sync() in a thread and sends signals to OWNER_CHAT_ID.
    """
    if not OWNER_CHAT_ID:
        # No owner configured; do nothing
        return

    loop = asyncio.get_running_loop()
    try:
        signals = await loop.run_in_executor(None, scan_market_sync)
    except Exception as e:
        # Optionally notify owner once
        await context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=f"❌ Scanner error: {e}",
        )
        return

    if not signals:
        return

    now = datetime.now(timezone.utc)
    for sig in signals:
        key = (sig["symbol"], sig["direction"])
        last = last_signal_time.get(key)
        # Avoid spamming same direction on same pair too often
        if last and (now - last).total_seconds() < 1800:  # 30 minutes
            continue

        msg = format_signal_message(sig)
        try:
            await context.bot.send_message(chat_id=OWNER_CHAT_ID, text=msg, parse_mode="Markdown")
            last_signal_time[key] = now
        except Exception:
            continue


# =========================
# MAIN
# =========================

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN env var is not set!")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Classic commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))

    # Catch all /whatever commands like /suiusdt, /suiusdt 4h
    app.add_handler(MessageHandler(filters.COMMAND, handle_command))

    # Scanner job every 5 min
    app.job_queue.run_repeating(
        scan_market_job,
        interval=SCAN_INTERVAL_SECONDS,
        first=30,
        name="market_scanner",
    )

    print("Bot started. Polling Telegram + running scanner...")
    app.run_polling()


if __name__ == "__main__":
    main()
