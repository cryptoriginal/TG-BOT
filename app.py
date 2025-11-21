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

SCAN_INTERVAL_SECONDS = 300
MIN_VOLUME = 35_000_000
MAX_SCAN_SYMBOLS = 25
MIN_PROBABILITY_FOR_TRADE = 65

DEFAULT_TIMEFRAMES = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]
SCAN_TIMEFRAMES = ["15m", "1h", "4h"]

BINANCE_FAPI = "https://fapi.binance.com"

last_signal_time = {}

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ===============================
# BINANCE API
# ===============================
def get_klines(symbol, interval, limit=120):
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def get_top_symbols():
    url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    pairs = [
        s for s in data
        if s["symbol"].endswith("USDT")
        and float(s["quoteVolume"]) >= MIN_VOLUME
    ]
    pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [p["symbol"] for p in pairs[:MAX_SCAN_SYMBOLS]]


def build_snapshot(symbol, timeframes):
    snapshot = {}
    current_price = None

    for tf in timeframes:
        kl = get_klines(symbol, tf, 100)
        candles = []
        for c in kl:
            candles.append({
                "open_time": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5])
            })
        snapshot[tf] = candles
        if candles:
            current_price = candles[-1]["close"]

    return snapshot, current_price


# ===============================
# GEMINI PROMPTS
# ===============================
def prompt_for_pair(symbol, timeframe, snapshot, price):
    return f"""
You are an elite crypto futures analyst.

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
   - direction
   - entry
   - stop_loss
   - take_profit_1
   - take_profit_2
   - rr_ratio
   - leverage_hint
   - confidence
   - reasoning

Return ONLY valid JSON in this exact schema:

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
    return f"""
Scan this crypto futures pair:

Symbol: {symbol}
Price: {price}

OHLCV JSON:
{json.dumps(snapshot)[:60000]}

TASK:
1. Determine upside/downside/flat probabilities.
2. Pick best_direction.
3. If best_direction != flat and probability >= {MIN_PROBABILITY_FOR_TRADE}, produce a short trade plan.

Return only JSON:
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
# GEMINI CALL
# ===============================
def call_gemini(prompt: str):
    r = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return r.text


def extract_json(text):
    try:
        start = text.index("{")
        end = text.rindex("}")
        return json.loads(text[start:end+1])
    except:
        return None


# ===============================
# ANALYSIS FUNCTIONS
# ===============================
def analyze_command(symbol, timeframe):
    tfs = [timeframe] if timeframe else DEFAULT_TIMEFRAMES
    snapshot, price = build_snapshot(symbol, tfs)
    if price is None:
        return "Invalid symbol or market data error."

    prompt = prompt_for_pair(symbol, timeframe, snapshot, price)
    raw = call_gemini(prompt)
    data = extract_json(raw)
    if not data:
        return "Gemini JSON error:\n" + raw[:3000]

    p = data["probabilities"]
    tp = data["trade_plan"]

    msg = f"📊 *{symbol} Analysis*\n"
    msg += f"Price: `{price}`\n\n"
    msg += "*Probabilities:*\n"
    msg += f"⬆️ Upside: `{p['upside']}%`\n"
    msg += f"⬇️ Downside: `{p['downside']}%`\n"
    msg += f"➖ Flat: `{p['flat']}%`\n"
    msg += f"🎯 Best direction: *{data['best_direction'].upper()}*\n\n"
    msg += f"*View:* {data['overall_view']}\n\n"

    if tp["direction"] != "none":
        msg += "*🔥 Trade Setup:*\n"
        msg += f"Direction: *{tp['direction'].upper()}*\n"
        msg += f"Entry: `{tp['entry']}`\n"
        msg += f"SL: `{tp['stop_loss']}`\n"
        msg += f"TP1: `{tp['take_profit_1']}`\n"
        msg += f"TP2: `{tp['take_profit_2']}`\n"
        msg += f"RR: `{tp['rr_ratio']}`\n"
        msg += f"Confidence: `{tp['confidence']}%`\n"
        msg += f"Reason: {tp['reasoning']}\n"
    else:
        msg += "🚫 No high probability setup (prob < 65%)."

    return msg


def analyze_scan(symbol):
    snapshot, price = build_snapshot(symbol, SCAN_TIMEFRAMES)
    if price is None:
        return None

    prompt = prompt_for_scan(symbol, snapshot, price)
    raw = call_gemini(prompt)
    data = extract_json(raw)
    if not data:
        return None

    best = data["best_direction"]
    prob = max(
        data["probabilities"]["upside"],
        data["probabilities"]["downside"]
    )

    tp = data["trade_plan"]

    if tp["direction"] == "none":
        return None
    if prob < MIN_PROBABILITY_FOR_TRADE:
        return None

    return {
        "symbol": symbol,
        "direction": tp["direction"],
        "probability": prob,
        "entry": tp["entry"],
        "sl": tp["stop_loss"],
        "tp1": tp["take_profit_1"],
        "tp2": tp["take_profit_2"],
        "confidence": tp["confidence"]
    }


# ===============================
# TELEGRAM BOT
# ===============================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Bot Online.\nUse /btcusdt or /btcusdt 4h"
    )


async def handle_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    parts = text.split()

    cmd = parts[0].replace("/", "").upper()
    timeframe = parts[1].lower() if len(parts) > 1 else None

    loop = asyncio.get_running_loop()
    msg = await loop.run_in_executor(
        None, analyze_command, cmd, timeframe
    )

    await update.message.reply_markdown(msg)


# ===============================
# SCANNER JOB
# ===============================
async def scanner(context):
    if OWNER_CHAT_ID == 0:
        return

    loop = asyncio.get_running_loop()
    symbols = await loop.run_in_executor(None, get_top_symbols)

    for sym in symbols:
        result = await loop.run_in_executor(None, analyze_scan, sym)
        if not result:
            continue

        key = (sym, result["direction"])
        now = datetime.now(timezone.utc)
        last = last_signal_time.get(key)

        if last and (now - last).total_seconds() < 1800:
            continue

        last_signal_time[key] = now

        msg = f"🚨 *AI Signal*\n"
        msg += f"Pair: *{sym}*\n"
        msg += f"Direction: *{result['direction'].upper()}*\n"
        msg += f"Probability: `{result['probability']}%`\n"
        msg += f"Entry: `{result['entry']}`\n"
        msg += f"SL: `{result['sl']}`\n"
        msg += f"TP1: `{result['tp1']}`\n"
        msg += f"TP2: `{result['tp2']}`\n"
        msg += f"Confidence: `{result['confidence']}%`"

        await context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=msg,
            parse_mode="Markdown"
        )


# ===============================
# MAIN
# ===============================
def main():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)
        .job_queue()
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.COMMAND, handle_pair))

    jq = app.job_queue
    jq.run_repeating(scanner, interval=SCAN_INTERVAL_SECONDS, first=20)

    print("Bot running with polling + scanner...")
    app.run_polling()


if __name__ == "__main__":
    main()
