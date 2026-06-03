# -*- coding: utf-8 -*-
"""
============================================================
  XAUUSD Gold Signal Bot — 5 MINUTE
  Strategy  : EMA(8/21/50) + RSI(14) + MACD(12,26,9)
  Data      : Yahoo Finance (FREE)
  Timeframe : 5-minute candles
============================================================
"""

import time
import sys
import io
import logging
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
from signal_config import TWELVEDATA_API_KEY

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from signal_config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    EMA_FAST, EMA_MID, EMA_SLOW,
    RSI_LEN, RSI_OB, RSI_OS,
    MACD_FAST, MACD_SLOW, MACD_SIG,
    ATR_LEN, ATR_SL_MULT, ATR_TP1_MULT, ATR_TP2_MULT,
)

TIMEFRAME       = "5m"
TIMEFRAME_LABEL = "5min"
LOOP_INTERVAL   = 120

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [5M]  %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler("gold_signals_5m.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


def send_telegram(message: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }, timeout=10)
        if r.status_code == 200:
            log.info("Telegram sent OK")
        else:
            log.warning("Telegram error: " + r.text)
    except Exception as e:
        log.error("Telegram failed: " + str(e))


def get_candles():
    params = {
        "symbol":     "XAU/USD",
        "interval":   "5min",
        "outputsize": 300,        # ~25 hours of 5m candles
        "format":     "JSON",
        "apikey":     TWELVEDATA_API_KEY,
    }
    r = requests.get(TWELVEDATA_URL, params=params, timeout=15)
    data = r.json()

    if data.get("status") == "error":
        raise ValueError(f"TwelveData error: {data.get('message')}")

    values = data.get("values")
    if not values:
        raise ValueError("No candle data returned from TwelveData")

    df = pd.DataFrame(values)
    df = df.rename(columns={
        "datetime": "time",
        "open":     "open",
        "high":     "high",
        "low":      "low",
        "close":    "close",
    })
    df["time"]  = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df = df[["open", "high", "low", "close"]].astype(float)
    df = df.dropna().iloc[:-1]   # drop last potentially incomplete candle

    log.info(f"Fetched {len(df)} candles from TwelveData. Last close: {round(float(df['close'].iloc[-1]), 2)}")
    return df


def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()

def calc_rsi(s, p):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - (100 / (1 + g / l))

def calc_macd(s, f, sl, sig):
    ml = calc_ema(s, f) - calc_ema(s, sl)
    return ml, calc_ema(ml, sig), ml - calc_ema(ml, sig)

def calc_atr(df, p):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=p-1, adjust=False).mean()


def compute_indicators(df):
    df = df.copy()
    df["ema_fast"]    = calc_ema(df["close"], EMA_FAST)
    df["ema_mid"]     = calc_ema(df["close"], EMA_MID)
    df["ema_slow"]    = calc_ema(df["close"], EMA_SLOW)
    df["rsi"]         = calc_rsi(df["close"], RSI_LEN)
    ml, sl, hist      = calc_macd(df["close"], MACD_FAST, MACD_SLOW, MACD_SIG)
    df["macd_line"]   = ml
    df["macd_signal"] = sl
    df["macd_hist"]   = hist
    df["atr"]         = calc_atr(df, ATR_LEN)
    return df


def get_signal(df):
    if len(df) < 3:
        return None, {}

    cur  = df.iloc[-1]
    prev = df.iloc[-2]

    def v(x):
        return float(x.iloc[0]) if hasattr(x, 'iloc') else float(x)

    cf   = v(cur["ema_fast"])
    cm   = v(cur["ema_mid"])
    cs   = v(cur["ema_slow"])
    cc   = v(cur["close"])
    cr   = v(cur["rsi"])
    cml  = v(cur["macd_line"])
    cms  = v(cur["macd_signal"])
    cmh  = v(cur["macd_hist"])
    catr = v(cur["atr"])
    pf   = v(prev["ema_fast"])
    pm   = v(prev["ema_mid"])

    # ── Crossover: fast crosses above/below mid (Pine: ta.crossover/crossunder)
    buy_cross  = pf <= pm and cf > cm
    sell_cross = pf >= pm and cf < cm

    # ── Trend filter: price vs slow EMA (matches Pine: bullTrend / bearTrend)
    bull = cc > cs
    bear = cc < cs

    # ── RSI: asymmetric exactly like Pine Script ──────────────
    buy_rsi  = cr > 45 and cr < RSI_OB   # was: cr > RSI_OS
    sell_rsi = cr < 55 and cr > RSI_OS   # was: cr < RSI_OB

    # ── MACD: line vs signal AND histogram sign (matches Pine) ─
    buy_macd  = cml > cms and cmh > 0    # was: cml > cms only
    sell_macd = cml < cms and cmh < 0    # was: cml < cms only

    info = {
        "price":     round(cc, 2),
        "atr":       round(catr, 2),
        "rsi":       round(cr, 1),
        "macd_hist": round(cmh, 4),
    }

    if buy_cross and buy_rsi and buy_macd and bull:
        info["sl"]  = round(cc - catr * ATR_SL_MULT,  2)
        info["tp1"] = round(cc + catr * ATR_TP1_MULT, 2)
        info["tp2"] = round(cc + catr * ATR_TP2_MULT, 2)
        return "BUY", info

    if sell_cross and sell_rsi and sell_macd and bear:
        info["sl"]  = round(cc + catr * ATR_SL_MULT,  2)
        info["tp1"] = round(cc - catr * ATR_TP1_MULT, 2)
        info["tp2"] = round(cc - catr * ATR_TP2_MULT, 2)
        return "SELL", info

    return None, info


def format_message(direction, info):
    arrow = "UP" if direction == "BUY" else "DOWN"
    prefix = "[BUY]" if direction == "BUY" else "[SELL]"
    return (
        f"{prefix} <b>GOLD {direction} SIGNAL [{arrow}]</b>\n"
        f"------------------------\n"
        f"<b>Entry :</b> {info['price']:.2f}\n"
        f"<b>SL    :</b> {info['sl']:.2f}\n"
        f"<b>TP1   :</b> {info['tp1']:.2f}\n"
        f"<b>TP2   :</b> {info['tp2']:.2f}\n"
        f"------------------------\n"
        f"RSI: {info['rsi']} | ATR: {info['atr']}\n"
        f"Timeframe: {TIMEFRAME_LABEL} | XAUUSD\n"
        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
    )


def get_current_price():
    """Fetch current spot price for verification on startup."""
    params = {
        "symbol":  "XAU/USD",
        "apikey":  TWELVEDATA_API_KEY,
    }
    r = requests.get("https://api.twelvedata.com/price", params=params, timeout=10)
    data = r.json()
    return float(data["price"])


def main():
    log.info("=" * 50)
    log.info(f"  Gold Signal Bot - {TIMEFRAME_LABEL} Starting")
    log.info("=" * 50)

    # ── Verify current price on startup ──────────────────
    try:
        current_price = get_current_price()
        price_line = f"Live Price : <b>${current_price:,.2f}</b> (verify on TradingView)"
        log.info(f"Startup price check: ${current_price:,.2f}")
    except Exception as e:
        price_line = f"Live Price : ⚠️ Could not fetch ({e})"
        log.warning(f"Startup price fetch failed: {e}")
    # ─────────────────────────────────────────────────────

    send_telegram(
        f"<b>Gold Signal Bot Started - {TIMEFRAME_LABEL}</b>\n"
        f"------------------------\n"
        f"Instrument : XAUUSD\n"
        f"Timeframe  : {TIMEFRAME_LABEL}\n"
        f"Strategy   : EMA {EMA_FAST}/{EMA_MID}/{EMA_SLOW} + RSI + MACD\n"
        f"Data       : TwelveData (Spot XAU/USD)\n"
        f"Mode       : Signal Only - No Trades\n"
        f"------------------------\n"
        f"{price_line}\n"
        f"------------------------\n"
        f"Watching for signals..."
    )

    last_bar = None

    while True:
        try:
            df       = compute_indicators(get_candles())
            bar_time = str(df.index[-1])
            signal, info = get_signal(df)

            if signal and bar_time == last_bar:
                log.info(f"Signal already sent for {bar_time}, skipping.")
                signal = None

            if signal:
                log.info(f"*** SIGNAL: {signal} | Price: {info['price']} RSI: {info['rsi']} ***")
                send_telegram(format_message(signal, info))
                last_bar = bar_time
            else:
                log.info(
                    f"No signal | Price: {info.get('price')} | "
                    f"RSI: {info.get('rsi')} | "
                    f"MACD: {info.get('macd_hist')}"
                )

        except KeyboardInterrupt:
            log.info("Bot stopped.")
            send_telegram("<b>Gold 5min Bot Stopped</b>")
            break

        except Exception as e:
            log.error("Error: " + str(e))
            time.sleep(30)
            continue

        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    main()
