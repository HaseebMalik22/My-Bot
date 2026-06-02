# -*- coding: utf-8 -*-

"""
=============================================================
XAUUSD Gold Signal Bot - 5 Minute
Strategy:
- EMA 8 / 21 / 50
- RSI 14
- MACD 12 / 26 / 9
- ATR SL/TP

Data Source:
Yahoo Finance (GC=F)

Deployment:
Railway

Author:
Optimized Version
=============================================================
"""

import os
import time
import sys
import io
import logging
import requests
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding="utf-8",
    errors="replace"
)

sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer,
    encoding="utf-8",
    errors="replace"
)

from signal_config import (
    EMA_FAST,
    EMA_MID,
    EMA_SLOW,
    RSI_LEN,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIG,
    ATR_LEN,
    ATR_SL_MULT,
    ATR_TP1_MULT,
    ATR_TP2_MULT
)

# ============================================================
# TELEGRAM CONFIG
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ============================================================
# SETTINGS
# ============================================================

SYMBOL = "GC=F"
TIMEFRAME = "5m"
TIMEFRAME_LABEL = "5 Minute"

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [5M] %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.getLogger(__name__)

# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if not TELEGRAM_BOT_TOKEN:
        return

    try:

        url = (
            f"https://api.telegram.org/bot"
            f"{TELEGRAM_BOT_TOKEN}/sendMessage"
        )

        r = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            },
            timeout=15
        )

        if r.status_code == 200:
            log.info("Telegram sent OK")

        else:
            log.warning(
                f"Telegram Error: {r.text}"
            )

    except Exception as e:
        log.error(
            f"Telegram Failed: {e}"
        )

# ============================================================
# SCHEDULER
# ============================================================

def wait_for_next_5m_close():

    now = datetime.utcnow()

    next_minute = (
        (now.minute // 5) + 1
    ) * 5

    if next_minute == 60:

        next_run = (
            now.replace(
                minute=0,
                second=15,
                microsecond=0
            )
            + timedelta(hours=1)
        )

    else:

        next_run = now.replace(
            minute=next_minute,
            second=15,
            microsecond=0
        )

    sleep_seconds = (
        next_run - now
    ).total_seconds()

    log.info(
        f"Waiting {int(sleep_seconds)}s "
        f"for next candle close..."
    )

    time.sleep(
        max(0, sleep_seconds)
    )

# ============================================================
# MARKET DATA
# ============================================================

def get_candles():

    df = yf.download(
        SYMBOL,
        period="5d",
        interval=TIMEFRAME,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        raise Exception(
            "Yahoo returned no data"
        )

    df = df[
        ["Open", "High", "Low", "Close"]
    ].copy()

    df.columns = [
        "open",
        "high",
        "low",
        "close"
    ]

    # Remove incomplete candle
    df = df.dropna().iloc[:-1]

    log.info(
        f"Candles={len(df)} | "
        f"Time={df.index[-1]} | "
        f"Close={float(df['close'].iloc[-1]):.2f}"
    )

    return df

# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def rsi(series, period):

    delta = series.diff()

    gain = (
        delta.clip(lower=0)
        .ewm(com=period - 1,
             adjust=False)
        .mean()
    )

    loss = (
        -delta.clip(upper=0)
        .ewm(com=period - 1,
             adjust=False)
        .mean()
    )

    rs = gain / loss

    return 100 - (
        100 / (1 + rs)
    )


def macd(series):

    fast = ema(
        series,
        MACD_FAST
    )

    slow = ema(
        series,
        MACD_SLOW
    )

    line = fast - slow

    signal = ema(
        line,
        MACD_SIG
    )

    hist = line - signal

    return line, signal, hist


def atr(df, period):

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"]
                - df["close"].shift()
            ).abs(),
            (
                df["low"]
                - df["close"].shift()
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        com=period - 1,
        adjust=False
    ).mean()

# ============================================================
# COMPUTE INDICATORS
# ============================================================

def compute(df):

    df = df.copy()

    df["ema_fast"] = ema(
        df["close"],
        EMA_FAST
    )

    df["ema_mid"] = ema(
        df["close"],
        EMA_MID
    )

    df["ema_slow"] = ema(
        df["close"],
        EMA_SLOW
    )

    df["rsi"] = rsi(
        df["close"],
        RSI_LEN
    )

    line, sig, hist = macd(
        df["close"]
    )

    df["macd_line"] = line
    df["macd_signal"] = sig
    df["macd_hist"] = hist

    df["atr"] = atr(
        df,
        ATR_LEN
    )

    return df

# ============================================================
# SIGNAL ENGINE
# ============================================================

def get_signal(df):

    cur = df.iloc[-1]
    prev = df.iloc[-2]

    buy_cross = (
        prev["ema_fast"] <= prev["ema_mid"]
        and
        cur["ema_fast"] > cur["ema_mid"]
    )

    sell_cross = (
        prev["ema_fast"] >= prev["ema_mid"]
        and
        cur["ema_fast"] < cur["ema_mid"]
    )

    bull_trend = (
        cur["close"] >
        cur["ema_slow"]
    )

    bear_trend = (
        cur["close"] <
        cur["ema_slow"]
    )

    buy_rsi = cur["rsi"] > 55
    sell_rsi = cur["rsi"] < 45

    buy_macd = (
        cur["macd_line"]
        >
        cur["macd_signal"]
    )

    sell_macd = (
        cur["macd_line"]
        <
        cur["macd_signal"]
    )

    info = {

        "price":
            round(
                float(cur["close"]),
                2
            ),

        "atr":
            round(
                float(cur["atr"]),
                2
            ),

        "rsi":
            round(
                float(cur["rsi"]),
                1
            ),

        "macd_line":
            round(
                float(cur["macd_line"]),
                4
            ),

        "macd_signal":
            round(
                float(cur["macd_signal"]),
                4
            )
    }

    if (
        buy_cross
        and buy_rsi
        and buy_macd
        and bull_trend
    ):

        price = float(
            cur["close"]
        )

        atr_val = float(
            cur["atr"]
        )

        info["sl"] = round(
            price -
            atr_val *
            ATR_SL_MULT,
            2
        )

        info["tp1"] = round(
            price +
            atr_val *
            ATR_TP1_MULT,
            2
        )

        info["tp2"] = round(
            price +
            atr_val *
            ATR_TP2_MULT,
            2
        )

        return "BUY", info

    if (
        sell_cross
        and sell_rsi
        and sell_macd
        and bear_trend
    ):

        price = float(
            cur["close"]
        )

        atr_val = float(
            cur["atr"]
        )

        info["sl"] = round(
            price +
            atr_val *
            ATR_SL_MULT,
            2
        )

        info["tp1"] = round(
            price -
            atr_val *
            ATR_TP1_MULT,
            2
        )

        info["tp2"] = round(
            price -
            atr_val *
            ATR_TP2_MULT,
            2
        )

        return "SELL", info

    return None, info

# ============================================================
# MESSAGE FORMAT
# ============================================================

def format_signal(signal, info):

    return (
        f"📈 <b>GOLD {signal}</b>\n\n"
        f"Entry: {info['price']}\n"
        f"SL: {info['sl']}\n"
        f"TP1: {info['tp1']}\n"
        f"TP2: {info['tp2']}\n\n"
        f"RSI: {info['rsi']}\n"
        f"ATR: {info['atr']}\n\n"
        f"{TIMEFRAME_LABEL}"
    )

# ============================================================
# MAIN
# ============================================================

def main():

    log.info(
        "Gold Signal Bot Started"
    )

    send_telegram(
        "✅ Gold Signal Bot Started\n"
        "Watching for signals..."
    )

    last_bar = None
    last_heartbeat_hour = None

    while True:

        wait_for_next_5m_close()

        try:

            df = compute(
                get_candles()
            )

            bar_time = str(
                df.index[-1]
            )

            signal, info = get_signal(df)

            current_hour = (
                datetime.utcnow().hour
            )

            if (
                current_hour
                !=
                last_heartbeat_hour
            ):

                send_telegram(
                    f"✅ Bot Alive\n"
                    f"Price: {info['price']}"
                )

                last_heartbeat_hour = (
                    current_hour
                )

            if signal:

                if (
                    bar_time
                    !=
                    last_bar
                ):

                    send_telegram(
                        format_signal(
                            signal,
                            info
                        )
                    )

                    log.info(
                        f"SIGNAL {signal} "
                        f"Price={info['price']}"
                    )

                    last_bar = bar_time

            else:

                log.info(
                    f"No Signal | "
                    f"Price={info['price']} | "
                    f"RSI={info['rsi']} | "
                    f"MACD={info['macd_line']} / "
                    f"{info['macd_signal']}"
                )

        except Exception as e:

            error_text = str(e)

            log.error(
                error_text
            )

            send_telegram(
                f"🚨 BOT ERROR\n\n"
                f"{error_text}"
            )

            time.sleep(30)

if __name__ == "__main__":
    main()
