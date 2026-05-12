import pandas as pd
import numpy as np
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["Close"]
    high = d["High"]
    low = d["Low"]
    volume = d["Volume"]

    # トレンド系
    d["SMA20"] = close.rolling(window=20).mean()
    d["SMA50"] = close.rolling(window=50).mean()
    d["SMA200"] = close.rolling(window=200).mean()
    d["EMA12"] = close.ewm(span=12, adjust=False).mean()
    d["EMA26"] = close.ewm(span=26, adjust=False).mean()

    # MACD
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    d["MACD"] = macd.macd()
    d["MACD_signal"] = macd.macd_signal()
    d["MACD_hist"] = macd.macd_diff()

    # RSI
    d["RSI14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # ボリンジャーバンド
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    d["BB_upper"] = bb.bollinger_hband()
    d["BB_mid"] = bb.bollinger_mavg()
    d["BB_lower"] = bb.bollinger_lband()

    # ストキャスティクス
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    d["STOCH_K"] = stoch.stoch()
    d["STOCH_D"] = stoch.stoch_signal()

    # ATR
    d["ATR14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # 出来高移動平均
    d["Volume_SMA20"] = volume.rolling(window=20).mean()

    return d


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["signal_score"] = 0

    # SMAゴールデンクロス
    if "SMA20" in d.columns and "SMA50" in d.columns:
        golden = (d["SMA20"] > d["SMA50"]) & (d["SMA20"].shift(1) <= d["SMA50"].shift(1))
        d.loc[golden, "signal_score"] += 30

    # RSI
    if "RSI14" in d.columns:
        rsi_buy = (d["RSI14"] > 30) & (d["RSI14"].shift(1) <= 30)
        rsi_sell = (d["RSI14"] < 70) & (d["RSI14"].shift(1) >= 70)
        d.loc[rsi_buy, "signal_score"] += 25
        d.loc[rsi_sell, "signal_score"] -= 25

    # MACDクロス
    if "MACD" in d.columns and "MACD_signal" in d.columns:
        macd_buy = (d["MACD"] > d["MACD_signal"]) & (d["MACD"].shift(1) <= d["MACD_signal"].shift(1))
        macd_sell = (d["MACD"] < d["MACD_signal"]) & (d["MACD"].shift(1) >= d["MACD_signal"].shift(1))
        d.loc[macd_buy, "signal_score"] += 25
        d.loc[macd_sell, "signal_score"] -= 25

    # ボリンジャーバンド
    if "BB_lower" in d.columns and "BB_upper" in d.columns:
        d.loc[d["Close"] <= d["BB_lower"], "signal_score"] += 15
        d.loc[d["Close"] >= d["BB_upper"], "signal_score"] -= 15

    # ストキャスティクス
    if "STOCH_K" in d.columns and "STOCH_D" in d.columns:
        stoch_buy = (d["STOCH_K"] > d["STOCH_D"]) & (d["STOCH_K"].shift(1) <= d["STOCH_D"].shift(1)) & (d["STOCH_K"] < 20)
        stoch_sell = (d["STOCH_K"] < d["STOCH_D"]) & (d["STOCH_K"].shift(1) >= d["STOCH_D"].shift(1)) & (d["STOCH_K"] > 80)
        d.loc[stoch_buy, "signal_score"] += 15
        d.loc[stoch_sell, "signal_score"] -= 15

    # 出来高急増
    if "Volume_SMA20" in d.columns:
        volume_surge = (d["Volume"] > d["Volume_SMA20"] * 1.5) & (d["Close"] > d["Open"])
        d.loc[volume_surge, "signal_score"] += 10

    # シグナル分類
    d["signal"] = "ホールド"
    d.loc[d["signal_score"] >= 30, "signal"] = "強い買い"
    d.loc[(d["signal_score"] >= 10) & (d["signal_score"] < 30), "signal"] = "買い"
    d.loc[d["signal_score"] <= -30, "signal"] = "強い売り"
    d.loc[(d["signal_score"] <= -10) & (d["signal_score"] > -30), "signal"] = "売り"

    return d
