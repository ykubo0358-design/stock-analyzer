import pandas as pd
import numpy as np
import pandas_ta as ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # トレンド系
    d["SMA20"] = ta.sma(d["Close"], length=20)
    d["SMA50"] = ta.sma(d["Close"], length=50)
    d["SMA200"] = ta.sma(d["Close"], length=200)
    d["EMA12"] = ta.ema(d["Close"], length=12)
    d["EMA26"] = ta.ema(d["Close"], length=26)

    # MACD
    macd = ta.macd(d["Close"], fast=12, slow=26, signal=9)
    if macd is not None:
        macd_col  = next((c for c in macd.columns if c.startswith("MACD_")), None)
        macds_col = next((c for c in macd.columns if c.startswith("MACDs_")), None)
        macdh_col = next((c for c in macd.columns if c.startswith("MACDh_")), None)
        if macd_col:  d["MACD"] = macd[macd_col]
        if macds_col: d["MACD_signal"] = macd[macds_col]
        if macdh_col: d["MACD_hist"] = macd[macdh_col]

    # RSI
    d["RSI14"] = ta.rsi(d["Close"], length=14)

    # ボリンジャーバンド
    bb = ta.bbands(d["Close"], length=20, std=2)
    if bb is not None:
        upper_col = next((c for c in bb.columns if c.startswith("BBU_")), None)
        mid_col   = next((c for c in bb.columns if c.startswith("BBM_")), None)
        lower_col = next((c for c in bb.columns if c.startswith("BBL_")), None)
        if upper_col: d["BB_upper"] = bb[upper_col]
        if mid_col:   d["BB_mid"]   = bb[mid_col]
        if lower_col: d["BB_lower"] = bb[lower_col]

    # ストキャスティクス
    stoch = ta.stoch(d["High"], d["Low"], d["Close"], k=14, d=3)
    if stoch is not None:
        k_col = next((c for c in stoch.columns if c.startswith("STOCHk_")), None)
        d_col = next((c for c in stoch.columns if c.startswith("STOCHd_")), None)
        if k_col: d["STOCH_K"] = stoch[k_col]
        if d_col: d["STOCH_D"] = stoch[d_col]

    # ATR（ボラティリティ）
    d["ATR14"] = ta.atr(d["High"], d["Low"], d["Close"], length=14)

    # 出来高移動平均
    d["Volume_SMA20"] = ta.sma(d["Volume"], length=20)

    return d


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["signal_score"] = 0  # -100〜+100のスコア

    # --- 買いシグナル要因 ---
    # SMAゴールデンクロス
    if "SMA20" in d.columns and "SMA50" in d.columns:
        golden = (d["SMA20"] > d["SMA50"]) & (d["SMA20"].shift(1) <= d["SMA50"].shift(1))
        d.loc[golden, "signal_score"] += 30

    # RSI 売られすぎ回復（30以下から上昇）
    if "RSI14" in d.columns:
        rsi_buy = (d["RSI14"] > 30) & (d["RSI14"].shift(1) <= 30)
        rsi_sell = (d["RSI14"] < 70) & (d["RSI14"].shift(1) >= 70)
        d.loc[rsi_buy, "signal_score"] += 25
        d.loc[rsi_sell, "signal_score"] -= 25

    # MACDゴールデンクロス
    if "MACD" in d.columns and "MACD_signal" in d.columns:
        macd_buy = (d["MACD"] > d["MACD_signal"]) & (d["MACD"].shift(1) <= d["MACD_signal"].shift(1))
        macd_sell = (d["MACD"] < d["MACD_signal"]) & (d["MACD"].shift(1) >= d["MACD_signal"].shift(1))
        d.loc[macd_buy, "signal_score"] += 25
        d.loc[macd_sell, "signal_score"] -= 25

    # ボリンジャーバンド タッチ
    if "BB_lower" in d.columns and "BB_upper" in d.columns:
        bb_buy = d["Close"] <= d["BB_lower"]
        bb_sell = d["Close"] >= d["BB_upper"]
        d.loc[bb_buy, "signal_score"] += 15
        d.loc[bb_sell, "signal_score"] -= 15

    # ストキャスティクス
    if "STOCH_K" in d.columns:
        stoch_buy = (d["STOCH_K"] > d["STOCH_D"]) & (d["STOCH_K"].shift(1) <= d["STOCH_D"].shift(1)) & (d["STOCH_K"] < 20)
        stoch_sell = (d["STOCH_K"] < d["STOCH_D"]) & (d["STOCH_K"].shift(1) >= d["STOCH_D"].shift(1)) & (d["STOCH_K"] > 80)
        d.loc[stoch_buy, "signal_score"] += 15
        d.loc[stoch_sell, "signal_score"] -= 15

    # 出来高急増（価格上昇を伴う）
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
