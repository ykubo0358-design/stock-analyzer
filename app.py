import streamlit as st
import pandas as pd
import numpy as np

from data import fetch_stock_data, get_stock_info
from indicators import add_indicators, generate_signals
from ml_model import train_model, predict_latest
from charts import make_main_chart, make_signal_score_chart

st.set_page_config(
    page_title="株価分析ツール",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 4px 0;
}
.signal-strong-buy  { color: #00cc44; font-size: 1.5rem; font-weight: bold; }
.signal-buy         { color: #66ff99; font-size: 1.5rem; font-weight: bold; }
.signal-sell        { color: #ff9999; font-size: 1.5rem; font-weight: bold; }
.signal-strong-sell { color: #ff1a1a; font-size: 1.5rem; font-weight: bold; }
.signal-hold        { color: #aaaaaa; font-size: 1.5rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─── サイドバー ───────────────────────────────────────────────────
with st.sidebar:
    st.title("設定")
    ticker = st.text_input("ティッカーシンボル", value="AAPL", help="例: AAPL, 7203.T, BTC-USD").upper().strip()
    period = st.selectbox("期間", ["6mo", "1y", "2y", "5y"], index=1,
                          format_func=lambda x: {"6mo": "6ヶ月", "1y": "1年", "2y": "2年", "5y": "5年"}[x])
    interval = st.selectbox("足種", ["1d", "1wk"], index=0,
                            format_func=lambda x: {"1d": "日足", "1wk": "週足"}[x])
    use_ml = st.toggle("MLシグナルを使う（ランダムフォレスト）", value=True)
    analyze = st.button("分析する", type="primary", use_container_width=True)

st.title("株価分析ツール")
st.caption("テクニカル指標 + 機械学習による売買シグナル")

if not analyze:
    st.info("左のサイドバーでティッカーシンボルを入力して「分析する」を押してください。")
    st.markdown("""
    **使い方**
    - 米国株: `AAPL`, `MSFT`, `NVDA`, `TSLA`
    - 日本株: `7203.T`（トヨタ）、`6758.T`（ソニー）
    - ETF: `SPY`, `QQQ`
    - 暗号通貨: `BTC-USD`, `ETH-USD`
    """)
    st.stop()

# ─── データ取得 ───────────────────────────────────────────────────
with st.spinner(f"{ticker} のデータを取得中..."):
    try:
        df_raw = fetch_stock_data(ticker, period=period, interval=interval)
        info = get_stock_info(ticker)
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        st.stop()

df = add_indicators(df_raw)
df = generate_signals(df)

# ─── 企業情報 ──────────────────────────────────────────────────────
col_info1, col_info2, col_info3 = st.columns([3, 1, 1])
with col_info1:
    st.subheader(f"{info['name']}  （{ticker}）")
    st.caption(f"{info['exchange']} | {info['sector']} | {info['industry']}")
with col_info2:
    latest_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    change = latest_close - prev_close
    change_pct = change / prev_close * 100
    st.metric("現在値", f"{latest_close:.2f} {info['currency']}", f"{change:+.2f} ({change_pct:+.2f}%)")
with col_info3:
    if info["market_cap"]:
        mc = info["market_cap"]
        label = f"{mc/1e12:.2f}兆" if mc >= 1e12 else f"{mc/1e8:.0f}億"
        st.metric("時価総額", label)

st.divider()

# ─── 最新シグナル ─────────────────────────────────────────────────
st.subheader("最新シグナル")
col_ta, col_ml = st.columns(2)

latest_signal = df["signal"].iloc[-1]
latest_score = df["signal_score"].iloc[-1]
signal_class = {
    "強い買い": "signal-strong-buy",
    "買い": "signal-buy",
    "売り": "signal-sell",
    "強い売り": "signal-strong-sell",
    "ホールド": "signal-hold",
}.get(latest_signal, "signal-hold")

with col_ta:
    st.markdown("**テクニカル分析シグナル**")
    st.markdown(f'<span class="{signal_class}">{latest_signal}</span>', unsafe_allow_html=True)
    st.caption(f"スコア: {latest_score:+d} / 100")

    # 各指標の状態
    with st.expander("指標詳細"):
        rsi = df["RSI14"].iloc[-1]
        macd_val = df["MACD"].iloc[-1]
        macd_sig = df["MACD_signal"].iloc[-1]
        close = df["Close"].iloc[-1]
        bb_u = df["BB_upper"].iloc[-1]
        bb_l = df["BB_lower"].iloc[-1]
        sma20 = df["SMA20"].iloc[-1]
        sma50 = df["SMA50"].iloc[-1]

        st.markdown(f"- **RSI14**: {rsi:.1f} {'(買われすぎ)' if rsi > 70 else '(売られすぎ)' if rsi < 30 else ''}")
        st.markdown(f"- **MACD**: {'上昇トレンド' if macd_val > macd_sig else '下降トレンド'} ({macd_val:.3f} vs {macd_sig:.3f})")
        st.markdown(f"- **ボリンジャー**: {'下限タッチ(買い)' if close <= bb_l else '上限タッチ(売り)' if close >= bb_u else '中間'}")
        st.markdown(f"- **SMA20 vs SMA50**: {'ゴールデンクロス' if sma20 > sma50 else 'デッドクロス'}")

# ─── ML予測 ───────────────────────────────────────────────────────
with col_ml:
    st.markdown("**機械学習（ランダムフォレスト）予測**")
    if use_ml:
        with st.spinner("学習中..."):
            clf, scaler, features, report = train_model(df)

        if clf is None:
            st.warning("データが少なすぎてMLモデルを学習できません（最低60日分必要）。")
        else:
            result = predict_latest(df, clf, scaler, features)
            ml_signal = result["label"]
            proba = result["proba"]

            ml_class = {
                "買い": "signal-buy", "売り": "signal-sell", "ホールド": "signal-hold"
            }.get(ml_signal, "signal-hold")
            st.markdown(f'<span class="{ml_class}">{ml_signal}</span>', unsafe_allow_html=True)

            with st.expander("確率・精度詳細"):
                for label, p in proba.items():
                    st.progress(p, text=f"{label}: {p*100:.1f}%")
                if report:
                    acc = report.get("accuracy", 0)
                    st.caption(f"バックテスト精度（最終fold）: {acc*100:.1f}%")
    else:
        st.info("MLシグナルはオフです。")

st.divider()

# ─── チャート ─────────────────────────────────────────────────────
st.subheader("チャート")
fig_main = make_main_chart(df)
st.plotly_chart(fig_main, use_container_width=True)

fig_score = make_signal_score_chart(df)
st.plotly_chart(fig_score, use_container_width=True)

# ─── シグナル履歴 ─────────────────────────────────────────────────
st.subheader("シグナル履歴（直近50件）")
signal_mask = df["signal"] != "ホールド"
signal_df = df[signal_mask][["Close", "signal", "signal_score", "RSI14", "MACD"]].tail(50).sort_index(ascending=False)
signal_df.index = signal_df.index.strftime("%Y-%m-%d")
signal_df.columns = ["終値", "シグナル", "スコア", "RSI14", "MACD"]

def color_signal(val):
    colors = {"強い買い": "color: #00cc44", "買い": "color: #66ff99", "売り": "color: #ff9999", "強い売り": "color: #ff1a1a"}
    return colors.get(val, "")

st.dataframe(
    signal_df.style.applymap(color_signal, subset=["シグナル"]).format({
        "終値": "{:.2f}", "スコア": "{:+d}", "RSI14": "{:.1f}", "MACD": "{:.3f}"
    }),
    use_container_width=True,
    height=350,
)

# ─── フッター ─────────────────────────────────────────────────────
st.divider()
st.caption("免責事項: このツールは情報提供のみを目的としており、投資助言ではありません。実際の投資判断はご自身の責任で行ってください。")
