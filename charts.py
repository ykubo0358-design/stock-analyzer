import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


SIGNAL_COLORS = {
    "強い買い": "#00cc44",
    "買い": "#66ff99",
    "強い売り": "#ff1a1a",
    "売り": "#ff9999",
}


def make_main_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.17, 0.17, 0.16],
        vertical_spacing=0.03,
        subplot_titles=("株価 + ボリンジャーバンド + 移動平均", "MACD", "RSI", "出来高"),
    )

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="株価", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # ボリンジャーバンド
    if "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], line=dict(color="rgba(100,100,255,0.4)", width=1), name="BB上限"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], line=dict(color="rgba(100,100,255,0.4)", width=1), fill="tonexty", fillcolor="rgba(100,100,255,0.05)", name="BB下限"), row=1, col=1)

    # 移動平均
    for col, color in [("SMA20", "#ffa500"), ("SMA50", "#1e90ff"), ("SMA200", "#ff69b4")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(color=color, width=1.2), name=col), row=1, col=1)

    # 売買シグナルマーカー
    if "signal" in df.columns:
        for sig, color in SIGNAL_COLORS.items():
            mask = df["signal"] == sig
            symbol = "triangle-up" if "買い" in sig else "triangle-down"
            y_pos = df.loc[mask, "Low"] * 0.99 if "買い" in sig else df.loc[mask, "High"] * 1.01
            fig.add_trace(go.Scatter(
                x=df.index[mask], y=y_pos,
                mode="markers",
                marker=dict(symbol=symbol, size=10, color=color),
                name=sig,
            ), row=1, col=1)

    # MACD
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#2196f3", width=1.5), name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], line=dict(color="#ff9800", width=1.5), name="シグナル"), row=2, col=1)
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], marker_color=colors, name="ヒスト"), row=2, col=1)

    # RSI
    if "RSI14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], line=dict(color="#9c27b0", width=1.5), name="RSI14"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # 出来高
    vol_colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="出来高"), row=4, col=1)
    if "Volume_SMA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Volume_SMA20"], line=dict(color="#ffa500", width=1), name="出来高MA20"), row=4, col=1)

    fig.update_layout(
        height=850,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=20),
    )
    return fig


def make_signal_score_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "signal_score" not in df.columns:
        return fig
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["signal_score"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["signal_score"], marker_color=colors, name="シグナルスコア"))
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", annotation_text="買いゾーン")
    fig.add_hline(y=-30, line_dash="dash", line_color="#ef5350", annotation_text="売りゾーン")
    fig.update_layout(
        height=250, template="plotly_dark",
        title="シグナルスコア（+100=強い買い、-100=強い売り）",
        margin=dict(l=40, r=40, t=50, b=20),
    )
    return fig
