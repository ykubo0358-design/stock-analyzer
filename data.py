import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"ティッカー '{ticker}' のデータが取得できませんでした。")
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def get_stock_info(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap"),
        "currency": info.get("currency", "USD"),
        "exchange": info.get("exchange", "N/A"),
    }
