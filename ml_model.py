import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report


FEATURE_COLS = [
    "RSI14", "MACD", "MACD_signal", "MACD_hist",
    "STOCH_K", "STOCH_D", "ATR14",
    "BB_upper", "BB_lower", "BB_mid",
    "SMA20", "SMA50",
    "Volume", "Volume_SMA20",
]

LOOKAHEAD = 5  # 5日後の騰落でラベル付け


def make_labels(df: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
    future_return = df["Close"].shift(-LOOKAHEAD) / df["Close"] - 1
    labels = pd.Series("ホールド", index=df.index)
    labels[future_return >= threshold] = "買い"
    labels[future_return <= -threshold] = "売り"
    return labels


def train_model(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = make_labels(df)

    # 欠損除去（先頭と末尾のlookahead分）
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid]

    if len(X) < 60:
        return None, None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )

    # 時系列クロスバリデーション（最後のfoldで評価）
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X_scaled))
    train_idx, test_idx = splits[-1]
    clf.fit(X_scaled[train_idx], y.iloc[train_idx])
    report = classification_report(
        y.iloc[test_idx],
        clf.predict(X_scaled[test_idx]),
        output_dict=True,
        zero_division=0,
    )

    # 全データで再学習
    clf.fit(X_scaled, y)
    return clf, scaler, available, report


def predict_latest(df: pd.DataFrame, clf, scaler, features: list) -> dict:
    row = df[features].iloc[-1:].copy()
    if row.isna().any(axis=1).values[0]:
        return {"label": "データ不足", "proba": {}}
    scaled = scaler.transform(row)
    label = clf.predict(scaled)[0]
    proba = dict(zip(clf.classes_, clf.predict_proba(scaled)[0]))
    return {"label": label, "proba": proba}
