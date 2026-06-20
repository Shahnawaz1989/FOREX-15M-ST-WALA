import MetaTrader5 as mt5
import pandas as pd


def _ensure_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")


def fetch_mt5_m15_atr(pair: str, lookback_days: int = 30, atr_period: int = 14) -> pd.DataFrame:
    _ensure_mt5()

    rates = mt5.copy_rates_from_pos(
        pair, mt5.TIMEFRAME_M15, 0, lookback_days * 96 + 300)
    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=["time", "atr"])

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)

    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values

    tr = [high[0] - low[0]]
    for i in range(1, len(df)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr.append(max(hl, hc, lc))

    tr = pd.Series(tr, dtype="float64")
    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

    df["atr"] = atr
    return df[["time", "atr"]].copy()


def fetch_mt5_h1_atr(pair: str, lookback_days: int = 30, atr_period: int = 14) -> pd.DataFrame:
    _ensure_mt5()

    rates = mt5.copy_rates_from_pos(
        pair, mt5.TIMEFRAME_H1, 0, lookback_days * 24 + 300)
    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=["time", "atr"])

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)

    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values

    tr = [high[0] - low[0]]
    for i in range(1, len(df)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr.append(max(hl, hc, lc))

    tr = pd.Series(tr, dtype="float64")
    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

    df["atr"] = atr
    return df[["time", "atr"]].copy()


def attach_mt5_m15_atr(df15m: pd.DataFrame, atr15m_df: pd.DataFrame) -> pd.DataFrame:
    if df15m is None or df15m.empty:
        return df15m

    x = df15m.copy()
    a = atr15m_df.copy()

    x["time"] = pd.to_datetime(x["time"])
    a["time"] = pd.to_datetime(a["time"])

    x = x.merge(a, on="time", how="left", suffixes=("", "_mt5"))
    if "atr_mt5" in x.columns:
        x["atr"] = x["atr_mt5"]
        x.drop(columns=["atr_mt5"], inplace=True)

    return x
