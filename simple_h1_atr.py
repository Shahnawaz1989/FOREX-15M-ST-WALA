from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

PAIR = "EURUSD.ecn"
CHECK_DATE = "2025-05-29"
ATR_PERIOD = 14


def _add_atr_rma(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)

    if len(df) == 0:
        df["tr"] = np.nan
        df["atr"] = np.nan
        return df

    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values

    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]

    for i in range(1, len(df)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = np.full(len(df), np.nan)

    if len(df) >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, len(df)):
            atr[i] = atr[i - 1] + (tr[i] - atr[i - 1]) / period

    df["tr"] = tr
    df["atr"] = atr
    return df


def _fetch_ohlc(symbol: str, timeframe, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, timeframe, start_utc, end_utc)
    print(f"copy_rates_range({symbol}, {timeframe}, {start_utc}, {end_utc})")
    print("last_error:", mt5.last_error())

    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(
        df["time"], unit="s", utc=True, errors="coerce")
    df = df[["time", "open", "high", "low", "close"]].dropna(subset=[
                                                             "time"]).copy()
    return df.sort_values("time").reset_index(drop=True)


def _dump_for_timeframe(symbol: str, timeframe_name: str, timeframe_const, expected_bars: int):
    day_start = datetime(2025, 5, 29, 0, 0, 0, tzinfo=timezone.utc)
    day_end = datetime(2025, 5, 30, 0, 0, 0, tzinfo=timezone.utc)

    fetch_start = day_start - timedelta(days=30)
    fetch_end = day_end

    print("\n" + "=" * 110)
    print(f"{symbol} | {timeframe_name} | {CHECK_DATE}")
    print("=" * 110)

    df = _fetch_ohlc(symbol, timeframe_const, fetch_start, fetch_end)
    if df.empty:
        print(f"{timeframe_name}: no OHLC fetched")
        return

    df = _add_atr_rma(df, ATR_PERIOD)

    out = df[(df["time"] >= day_start) & (df["time"] < day_end)].copy()

    print(f"Bars on target day: {len(out)} | expected approx: {expected_bars}")
    if out.empty:
        print("No target-day bars found")
        return

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print(out[["time", "open", "high", "low", "close",
          "tr", "atr"]].to_string(index=False))

    print("\nMATCH THESE ATR VALUES WITH MT5 CHART ATR(14)")
    print("Tip: MT5 Data Window me same candle par ATR value dekho.")


if __name__ == "__main__":
    ok = mt5.initialize()
    print("initialize:", ok, "| last_error:", mt5.last_error())
    if not ok:
        raise SystemExit

    try:
        selected = mt5.symbol_select(PAIR, True)
        print("symbol_select:", selected, "| last_error:", mt5.last_error())

        info = mt5.symbol_info(PAIR)
        print("symbol_info exists:", info is not None)

        _dump_for_timeframe(PAIR, "H1", mt5.TIMEFRAME_H1, 24)
        _dump_for_timeframe(PAIR, "M15", mt5.TIMEFRAME_M15, 96)

    finally:
        mt5.shutdown()
