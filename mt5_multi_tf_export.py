from pathlib import Path
from datetime import datetime
from typing import Dict, List

import MetaTrader5 as mt5
import pandas as pd
import pytz


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "mt5_exports"
UTC = pytz.UTC

TIMEFRAME_MAP: Dict[str, int] = {
    "1m": mt5.TIMEFRAME_M1,
    "2m": mt5.TIMEFRAME_M2,
    "3m": mt5.TIMEFRAME_M3,
    "4m": mt5.TIMEFRAME_M4,
    "5m": mt5.TIMEFRAME_M5,
    "6m": mt5.TIMEFRAME_M6,
    "10m": mt5.TIMEFRAME_M10,
    "12m": mt5.TIMEFRAME_M12,
    "15m": mt5.TIMEFRAME_M15,
    "20m": mt5.TIMEFRAME_M20,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "2h": mt5.TIMEFRAME_H2,
    "3h": mt5.TIMEFRAME_H3,
    "4h": mt5.TIMEFRAME_H4,
    "6h": mt5.TIMEFRAME_H6,
    "8h": mt5.TIMEFRAME_H8,
    "12h": mt5.TIMEFRAME_H12,
    "1d": mt5.TIMEFRAME_D1,
    "1w": mt5.TIMEFRAME_W1,
    "1mn": mt5.TIMEFRAME_MN1,
}


def ensure_mt5() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")


def shutdown_mt5() -> None:
    try:
        mt5.shutdown()
    except Exception:
        pass


def parse_date_to_utc(date_str: str, end_of_day: bool = False) -> datetime:
    date_str = date_str.strip()
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    else:
        dt = dt.replace(hour=0, minute=0, second=0)
    return UTC.localize(dt)


def fetch_rates(symbol: str, timeframe_label: str, start_date: str, end_date: str) -> pd.DataFrame:
    tf_key = timeframe_label.strip().lower()

    if tf_key not in TIMEFRAME_MAP:
        raise ValueError(
            f"Unsupported timeframe '{timeframe_label}'. Supported values: {list(TIMEFRAME_MAP.keys())}"
        )

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Failed to select symbol: {symbol}")

    utc_from = parse_date_to_utc(start_date, end_of_day=False)
    utc_to = parse_date_to_utc(end_date, end_of_day=True)

    rates = mt5.copy_rates_range(
        symbol, TIMEFRAME_MAP[tf_key], utc_from, utc_to)

    if rates is None:
        raise RuntimeError(
            f"copy_rates_range failed for {symbol} {timeframe_label}: {mt5.last_error()}"
        )

    df = pd.DataFrame(rates)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
                "spread",
                "real_volume",
            ]
        )

    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[
        ["datetime", "open", "high", "low", "close",
            "tick_volume", "spread", "real_volume"]
    ].copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def save_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def export_symbol_timeframes(
    symbol: str,
    timeframes: List[str],
    start_date: str,
    end_date: str,
    output_dir: Path = OUTPUT_DIR,
    make_combined_csv: bool = True,
) -> None:
    symbol_safe = symbol.replace(".", "_")
    symbol_dir = output_dir / symbol_safe
    combined_frames: List[pd.DataFrame] = []

    print("\n" + "=" * 70)
    print(f"Exporting symbol={symbol} | range={start_date} -> {end_date}")

    for timeframe_label in timeframes:
        tf_key = timeframe_label.strip().lower()
        print(f"  -> Fetching {symbol} {tf_key}")

        df = fetch_rates(symbol, tf_key, start_date, end_date)

        if df.empty:
            print(f"     No data returned for {symbol} {tf_key}")
        else:
            print(
                f"     rows={len(df)} | "
                f"min={df['datetime'].min()} | "
                f"max={df['datetime'].max()}"
            )

        file_name = f"{symbol_safe}_{tf_key.upper()}.csv"
        save_path = symbol_dir / file_name
        save_dataframe_csv(df, save_path)
        print(f"     Saved -> {save_path}")

        if make_combined_csv and not df.empty:
            temp = df.copy()
            temp["symbol"] = symbol
            temp["timeframe"] = tf_key.upper()
            combined_frames.append(temp)

    if make_combined_csv:
        if combined_frames:
            combined_df = pd.concat(combined_frames, ignore_index=True)
        else:
            combined_df = pd.DataFrame(
                columns=[
                    "datetime",
                    "open",
                    "high",
                    "low",
                    "close",
                    "tick_volume",
                    "spread",
                    "real_volume",
                    "symbol",
                    "timeframe",
                ]
            )

        combined_path = symbol_dir / f"{symbol_safe}_COMBINED.csv"
        save_dataframe_csv(combined_df, combined_path)
        print(f"  -> Combined saved -> {combined_path}")


def export_multiple_symbols(
    symbols: List[str],
    timeframes: List[str],
    start_date: str,
    end_date: str,
    output_dir: Path = OUTPUT_DIR,
    make_combined_csv: bool = True,
) -> None:
    ensure_mt5()
    try:
        for symbol in symbols:
            export_symbol_timeframes(
                symbol=symbol,
                timeframes=timeframes,
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
                make_combined_csv=make_combined_csv,
            )
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    symbols = ["EURUSD.ecn"]
    timeframes = ["3m"]
    start_date = "2025-06-01"
    end_date = "2026-06-08"

    export_multiple_symbols(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        output_dir=OUTPUT_DIR,
        make_combined_csv=True,
    )
