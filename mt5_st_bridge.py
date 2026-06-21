# mt5_st_bridge.py
import os
import time
from datetime import datetime
from typing import List, Dict

import pandas as pd

from st_config import (
    DEBUG_ST,
    COMMON_DIR,
    ST_REQUEST_FILE,
    ST_DONE_FILE,
    ST_ERROR_FILE,
    ST_SUBFOLDER,
    ST_PERIODS,
    ST_MULTIPLIER,
    ST_BARS_TO_EXPORT,
    ST_TIMEOUT_SEC,
    ST_DEFAULT_TIMEFRAMES,
    ST_SOURCE,
    ST_CHANGE_ATR,
    ST_SHOW_SIGNALS,
    ST_ENABLE_ALERTS,
)


def _st_log(*args):
    if DEBUG_ST:
        print(*args)


def _get_common_dir() -> str:
    return COMMON_DIR


def _ensure_st_dirs():
    base = _get_common_dir()
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, ST_SUBFOLDER), exist_ok=True)


def _write_st_request(
    run_id: str,
    day: datetime,
    symbols: List[str],
    timeframes: List[str],
    periods: int = ST_PERIODS,
    multiplier: float = ST_MULTIPLIER,
    bars_to_export: int = ST_BARS_TO_EXPORT,
):
    _ensure_st_dirs()
    base = _get_common_dir()
    path = os.path.join(base, ST_REQUEST_FILE)

    day_str = day.strftime("%Y.%m.%d")
    symbols_csv = ",".join(symbols)
    tfs_csv = ",".join(timeframes)

    content = (
        f"run_id={run_id}\n"
        f"day={day_str}\n"
        f"symbols={symbols_csv}\n"
        f"timeframes={tfs_csv}\n"
        f"periods={periods}\n"
        f"multiplier={multiplier}\n"
        f"source={ST_SOURCE}\n"
        f"change_atr={str(ST_CHANGE_ATR).lower()}\n"
        f"show_signals={str(ST_SHOW_SIGNALS).lower()}\n"
        f"enable_alerts={str(ST_ENABLE_ALERTS).lower()}\n"
        f"bars_to_export={bars_to_export}\n"
    )

    print(
        f"[ST BRIDGE] request sent | run_id={run_id} | day={day_str} "
        f"| symbols={symbols_csv} | tfs={tfs_csv}"
    )
    _st_log("[ST BRIDGE] request content:\n" + content)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _wait_for_st_done(run_id: str, timeout_sec: int = ST_TIMEOUT_SEC) -> None:
    base = _get_common_dir()
    done_path = os.path.join(base, ST_DONE_FILE)
    err_path = os.path.join(base, ST_ERROR_FILE)

    _st_log(f"[ST BRIDGE] waiting for done file in: {base}")
    _st_log(f"[ST BRIDGE] expecting run_id: {run_id}")

    start = time.time()

    while True:
        elapsed = time.time() - start

        if elapsed > timeout_sec:
            raise TimeoutError(
                f"ST bridge timeout after {timeout_sec}s | "
                f"done_exists={os.path.exists(done_path)} | "
                f"err_exists={os.path.exists(err_path)}"
            )

        if os.path.exists(err_path):
            try:
                with open(err_path, "r", encoding="utf-8") as f:
                    err_txt = f.read().strip()
            except PermissionError:
                time.sleep(0.2)
                continue

            print(f"[ST BRIDGE] error file found: {err_txt}")
            if err_txt:
                raise RuntimeError(f"ST bridge error: {err_txt}")

        if os.path.exists(done_path):
            try:
                with open(done_path, "r", encoding="utf-8") as f:
                    done_txt = f.read().strip()
            except PermissionError:
                time.sleep(0.2)
                continue

            _st_log(f"[ST BRIDGE] done file content:\n{done_txt}")

            kv = {}
            for line in done_txt.splitlines():
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    kv[k.strip()] = v.strip()

            if kv.get("status") == "done" and kv.get("run_id") == run_id:
                print("[ST BRIDGE] matching done file received")
                return

        time.sleep(0.5)


def _load_single_st_csv(run_id: str, symbol: str, tf_name: str) -> pd.DataFrame:
    base = _get_common_dir()
    fname = f"{run_id}_{symbol}_{tf_name}_ST.csv"
    path = os.path.join(base, ST_SUBFOLDER, fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"ST CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    print(
        f"[ST BRIDGE] csv loaded | symbol={symbol} | tf={tf_name} "
        f"| rows={len(df)} | cols={df.columns.tolist()}"
    )

    if "time" not in df.columns:
        if "bar_time" in df.columns:
            df["time"] = pd.to_datetime(df["bar_time"], errors="coerce")
        elif "datetime" in df.columns:
            df["time"] = pd.to_datetime(df["datetime"], errors="coerce")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    for col in ["open", "high", "low", "close", "st_line", "buy_buffer", "sell_buffer"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["trend", "signal"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    required = ["time", "trend", "signal"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required ST columns {missing} in {path} | "
            f"actual={df.columns.tolist()}"
        )

    keep_cols = [
        c for c in [
            "time",
            "open",
            "high",
            "low",
            "close",
            "st_line",
            "buy_buffer",
            "sell_buffer",
            "trend",
            "signal",
        ] if c in df.columns
    ]

    df = (
        df[keep_cols]
        .dropna(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    return df


def fetch_mt5_st_for_pairs(
    day: datetime,
    symbols: List[str],
    timeframes: List[str],
    periods: int = ST_PERIODS,
    multiplier: float = ST_MULTIPLIER,
    bars_to_export: int = ST_BARS_TO_EXPORT,
    timeout_sec: int = ST_TIMEOUT_SEC,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    _write_st_request(
        run_id=run_id,
        day=day,
        symbols=symbols,
        timeframes=timeframes,
        periods=periods,
        multiplier=multiplier,
        bars_to_export=bars_to_export,
    )
    _wait_for_st_done(run_id=run_id, timeout_sec=timeout_sec)

    out: Dict[str, Dict[str, pd.DataFrame]] = {}

    for symbol in symbols:
        symbol_data: Dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            try:
                df_st = _load_single_st_csv(run_id, symbol, tf)
                symbol_data[tf] = df_st
            except Exception as e:
                print(
                    f"[ST BRIDGE] failed | symbol={symbol} | tf={tf} | error={e}")

        out[symbol] = symbol_data

    return out


def fetch_mt5_st(
    symbol: str,
    day: datetime,
    timeframes: List[str] = ST_DEFAULT_TIMEFRAMES,
    periods: int = ST_PERIODS,
    multiplier: float = ST_MULTIPLIER,
    bars_to_export: int = ST_BARS_TO_EXPORT,
    timeout_sec: int = ST_TIMEOUT_SEC,
) -> Dict[str, pd.DataFrame]:
    res = fetch_mt5_st_for_pairs(
        day=day,
        symbols=[symbol],
        timeframes=timeframes,
        periods=periods,
        multiplier=multiplier,
        bars_to_export=bars_to_export,
        timeout_sec=timeout_sec,
    )
    return res.get(symbol, {})
