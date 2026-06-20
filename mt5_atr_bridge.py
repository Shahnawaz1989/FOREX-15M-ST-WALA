# mt5_atr_bridge.py

import os
import time
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd


def _get_common_dir() -> str:
    return r"C:\Users\Uzair Khan\AppData\Roaming\MetaQuotes\Terminal\Common\Files"


REQUEST_FILE = "atr_request.txt"
DONE_FILE = "atr_done.txt"
ERROR_FILE = "atr_error.txt"
SUBFOLDER = os.path.join("ATRBridge")  # inside Common\Files


def _ensure_dirs():
    base = _get_common_dir()
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, SUBFOLDER), exist_ok=True)


def _write_request(run_id: str, day: datetime, symbols: List[str], atr_period: int = 14):
    _ensure_dirs()
    base = _get_common_dir()
    path = os.path.join(base, REQUEST_FILE)

    day_str = day.strftime("%Y.%m.%d")
    symbols_csv = ",".join(symbols)

    content = (
        f"run_id={run_id}\n"
        f"day={day_str}\n"
        f"atr_period={atr_period}\n"
        f"symbols={symbols_csv}\n"
    )

    print(f"[ATR BRIDGE] writing request -> {path}")
    print("[ATR BRIDGE] request content:")
    print(content)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _wait_for_done(run_id: str, timeout_sec: int = 30) -> None:
    base = _get_common_dir()
    done_path = os.path.join(base, DONE_FILE)
    err_path = os.path.join(base, ERROR_FILE)

    print(f"[ATR BRIDGE] waiting for done file in: {base}")
    print(f"[ATR BRIDGE] expecting run_id: {run_id}")

    start = time.time()

    while True:
        elapsed = time.time() - start

        if elapsed > timeout_sec:
            raise TimeoutError(
                f"ATR bridge timeout after {timeout_sec}s | "
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

            print(f"[ATR BRIDGE] error file found: {err_txt}")
            if err_txt:
                raise RuntimeError(f"ATR bridge error: {err_txt}")

        if os.path.exists(done_path):
            try:
                with open(done_path, "r", encoding="utf-8") as f:
                    done_txt = f.read().strip()
            except PermissionError:
                time.sleep(0.2)
                continue

            print(f"[ATR BRIDGE] done file content:\n{done_txt}")

            kv = {}
            for line in done_txt.splitlines():
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    kv[k.strip()] = v.strip()

            if kv.get("status") == "done" and kv.get("run_id") == run_id:
                print("[ATR BRIDGE] matching done file received")
                return

        time.sleep(0.5)


def _load_single_csv(run_id: str, symbol: str, tf_name: str) -> pd.DataFrame:
    base = _get_common_dir()
    fname = f"{run_id}_{symbol}_{tf_name}_ATR.csv"
    path = os.path.join(base, SUBFOLDER, fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"ATR CSV not found: {path}")

    # Try default
    df = pd.read_csv(path)

    # Retry with semicolon if header collapsed (non-tab)
    if len(df.columns) == 1 and ("\t" not in str(df.columns[0])):
        df = pd.read_csv(path, sep=";")

    # Retry with tab if header is tab-joined
    if len(df.columns) == 1 and "\t" in str(df.columns[0]):
        df = pd.read_csv(path, sep="\t")

    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    print(
        f"[MT5 ATR BRIDGE] loaded columns for {symbol} {tf_name}: {df.columns.tolist()}")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    elif "datetime" in df.columns:
        df["time"] = pd.to_datetime(df["datetime"], errors="coerce")

    if "atr_mt5" in df.columns and "atr" not in df.columns:
        df["atr"] = pd.to_numeric(df["atr_mt5"], errors="coerce")

    for col in ["open", "high", "low", "close", "atr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["time", "atr"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {path} | actual={df.columns.tolist()}"
        )

    keep_cols = [c for c in ["time", "open", "high",
                             "low", "close", "atr"] if c in df.columns]
    df = (
        df[keep_cols]
        .dropna(subset=["time", "atr"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    return df


def fetch_mt5_h1_m15_atr_for_pairs(
    day: datetime,
    symbols: List[str],
    atr_period: int = 14,
    timeout_sec: int = 30,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    High-level bridge:
    - Python request -> ATRBridgeEA -> CSVs -> Python load
    Returns: dict[symbol] = {"h1": df_h1, "m15": df_m15}
    """
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    _write_request(run_id=run_id, day=day,
                   symbols=symbols, atr_period=atr_period)
    _wait_for_done(run_id=run_id, timeout_sec=timeout_sec)

    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        symbol_data: Dict[str, pd.DataFrame] = {}
        try:
            df_h1 = _load_single_csv(run_id, symbol, "H1")
            symbol_data["h1"] = df_h1
        except Exception as e:
            print(f"[MT5 ATR BRIDGE] failed for {symbol} H1: {e}")

        try:
            df_m15 = _load_single_csv(run_id, symbol, "M15")
            symbol_data["m15"] = df_m15
        except Exception as e:
            print(f"[MT5 ATR BRIDGE] failed for {symbol} M15: {e}")

        out[symbol] = symbol_data

    return out


def fetch_mt5_h1_m15_atr(
    symbol: str,
    day: datetime,
    atr_period: int = 14,
    timeout_sec: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience wrapper for single pair.
    Returns: {"h1": df_h1, "m15": df_m15}
    """
    res = fetch_mt5_h1_m15_atr_for_pairs(
        day=day,
        symbols=[symbol],
        atr_period=atr_period,
        timeout_sec=timeout_sec,
    )
    return res.get(symbol, {})
