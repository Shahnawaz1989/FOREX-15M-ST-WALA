# st_test_reader.py
import os
import time
from datetime import datetime
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


def enrich_with_mt5_st(
    candles_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    day: datetime,
    periods: int = ST_PERIODS,
    multiplier: float = ST_MULTIPLIER,
    bars_to_export: int = ST_BARS_TO_EXPORT,
    timeout_sec: int = ST_TIMEOUT_SEC,
) -> pd.DataFrame:

    def ensure_dirs():
        base = _get_common_dir()
        os.makedirs(base, exist_ok=True)
        os.makedirs(os.path.join(base, ST_SUBFOLDER), exist_ok=True)

    def write_request(run_id: str):
        ensure_dirs()
        base = _get_common_dir()
        path = os.path.join(base, ST_REQUEST_FILE)
        done_path = os.path.join(base, ST_DONE_FILE)
        err_path = os.path.join(base, ST_ERROR_FILE)

        for p in (done_path, err_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                    _st_log(f"[ST BRIDGE] removed stale file: {p}")
                except Exception as e:
                    _st_log(
                        f"[ST BRIDGE] could not remove stale file {p}: {e}")

        content = (
            f"run_id={run_id}\n"
            f"day={day.strftime('%Y.%m.%d')}\n"
            f"symbols={symbol}\n"
            f"timeframes={timeframe}\n"
            f"periods={periods}\n"
            f"multiplier={multiplier}\n"
            f"source={ST_SOURCE}\n"
            f"change_atr={str(ST_CHANGE_ATR).lower()}\n"
            f"show_signals={str(ST_SHOW_SIGNALS).lower()}\n"
            f"enable_alerts={str(ST_ENABLE_ALERTS).lower()}\n"
            f"bars_to_export={bars_to_export}\n"
        )

        print(
            f"[ST BRIDGE] request sent | run_id={run_id} "
            f"| symbol={symbol} | tf={timeframe} | day={day.strftime('%Y.%m.%d')}"
        )
        _st_log("[ST BRIDGE] request content:\n" + content)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def wait_done(run_id: str):
        base = _get_common_dir()
        done_path = os.path.join(base, ST_DONE_FILE)
        err_path = os.path.join(base, ST_ERROR_FILE)

        _st_log(f"[ST BRIDGE] waiting for done file in: {base}")
        _st_log(f"[ST BRIDGE] expecting run_id: {run_id}")

        start = time.time()
        while True:
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"ST bridge timeout | run_id={run_id}")

            if os.path.exists(err_path):
                try:
                    with open(err_path, "r", encoding="utf-8") as f:
                        txt = f.read().strip()
                except PermissionError:
                    time.sleep(0.2)
                    continue

                print(f"[ST BRIDGE] error file found: {txt}")
                if txt:
                    raise RuntimeError(f"ST bridge error: {txt}")

            if os.path.exists(done_path):
                try:
                    with open(done_path, "r", encoding="utf-8") as f:
                        done_txt = f.read().strip()
                except PermissionError:
                    time.sleep(0.2)
                    continue

                _st_log("[ST BRIDGE] done file content:\n" + done_txt)

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

    def load_csv(run_id: str) -> pd.DataFrame:
        base = _get_common_dir()
        fname = f"{run_id}_{symbol}_{timeframe}_ST.csv"
        folder_path = os.path.join(base, ST_SUBFOLDER)
        path = os.path.join(folder_path, fname)

        _st_log(f"[ST BRIDGE] common dir  = {base}")
        _st_log(f"[ST BRIDGE] ST folder   = {folder_path}")
        _st_log(f"[ST BRIDGE] expected csv= {path}")

        if not os.path.exists(path):
            if os.path.exists(folder_path):
                try:
                    _st_log("[ST BRIDGE] ST folder files = " +
                            str(os.listdir(folder_path)))
                except Exception as e:
                    _st_log(f"[ST BRIDGE] failed to list ST folder files: {e}")
            raise FileNotFoundError(f"ST CSV not found: {path}")

        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(path, sep="\t")

        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

        if "time" not in df.columns:
            raise ValueError(
                f"'time' column missing in ST CSV: {df.columns.tolist()}")

        df["time"] = pd.to_datetime(df["time"], errors="coerce")

        for col in ["open", "high", "low", "close", "st_line", "buy_buffer", "sell_buffer"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ["trend", "signal"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()

        df = (
            df.dropna(subset=["time"])
            .sort_values("time")
            .reset_index(drop=True)
        )

        print(
            f"[ST BRIDGE] csv loaded | symbol={symbol} | tf={timeframe} "
            f"| rows={len(df)} | cols={list(df.columns)}"
        )
        return df

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    write_request(run_id)
    wait_done(run_id)
    st_df = load_csv(run_id)

    print("[ST BRIDGE] request day =", day.strftime("%Y.%m.%d"))
    print("[ST BRIDGE] bars_to_export =", bars_to_export)
    print("[ST BRIDGE] csv min_time =", st_df["time"].min(),
          "| max_time =", st_df["time"].max())
    print("[ST BRIDGE] candle min_time =", pd.to_datetime(candles_df["time"], errors="coerce").min(),
          "| candle max_time =", pd.to_datetime(candles_df["time"], errors="coerce").max())

    x = candles_df.copy()
    x["time"] = pd.to_datetime(x["time"], errors="coerce").dt.floor("min")
    st_df["time"] = pd.to_datetime(
        st_df["time"], errors="coerce").dt.floor("min")

    cols = ["time", "st_line", "buy_buffer", "sell_buffer", "trend", "signal"]
    cols = [c for c in cols if c in st_df.columns]

    print("[ST-ENRICH] merging cols from ST:", cols)
    x = x.merge(st_df[cols], on="time", how="left")

    x["supertrend_direction"] = x["trend"] if "trend" in x.columns else None
    x["supertrend_signal"] = x["signal"] if "signal" in x.columns else None

    nn_dir = int(x["supertrend_direction"].notna().sum()
                 ) if "supertrend_direction" in x.columns else 0
    nn_sig = int(x["supertrend_signal"].notna().sum()
                 ) if "supertrend_signal" in x.columns else 0

    print(
        f"[ST-ENRICH] result | symbol={symbol} | tf={timeframe} "
        f"| nn_dir={nn_dir} | nn_sig={nn_sig}"
    )

    if DEBUG_ST and "supertrend_direction" in x.columns and "supertrend_signal" in x.columns:
        _st_log("[ST-ENRICH] tail preview (time, close, dir, sig):")
        _st_log(
            x[["time", "close", "supertrend_direction", "supertrend_signal"]]
            .tail(15)
            .to_string(index=False)
        )

    if nn_dir == 0 or nn_sig == 0:
        raise ValueError(
            f"ST merge produced empty signals | nn_dir={nn_dir} | nn_sig={nn_sig}"
        )

    return x
