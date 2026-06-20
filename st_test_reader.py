import pandas as pd
from pathlib import Path

CSV_PATH = Path(
    r"C:\Users\Uzair Khan\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\st_test_report.csv")

PAIR = "EURUSD.ecn"
START = "2026-06-17 12:30"
END = "2026-06-17 14:30"


def main():
    if not CSV_PATH.exists():
        print("CSV not found:", CSV_PATH)
        return

    df = pd.read_csv(CSV_PATH, sep=r"\s{2,}|\t+|,", engine="python")
    df.columns = df.columns.str.strip()

    print("COLUMNS:", df.columns.tolist())

    if df.empty:
        print("CSV empty")
        return

    required = ["symbol", "timeframe", "bar_time", "close",
                "st_line", "buy_buffer", "sell_buffer", "trend", "signal"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("Missing columns:", missing)
        return

    df["bar_time"] = pd.to_datetime(
        df["bar_time"], format="%Y.%m.%d %H:%M", errors="coerce")
    df = df.dropna(subset=["bar_time"])

    start_dt = pd.to_datetime(START)
    end_dt = pd.to_datetime(END)

    out = df[
        (df["symbol"].astype(str).str.strip() == PAIR) &
        (df["bar_time"] >= start_dt) &
        (df["bar_time"] <= end_dt)
    ].copy()

    if out.empty:
        print("No rows found")
        return

    cols = [
        "symbol", "timeframe", "bar_time", "close",
        "st_line", "buy_buffer", "sell_buffer", "trend", "signal"
    ]
    out = out[cols].sort_values("bar_time").reset_index(drop=True)

    print("\nFILTERED REPORT\n")
    print(out.to_string(index=False))

    print("\nSIGNAL ONLY\n")
    sig = out[out["signal"].isin(["BUY", "SELL"])]
    if sig.empty:
        print("No BUY/SELL signal in this range")
    else:
        print(sig.to_string(index=False))


if __name__ == "__main__":
    main()
