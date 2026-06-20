import os
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd

from backtest_engine_1h_orb import BacktestEngine1HORB
from backtest_orb_setup_builder import build_setup_for_day
from mt5_atr_bridge import fetch_mt5_h1_m15_atr


PAIRS = [
    "AUDCAD.ecn",
    "AUDUSD.ecn",
    "AUDCHF.ecn",
    "CADCHF.ecn",
    "EURAUD.ecn",
    "EURCAD.ecn",
    "EURCHF.ecn",
    "EURUSD.ecn",
    "EURGBP.ecn",
    "GBPAUD.ecn",
    "GBPCAD.ecn",
    "GBPCHF.ecn",
    "GBPUSD.ecn",
    "NZDCAD.ecn",
    "NZDUSD.ecn",
    "NZDCHF.ecn",
    "USDCAD.ecn",
    "USDCHF.ecn",
]

INITIAL_FUND = 30.0
INITIAL_RISK = 8.0
DATA_DIR = "."
START_DATE = "2026-05-01"
END_DATE = "2026-05-31"
TIMEFRAME = mt5.TIMEFRAME_M15
ONLY_PAIRS = None


def pair_to_temp_csv(pair: str) -> str:
    pair_name = pair.replace(".", "_")
    return f"_temp_{pair_name}.csv"


def init_mt5() -> bool:
    if not mt5.initialize():
        print("MT5 initialization failed")
        return False

    terminal = mt5.terminal_info()
    print("MT5 initialized")
    if terminal:
        print(f"Terminal: {terminal.name}")
    return True


def fetch_pair_data(pair: str, start_date: str, end_date: str) -> bool:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=7)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    print(f"\n[{pair}] Fetching data from {start_dt.date()} to {end_dt.date()}")
    rates = mt5.copy_rates_range(pair, TIMEFRAME, start_dt, end_dt)

    if rates is None or len(rates) == 0:
        print(f"  -> No MT5 data returned for {pair}")
        return False

    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = df[["datetime", "open", "high", "low", "close"]].copy()

    output_file = os.path.join(DATA_DIR, pair_to_temp_csv(pair))
    df.to_csv(output_file, index=False)

    print(f"  -> Saved {len(df)} rows to {output_file}")
    print(f"  -> Range: {df['datetime'].min()} to {df['datetime'].max()}")
    return True


def refresh_csv_data(pairs: list[str], start_date: str, end_date: str) -> list[str]:
    updated_pairs = []
    print("\n" + "=" * 70)
    print("REFRESHING CSV DATA FROM MT5")
    print("=" * 70)

    for pair in pairs:
        ok = fetch_pair_data(pair, start_date, end_date)
        if ok:
            updated_pairs.append(pair)

    print("\n" + "=" * 70)
    print(
        f"CSV refresh complete: {len(updated_pairs)}/{len(pairs)} pairs updated")
    print("=" * 70)
    return updated_pairs


def build_specs(data_dir: str, pairs: list[str]) -> list[dict]:
    specs = []
    for pair in pairs:
        filename = pair_to_temp_csv(pair)
        csv_path = os.path.join(data_dir, filename)
        if not os.path.exists(csv_path):
            continue
        specs.append({"pair": pair, "csv": csv_path})
    return specs


def get_day_df(engine, df, day):
    day_df = df[df["time"].dt.date == day].copy()
    if day_df.empty:
        return day_df

    if "atr" not in day_df.columns:
        raise RuntimeError(
            f"M15 ATR column missing in dataframe for {engine.pair} on {day}. "
            f"Strategy currently requires candidate 15M ATR."
        )

    day_df["atr"] = pd.to_numeric(day_df["atr"], errors="coerce").ffill()
    return day_df


def prepare_data_for_engine(engine, specs):
    data_by_pair = {}
    all_dates = set()

    for spec in specs:
        pair = spec["pair"]
        csv_path = spec["csv"]

        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("time").reset_index(drop=True)

        full_df = df.copy()
        filtered_df = df.copy()

        if engine.start_date is not None:
            filtered_df = filtered_df[filtered_df["time"].dt.date >=
                                      engine.start_date]
        if engine.end_date is not None:
            filtered_df = filtered_df[filtered_df["time"].dt.date <=
                                      engine.end_date]

        if filtered_df.empty:
            continue

        data_by_pair[pair] = full_df
        all_dates.update(filtered_df["time"].dt.date.unique())

    return data_by_pair, sorted(all_dates)


def attach_h1_and_m15_atr(engine, pair, df, day):
    bridge_data = fetch_mt5_h1_m15_atr(
        symbol=pair,
        day=datetime.strptime(str(day), "%Y-%m-%d"),
        atr_period=engine.atr_period,
        timeout_sec=30,
    )

    h1_atr = bridge_data.get("h1")
    m15_atr = bridge_data.get("m15")

    if h1_atr is None or h1_atr.empty:
        return None

    engine.h1_atr_df = h1_atr.copy()
    engine.h1_atr_df["time"] = pd.to_datetime(
        engine.h1_atr_df["time"], errors="coerce")
    engine.h1_atr_df["atr"] = pd.to_numeric(
        engine.h1_atr_df["atr"], errors="coerce")
    engine.h1_atr_df = (
        engine.h1_atr_df
        .dropna(subset=["time", "atr"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    xdf = df.copy()
    xdf["time"] = pd.to_datetime(xdf["time"], errors="coerce")

    if m15_atr is not None and not m15_atr.empty:
        m15 = m15_atr.copy()
        m15["time"] = pd.to_datetime(m15["time"], errors="coerce")
        m15["atr"] = pd.to_numeric(m15["atr"], errors="coerce")
        xdf = xdf.merge(m15[["time", "atr"]], on="time", how="left")

    return xdf


def main():
    pairs = PAIRS
    if ONLY_PAIRS:
        wanted = {p.strip() for p in ONLY_PAIRS}
        pairs = [p for p in PAIRS if p in wanted]

    if not init_mt5():
        raise RuntimeError("MT5 not initialized. Open MT5 and login first.")

    try:
        updated_pairs = refresh_csv_data(pairs, START_DATE, END_DATE)
        if not updated_pairs:
            raise RuntimeError(
                "No pair data fetched from MT5 for selected date range.")

        engine = BacktestEngine1HORB(
            initial_fund=INITIAL_FUND,
            initial_risk_percent=INITIAL_RISK,
            pair=updated_pairs[0],
        )
        engine.start_date = datetime.strptime(START_DATE, "%Y-%m-%d").date()
        engine.end_date = datetime.strptime(END_DATE, "%Y-%m-%d").date()

        specs = build_specs(DATA_DIR, updated_pairs)
        if not specs:
            raise RuntimeError(
                "No matching CSV files found for selected pairs.")

        data_by_pair, all_dates = prepare_data_for_engine(engine, specs)

        total_early = 0
        records = []

        print("\n" + "=" * 70)
        print("COUNTING TP_EARLY SETUPS")
        print(f"Date range: {engine.start_date} -> {engine.end_date}")
        print(f"Pairs loaded: {len(specs)}")
        print("=" * 70)

        for day in all_dates:
            for pair, df in data_by_pair.items():
                engine.pair = pair
                try:
                    xdf = attach_h1_and_m15_atr(engine, pair, df, day)
                    if xdf is None:
                        print(f"[SKIP] {day} | {pair} | no H1 ATR")
                        continue

                    day_df = get_day_df(engine, xdf, day)
                    if day_df.empty:
                        continue

                    if not engine.validate_day(day_df):
                        continue

                    setup = build_setup_for_day(
                        engine=engine,
                        day_df=day_df,
                        fund=INITIAL_FUND,
                        risk_percent=INITIAL_RISK,
                        verbose=False,
                        hh_debug=False,
                    )

                    if not setup:
                        continue

                    if str(setup.get("tp_mode", "")).strip().upper() == "TP_EARLY":
                        total_early += 1
                        records.append({
                            "date": str(day),
                            "pair": pair,
                            "side": setup.get("side"),
                            "pattern": setup.get("pattern"),
                            "trigger_time": setup.get("trigger_time"),
                            "picked_candle_time": setup.get("picked_candle_time"),
                            "breakout_candle_time": setup.get("breakout_candle_time"),
                            "entry": setup.get("entry"),
                            "sl": setup.get("sl"),
                            "tp": setup.get("tp"),
                            "tp_mode": setup.get("tp_mode"),
                            "tp_source": setup.get("tp_source"),
                            "target_result_price": setup.get("target_result_price"),
                            "gann_cmp": setup.get("gann_cmp"),
                        })
                        print(
                            f"[TP_EARLY] {day} | {pair} | side={setup.get('side')} | "
                            f"trigger={setup.get('trigger_time')} | entry={setup.get('entry')} | tp={setup.get('tp')}"
                        )

                except Exception as e:
                    print(f"[ERROR] {day} | {pair} | {e}")

        print("\n" + "=" * 70)
        print(f"TP_EARLY TOTAL COUNT: {total_early}")
        print("=" * 70)

        if records:
            out_df = pd.DataFrame(records)
            export_name = f"tp_early_count_{START_DATE}_to_{END_DATE}.xlsx"
            out_df.to_excel(export_name, index=False)
            print(f"Exported: {export_name}")

            print("\nCount by pair:")
            print(out_df.groupby("pair").size().sort_values(
                ascending=False).to_string())
        else:
            print("No TP_EARLY setups found.")

    finally:
        mt5.shutdown()
        print("MT5 shutdown")


if __name__ == "__main__":
    main()
