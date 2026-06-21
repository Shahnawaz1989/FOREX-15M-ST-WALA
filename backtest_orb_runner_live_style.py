import pandas as pd

from backtest_orb_runner_helpers import (
    get_day_df,
    get_weekly_risk_percent,
    process_setup_candidate,
    detect_day_gap,
    _print_total_setups_summary,
)

from st_test_reader import enrich_with_mt5_st


def _dedup_live_candidates(candidates):
    seen = set()
    unique = []

    for s in candidates:
        side = str(s.get("side", "")).upper().strip()

        picked_time = pd.Timestamp(s.get("picked_candle_time")) if s.get(
            "picked_candle_time") is not None else None
        trigger_time = pd.Timestamp(s.get("trigger_time")) if s.get(
            "trigger_time") is not None else None

        try:
            entry = round(float(s.get("entry", 0.0)), 5)
        except Exception:
            entry = None

        try:
            sl = round(float(s.get("sl", 0.0)), 5)
        except Exception:
            sl = None

        try:
            tp = round(float(s.get("tp", 0.0)), 5)
        except Exception:
            tp = None

        key = (side, picked_time, trigger_time, entry, sl, tp)

        if key in seen:
            continue

        seen.add(key)
        unique.append(s)

    print(f" -> DEDUP: before={len(candidates)} | after={len(unique)}")
    return unique


def build_intraday_live_style_candidates(engine, day_df, fund, risk_percent, gap_info=None):
    xdf = day_df.copy()
    xdf["time"] = pd.to_datetime(xdf["time"], errors="coerce")
    xdf = xdf.sort_values("time").reset_index(drop=True)

    all_candidates = []
    all_attempts = []

    for i in range(len(xdf)):
        partial_df = xdf.iloc[:i + 1].copy()

        result = engine._build_setup_for_day(
            partial_df,
            fund=fund,
            risk_percent=risk_percent,
            gap_info=gap_info,
        )

        chosen = result.get("chosen_setups", [])
        attempts = result.get("all_setups", [])

        if attempts:
            all_attempts.extend(attempts)
        if chosen:
            all_candidates.extend(chosen)

    all_candidates = _dedup_live_candidates(all_candidates)
    return all_candidates, all_attempts


def process_pair_day_live_style(engine, day, pair, df):
    day_df = get_day_df(engine, df, day)
    if day_df.empty:
        return []

    engine.pair = pair
    engine._debug_last_signature = {}
    engine._debug_seen_messages = set()

    print("\n" + "-" * 40)
    print(f"[{pair}] Processing LIVE-STYLE: {day}")
    print(f"Current Fund: ${engine.current_fund:.2f}")

    gap_info = detect_day_gap(df, day)
    print(f" -> GAP_STATUS={gap_info.get('status')}")

    if gap_info.get("status") in ("GAP_UP", "GAP_DOWN", "NO_GAP"):
        print(
            f" -> {gap_info['status']} | "
            f"prev_last_high={gap_info['prev_high']:.5f}, "
            f"prev_last_low={gap_info['prev_low']:.5f} @ {gap_info['prev_time']} | "
            f"today_first_high={gap_info['today_high']:.5f}, "
            f"today_first_low={gap_info['today_low']:.5f} @ {gap_info['today_time']} | "
            f"gap={gap_info['gap_points']:.5f} ({gap_info['gap_percent']:.4f}%)"
        )

    if not engine._validate_day(day_df):
        print(" -> Day invalid, skipping")
        return []

    engine.h1_atr_df = engine.h1_atr_df.copy()
    engine.h1atrdf = engine.h1_atr_df.copy()

    try:
        first_time = pd.to_datetime(day_df.iloc[0]["time"], errors="coerce")

        day_df = enrich_with_mt5_st(
            candles_df=day_df,
            symbol=pair,
            timeframe="M15",
            day=first_time.to_pydatetime(),
        )

        present_st_cols = [
            c for c in [
                "st_line",
                "buy_buffer",
                "sell_buffer",
                "trend",
                "signal",
                "supertrend_direction",
                "supertrend_signal",
            ] if c in day_df.columns
        ]

        nn_dir = int(day_df["supertrend_direction"].notna().sum(
        )) if "supertrend_direction" in day_df.columns else 0
        nn_sig = int(day_df["supertrend_signal"].notna().sum()
                     ) if "supertrend_signal" in day_df.columns else 0

        print(
            f" -> ST READY: cols={present_st_cols} | nn_dir={nn_dir} | nn_sig={nn_sig}")

    except Exception as e:
        print(f" -> ST ENRICH FAILED: {pair} {day} | {e}")

    week_num, risk_percent = get_weekly_risk_percent(engine, day)
    raw_fund = engine.current_fund

    print(f" -> Week {week_num}: Risk={risk_percent:.1f}%")
    print(f" -> Sizing Fund: ${raw_fund:,.2f}")
    print(" -> Live-style chronological HH/LL replay")

    candidates, all_setups = build_intraday_live_style_candidates(
        engine=engine,
        day_df=day_df,
        fund=raw_fund,
        risk_percent=risk_percent,
        gap_info=gap_info,
    )

    _print_total_setups_summary(all_setups)

    if not candidates:
        print(" -> No live-style HH/LL candidates for this day")
        return []

    print(f" -> Candidate setups found: {len(candidates)}")

    trades_for_pair_day = []
    last_exit_time_by_side = {"BUY": None, "SELL": None}

    for setup in candidates:
        if getattr(engine, "stop_requested", False):
            break

        side = str(setup.get("side", "")).upper().strip()

        trade = process_setup_candidate(
            engine=engine,
            df=df,
            day_df=day_df,
            setup=setup,
            last_exit_time=last_exit_time_by_side.get(side),
        )

        if trade is None:
            continue

        trades_for_pair_day.append(trade)
        last_exit_time_by_side[side] = trade["exit_time"]

    return trades_for_pair_day
