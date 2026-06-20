from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from strategy_calculator import StrategyCalculator


def _dbg(verbose: bool, msg: str):
    if verbose:
        print(msg)


def _round5(x):
    return round(float(x), 5)


def _gann_lookup_price(x: float) -> float:
    return np.floor(float(x) * 10000) / 10000.0


def _record_attempt(
    all_attempts: List[Dict],
    side: str,
    picked_time,
    status: str,
    reason: str,
    extra: Optional[Dict] = None,
):
    row = {
        "side": str(side).upper().strip(),
        "picked_candle_time": pd.Timestamp(picked_time) if picked_time is not None else None,
        "status": str(status).upper().strip(),
        "reason": str(reason),
    }
    if extra:
        row.update(extra)
    all_attempts.append(row)


def _find_nearest_buy_target(
    levels: Dict[str, float],
    target_price: float,
    entry_price: float,
) -> Optional[Dict]:
    order = ["buy_t05", "buy_t1", "buy_t125", "buy_t15", "buy_t2"]
    picks = []
    for key in order:
        px = levels.get(key)
        if px is None:
            continue
        px = float(px)
        if px <= float(entry_price):
            continue
        picks.append((abs(px - float(target_price)), px, key))
    if not picks:
        return None
    picks.sort(key=lambda x: (x[0], x[1]))
    _, px, key = picks[0]
    return {"target_key": key, "target_price": _round5(px)}


def _find_nearest_sell_target(
    levels: Dict[str, float],
    target_price: float,
    entry_price: float,
) -> Optional[Dict]:
    order = ["sell_t05", "sell_t1", "sell_t125", "sell_t15", "sell_t2"]
    picks = []
    for key in order:
        px = levels.get(key)
        if px is None:
            continue
        px = float(px)
        if px >= float(entry_price):
            continue
        picks.append((abs(px - float(target_price)), px, key))
    if not picks:
        return None
    picks.sort(key=lambda x: (x[0], x[1]))
    _, px, key = picks[0]
    return {"target_key": key, "target_price": _round5(px)}


def _get_h1_context_for_time(engine, ref_time, verbose=False):
    result = {
        "valid": False,
        "h1_atr_raw": None,
        "h1_atr_cmp": None,
        "prev_h1_time": None,
        "reason": None,
    }

    xh1 = getattr(engine, "h1_atr_df", None)
    if xh1 is None or len(xh1) == 0:
        result["reason"] = "no_h1_atr_dataframe"
        return result

    xh1 = xh1.copy()

    time_col = None
    if "time" in xh1.columns:
        time_col = "time"
    elif "datetime" in xh1.columns:
        time_col = "datetime"
    else:
        result["reason"] = "h1_time_column_missing"
        return result

    atr_col = None
    if "atr" in xh1.columns:
        atr_col = "atr"
    elif "atr_mt5" in xh1.columns:
        atr_col = "atr_mt5"
    else:
        result["reason"] = "h1_atr_column_missing"
        return result

    xh1[time_col] = pd.to_datetime(xh1[time_col], errors="coerce")
    xh1[atr_col] = pd.to_numeric(xh1[atr_col], errors="coerce")
    xh1 = (
        xh1.dropna(subset=[time_col, atr_col])
        .sort_values(time_col)
        .reset_index(drop=True)
    )

    if xh1.empty:
        result["reason"] = "h1_atr_dataframe_empty_after_cleanup"
        return result

    bt = pd.to_datetime(ref_time, errors="coerce")
    if pd.isna(bt) or bt.year <= 1971:
        result["reason"] = "invalid_reference_time"
        return result

    current_h1_open = bt.floor("1h")
    prev_candidates = xh1.loc[xh1[time_col] < current_h1_open]
    if prev_candidates.empty:
        result["reason"] = "no_previous_available_closed_h1_atr"
        return result

    prev_row = prev_candidates.iloc[-1]
    prev_time = pd.Timestamp(prev_row[time_col])

    try:
        prev_h1_atr = float(prev_row[atr_col])
    except Exception:
        result["prev_h1_time"] = prev_time
        result["reason"] = "invalid_previous_h1_atr"
        return result

    result["valid"] = True
    result["h1_atr_raw"] = _round5(prev_h1_atr)
    result["h1_atr_cmp"] = round(prev_h1_atr * 100000, 2)
    result["prev_h1_time"] = prev_time
    result["reason"] = "passed"

    if verbose:
        print(
            f" -> H1_CONTEXT | ref_time={bt} | used_prev_h1={prev_time} | "
            f"h1_atr_raw={prev_h1_atr:.6f} | h1_atr_cmp={result['h1_atr_cmp']:.2f}"
        )

    return result


def _pickup_filter_pass(engine, pickup_time, pickup_atr, verbose=False):
    result = {
        "valid": False,
        "pickup_atr_raw": None,
        "pickup_atr_cmp": None,
        "h1_atr_raw": None,
        "h1_atr_cmp": None,
        "threshold_cmp": None,
        "prev_h1_time": None,
        "reason": None,
    }

    try:
        pickup_atr = float(pickup_atr)
    except Exception:
        result["reason"] = "pickup_atr_invalid"
        return result

    if not np.isfinite(pickup_atr) or pickup_atr <= 0:
        result["reason"] = "pickup_atr_invalid"
        return result

    h1ctx = _get_h1_context_for_time(engine, pickup_time, verbose=verbose)
    if not h1ctx["valid"]:
        result["reason"] = h1ctx["reason"]
        result["prev_h1_time"] = h1ctx.get("prev_h1_time")
        return result

    pickup_cmp = round(pickup_atr * 100000, 2)
    h1_cmp = float(h1ctx["h1_atr_cmp"])
    threshold_cmp = round(h1_cmp * 0.70, 2)

    compare_ok = pickup_cmp <= threshold_cmp

    result["valid"] = compare_ok
    result["pickup_atr_raw"] = _round5(pickup_atr)
    result["pickup_atr_cmp"] = pickup_cmp
    result["h1_atr_raw"] = h1ctx["h1_atr_raw"]
    result["h1_atr_cmp"] = h1_cmp
    result["threshold_cmp"] = threshold_cmp
    result["prev_h1_time"] = h1ctx["prev_h1_time"]
    result["reason"] = "passed" if compare_ok else "pickup_atr_compare_failed"

    if verbose:
        print(
            f" -> PICKUP_ATR_FILTER | pickup_time={pickup_time} | "
            f"prev_h1_time={result['prev_h1_time']} | "
            f"h1_cmp={h1_cmp:.2f} | threshold_cmp={threshold_cmp:.2f} | "
            f"pickup_cmp={pickup_cmp:.2f} | pass={compare_ok}"
        )

    return result


def _candidate_buffer_from_pickup_atr(pickup_atr: float):
    cmp_val = round(float(pickup_atr) * 100000, 2)

    if cmp_val <= 30:
        divisor = 1.0
    elif cmp_val <= 50:
        divisor = 2.0
    elif cmp_val <= 70:
        divisor = 3.0
    elif cmp_val <= 100:
        divisor = 4.0
    else:
        divisor = 5.0

    return {
        "pickup_atr_cmp": cmp_val,
        "divisor": divisor,
        "buffer_raw": float(pickup_atr) / divisor,
        "buffer_cmp": round(cmp_val / divisor, 2),
    }


def _is_st_flip_buy(row):
    trend = str(row.get("supertrend_direction", "")).upper().strip()
    signal = str(row.get("supertrend_signal", "")).upper().strip()
    return signal == "BUY" or trend == "BUY_TREND"


def _is_st_flip_sell(row):
    trend = str(row.get("supertrend_direction", "")).upper().strip()
    signal = str(row.get("supertrend_signal", "")).upper().strip()
    return signal == "SELL" or trend == "SELL_TREND"


def _find_buy_candidate(xdf: pd.DataFrame, start_idx: int, pickup_high: float, pickup_atr: float):
    pickup_row = xdf.iloc[start_idx]
    buf = _candidate_buffer_from_pickup_atr(pickup_atr)
    breakout_level = float(pickup_high) + float(buf["buffer_raw"])

    if _is_st_flip_buy(pickup_row):
        return {
            "candidate_row": pickup_row,
            "candidate_index": start_idx,
            "candidate_mode": "PICKUP_ST_SAME_CANDLE",
            "breakout_level": _round5(breakout_level),
            "buffer_info": buf,
        }

    for j in range(start_idx + 1, len(xdf)):
        r = xdf.iloc[j]
        c = float(r["close"])
        if c >= breakout_level and _is_st_flip_buy(r):
            return {
                "candidate_row": r,
                "candidate_index": j,
                "candidate_mode": "BREAKOUT_CLOSE_WITH_ST_FLIP",
                "breakout_level": _round5(breakout_level),
                "buffer_info": buf,
            }

    return None


def _find_sell_candidate(xdf: pd.DataFrame, start_idx: int, pickup_low: float, pickup_atr: float):
    pickup_row = xdf.iloc[start_idx]
    buf = _candidate_buffer_from_pickup_atr(pickup_atr)
    breakout_level = float(pickup_low) - float(buf["buffer_raw"])

    if _is_st_flip_sell(pickup_row):
        return {
            "candidate_row": pickup_row,
            "candidate_index": start_idx,
            "candidate_mode": "PICKUP_ST_SAME_CANDLE",
            "breakout_level": _round5(breakout_level),
            "buffer_info": buf,
        }

    for j in range(start_idx + 1, len(xdf)):
        r = xdf.iloc[j]
        c = float(r["close"])
        if c <= breakout_level and _is_st_flip_sell(r):
            return {
                "candidate_row": r,
                "candidate_index": j,
                "candidate_mode": "BREAKOUT_CLOSE_WITH_ST_FLIP",
                "breakout_level": _round5(breakout_level),
                "buffer_info": buf,
            }

    return None


def _build_ll_buy_setup(
    engine,
    xdf: pd.DataFrame,
    all_attempts: List[Dict],
    hh_debug: bool = False,
    gap_info: Optional[Dict] = None,
) -> Optional[Dict]:
    side = "BUY"
    ll_idx = xdf["low"].idxmin()
    ll_row = xdf.loc[ll_idx]

    ll_low = float(ll_row["low"])
    ll_high = float(ll_row["high"])
    picked_time = pd.Timestamp(ll_row["time"])

    pickup_atr = float(ll_row.get("atr", np.nan))
    if not np.isfinite(pickup_atr) or pickup_atr <= 0:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "pickup_atr_invalid")
        return None

    pickup_filter = _pickup_filter_pass(
        engine, picked_time, pickup_atr, verbose=hh_debug)
    if not pickup_filter["valid"]:
        _record_attempt(
            all_attempts,
            side,
            picked_time,
            "REJECTED",
            pickup_filter["reason"],
            extra=pickup_filter,
        )
        return None

    candidate_info = _find_buy_candidate(xdf, ll_idx, ll_high, pickup_atr)
    if candidate_info is None:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "no_candidate_found")
        return None

    breakout_row = candidate_info["candidate_row"]
    breakout_time = pd.Timestamp(breakout_row["time"])
    breakout_high = float(breakout_row["high"])
    breakout_close = float(breakout_row["close"])
    candidate_atr = float(breakout_row.get("atr", np.nan))

    if not np.isfinite(candidate_atr) or candidate_atr <= 0:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "candidate_atr_invalid")
        return None

    gann_cmp = _gann_lookup_price(breakout_high)
    gann_levels = engine._get_gann_from_lookup(gann_cmp)
    if not gann_levels:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "gann_lookup_failed")
        return None

    levels = StrategyCalculator._extract_levels(gann_levels)
    if gann_levels.get("buy_at") is None:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "buy_at_missing")
        return None

    final_entry_price = _round5(float(gann_levels["buy_at"]))

    if levels.get("sell_super_middle") is None:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "sell_super_middle_missing")
        return None

    final_sl = _round5(
        float(levels["sell_super_middle"]) - (candidate_atr / 10.0))

    raw_target_result = _round5(breakout_high + candidate_atr * 5.0)
    nearest = _find_nearest_buy_target(
        levels, raw_target_result, final_entry_price)

    final_tp_key = None
    final_tp = None
    if nearest:
        final_tp_key = nearest["target_key"]
        final_tp = float(nearest["target_price"])
    elif levels.get("buy_t1") is not None:
        final_tp_key = "buy_t1"
        final_tp = float(levels["buy_t1"])

    if final_tp is None:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "buy_t1_missing")
        return None

    sl_pips = StrategyCalculator._calc_sl_pips(
        final_entry_price, final_sl, engine.pair)
    lot = StrategyCalculator.calculate_lot_size(
        fund=float(getattr(engine, "current_fund", 0.0)),
        risk_percent=float(getattr(engine, "initial_risk_percent", 0.0)),
        sl_pips=sl_pips,
        pair=engine.pair,
        entry=final_entry_price,
    )

    setup = {
        "side": "BUY",
        "pattern": "LL",
        "entry": _round5(final_entry_price),
        "sl": _round5(final_sl),
        "tp": _round5(final_tp),
        "lot_size": lot,
        "sl_pips": round(float(sl_pips), 1),
        "entry_mode": "BUY_AT",
        "entry_key": "buy_at",
        "target_mode": final_tp_key.upper() if final_tp_key else None,
        "tp_mode": "OLD_PLACEHOLDER",
        "trigger_time": breakout_time,
        "picked_candle_time": picked_time,
        "breakout_candle_time": breakout_time,
        "candidate_mode": candidate_info["candidate_mode"],
        "pivot_low": _round5(ll_low),
        "pivot_ref": _round5(ll_low),
        "base_level": _round5(ll_high),
        "breakout_extreme": _round5(breakout_high),
        "breakout_close": _round5(breakout_close),
        "pickup_atr": _round5(pickup_atr),
        "candidate_breakout_atr": _round5(candidate_atr),
        "pickup_filter_valid": pickup_filter["valid"],
        "pickup_filter_h1_raw": pickup_filter["h1_atr_raw"],
        "pickup_filter_h1_cmp": pickup_filter["h1_atr_cmp"],
        "pickup_filter_threshold_cmp": pickup_filter["threshold_cmp"],
        "pickup_filter_m15_cmp": pickup_filter["pickup_atr_cmp"],
        "pickup_filter_prev_h1_time": pickup_filter["prev_h1_time"],
        "pickup_buffer_divisor": candidate_info["buffer_info"]["divisor"],
        "pickup_buffer_atr": _round5(candidate_info["buffer_info"]["buffer_raw"]),
        "pickup_breakout_level": candidate_info["breakout_level"],
        "target_result_price": _round5(raw_target_result),
        "gann_cmp": _round5(gann_cmp),
        "gann_lookup_cmp": _round5(gann_cmp),
        "gann_raw_lookup_input": gann_levels.get("raw_lookup_input"),
        "gann_matched_price": gann_levels.get("matched_price"),
        "gann_levels": levels,
        "sl_source": "SELL_SUPER_MIDDLE_MINUS_CANDIDATE_ATR10",
        "tp_source": final_tp_key,
        "status": "PENDING",
        "setup_valid": True,
    }

    _record_attempt(
        all_attempts,
        side,
        picked_time,
        "VALID",
        "valid",
        extra={
            "trigger_time": breakout_time,
            "entry": setup["entry"],
            "sl": setup["sl"],
            "tp": setup["tp"],
        },
    )
    return setup


def _build_hh_sell_setup(
    engine,
    xdf: pd.DataFrame,
    all_attempts: List[Dict],
    hh_debug: bool = False,
    gap_info: Optional[Dict] = None,
) -> Optional[Dict]:
    side = "SELL"
    hh_idx = xdf["high"].idxmax()
    hh_row = xdf.loc[hh_idx]

    hh_high = float(hh_row["high"])
    hh_low = float(hh_row["low"])
    picked_time = pd.Timestamp(hh_row["time"])

    pickup_atr = float(hh_row.get("atr", np.nan))
    if not np.isfinite(pickup_atr) or pickup_atr <= 0:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "pickup_atr_invalid")
        return None

    pickup_filter = _pickup_filter_pass(
        engine, picked_time, pickup_atr, verbose=hh_debug)
    if not pickup_filter["valid"]:
        _record_attempt(
            all_attempts,
            side,
            picked_time,
            "REJECTED",
            pickup_filter["reason"],
            extra=pickup_filter,
        )
        return None

    candidate_info = _find_sell_candidate(xdf, hh_idx, hh_low, pickup_atr)
    if candidate_info is None:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "no_candidate_found")
        return None

    breakout_row = candidate_info["candidate_row"]
    breakout_time = pd.Timestamp(breakout_row["time"])
    breakout_low = float(breakout_row["low"])
    breakout_close = float(breakout_row["close"])
    candidate_atr = float(breakout_row.get("atr", np.nan))

    if not np.isfinite(candidate_atr) or candidate_atr <= 0:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "candidate_atr_invalid")
        return None

    gann_cmp = _gann_lookup_price(breakout_low)
    gann_levels = engine._get_gann_from_lookup(gann_cmp)
    if not gann_levels:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "gann_lookup_failed")
        return None

    levels = StrategyCalculator._extract_levels(gann_levels)
    if gann_levels.get("sell_at") is None:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "sell_at_missing")
        return None

    final_entry_price = _round5(float(gann_levels["sell_at"]))

    if levels.get("buy_super_middle") is None:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "buy_super_middle_missing")
        return None

    final_sl = _round5(
        float(levels["buy_super_middle"]) + (candidate_atr / 10.0))

    raw_target_result = _round5(breakout_low - candidate_atr * 5.0)
    nearest = _find_nearest_sell_target(
        levels, raw_target_result, final_entry_price)

    final_tp_key = None
    final_tp = None
    if nearest:
        final_tp_key = nearest["target_key"]
        final_tp = float(nearest["target_price"])
    elif levels.get("sell_t1") is not None:
        final_tp_key = "sell_t1"
        final_tp = float(levels["sell_t1"])

    if final_tp is None:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "sell_t1_missing")
        return None

    sl_pips = StrategyCalculator._calc_sl_pips(
        final_entry_price, final_sl, engine.pair)
    lot = StrategyCalculator.calculate_lot_size(
        fund=float(getattr(engine, "current_fund", 0.0)),
        risk_percent=float(getattr(engine, "initial_risk_percent", 0.0)),
        sl_pips=sl_pips,
        pair=engine.pair,
        entry=final_entry_price,
    )

    setup = {
        "side": "SELL",
        "pattern": "HH",
        "entry": _round5(final_entry_price),
        "sl": _round5(final_sl),
        "tp": _round5(final_tp),
        "lot_size": lot,
        "sl_pips": round(float(sl_pips), 1),
        "entry_mode": "SELL_AT",
        "entry_key": "sell_at",
        "target_mode": final_tp_key.upper() if final_tp_key else None,
        "tp_mode": "OLD_PLACEHOLDER",
        "trigger_time": breakout_time,
        "picked_candle_time": picked_time,
        "breakout_candle_time": breakout_time,
        "candidate_mode": candidate_info["candidate_mode"],
        "pivot_high": _round5(hh_high),
        "pivot_ref": _round5(hh_high),
        "base_level": _round5(hh_low),
        "breakout_extreme": _round5(breakout_low),
        "breakout_close": _round5(breakout_close),
        "pickup_atr": _round5(pickup_atr),
        "candidate_breakout_atr": _round5(candidate_atr),
        "pickup_filter_valid": pickup_filter["valid"],
        "pickup_filter_h1_raw": pickup_filter["h1_atr_raw"],
        "pickup_filter_h1_cmp": pickup_filter["h1_atr_cmp"],
        "pickup_filter_threshold_cmp": pickup_filter["threshold_cmp"],
        "pickup_filter_m15_cmp": pickup_filter["pickup_atr_cmp"],
        "pickup_filter_prev_h1_time": pickup_filter["prev_h1_time"],
        "pickup_buffer_divisor": candidate_info["buffer_info"]["divisor"],
        "pickup_buffer_atr": _round5(candidate_info["buffer_info"]["buffer_raw"]),
        "pickup_breakout_level": candidate_info["breakout_level"],
        "target_result_price": _round5(raw_target_result),
        "gann_cmp": _round5(gann_cmp),
        "gann_lookup_cmp": _round5(gann_cmp),
        "gann_raw_lookup_input": gann_levels.get("raw_lookup_input"),
        "gann_matched_price": gann_levels.get("matched_price"),
        "gann_levels": levels,
        "sl_source": "BUY_SUPER_MIDDLE_PLUS_CANDIDATE_ATR10",
        "tp_source": final_tp_key,
        "status": "PENDING",
        "setup_valid": True,
    }

    _record_attempt(
        all_attempts,
        side,
        picked_time,
        "VALID",
        "valid",
        extra={
            "trigger_time": breakout_time,
            "entry": setup["entry"],
            "sl": setup["sl"],
            "tp": setup["tp"],
        },
    )
    return setup


def build_setup_for_day(
    engine,
    day_df,
    fund: float = 0.0,
    risk_percent: float = 0.0,
    verbose: bool = False,
    hh_debug: bool = False,
    gap_info: Optional[Dict] = None,
):
    if day_df is None or day_df.empty:
        print(" [SETUP DEBUG] day_df empty -> no setup")
        return {"chosen_setups": [], "all_setups": []}

    xdf = day_df.copy()
    xdf["time"] = pd.to_datetime(xdf["time"], errors="coerce")
    xdf = (
        xdf.dropna(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    required_cols = ["open", "high", "low", "close", "atr"]
    for col in required_cols:
        if col in xdf.columns:
            xdf[col] = pd.to_numeric(xdf[col], errors="coerce")

    xdf = xdf.dropna(subset=required_cols).reset_index(drop=True)

    if xdf.empty:
        print(" [SETUP DEBUG] xdf empty after numeric cleanup -> no setup")
        return {"chosen_setups": [], "all_setups": []}

    all_attempts: List[Dict] = []

    buy_setup = _build_ll_buy_setup(
        engine, xdf, all_attempts=all_attempts, hh_debug=hh_debug, gap_info=gap_info
    )
    sell_setup = _build_hh_sell_setup(
        engine, xdf, all_attempts=all_attempts, hh_debug=hh_debug, gap_info=gap_info
    )

    chosen_setups: List[Dict] = []
    if buy_setup:
        chosen_setups.append(buy_setup)
    if sell_setup:
        chosen_setups.append(sell_setup)

    chosen_setups.sort(key=lambda s: pd.Timestamp(s["trigger_time"]))

    return {
        "chosen_setups": chosen_setups,
        "all_setups": all_attempts,
    }


def invalidate_pending_setup_on_new_pivot(
    setup: Optional[Dict],
    intraday_df: pd.DataFrame,
) -> Dict:
    if not setup:
        return {"active_setup": None, "cancelled": False, "reason": "setup_missing"}

    if intraday_df is None or intraday_df.empty:
        return {"active_setup": setup, "cancelled": False, "reason": "intraday_df_missing"}

    if str(setup.get("status", "")).upper() != "PENDING":
        return {"active_setup": setup, "cancelled": False, "reason": "setup_not_pending"}

    side = str(setup.get("side", "")).upper().strip()
    trigger_time = pd.Timestamp(setup["trigger_time"])
    entry_price = float(setup["entry"])

    xdf = intraday_df.copy()
    xdf["time"] = pd.to_datetime(xdf["time"], errors="coerce")

    if "low" in xdf.columns:
        xdf["low"] = pd.to_numeric(xdf["low"], errors="coerce")
    if "high" in xdf.columns:
        xdf["high"] = pd.to_numeric(xdf["high"], errors="coerce")

    xdf = (
        xdf.dropna(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    after_trigger = xdf.loc[xdf["time"] > trigger_time].copy()
    if after_trigger.empty:
        return {"active_setup": setup, "cancelled": False, "reason": "no_data_after_trigger"}

    cancelled_setup = dict(setup)

    if side == "BUY":
        pivot_low = float(setup["pivot_low"])

        fill_chk = after_trigger.dropna(subset=["high"])
        fill_hit = fill_chk.loc[fill_chk["high"] >= entry_price]
        first_fill_time = pd.Timestamp(
            fill_hit.iloc[0]["time"]) if not fill_hit.empty else None

        inv_chk = after_trigger.dropna(subset=["low"])
        inv_hit = inv_chk.loc[inv_chk["low"] < pivot_low]
        first_inv_time = pd.Timestamp(
            inv_hit.iloc[0]["time"]) if not inv_hit.empty else None

        if first_inv_time is None:
            return {"active_setup": setup, "cancelled": False, "reason": "no_new_ll"}

        if first_fill_time is not None and first_fill_time <= first_inv_time:
            return {"active_setup": setup, "cancelled": False, "reason": "entry_hit_before_new_ll"}

        first_hit = inv_hit.iloc[0]
        cancelled_setup["status"] = "CANCELLED"
        cancelled_setup["setup_valid"] = False
        cancelled_setup["cancel_reason"] = "new_lower_low_found_while_pending"
        cancelled_setup["cancel_time"] = pd.Timestamp(first_hit["time"])
        cancelled_setup["new_pivot_low"] = round(float(first_hit["low"]), 5)

        return {
            "active_setup": cancelled_setup,
            "cancelled": True,
            "reason": "new_lower_low_found_while_pending",
        }

    if side == "SELL":
        pivot_high = float(setup["pivot_high"])

        fill_chk = after_trigger.dropna(subset=["low"])
        fill_hit = fill_chk.loc[fill_chk["low"] <= entry_price]
        first_fill_time = pd.Timestamp(
            fill_hit.iloc[0]["time"]) if not fill_hit.empty else None

        inv_chk = after_trigger.dropna(subset=["high"])
        inv_hit = inv_chk.loc[inv_chk["high"] > pivot_high]
        first_inv_time = pd.Timestamp(
            inv_hit.iloc[0]["time"]) if not inv_hit.empty else None

        if first_inv_time is None:
            return {"active_setup": setup, "cancelled": False, "reason": "no_new_hh"}

        if first_fill_time is not None and first_fill_time <= first_inv_time:
            return {"active_setup": setup, "cancelled": False, "reason": "entry_hit_before_new_hh"}

        first_hit = inv_hit.iloc[0]
        cancelled_setup["status"] = "CANCELLED"
        cancelled_setup["setup_valid"] = False
        cancelled_setup["cancel_reason"] = "new_higher_high_found_while_pending"
        cancelled_setup["cancel_time"] = pd.Timestamp(first_hit["time"])
        cancelled_setup["new_pivot_high"] = round(float(first_hit["high"]), 5)

        return {
            "active_setup": cancelled_setup,
            "cancelled": True,
            "reason": "new_higher_high_found_while_pending",
        }

    return {"active_setup": setup, "cancelled": False, "reason": "unknown_side"}
