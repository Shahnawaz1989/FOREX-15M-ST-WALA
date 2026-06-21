from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from strategy_calculator import StrategyCalculator

print(">>> backtest_orb_setup_builder loaded")


def _dbg(verbose: bool, msg: str):
    if verbose:
        print(msg)


def _round5(x):
    return round(float(x), 5)


def _gann_lookup_price(x: float) -> float:
    return np.floor(float(x) * 10000) / 10000.0


def _atr_cmp(x: float) -> Optional[int]:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v) or v <= 0:
        return None
    return int(round(v * 100000))


def _digit_count_from_cmp(cmp_val: Optional[int]) -> Optional[int]:
    if cmp_val is None:
        return None
    cmp_val = abs(int(cmp_val))
    return len(str(cmp_val))


def _is_digit_imbalance(pickup_atr: float, candidate_atr: float) -> bool:
    p_cmp = _atr_cmp(pickup_atr)
    c_cmp = _atr_cmp(candidate_atr)
    p_digits = _digit_count_from_cmp(p_cmp)
    c_digits = _digit_count_from_cmp(c_cmp)
    return p_digits == 2 and c_digits == 3


def _reduce_atr_by_30_percent(x: float) -> float:
    return float(x) * 0.70


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

    print(
        f" -> ATTEMPT | side={row['side']} | picked={row['picked_candle_time']} | "
        f"status={row['status']} | reason={row['reason']}"
    )


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
            f" -> H1_CONTEXT | ref_time={bt} | prev_h1={prev_time} | "
            f"h1_raw={result['h1_atr_raw']} | h1_cmp={result['h1_atr_cmp']}"
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

    # 15m ATR validate
    try:
        pickup_atr = float(pickup_atr)
    except Exception:
        result["reason"] = "pickup_atr_invalid"
        return result

    if not np.isfinite(pickup_atr) or pickup_atr <= 0:
        result["reason"] = "pickup_atr_invalid"
        return result

    # H1 context
    h1ctx = _get_h1_context_for_time(engine, pickup_time, verbose=verbose)
    if not h1ctx["valid"]:
        result["reason"] = h1ctx["reason"]
        result["prev_h1_time"] = h1ctx.get("prev_h1_time")
        return result

    # Compare values (x100000 cmp units)
    pickup_cmp = round(pickup_atr * 100000, 2)
    h1_cmp = float(h1ctx["h1_atr_cmp"])

    # RULE: 15m ATR < 70% of previous closed 1H ATR
    pickup_filter_multiplier = 0.70
    threshold_cmp = round(h1_cmp * pickup_filter_multiplier, 2)
    compare_ok = pickup_cmp < threshold_cmp  # strictly less than 70%

    result["valid"] = compare_ok
    result["pickup_atr_raw"] = _round5(pickup_atr)
    result["pickup_atr_cmp"] = pickup_cmp
    result["h1_atr_raw"] = h1ctx["h1_atr_raw"]
    result["h1_atr_cmp"] = h1_cmp
    result["threshold_cmp"] = threshold_cmp
    result["prev_h1_time"] = h1ctx["prev_h1_time"]
    result["reason"] = "passed" if compare_ok else "pickup_atr_compare_failed"

    print(
        f" -> PICKUP_FILTER | time={pickup_time} | "
        f"pickup_cmp={pickup_cmp} | h1_cmp={h1_cmp} | "
        f"threshold={threshold_cmp} | pass={compare_ok}"
    )

    return result


def _candidate_buffer_from_pickup_atr(pickup_atr: float):
    cmp_val = round(float(pickup_atr) * 100000, 2)

    if cmp_val <= 35:
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


def _st_state(row):
    """Normalize ST state from direction/signal."""
    trend = str(row.get("supertrend_direction", "")).upper().strip()
    signal = str(row.get("supertrend_signal", "")).upper().strip()

    if signal in ("BUY", "SELL"):
        return signal

    if trend == "BUY_TREND":
        return "BUY"
    if trend == "SELL_TREND":
        return "SELL"

    return "NONE"


def _is_st_flip_buy(prev_row, curr_row) -> bool:
    """True only when this candle flips INTO BUY."""
    prev = _st_state(prev_row) if prev_row is not None else "NONE"
    curr = _st_state(curr_row)
    return prev != "BUY" and curr == "BUY"


def _is_st_flip_sell(prev_row, curr_row) -> bool:
    """True only when this candle flips INTO SELL."""
    prev = _st_state(prev_row) if prev_row is not None else "NONE"
    curr = _st_state(curr_row)
    return prev != "SELL" and curr == "SELL"


def _find_buy_candidate(xdf: pd.DataFrame, start_idx: int, pickup_high: float, pickup_atr: float):
    """
    BUY side:
    - PICKUP_ST_SAME_CANDLE   -> LL pickup candle par hi ST BUY flip.
    - BREAKOUT_ST_SAME_CANDLE -> breakout candle par naya ST BUY flip.
    """
    pickup_row = xdf.iloc[start_idx]
    prev_pickup = xdf.iloc[start_idx - 1] if start_idx > 0 else None

    buf = _candidate_buffer_from_pickup_atr(pickup_atr)
    breakout_level = float(pickup_high) + float(buf["buffer_raw"])

    # LOGIC A: pickup LL candle par flip
    if _is_st_flip_buy(prev_pickup, pickup_row):
        return {
            "candidate_row": pickup_row,
            "candidate_index": start_idx,
            "candidate_mode": "PICKUP_ST_SAME_CANDLE",
            "breakout_level": _round5(breakout_level),
            "buffer_info": buf,
        }

    # LOGIC B: breakout candle par hi naya flip
    for j in range(start_idx + 1, len(xdf)):
        r = xdf.iloc[j]
        c = float(r["close"])

        if c >= breakout_level:
            prev = xdf.iloc[j - 1] if j > 0 else None

            # breakout candle par naya BUY flip?
            if _is_st_flip_buy(prev, r):
                return {
                    "candidate_row": r,
                    "candidate_index": j,
                    "candidate_mode": "BREAKOUT_ST_SAME_CANDLE",
                    "breakout_level": _round5(breakout_level),
                    "buffer_info": buf,
                }

            # Breakout mil gaya, lekin is candle par flip nahi -> INVALID
            return None

    return None


def _find_sell_candidate(xdf: pd.DataFrame, start_idx: int, pickup_low: float, pickup_atr: float):
    """
    SELL side:
    - PICKUP_ST_SAME_CANDLE   -> HH pickup candle par hi ST SELL flip.
    - BREAKOUT_ST_SAME_CANDLE -> breakout candle par naya ST SELL flip.
    """
    pickup_row = xdf.iloc[start_idx]
    prev_pickup = xdf.iloc[start_idx - 1] if start_idx > 0 else None

    buf = _candidate_buffer_from_pickup_atr(pickup_atr)
    breakout_level = float(pickup_low) - float(buf["buffer_raw"])

    # LOGIC A: pickup HH candle par flip
    if _is_st_flip_sell(prev_pickup, pickup_row):
        return {
            "candidate_row": pickup_row,
            "candidate_index": start_idx,
            "candidate_mode": "PICKUP_ST_SAME_CANDLE",
            "breakout_level": _round5(breakout_level),
            "buffer_info": buf,
        }

    # LOGIC B: breakout candle par hi naya flip
    for j in range(start_idx + 1, len(xdf)):
        r = xdf.iloc[j]
        c = float(r["close"])

        if c <= breakout_level:
            prev = xdf.iloc[j - 1] if j > 0 else None

            # breakout candle par naya SELL flip?
            if _is_st_flip_sell(prev, r):
                return {
                    "candidate_row": r,
                    "candidate_index": j,
                    "candidate_mode": "BREAKOUT_ST_SAME_CANDLE",
                    "breakout_level": _round5(breakout_level),
                    "buffer_info": buf,
                }

            # Breakout mil gaya, lekin is candle par flip nahi -> INVALID
            return None

    return None


def _resolve_buy_target(levels: Dict[str, float], breakout_high: float, candidate_atr: float, final_entry_price: float, pickup_atr: float):
    candidate_cmp = _atr_cmp(candidate_atr)
    pickup_cmp = _atr_cmp(pickup_atr)

    digit_imbalance = _is_digit_imbalance(pickup_atr, candidate_atr)
    target_calc_mode = None
    target_atr_used = None

    if digit_imbalance:
        new_atr = _reduce_atr_by_30_percent(candidate_atr)
        raw_target = _round5(float(breakout_high) + new_atr * 4.0)
        target_calc_mode = "DIGIT_IMBALANCE_NEWATR_X4"
        target_atr_used = _round5(new_atr)
    elif candidate_cmp is not None and candidate_cmp <= 50:
        raw_target = _round5(float(breakout_high) +
                             float(candidate_atr) * 10.0)
        target_calc_mode = "ATR_LE_50_X10"
        target_atr_used = _round5(candidate_atr)
    else:
        raw_target = _round5(float(breakout_high) + float(candidate_atr) * 5.0)
        target_calc_mode = "NORMAL_X5"
        target_atr_used = _round5(candidate_atr)

    early_mode = raw_target <= float(final_entry_price)

    final_tp_key = None
    final_tp = None
    if early_mode and levels.get("buy_t1") is not None:
        final_tp_key = "buy_t1"
        final_tp = float(levels["buy_t1"])
    else:
        nearest = _find_nearest_buy_target(
            levels, raw_target, final_entry_price)
        if nearest:
            final_tp_key = nearest["target_key"]
            final_tp = float(nearest["target_price"])
        elif levels.get("buy_t1") is not None:
            final_tp_key = "buy_t1"
            final_tp = float(levels["buy_t1"])

    return {
        "final_tp": _round5(final_tp) if final_tp is not None else None,
        "final_tp_key": final_tp_key,
        "raw_target": _round5(raw_target),
        "early_mode": early_mode,
        "target_calc_mode": target_calc_mode,
        "target_atr_used": target_atr_used,
        "digit_imbalance": digit_imbalance,
        "pickup_atr_cmp_int": pickup_cmp,
        "candidate_atr_cmp_int": candidate_cmp,
    }


def _resolve_sell_target(levels: Dict[str, float], breakout_low: float, candidate_atr: float, final_entry_price: float, pickup_atr: float):
    candidate_cmp = _atr_cmp(candidate_atr)
    pickup_cmp = _atr_cmp(pickup_atr)

    digit_imbalance = _is_digit_imbalance(pickup_atr, candidate_atr)
    target_calc_mode = None
    target_atr_used = None

    if digit_imbalance:
        new_atr = _reduce_atr_by_30_percent(candidate_atr)
        raw_target = _round5(float(breakout_low) - new_atr * 4.0)
        target_calc_mode = "DIGIT_IMBALANCE_NEWATR_X4"
        target_atr_used = _round5(new_atr)
    elif candidate_cmp is not None and candidate_cmp <= 50:
        raw_target = _round5(float(breakout_low) - float(candidate_atr) * 10.0)
        target_calc_mode = "ATR_LE_50_X10"
        target_atr_used = _round5(candidate_atr)
    else:
        raw_target = _round5(float(breakout_low) - float(candidate_atr) * 5.0)
        target_calc_mode = "NORMAL_X5"
        target_atr_used = _round5(candidate_atr)

    early_mode = raw_target >= float(final_entry_price)

    final_tp_key = None
    final_tp = None
    if early_mode and levels.get("sell_t1") is not None:
        final_tp_key = "sell_t1"
        final_tp = float(levels["sell_t1"])
    else:
        nearest = _find_nearest_sell_target(
            levels, raw_target, final_entry_price)
        if nearest:
            final_tp_key = nearest["target_key"]
            final_tp = float(nearest["target_price"])
        elif levels.get("sell_t1") is not None:
            final_tp_key = "sell_t1"
            final_tp = float(levels["sell_t1"])

    return {
        "final_tp": _round5(final_tp) if final_tp is not None else None,
        "final_tp_key": final_tp_key,
        "raw_target": _round5(raw_target),
        "early_mode": early_mode,
        "target_calc_mode": target_calc_mode,
        "target_atr_used": target_atr_used,
        "digit_imbalance": digit_imbalance,
        "pickup_atr_cmp_int": pickup_cmp,
        "candidate_atr_cmp_int": candidate_cmp,
    }


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
    if gann_levels.get("sell_at") is None:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "sell_at_missing")
        return None

    final_entry_price = _round5(float(gann_levels["buy_at"]))
    final_sl = _round5(float(gann_levels["sell_at"]))

    tp_info = _resolve_buy_target(
        levels=levels,
        breakout_high=breakout_high,
        candidate_atr=candidate_atr,
        final_entry_price=final_entry_price,
        pickup_atr=pickup_atr,
    )

    final_tp = tp_info["final_tp"]
    final_tp_key = tp_info["final_tp_key"]
    raw_target_result = tp_info["raw_target"]

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
        "tp_mode": tp_info["target_calc_mode"],
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
        "pickup_atr_cmp_int": tp_info["pickup_atr_cmp_int"],
        "candidate_atr_cmp_int": tp_info["candidate_atr_cmp_int"],
        "digit_imbalance": tp_info["digit_imbalance"],
        "early_mode": tp_info["early_mode"],
        "target_atr_used": tp_info["target_atr_used"],
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
        "sl_source": "SELL_AT",
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
            "early_mode": setup["early_mode"],
            "digit_imbalance": setup["digit_imbalance"],
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
    if gann_levels.get("buy_at") is None:
        _record_attempt(all_attempts, side, picked_time,
                        "REJECTED", "buy_at_missing")
        return None

    final_entry_price = _round5(float(gann_levels["sell_at"]))
    final_sl = _round5(float(gann_levels["buy_at"]))

    tp_info = _resolve_sell_target(
        levels=levels,
        breakout_low=breakout_low,
        candidate_atr=candidate_atr,
        final_entry_price=final_entry_price,
        pickup_atr=pickup_atr,
    )

    final_tp = tp_info["final_tp"]
    final_tp_key = tp_info["final_tp_key"]
    raw_target_result = tp_info["raw_target"]

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
        "tp_mode": tp_info["target_calc_mode"],
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
        "pickup_atr_cmp_int": tp_info["pickup_atr_cmp_int"],
        "candidate_atr_cmp_int": tp_info["candidate_atr_cmp_int"],
        "digit_imbalance": tp_info["digit_imbalance"],
        "early_mode": tp_info["early_mode"],
        "target_atr_used": tp_info["target_atr_used"],
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
        "sl_source": "BUY_AT",
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
            "early_mode": setup["early_mode"],
            "digit_imbalance": setup["digit_imbalance"],
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
        return {"chosen_setups": [], "all_setups": []}

    xdf = day_df.copy()
    xdf["time"] = pd.to_datetime(xdf["time"], errors="coerce")
    xdf = (
        xdf.dropna(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    st_cols = [c for c in ["supertrend_direction",
                           "supertrend_signal", "trend", "signal"] if c in xdf.columns]
    nn_dir = int(xdf["supertrend_direction"].notna().sum()
                 ) if "supertrend_direction" in xdf.columns else 0
    nn_sig = int(xdf["supertrend_signal"].notna().sum()
                 ) if "supertrend_signal" in xdf.columns else 0
    print(
        f" -> BUILDER INPUT | st_cols={st_cols} | nn_dir={nn_dir} | nn_sig={nn_sig}")

    required_cols = ["open", "high", "low", "close", "atr"]
    for col in required_cols:
        if col in xdf.columns:
            xdf[col] = pd.to_numeric(xdf[col], errors="coerce")

    xdf = xdf.dropna(subset=required_cols).reset_index(drop=True)

    if xdf.empty:
        return {"chosen_setups": [], "all_setups": []}

    all_attempts: List[Dict] = []

    buy_setup = _build_ll_buy_setup(
        engine,
        xdf,
        all_attempts=all_attempts,
        hh_debug=hh_debug,
        gap_info=gap_info,
    )

    sell_setup = _build_hh_sell_setup(
        engine,
        xdf,
        all_attempts=all_attempts,
        hh_debug=hh_debug,
        gap_info=gap_info,
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
