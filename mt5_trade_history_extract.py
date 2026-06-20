from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception as e:
    raise RuntimeError(f"MetaTrader5 import failed: {e}")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
DETAIL_CSV = OUTPUT_DIR / "mt5_3months_trade_and_spread_detail.csv"
SYMBOL_SUMMARY_CSV = OUTPUT_DIR / "mt5_3months_symbol_summary.csv"
SUMMARY_MD = OUTPUT_DIR / "mt5_3months_summary.md"

PAIRS = [
    "AUDCAD.ecn", "AUDUSD.ecn", "AUDCHF.ecn", "CADCHF.ecn", "EURAUD.ecn",
    "EURCAD.ecn", "EURCHF.ecn", "EURUSD.ecn", "EURGBP.ecn", "GBPAUD.ecn",
    "GBPCAD.ecn", "GBPCHF.ecn", "GBPUSD.ecn", "NZDCAD.ecn", "NZDUSD.ecn",
    "NZDCHF.ecn", "USDCAD.ecn", "USDCHF.ecn",
]

now = datetime.now()
start = now - timedelta(days=92)

if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

account = mt5.account_info()
if account is None:
    mt5.shutdown()
    raise RuntimeError(f"account_info failed: {mt5.last_error()}")


def pip_size_from_symbol_info(info):
    if info is None:
        return None
    point = getattr(info, "point", None)
    digits = getattr(info, "digits", None)
    if point is None:
        return None
    return point * 10 if digits in (3, 5) else point


def estimate_pip_value_001(symbol, info, reference_price=None):
    if info is None:
        return None
    contract_size = getattr(info, "trade_contract_size", None)
    pip_size = pip_size_from_symbol_info(info)
    if contract_size in (None, 0) or pip_size in (None, 0):
        return None
    try:
        if symbol.endswith("JPY"):
            px = reference_price or getattr(
                info, "bid", None) or getattr(info, "ask", None)
            if px in (None, 0):
                return None
            return (contract_size * pip_size / px) * 0.01
        return contract_size * pip_size * 0.01
    except Exception:
        return None


def spread_stats_from_ticks(symbol, start_dt, end_dt):
    info = mt5.symbol_info(symbol)
    if info is None:
        return {
            "tick_count": 0,
            "spread_points_mean": None,
            "spread_points_median": None,
            "spread_points_p90": None,
            "spread_points_max": None,
            "spread_price_mean": None,
        }

    ticks = mt5.copy_ticks_range(symbol, start_dt, end_dt, mt5.COPY_TICKS_INFO)
    if ticks is None or len(ticks) == 0:
        return {
            "tick_count": 0,
            "spread_points_mean": None,
            "spread_points_median": None,
            "spread_points_p90": None,
            "spread_points_max": None,
            "spread_price_mean": None,
        }

    tdf = pd.DataFrame(ticks)
    if tdf.empty or "bid" not in tdf.columns or "ask" not in tdf.columns:
        return {
            "tick_count": 0,
            "spread_points_mean": None,
            "spread_points_median": None,
            "spread_points_p90": None,
            "spread_points_max": None,
            "spread_price_mean": None,
        }

    point = getattr(info, "point", None)
    tdf = tdf[(tdf["bid"] > 0) & (tdf["ask"] > 0)].copy()

    if tdf.empty or point in (None, 0):
        return {
            "tick_count": 0,
            "spread_points_mean": None,
            "spread_points_median": None,
            "spread_points_p90": None,
            "spread_points_max": None,
            "spread_price_mean": None,
        }

    tdf["spread_price"] = tdf["ask"] - tdf["bid"]
    tdf["spread_points"] = tdf["spread_price"] / point

    return {
        "tick_count": int(len(tdf)),
        "spread_points_mean": float(tdf["spread_points"].mean()),
        "spread_points_median": float(tdf["spread_points"].median()),
        "spread_points_p90": float(tdf["spread_points"].quantile(0.90)),
        "spread_points_max": float(tdf["spread_points"].max()),
        "spread_price_mean": float(tdf["spread_price"].mean()),
    }


all_deals = mt5.history_deals_get(start, now)
if all_deals is None:
    err = mt5.last_error()
    mt5.shutdown()
    raise RuntimeError(f"history_deals_get failed: {err}")

deal_rows = []
for d in all_deals:
    x = d._asdict()
    symbol = x.get("symbol")
    if not symbol:
        continue

    info = mt5.symbol_info(symbol)
    pip_size = pip_size_from_symbol_info(info)
    pip_value_001 = estimate_pip_value_001(symbol, info, x.get("price"))

    deal_rows.append({
        "time": pd.to_datetime(x.get("time"), unit="s", errors="coerce"),
        "ticket": x.get("ticket"),
        "position_id": x.get("position_id"),
        "symbol": symbol,
        "entry": x.get("entry"),
        "type": x.get("type"),
        "volume": x.get("volume"),
        "price": x.get("price"),
        "profit": x.get("profit"),
        "commission": x.get("commission"),
        "swap": x.get("swap"),
        "fee": x.get("fee"),
        "comment": x.get("comment"),
        "point": getattr(info, "point", None) if info else None,
        "digits": getattr(info, "digits", None) if info else None,
        "pip_size": pip_size,
        "pip_value_001_est": pip_value_001,
    })

deals_df = pd.DataFrame(deal_rows)
if deals_df.empty:
    deals_df = pd.DataFrame(columns=["symbol"])

spread_rows = []
for symbol in PAIRS:
    info = mt5.symbol_info(symbol)
    if info is None:
        spread_rows.append({"symbol": symbol, "available": False})
        continue

    stats = spread_stats_from_ticks(symbol, start, now)
    spread_rows.append({
        "symbol": symbol,
        "available": True,
        "point": getattr(info, "point", None),
        "digits": getattr(info, "digits", None),
        "pip_size": pip_size_from_symbol_info(info),
        "contract_size": getattr(info, "trade_contract_size", None),
        "currency_profit": getattr(info, "currency_profit", None),
        "pip_value_001_est": estimate_pip_value_001(
            symbol, info, getattr(
                info, "bid", None) or getattr(info, "ask", None)
        ),
        **stats,
        "current_spread_points": getattr(info, "spread", None),
    })

mt5.shutdown()

spread_df = pd.DataFrame(spread_rows)

deal_sym = (
    deals_df.groupby("symbol", dropna=False).agg(
        deals=("ticket", "count"),
        total_profit=("profit", "sum"),
        avg_profit=("profit", "mean"),
        avg_volume=("volume", "mean"),
        pip_value_001_from_deals=("pip_value_001_est", "mean"),
    ).reset_index()
    if not deals_df.empty
    else pd.DataFrame(columns=["symbol"])
)

summary_df = spread_df.merge(deal_sym, on="symbol", how="left")
summary_df = summary_df.sort_values("symbol").reset_index(drop=True)
summary_df.to_csv(SYMBOL_SUMMARY_CSV, index=False)

if not deals_df.empty:
    detail_df = deals_df.merge(
        summary_df[
            ["symbol", "spread_points_mean", "spread_points_median",
                "spread_points_p90", "current_spread_points"]
        ],
        on="symbol",
        how="left",
    )
else:
    detail_df = summary_df.copy()

detail_df.to_csv(DETAIL_CSV, index=False)

lines = []
lines.append(
    f"Analysis window: {start:%Y-%m-%d %H:%M:%S} to {now:%Y-%m-%d %H:%M:%S}")
lines.append(f"Account login: {getattr(account, 'login', 'NA')}")
lines.append(f"Total deals found: {0 if deals_df.empty else len(deals_df)}")
lines.append("")
lines.append("Columns that matter for backtest calibration:")
lines.append(
    "- pip_value_001_est = estimated money impact of 1 pip move on 0.01 lot")
lines.append(
    "- spread_points_mean = mean historical tick spread in points over ~3 months")
lines.append("- spread_points_median = median historical tick spread in points")
lines.append("- spread_points_p90 = 90th percentile spread in points")
lines.append("- current_spread_points = current terminal spread points")
lines.append("")
lines.append(summary_df.to_string(index=False))

SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")

print("\nDONE")
print(f"Detail CSV: {DETAIL_CSV.resolve()}")
print(f"Summary CSV: {SYMBOL_SUMMARY_CSV.resolve()}")
print(f"Summary MD : {SUMMARY_MD.resolve()}")
