import json

INPUT_FILE = "forex_gann_lookup_1_3.json"
OUTPUT_FILE = "forex_gann_lookup_1_3_ready.json"

ROUND_DIGITS = 10


def r(x, digits=ROUND_DIGITS):
    return round(float(x), digits)


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Aapki file ka structure:
# {
#   "0.1000": { "input_price": ..., "buy_at": ..., "sell_at": ... },
#   "0.1001": { ... }
# }

if not isinstance(data, dict):
    raise ValueError("Expected top-level JSON object")

for price_key, row in data.items():
    if not isinstance(row, dict):
        continue

    if "buy_at" not in row or "sell_at" not in row:
        continue

    buy_at = float(row["buy_at"])
    sell_at = float(row["sell_at"])

    middle = (buy_at + sell_at) / 2.0
    buy_super_middle = (buy_at + middle) / 2.0
    sell_super_middle = (sell_at + middle) / 2.0

    row["middle"] = r(middle)
    row["buy_super_middle"] = r(buy_super_middle)
    row["sell_super_middle"] = r(sell_super_middle)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Done: {OUTPUT_FILE}")
