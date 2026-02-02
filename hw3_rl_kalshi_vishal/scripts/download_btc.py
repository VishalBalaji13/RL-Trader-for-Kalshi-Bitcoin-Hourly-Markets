import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

# Coinbase Exchange (public) candles endpoint
# Docs: GET /products/{product_id}/candles
COINBASE_BASE = "https://api.exchange.coinbase.com"

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def fetch_candles(product_id: str, start: datetime, end: datetime, granularity: int):
    """
    Returns candles: [ time, low, high, open, close, volume ]
    time is Unix epoch seconds (UTC)
    """
    url = f"{COINBASE_BASE}/products/{product_id}/candles"
    params = {
        "start": iso(start),
        "end": iso(end),
        "granularity": granularity,
    }
    headers = {"User-Agent": "kalshi-rl-hw3"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="BTC-USD", help="Coinbase product id, e.g., BTC-USD")
    ap.add_argument("--interval", default="1m", choices=["1m","5m"], help="1m or 5m")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC), inclusive end date")
    ap.add_argument("--out", default="data/btc_minute.csv")
    args = ap.parse_args()

    granularity = 60 if args.interval == "1m" else 300

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    # Coinbase candles endpoint limits range; we chunk by ~6 hours for 1m, ~24 hours for 5m
    chunk = timedelta(hours=6) if granularity == 60 else timedelta(hours=24)

    rows = []
    cur = start_dt
    final_end = end_dt + timedelta(days=1)  # make inclusive end cover full day

    while cur < final_end:
        nxt = min(cur + chunk, final_end)
        data = fetch_candles(args.product, cur, nxt, granularity)
        # API returns newest-first
        rows.extend(data)
        cur = nxt
        time.sleep(0.2)

    if not rows:
        raise RuntimeError("No data returned. Try a shorter date range or 5m interval.")

    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    df = df[["open_time", "open", "high", "low", "close", "volume"]].sort_values("open_time").reset_index(drop=True)

    # Keep only requested date range (UTC)
    df = df[(df["open_time"] >= start_dt) & (df["open_time"] < final_end)]

    # Ensure output directory exists
    out_path = args.out
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")

if __name__ == "__main__":
    main()
