# live_trading/trader.py
from __future__ import annotations
import numpy as np

import os
import json
import time
import argparse
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from stable_baselines3 import PPO

from live_trading.kalshi_auth import KalshiAuth
from live_trading.kalshi_client import KalshiClient

LOG_DIR = "logs/live"
STATE_PATH = f"{LOG_DIR}/state.json"
TRADES_CSV = f"{LOG_DIR}/trades.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_logs():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(TRADES_CSV):
        pd.DataFrame(
            columns=["time", "market_ticker", "action", "count", "yes_price_cents", "resp"]
        ).to_csv(TRADES_CSV, index=False)


def write_state(d: dict):
    with open(STATE_PATH, "w") as f:
        json.dump(d, f, indent=2)


def safe_get_markets_list(markets_resp):
    """
    Kalshi responses sometimes wrap in {"markets":[...]} or return list directly.
    """
    if isinstance(markets_resp, dict):
        if "markets" in markets_resp and isinstance(markets_resp["markets"], list):
            return markets_resp["markets"]
        if "market" in markets_resp and isinstance(markets_resp["market"], list):
            return markets_resp["market"]
    if isinstance(markets_resp, list):
        return markets_resp
    return []


def safe_get_positions_list(pos_resp):
    """
    Kalshi responses sometimes wrap in {"positions":[...]}.
    """
    if isinstance(pos_resp, dict):
        if "positions" in pos_resp and isinstance(pos_resp["positions"], list):
            return pos_resp["positions"]
    if isinstance(pos_resp, list):
        return pos_resp
    return []


def safe_get_orders_list(orders_resp):
    """
    Kalshi responses sometimes wrap in {"orders":[...]} or {"data":[...]}.
    """
    if isinstance(orders_resp, dict):
        if "orders" in orders_resp and isinstance(orders_resp["orders"], list):
            return orders_resp["orders"]
        if "data" in orders_resp and isinstance(orders_resp["data"], list):
            return orders_resp["data"]
    if isinstance(orders_resp, list):
        return orders_resp
    return []


def extract_best_yes_prices(market_obj):
    """
    Try a few common fields; fall back to mid-ish defaults.
    Prices can be in cents (int) or decimals (float) depending on API response.
    We'll normalize to decimal in [0,1] for decision, then convert to cents for orders.
    """
    candidates_bid = [
        market_obj.get("yes_bid"),
        market_obj.get("best_yes_bid"),
        market_obj.get("best_bid"),
    ]
    candidates_ask = [
        market_obj.get("yes_ask"),
        market_obj.get("best_yes_ask"),
        market_obj.get("best_ask"),
    ]

    bid = next((x for x in candidates_bid if x is not None), None)
    ask = next((x for x in candidates_ask if x is not None), None)

    ob = market_obj.get("orderbook") or market_obj.get("order_book") or {}
    if bid is None:
        bid = ob.get("yes_bid") or ob.get("best_yes_bid")
    if ask is None:
        ask = ob.get("yes_ask") or ob.get("best_yes_ask")

    if bid is None:
        bid = 0.49
    if ask is None:
        ask = 0.51

    bid = float(bid)
    ask = float(ask)

    if bid > 1.0 or ask > 1.0:
        bid /= 100.0
        ask /= 100.0

    bid = max(0.01, min(0.99, bid))
    ask = max(0.01, min(0.99, ask))
    return bid, ask


def market_ticker_from_obj(m):
    return m.get("ticker") or m.get("market_ticker") or m.get("symbol")


def pick_candidate_market(markets_list):
    """
    Placeholder selection: pick the first market.
    TODO: Replace with filter for BTC hourly threshold markets once you inspect tickers.
    """
    return markets_list[0] if markets_list else None


def main():
    load_dotenv(".env")
    ensure_logs()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ppo_btc_kalshi.zip")
    ap.add_argument("--base_url", default="https://demo-api.kalshi.co")
    ap.add_argument("--poll_sec", type=int, default=30)
    args = ap.parse_args()

    api_key_id = os.environ.get("KALSHI_API_KEY_ID", "").strip()
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()

    if not api_key_id:
        raise RuntimeError("Missing KALSHI_API_KEY_ID in .env")
    if not key_path:
        raise RuntimeError("Missing KALSHI_PRIVATE_KEY_PATH in .env")
    if not os.path.exists(key_path):
        raise RuntimeError(f"Private key file not found at: {key_path}")

    auth = KalshiAuth(api_key_id=api_key_id, private_key_path=key_path)
    client = KalshiClient(base_url=args.base_url, auth=auth)

    if not os.path.exists(args.model):
        raise RuntimeError(f"Model not found at {args.model}. Train first.")

    model = PPO.load(args.model)

    max_daily_loss = float(os.environ.get("MAX_DAILY_LOSS", "200"))
    max_notional = float(os.environ.get("MAX_NOTIONAL", "1000"))
    max_count = int(os.environ.get("MAX_CONTRACTS_PER_TRADE", "50"))

    logger.info("Starting live demo trader (polling)...")

    while True:
        try:
            # 1) balance + positions
            balance = client.get_balance()
            pos_resp = client.get_positions(limit=200)
            positions = safe_get_positions_list(pos_resp)

            # 1b) resting/pending orders (SAFE)
            resting_orders = []
            orders_error = None
            try:
                orders_resp = client.get_orders(status="resting", limit=200)
                resting_orders = safe_get_orders_list(orders_resp)
            except Exception as oe:
                orders_error = str(oe)
                logger.warning(f"Could not fetch resting orders: {orders_error}")

            # balance fields are usually in cents; normalize to dollars
            balance_cents = balance.get("balance", 10_000_00)
            pv_cents = balance.get("portfolio_value", balance_cents)

            cash = float(balance_cents) / 100.0
            portfolio_value = float(pv_cents) / 100.0

            # 2) markets
            markets_resp = client.get_markets(limit=100)
            markets_list = safe_get_markets_list(markets_resp)

            if not markets_list:
                write_state({"time": utc_now(), "status": "no_markets"})
                time.sleep(args.poll_sec)
                continue

            m = pick_candidate_market(markets_list)
            if not m:
                write_state({"time": utc_now(), "status": "no_candidate_market"})
                time.sleep(args.poll_sec)
                continue

            market_ticker = market_ticker_from_obj(m)
            if not market_ticker:
                write_state({"time": utc_now(), "status": "missing_market_ticker"})
                time.sleep(args.poll_sec)
                continue

            best_bid, best_ask = extract_best_yes_prices(m)
            mid = (best_bid + best_ask) / 2.0

            # 3) current position qty for this market
            pos_qty = 0
            for p in positions:
                tkr = p.get("ticker") or p.get("market_ticker")
                if tkr == market_ticker:
                    pos_qty = int(p.get("position", p.get("net_position", p.get("quantity", 0))))
                    break

            # 4) observation
            obs_dim = model.policy.observation_space.shape[0]
            obs = np.zeros((obs_dim,), dtype=np.float32)

            if obs_dim >= 6:
                obs[-6] = 0.0
                obs[-5] = float(mid)
                obs[-4] = float(cash / 10000.0)
                obs[-3] = float(pos_qty / 100.0)
                obs[-2] = float(mid)
                obs[-1] = float((portfolio_value - 10000.0) / 1000.0)

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            # 5) Risk checks + decide trade
            do_trade = False
            trade = None

            if action in (1, 2):  # buy
                count = 10 if action == 1 else 30
                count = min(count, max_count)

                est_notional = count * best_ask
                if est_notional <= (max_notional / 100.0):
                    do_trade = True
                    trade = ("buy", count, best_ask)

            elif action in (3, 4) and pos_qty > 0:  # sell/close
                count = min(10 if action == 3 else pos_qty, max_count)
                do_trade = True
                trade = ("sell", count, best_bid)

            resp = None
            if do_trade and trade:
                act, count, px = trade
                yes_px_cents = int(round(px * 100))
                yes_px_cents = max(1, min(99, yes_px_cents))

                resp = client.create_order_yes(
                    market_ticker=market_ticker,
                    action=act,
                    count=count,
                    yes_price=yes_px_cents,
                )

                df = pd.read_csv(TRADES_CSV)
                df.loc[len(df)] = [utc_now(), market_ticker, act, count, yes_px_cents, json.dumps(resp)[:800]]
                df.to_csv(TRADES_CSV, index=False)

            # state.json (NOW includes resting orders)
            state = {
                "time": utc_now(),
                "market_ticker": market_ticker,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid": mid,
                "cash": cash,
                "portfolio_value": portfolio_value,
                "pos_qty": pos_qty,
                "agent_action": action,
                "trade_resp": resp,
                "resting_orders_count": len(resting_orders),
                "resting_orders": resting_orders,
                "orders_error": orders_error,
                "note": "Market selection is placeholder. Next: filter BTC hourly threshold tickers.",
            }
            write_state(state)

        except Exception as e:
            logger.exception(e)
            write_state({"time": utc_now(), "status": "error", "error": str(e)})

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
