import os
import json
import pandas as pd
import streamlit as st

LOG_DIR = "logs/live"
STATE_PATH = f"{LOG_DIR}/state.json"
TRADES_CSV = f"{LOG_DIR}/trades.csv"

st.set_page_config(page_title="Kalshi RL Trader (Demo)", layout="wide")
st.title("Kalshi RL Trader (Demo) — BTC Hourly Thresholds")

# -----------------------
# Helpers
# -----------------------
def load_state():
    if not os.path.exists(STATE_PATH):
        return None
    with open(STATE_PATH, "r") as f:
        return json.load(f)

def load_trades():
    if not os.path.exists(TRADES_CSV):
        return pd.DataFrame(columns=["time","market_ticker","action","count","price","resp"])
    return pd.read_csv(TRADES_CSV)

state = load_state()
trades = load_trades()

# -----------------------
# Layout
# -----------------------
left, right = st.columns([1, 1])

# ===== LEFT: Live State =====
with left:
    st.subheader("Live State")
    if state:
        st.json(state)
    else:
        st.info("No live state yet. Start the trader: `python -m live_trading.trader`")

# ===== RIGHT: Resting Orders (TOP) =====
with right:
    st.subheader("Resting / Pending Orders")

    if state:
        resting = state.get("resting_orders", [])
        resting_count = state.get("resting_orders_count", 0)

        st.caption(f"Open limit orders: {resting_count}")

        if resting:
            rows = []
            for o in resting:
                rows.append({
                    "market": o.get("ticker") or o.get("market_ticker"),
                    "side": o.get("action"),
                    "contracts": o.get("initial_count"),
                    "filled": o.get("fill_count"),
                    "remaining": o.get("remaining_count"),
                    "yes_price": o.get("yes_price"),
                    "status": o.get("status"),
                    "created": o.get("created_time"),
                })

            df_rest = pd.DataFrame(rows)
            #st.dataframe(df_rest, use_container_width=True)
            st.table(df_rest)

        else:
            st.info("No resting orders.")
    else:
        st.info("No live state yet.")

# -----------------------
# Recent Trades BELOW
# -----------------------
st.divider()
st.subheader("Recent Trades (Executed)")
st.dataframe(trades.tail(50), use_container_width=True)

# -----------------------
# Agent Recommendation
# -----------------------
st.divider()
st.subheader("Agent Recommendation")

if state and "agent_action" in state:
    a = state["agent_action"]
    mapping = {
        0: "HOLD",
        1: "BUY YES (small)",
        2: "BUY YES (large)",
        3: "SELL YES (small)",
        4: "CLOSE ALL",
    }
    st.metric("Action", mapping.get(a, str(a)))
else:
    st.write("No action yet.")
