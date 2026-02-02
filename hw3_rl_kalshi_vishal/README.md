# HW3 RL Kalshi Demo Trader (BTC Hourly Threshold)

This project builds a reinforcement learning agent that:
1) simulates Kalshi-style hourly BTC threshold markets using historical BTC candles,
2) trains an RL policy (PPO) in a Gymnasium environment,
3) connects to the Kalshi **demo** API to place paper trades,
4) provides a Streamlit GUI for markets, positions, P&L, and agent decisions.

Kalshi demo environment and demo API root are documented by Kalshi:
- Demo: https://demo.kalshi.co/
- Demo API root: https://demo-api.kalshi.co/trade-api/v2

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
mkdir -p secrets data
# Place your Kalshi demo private key file in: ./secrets/kalshi_demo_private_key.key
# Fill KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY_PATH in .env
