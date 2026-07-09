# Reinforcement Learning Trader for Kalshi Bitcoin Hourly Markets

This project details the design, implementation, and deployment of a Deep Reinforcement Learning (DRL) agent capable of trading Bitcoin price threshold event contracts on the Kalshi prediction market. The objective was to create an autonomous system that backtests a trading strategy using historical data, trains a policy using Proximal Policy Optimization (PPO), and deploys this policy in real-time to the Kalshi Demo environment.

![Live demo — the PPO agent placing a limit order on a Bitcoin hourly contract in the Kalshi Demo environment](live_trading_demo.jpeg)

*The deployed agent in action: a live limit order ("Buy YES" at 60¢, Good 'til Canceled) queued autonomously in the Kalshi Demo environment via the Kalshi v2 API.*

## System Architecture

The system includes a custom simulation environment, a live trading engine connected to the Kalshi v2 API, and a Streamlit-based GUI for monitoring agent performance and market states.

### Problem Formulation

The trading task is formulated as a Markov Decision Process (MDP) tuple $(S, A, R, \gamma)$.

**State Space:**

* **Market Data:** Normalized recent Bitcoin price history, current implied probability of the "YES" contract, and the strike price.
* **Time Features:** Time remaining until the contract expires.
* **Account State:** Current inventory and available cash balance.
* **Technical Indicators:** Rolling volatility and Relative Strength Index (RSI).

**Action Space:**

* The system utilizes a discrete action space.
* The available actions are 0 (Hold), 1 (Buy YES), 2 (Buy NO), and 3 (Close).
* Position sizing is fixed to isolate decision quality from bankroll management variance.

**Reward Function:**

* The reward function is the change in the agent's net portfolio value, or Mark-to-Market P&L.
* A small penalty is applied for excessive switching of positions to account for spread costs and fees.

### Algorithm & Network

* **Algorithm:** The agent is trained using Proximal Policy Optimization (PPO), an on-policy gradient method.
* **Actor Network:** Utilizes a Multi-Layer Perceptron (MLP) with two hidden layers of 64 units each, outputting a softmax distribution over the 4 discrete actions.
* **Critic Network:** Utilizes a similar MLP to estimate the Value function $V(s)$.
* **Hyperparameters:** Key parameters include a Learning Rate of 3e-4, a Gamma of 0.99, a Clip Range of 0.2, and an Entropy Coefficient of 0.01.

## Deployment & Monitoring

### Backtesting Environment

* Built using a custom Gymnasium environment (KalshiEnv) that loads historical BTC minute-data.

### Live Demo Trading

* The trained model is deployed to the Kalshi Demo environment using the official kalshi-python SDK.
* The live trading script runs a continuous loop to poll the market, preprocess data, query the PPO agent for an action, and submit limit orders to the Kalshi API.

### GUI Front-End

* A web interface was built using Streamlit.
* The dashboard features Market Dashboard displays, Live Prediction confidences, PnL Tracking, and Agent State visualizations.
