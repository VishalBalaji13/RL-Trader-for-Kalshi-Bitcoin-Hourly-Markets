# [cite_start]Reinforcement Learning Trader for Kalshi Bitcoin Hourly Markets [cite: 1]

[cite_start]This project details the design, implementation, and deployment of a Deep Reinforcement Learning (DRL) agent capable of trading Bitcoin price threshold event contracts on the Kalshi prediction market[cite: 3]. [cite_start]The objective was to create an autonomous system that backtests a trading strategy using historical data, trains a policy using Proximal Policy Optimization (PPO), and deploys this policy in real-time to the Kalshi Demo environment[cite: 4].

## System Architecture

[cite_start]The system includes a custom simulation environment, a live trading engine connected to the Kalshi v2 API, and a Streamlit-based GUI for monitoring agent performance and market states[cite: 5].

### Problem Formulation
[cite_start]The trading task is formulated as a Markov Decision Process (MDP) tuple $(S, A, R, \gamma)$[cite: 8].

**State Space:**
* [cite_start]**Market Data:** Normalized recent Bitcoin price history, current implied probability of the "YES" contract, and the strike price[cite: 11].
* [cite_start]**Time Features:** Time remaining until the contract expires[cite: 12].
* [cite_start]**Account State:** Current inventory and available cash balance[cite: 13].
* [cite_start]**Technical Indicators:** Rolling volatility and Relative Strength Index (RSI)[cite: 14].

**Action Space:**
* [cite_start]The system utilizes a discrete action space[cite: 16].
* [cite_start]The available actions are 0 (Hold), 1 (Buy YES), 2 (Buy NO), and 3 (Close)[cite: 17, 18, 19, 20].
* [cite_start]Position sizing is fixed to isolate decision quality from bankroll management variance[cite: 21].

**Reward Function:**
* [cite_start]The reward function is the change in the agent's net portfolio value, or Mark-to-Market P&L[cite: 23].
* [cite_start]A small penalty is applied for excessive switching of positions to account for spread costs and fees[cite: 26].

### Algorithm & Network
* [cite_start]**Algorithm:** The agent is trained using Proximal Policy Optimization (PPO), an on-policy gradient method[cite: 28, 29].
* [cite_start]**Actor Network:** Utilizes a Multi-Layer Perceptron (MLP) with two hidden layers of 64 units each, outputting a softmax distribution over the 4 discrete actions[cite: 33].
* [cite_start]**Critic Network:** Utilizes a similar MLP to estimate the Value function $V(s)$[cite: 34].
* [cite_start]**Hyperparameters:** Key parameters include a Learning Rate of 3e-4, a Gamma of 0.99, a Clip Range of 0.2, and an Entropy Coefficient of 0.01[cite: 36, 37, 38, 39].

## Deployment & Monitoring

### Backtesting Environment
* [cite_start]Built using a custom Gymnasium environment (KalshiEnv) that loads historical BTC minute-data[cite: 42].

### Live Demo Trading
* [cite_start]The trained model is deployed to the Kalshi Demo environment using the official kalshi-python SDK[cite: 71].
* [cite_start]The live trading script runs a continuous loop to poll the market, preprocess data, query the PPO agent for an action, and submit limit orders to the Kalshi API[cite: 73, 74, 75, 76, 77].

### GUI Front-End
* [cite_start]A web interface was built using Streamlit[cite: 81].
* [cite_start]The dashboard features Market Dashboard displays, Live Prediction confidences, PnL Tracking, and Agent State visualizations[cite: 83, 84, 85, 86].
