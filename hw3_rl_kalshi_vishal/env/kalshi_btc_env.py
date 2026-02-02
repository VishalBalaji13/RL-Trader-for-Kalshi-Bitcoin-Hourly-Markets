# env/kalshi_btc_env.py
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from agent.features import make_price_features
from env.market_simulator import implied_prob_above_threshold, make_quote, settle_yes


class KalshiBTCHourlyEnv(gym.Env):
    """
    Simulates trading hourly "BTC > threshold at HH:00" YES contracts.

    Episode: one (simulated) day covering hourly intervals from 9:00 to 24:00 (inclusive end at midnight).
    Step: one decision for the current hour's contract.

    Action space:
      0 = hold
      1 = buy YES small
      2 = buy YES large
      3 = sell/close YES small
      4 = sell/close YES all

    Observation (float vector):
      - last n_returns log returns + 1 realized vol
      - 6 extra features: [dist_to_threshold, mid, cash_norm, pos_norm, pos_avg, realized_pnl_norm]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        btc_df: pd.DataFrame,
        initial_cash: float = 10_000.0,
        spread: float = 0.03,
        fee_per_contract: float = 0.01,
        n_returns: int = 12,  # 12 * 5m candles ~= 60 minutes history
        small_size: int = 10,
        large_size: int = 30,
        seed: int = 42,
    ):
        super().__init__()

        self.df = btc_df.copy()
        self.df["open_time"] = pd.to_datetime(self.df["open_time"], utc=True)
        self.df = self.df.sort_values("open_time").reset_index(drop=True)

        self.initial_cash = float(initial_cash)
        self.spread = float(spread)
        self.fee = float(fee_per_contract)
        self.n_returns = int(n_returns)
        self.small_size = int(small_size)
        self.large_size = int(large_size)

        self.rng = np.random.default_rng(seed)

        # observation = n_returns returns + vol(1) + 6 extra features
        obs_dim = self.n_returns + 1 + 6
        self.observation_space = spaces.Box(low=-10, high=10, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        self._reset_state()

    def _reset_state(self):
        self.cash = self.initial_cash
        self.position_qty = 0
        self.position_avg = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

        self.cur_day = None
        # 9..23 plus "24" representing midnight end event
        self.hours = list(range(9, 24)) + [24]
        self.step_idx = 0

        self.prev_portfolio_value = self.initial_cash

    def _pick_random_day(self) -> pd.DataFrame:
        """
        Pick a random date from the dataset. We assume your CSV spans multiple days.
        If you only downloaded one month, it's fine.
        """
        dates = self.df["open_time"].dt.date.unique()
        self.cur_day = self.rng.choice(dates)
        day_df = self.df[self.df["open_time"].dt.date == self.cur_day]
        return day_df

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        self.day_df = self._pick_random_day()
        self.step_idx = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_hour_timestamp(self, hour: int) -> pd.Timestamp:
        """
        Hour in 9..24 (where 24 means midnight of next day).
        This environment uses UTC timestamps for simplicity.
        """
        t = pd.Timestamp(
            year=self.cur_day.year,
            month=self.cur_day.month,
            day=self.cur_day.day,
            hour=hour % 24,
            tz="UTC",
        )
        if hour == 24:
            t = t + pd.Timedelta(days=1)
        return t

    def _get_prices_window(self, asof_time: pd.Timestamp) -> np.ndarray:
        """
        Returns last (n_returns+1) close prices up to asof_time (inclusive).
        Pads with earliest close if not enough history.
        """
        sub = self.df[self.df["open_time"] <= asof_time].tail(self.n_returns + 1)
        if len(sub) < self.n_returns + 1:
            first = float(self.df.iloc[0]["close"])
            pad_n = self.n_returns + 1 - len(sub)
            pad = np.full((pad_n,), first, dtype=np.float32)
            arr = np.concatenate([pad, sub["close"].to_numpy(dtype=np.float32)])
            return arr
        return sub["close"].to_numpy(dtype=np.float32)

    def _spot_at_or_after(self, t: pd.Timestamp) -> float:
        sub = self.df[self.df["open_time"] >= t]
        if len(sub) == 0:
            return float(self.df.iloc[-1]["close"])
        return float(sub.iloc[0]["close"])

    def _mark_to_market(self, mid: float):
        self.unrealized_pnl = self.position_qty * (mid - self.position_avg)

    def _portfolio_value(self, mid: float) -> float:
        return self.cash + self.position_qty * mid

    def _compute_quote_and_threshold(self):
        """
        Compute (time, spot, threshold, quote) for the CURRENT step.
        IMPORTANT: clamp index to avoid out-of-range at terminal.
        """
        idx = min(self.step_idx, len(self.hours) - 1)
        now_hour = self.hours[idx]
        now_t = self._get_hour_timestamp(now_hour)

        prices = self._get_prices_window(now_t)
        spot = float(prices[-1])

        # realized vol from recent returns
        r = np.diff(np.log(np.maximum(prices, 1e-12)))

        # If you are using 5m candles, n_returns=12 gives ~1hr history.
        # Use the last ~1hr returns for vol. For 5m candles, 12 returns ~ 60m.
        window = min(len(r), 12)
        vol = float(np.std(r[-window:]) + 1e-8)

        # Threshold near the money so trades matter:
        band = self.rng.choice([0.0025, 0.005, 0.01])
        direction = self.rng.choice([-1.0, 1.0])
        threshold = spot * (1.0 + direction * band)

        # time to expiry ~1 hour
        tau_hours = 1.0

        p = implied_prob_above_threshold(spot, threshold, vol, tau_hours)
        q = make_quote(p, spread=self.spread)

        return now_t, spot, threshold, q

    def _buy(self, qty: int, price: float):
        if qty <= 0:
            return
        cost = qty * price + qty * self.fee
        if cost > self.cash:
            qty = int(self.cash // (price + self.fee))
            if qty <= 0:
                return
            cost = qty * price + qty * self.fee

        new_qty = self.position_qty + qty
        if new_qty > 0:
            self.position_avg = (self.position_avg * self.position_qty + price * qty) / new_qty
        self.position_qty = new_qty
        self.cash -= cost

    def _sell(self, qty: int, price: float, is_settlement: bool = False):
        if qty <= 0:
            return
        qty = min(qty, self.position_qty)
        proceeds = qty * price - qty * (0.0 if is_settlement else self.fee)
        self.realized_pnl += qty * (price - self.position_avg)
        self.position_qty -= qty
        if self.position_qty == 0:
            self.position_avg = 0.0
        self.cash += proceeds

    def step(self, action: int):
        now_t, spot, threshold, q = self._compute_quote_and_threshold()
        self._mark_to_market(q.mid)

        trade_qty = 0
        trade_price = None

        # execute action with bid/ask + fees
        if action == 1:  # buy small
            trade_qty = self.small_size
            trade_price = q.ask
            self._buy(trade_qty, trade_price)
        elif action == 2:  # buy large
            trade_qty = self.large_size
            trade_price = q.ask
            self._buy(trade_qty, trade_price)
        elif action == 3:  # sell small
            trade_qty = min(self.small_size, self.position_qty)
            trade_price = q.bid
            self._sell(trade_qty, trade_price)
        elif action == 4:  # sell all
            trade_qty = self.position_qty
            trade_price = q.bid
            self._sell(trade_qty, trade_price)

        # settlement at expiry time for this hour
        expiry_t = now_t
        spot_exp = self._spot_at_or_after(expiry_t)
        payoff = settle_yes(spot_exp, threshold)

        # settle any remaining position fully at payoff
        if self.position_qty != 0:
            self._sell(self.position_qty, payoff, is_settlement=True)

        # reward: change in portfolio value (cash at end of step)
        pv = self.cash
        reward = pv - self.prev_portfolio_value
        self.prev_portfolio_value = pv

        # light turnover penalty
        reward -= 0.001 * abs(trade_qty)

        self.step_idx += 1
        terminated = self.step_idx >= len(self.hours)
        truncated = False

        # IMPORTANT: don't compute a "next" obs when terminated
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()

        info = {
            "time": str(now_t),
            "spot": spot,
            "threshold": threshold,
            "bid": q.bid,
            "ask": q.ask,
            "mid": q.mid,
            "trade_qty": trade_qty,
            "trade_price": trade_price,
            "settle_spot": spot_exp,
            "payoff": payoff,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
        }
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        idx = min(self.step_idx, len(self.hours) - 1)
        hour = self.hours[idx]
        t = self._get_hour_timestamp(hour)

        prices = self._get_prices_window(t)
        feat = make_price_features(prices, n_returns=self.n_returns)

        spot = float(prices[-1])
        _, _, threshold, q = self._compute_quote_and_threshold()
        dist = (spot - threshold) / max(spot, 1e-6)

        obs_extra = np.array(
            [
                dist,
                q.mid,
                self.cash / 10_000.0,
                self.position_qty / 100.0,
                self.position_avg,
                self.realized_pnl / 1000.0,
            ],
            dtype=np.float32,
        )

        return np.concatenate([feat, obs_extra], axis=0).astype(np.float32)
