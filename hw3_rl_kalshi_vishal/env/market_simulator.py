from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class MarketQuote:
    bid: float
    ask: float
    mid: float

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def implied_prob_above_threshold(spot: float, threshold: float, vol: float, tau_hours: float) -> float:
    """
    Heuristic: map distance-to-threshold into a probability using a volatility-scaled sigmoid.
    - spot, threshold in USD
    - vol is realized vol of returns per sqrt(hour)
    - tau_hours time to expiry in hours
    """
    tau = max(tau_hours, 1e-6)
    # distance normalized by uncertainty
    denom = max(vol * np.sqrt(tau), 1e-6)
    z = (spot - threshold) / (spot * denom)
    p = sigmoid(2.0 * z)  # scale factor makes it less flat
    return float(np.clip(p, 0.01, 0.99))

def make_quote(p: float, spread: float = 0.03) -> MarketQuote:
    mid = float(np.clip(p, 0.01, 0.99))
    bid = float(np.clip(mid - spread / 2, 0.01, 0.99))
    ask = float(np.clip(mid + spread / 2, 0.01, 0.99))
    return MarketQuote(bid=bid, ask=ask, mid=mid)

def settle_yes(spot_at_expiry: float, threshold: float) -> float:
    return 1.0 if spot_at_expiry > threshold else 0.0
