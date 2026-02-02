import numpy as np

def returns(x: np.ndarray) -> np.ndarray:
    # x = prices
    r = np.diff(np.log(np.maximum(x, 1e-12)))
    return r

def realized_vol(r: np.ndarray) -> float:
    # std of log returns; avoid zero
    return float(np.std(r) + 1e-8)

def make_price_features(prices: np.ndarray, n_returns: int = 60) -> np.ndarray:
    """
    prices: last N+1 prices (minute bars)
    output: last n_returns log returns + vol
    """
    r = returns(prices)
    if len(r) < n_returns:
        pad = np.zeros(n_returns - len(r), dtype=np.float32)
        r = np.concatenate([pad, r.astype(np.float32)])
    else:
        r = r[-n_returns:].astype(np.float32)

    vol = np.array([realized_vol(r)], dtype=np.float32)
    return np.concatenate([r, vol], axis=0)
