import numpy as np

def sharpe(pnl_series, eps=1e-9):
    r = np.diff(pnl_series)
    if len(r) < 2:
        return 0.0
    return float(np.mean(r) / (np.std(r) + eps) * np.sqrt(252))

def max_drawdown(pnl_series):
    x = np.array(pnl_series, dtype=float)
    peak = np.maximum.accumulate(x)
    dd = (x - peak)
    return float(dd.min())
