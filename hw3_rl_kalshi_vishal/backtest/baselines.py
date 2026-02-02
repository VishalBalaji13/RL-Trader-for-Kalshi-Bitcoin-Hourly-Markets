import numpy as np

def baseline_hold(obs):
    return 0

def baseline_momentum(obs):
    # last returns are at beginning, vol is at index 60
    # if recent average return positive -> buy small
    rets = obs[:60]
    m = float(np.mean(rets[-10:]))
    if m > 0:
        return 1
    return 0
