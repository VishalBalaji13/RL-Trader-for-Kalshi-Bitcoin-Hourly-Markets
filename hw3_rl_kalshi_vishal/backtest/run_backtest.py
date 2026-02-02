import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env.kalshi_btc_env import KalshiBTCHourlyEnv
from backtest.baselines import baseline_hold, baseline_momentum
from backtest.metrics import sharpe, max_drawdown

def load_btc(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df

def run_policy(env, policy_fn, episodes=50):
    all_final = []
    all_paths = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        pv_path = [env.cash]
        while not done:
            a = policy_fn(obs)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            pv_path.append(env.cash)
        all_final.append(env.cash)
        all_paths.append(pv_path)
    return all_final, all_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/btc_minute.csv")
    ap.add_argument("--model", default="models/ppo_btc_kalshi.zip")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--outdir", default="logs/backtest")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    btc_df = load_btc(args.data)
    env = KalshiBTCHourlyEnv(btc_df)

    # baselines
    hold_final, hold_paths = run_policy(env, baseline_hold, args.episodes)
    mom_final, mom_paths = run_policy(env, baseline_momentum, args.episodes)

    # RL
    if os.path.exists(args.model):
        model = PPO.load(args.model)
        rl_final, rl_paths = run_policy(env, lambda o: int(model.predict(o, deterministic=True)[0]), args.episodes)
    else:
        rl_final, rl_paths = [], []

    def summarize(name, finals, paths):
        if not finals:
            return None
        pnl = [f - 10_000.0 for f in finals]
        # use mean path as proxy series for sharpe/dd
        avg_path = [sum(x)/len(x) for x in zip(*paths)]
        s = sharpe(avg_path)
        dd = max_drawdown(avg_path)
        return {
            "name": name,
            "mean_final": float(sum(finals)/len(finals)),
            "mean_pnl": float(sum(pnl)/len(pnl)),
            "sharpe_proxy": s,
            "max_drawdown_proxy": dd,
        }

    rows = []
    rows.append(summarize("hold", hold_final, hold_paths))
    rows.append(summarize("momentum", mom_final, mom_paths))
    if rl_final:
        rows.append(summarize("ppo_rl", rl_final, rl_paths))
    rows = [r for r in rows if r is not None]

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.outdir}/summary.csv", index=False)
    print(df)

    # plot average equity curves
    plt.figure()
    for name, paths in [("hold", hold_paths), ("momentum", mom_paths)] + ([("ppo_rl", rl_paths)] if rl_paths else []):
        avg = [sum(x)/len(x) for x in zip(*paths)]
        plt.plot(avg, label=name)
    plt.title("Average Equity Curve (proxy)")
    plt.xlabel("Step (hour)")
    plt.ylabel("Cash (end-of-step)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/equity_curves.png")
    print(f"Saved {args.outdir}/equity_curves.png and summary.csv")

if __name__ == "__main__":
    main()
