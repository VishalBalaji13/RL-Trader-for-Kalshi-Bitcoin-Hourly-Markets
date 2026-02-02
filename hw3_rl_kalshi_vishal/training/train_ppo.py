import os
import argparse
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.kalshi_btc_env import KalshiBTCHourlyEnv

def load_btc(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/btc_minute.csv")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--model_out", default="models/ppo_btc_kalshi.zip")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)

    btc_df = load_btc(args.data)

    def make_env():
        return KalshiBTCHourlyEnv(btc_df)

    venv = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
    )

    model.learn(total_timesteps=args.steps)
    model.save(args.model_out)
    print(f"Saved model to {args.model_out}")

if __name__ == "__main__":
    main()
