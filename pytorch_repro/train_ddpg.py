import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from environment import Simulation
from ddpg import DDPGAgent
from model import ActorNet


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_pretrain_state(x, time_steps=5):
    # Legacy generator stores [utilization_fraction, remaining_round].
    # Map utilization to approximately the 0..200 production scale and use
    # the semantically correct state ordering [price, utilization, remaining].
    x = x[:, -time_steps:, :].astype(np.float32)
    state = np.zeros((x.shape[0], time_steps, 3), dtype=np.float32)
    state[:, :, 1] = x[:, :, 0] * 200.0
    state[:, :, 2] = x[:, :, 1]
    return state


def supervised_pretrain(agent, data_dir, epochs, batch_size, lr, seed):
    x = np.load(data_dir / "x_pretrain.npy")
    y = np.load(data_dir / "y_pretrain.npy").astype(np.float32)
    state = make_pretrain_state(x, agent.time_steps)
    y = np.clip(y, -agent.action_range, agent.action_range)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(state))
    split = int(0.8 * len(idx))
    tr, va = idx[:split], idx[split:]

    ds = TensorDataset(
        torch.from_numpy(state[tr]),
        torch.from_numpy(y[tr, None]),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    vx = torch.from_numpy(state[va]).to(agent.device)
    vy = torch.from_numpy(y[va, None]).to(agent.device)
    c_train = torch.zeros((batch_size, agent.c_dim), device=agent.device)
    c_val = torch.zeros((len(va), agent.c_dim), device=agent.device)

    opt = torch.optim.Adam(agent.actor.parameters(), lr=lr)
    mse = nn.MSELoss()
    best = float("inf")
    best_state = None

    for epoch in range(epochs):
        agent.actor.train()
        running = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(agent.device), yb.to(agent.device)
            c = c_train[: len(xb)]
            pred = agent.actor(xb, c)
            loss = mse(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu()) * len(xb)
            n += len(xb)

        agent.actor.eval()
        with torch.no_grad():
            val_pred = agent.actor(vx, c_val)
            val_mse = float(mse(val_pred, vy).cpu())
        if val_mse < best:
            best = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(json.dumps({
                "stage": "pretrain",
                "epoch": epoch + 1,
                "train_mse": running / max(n, 1),
                "val_mse": val_mse,
            }, sort_keys=True))

    agent.load_actor_state(best_state)
    return best


def load_real_transitions(agent, path):
    df = pd.read_excel(path, index_col=[0])
    c = np.zeros(agent.c_dim, dtype=np.float32)
    added = 0
    for i in range(max(0, df.shape[0] - 6)):
        seq = []
        valid = True
        for j in range(agent.time_steps):
            row = df.iloc[i + j]
            try:
                seq.append([float(row.iloc[1]), float(row.iloc[4]), float(row.iloc[5])])
                if float(row.iloc[7]) == 1.0 and j < agent.time_steps - 1:
                    valid = False
                    break
            except Exception:
                valid = False
                break
        if not valid or len(seq) != agent.time_steps:
            continue

        try:
            action = float(df.iloc[i + 6].iloc[1]) - float(df.iloc[i + 5].iloc[1])
            reward = float(df.iloc[i + 6].iloc[3]) * float(df.iloc[i + 5].iloc[1])
            next_seq = seq[1:] + [[
                float(df.iloc[i + 6].iloc[1]),
                float(df.iloc[i + 6].iloc[4]),
                float(df.iloc[i + 6].iloc[5]),
            ]]
            done = float(df.iloc[i + 6].iloc[7])
        except Exception:
            continue

        action = float(np.clip(action, -80.0, 80.0))
        agent.add(np.asarray(seq), c, action, reward, np.asarray(next_seq), c, done)
        added += 1

    print(json.dumps({
        "stage": "real_replay",
        "rows": int(df.shape[0]),
        "transitions_added": added,
        "columns": [str(x) for x in df.columns],
    }, sort_keys=True))
    return added


def make_simulation(price, sale_path, count_path):
    """Create the simulator and normalize only legacy TF-era scalar quirks.

    The modern environment.Simulation is already scalar-safe. The compatibility
    branch below is retained only if someone swaps back to the legacy simulator.
    """
    sim = Simulation(price, str(sale_path), str(count_path))
    if hasattr(sim.env, "beta_est"):
        beta = float(np.asarray(sim.env.beta_est).reshape(-1)[0])
        if not np.isfinite(beta) or beta <= 0:
            beta = 8.0
        sim.env.beta_est = beta
        if hasattr(sim.env, "popt"):
            sim.env.popt = np.asarray(sim.env.popt, dtype=np.float64)
    return sim


def sim_training(agent, sim, steps, warmup, batch_size, updates_per_step, noise_std):
    c = np.zeros(agent.c_dim, dtype=np.float32)
    state = np.zeros((agent.time_steps - 1, 3), dtype=np.float32)
    state = np.concatenate([state, np.asarray(sim.get_sample(), dtype=np.float32)[None, :]], axis=0)
    metrics = []

    for i in range(steps):
        origin = state.copy()
        if i < warmup:
            action = random.uniform(-80.0, 80.0)
        else:
            action = agent.act(origin, c, noise_std=noise_std)

        reward, sample, done = sim.step(action)
        sample = np.asarray(sample, dtype=np.float32)
        next_state = np.concatenate([origin[1:], sample[None, :]], axis=0)
        agent.add(origin, c, action, float(reward), next_state, c, float(done))
        state = np.zeros_like(next_state) if done else next_state

        last = None
        if len(agent.replay) >= batch_size:
            for _ in range(updates_per_step):
                last = agent.update(batch_size)
        if last:
            metrics.append(last)

        if i == 0 or (i + 1) % 500 == 0 or i + 1 == steps:
            recent = metrics[-100:] if metrics else []
            rec = {
                "stage": "ddpg",
                "step": i + 1,
                "replay": len(agent.replay),
                "epsilon_noise": noise_std,
            }
            if recent:
                for k in recent[0]:
                    rec[k] = float(np.mean([m[k] for m in recent]))
            print(json.dumps(rec, sort_keys=True))


def evaluate_policy(agent, sim, episodes=20):
    c = np.zeros(agent.c_dim, dtype=np.float32)
    rewards = []
    for ep in range(episodes):
        state = np.zeros((agent.time_steps - 1, 3), dtype=np.float32)
        state = np.concatenate([state, np.asarray(sim.get_sample(), dtype=np.float32)[None, :]], axis=0)
        total = 0.0
        for _ in range(200):
            action = agent.act(state, c)
            reward, sample, done = sim.step(action)
            total += float(reward)
            sample = np.asarray(sample, dtype=np.float32)
            state = np.concatenate([state[1:], sample[None, :]], axis=0)
            if done:
                break
        rewards.append(total)
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../RL-DDPG/data")
    ap.add_argument("--output", default="ddpg_actor_critic.pt")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pretrain-epochs", type=int, default=40)
    ap.add_argument("--sim-steps", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--updates-per-step", type=int, default=1)
    ap.add_argument("--noise-std", type=float, default=5.0)
    args = ap.parse_args()

    seed_all(args.seed)
    data_dir = Path(args.data_dir)

    agent = DDPGAgent(device=args.device)
    best_pretrain = supervised_pretrain(
        agent,
        data_dir,
        args.pretrain_epochs,
        256,
        1e-3,
        args.seed,
    )

    real_added = load_real_transitions(
        agent,
        data_dir / "YIK_QZH_training_data_source.xlsx",
    )

    sim = make_simulation(
        2500,
        data_dir / "2hours_price_setting_env.csv",
        data_dir / "num_count.pkl",
    )
    sim_training(
        agent,
        sim,
        steps=args.sim_steps,
        warmup=args.warmup,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
        noise_std=args.noise_std,
    )

    # A short deterministic policy evaluation on the same fitted simulator.
    eval_sim = make_simulation(
        2500,
        data_dir / "2hours_price_setting_env.csv",
        data_dir / "num_count.pkl",
    )
    rewards = evaluate_policy(agent, eval_sim, episodes=20)
    summary = {
        "pretrain_best_mse": best_pretrain,
        "real_transitions": real_added,
        "replay_size": len(agent.replay),
        "eval_mean_reward": float(np.mean(rewards)),
        "eval_std_reward": float(np.std(rewards)),
        "eval_min_reward": float(np.min(rewards)),
        "eval_max_reward": float(np.max(rewards)),
        "seed": args.seed,
    }
    print(json.dumps({"stage": "summary", **summary}, sort_keys=True))

    torch.save(agent.checkpoint(summary), args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
