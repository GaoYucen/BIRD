import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RL-DDPG"))

from DP import AirPrice
import RL
from model import ActorNet


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def expert(base, expert_range):
    if random.random() < 0.5:
        return base + expert_range * random.random()
    return base - expert_range * random.random()


def decision_table(volume, volume_round, C, i, T, price):
    utilize = volume / C * 100
    utilize_round = (volume - volume_round) / C * 100
    fix_rate_up = T / 3

    if i < int(fix_rate_up):
        if (utilize_round < 50) and (utilize >= 50):
            price += 30
        elif (utilize_round < 70) and (utilize >= 70):
            price += 20
        elif (utilize_round < 80) and (utilize >= 80):
            price += 40
    elif i < int(fix_rate_up * 2):
        if (utilize_round < 60) and (utilize >= 60):
            price += 50
        elif (utilize_round < 70) and (utilize >= 70):
            price += 30
        elif (utilize_round < 80) and (utilize >= 80):
            price += 60
    else:
        if (utilize_round < 50) and (utilize >= 50):
            price += 30
        elif (utilize_round < 60) and (utilize >= 60):
            price += 40
        elif (utilize_round < 70) and (utilize >= 70):
            price += 50
        elif (utilize_round < 80) and (utilize >= 80):
            price += 60

    fix_rate_down = T / 28
    if i < int(T / 2):
        for j in range(14):
            if i == int(fix_rate_down * (j + 1)) and utilize < 10:
                price -= 90
    else:
        for j in range(14, 28):
            if i == int(fix_rate_down * (j + 1)) and utilize < 20:
                price -= 92
    return price


class TorchActor:
    def __init__(self, checkpoint, device="cpu"):
        ckpt = torch.load(checkpoint, map_location=device)
        self.device = torch.device(device)
        self.model = ActorNet(
            state_dim=ckpt.get("state_dim", 3),
            hidden_size=ckpt.get("hidden_size", 128),
            num_layers=ckpt.get("num_layers", 2),
            c_dim=ckpt.get("c_dim", 128),
            action_range=ckpt.get("action_range", 40.0),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def action(self, state, c):
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None, ...]
        cc = torch.as_tensor(c, dtype=torch.float32, device=self.device)[None, ...]
        with torch.no_grad():
            return float(self.model(x, cc).cpu().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default="actor_pretrained.pt")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--inventory", type=int, default=200)
    ap.add_argument("--legacy-epsilon-zero", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)

    T = args.T
    N = 5
    m = 5
    W = 2
    Gamma = 5

    c = np.full(N, args.inventory + m, dtype=float)
    t_a = np.arange(0, T / 2, T / 10)
    t_e = np.arange(T / 5, T / 10 * 7, T / 10)
    T_end = int(T / 10 * 6) + 1

    p = np.zeros((Gamma, T, N))
    buy = np.zeros((Gamma, T, N))
    profit = np.zeros((Gamma, T, N))
    c_Gamma = np.zeros((Gamma, N))
    t_Gamma = np.zeros((Gamma, N))
    for gamma in range(1, Gamma):
        c_Gamma[gamma] = c.copy()

    base_price = np.array([2000, 2500, 3000, 3500, 4000], dtype=float)
    v = np.zeros((T, N))
    for i in range(T_end):
        for j in range(N):
            v[i, j] = random.random() * base_price[j] + base_price[j] * 0.5

    base = base_price.copy()
    range_expert = base_price / 40
    expert_upper_bound = base_price * 1.5
    expert_lower_bound = base_price * 0.5

    DP = [
        AirPrice(
            real_min_demand_level=base_price[i] * 0.5,
            real_max_demand_level=base_price[i] * 1.5,
            max_days=14,
            num_tickets=c[i],
        )
        for i in range(N)
    ]
    fix_rate_DP = T / 5 / 14
    round_RL = 14
    fix_rate_RL = np.array([(t_e[i] - t_a[i]) / round_RL for i in range(N)]).astype(int)
    actor = TorchActor(args.actor)
    c_RL = np.zeros(128, dtype=float)
    input_array = np.zeros((5, 3), dtype=float)

    # Baseline strategy simulation.
    for gamma in range(1, Gamma):
        # These were accidentally re-created inside the innermost legacy loop.
        # Keep cumulative volumes here so the decision-table thresholds work as intended.
        volume = np.zeros(N)
        volume_round = np.zeros(N)

        for i in range(T_end):
            for j in range(N):
                if gamma == 1:
                    base[j] = expert(base[j], range_expert[j])
                    base[j] = min(max(base[j], expert_lower_bound[j]), expert_upper_bound[j])
                    p[gamma, i, j] = base[j]

                elif gamma == 2:
                    if i == t_a[j] or i == t_a[j] + 1:
                        p[gamma, i, j] = 2000 + 500 * j
                    elif (i >= t_a[j]) and (i < t_e[j]):
                        p[gamma, i, j] = decision_table(
                            volume[j],
                            volume_round[j],
                            c[j],
                            i - t_a[j],
                            T / 5,
                            p[gamma, i - 1, j],
                        )

                elif gamma == 3 and (i >= t_a[j]) and (i < t_e[j]):
                    days_left = max(1, 14 - int((i - t_a[j]) / fix_rate_DP))
                    p[gamma, i, j] = DP[j].get_price(days_left, c_Gamma[gamma, j])

                elif gamma == 4 and (i >= t_a[j]) and (i < t_e[j]):
                    if i == t_a[j]:
                        p[gamma, i, j] = base_price[j]
                    elif (i - t_a[j]) % max(1, fix_rate_RL[j]) == 0:
                        start_index = max(0, i - fix_rate_RL[j])
                        volume_RL = buy[gamma, start_index:i, j].sum()
                        p[gamma, i, j] = RL.get_price(
                            volume_RL,
                            max(0, round_RL - int((i - t_a[j]) / max(1, fix_rate_RL[j]))),
                            c[j],
                            p[gamma, i - 1, j],
                            base_price[j] * 1.5,
                            base_price[j] * 0.5,
                            round_RL,
                        )
                        input_array[:, 0] = p[gamma, i - 1, j]
                        input_array[:, 1] = volume_RL
                        input_array[:, 2] = max(
                            0,
                            round_RL - int((i - t_a[j]) / max(1, fix_rate_RL[j])),
                        )
                        p[gamma, i, j] += actor.action(input_array, c_RL)
                    else:
                        p[gamma, i, j] = p[gamma, i - 1, j]

                if (c_Gamma[gamma, j] >= m) and (i >= t_a[j]) and (i < t_e[j]):
                    sold = 0
                    for k in range(m):
                        if math.pow(0.95, k) * v[i, j] >= p[gamma, i, j]:
                            sold = k + 1
                    buy[gamma, i, j] = sold
                    c_Gamma[gamma, j] -= sold
                    profit[gamma, i, j] = sold * p[gamma, i, j]
                    if gamma == 2:
                        volume_round[j] = volume[j]
                        volume[j] += sold
                elif (t_Gamma[gamma, j] == 0) and (i >= t_a[j]) and (i < t_e[j]):
                    t_Gamma[gamma, j] = i

    max_inventory = float(c[0])
    epsilon = 0.0 if args.legacy_epsilon_zero else math.sqrt(max_inventory * W / T)
    p_chasing = np.zeros((T, N))
    c_chasing = c.copy()
    buy_chasing = np.zeros((T, N))
    profit_chasing = np.zeros((T, N))

    profit_gamma = np.zeros(Gamma)
    D = 2
    for i in range(1, T):
        if random.random() <= epsilon:
            p_chasing[i, :] = 1
            continue

        R_gamma = np.zeros(Gamma)
        for gamma in range(1, Gamma):
            for j in range(N):
                profit_gamma[gamma] += profit[gamma, i - 1, j]
                R_gamma[gamma] += profit[gamma, i - 1, j]
        R = max(0.05, np.max(R_gamma))

        profit_add_gamma = np.zeros(Gamma)
        for gamma in range(1, Gamma):
            for j in range(N):
                profit_add_gamma[gamma] += p[gamma, i, j] * m / math.sqrt(D / (R * R * T))

        select_gamma = int(np.argmax(profit_gamma + profit_add_gamma))
        for j in range(N):
            if c_chasing[j] >= c_Gamma[select_gamma, j]:
                p_chasing[i, j] = p[select_gamma, i, j]
            else:
                p_chasing[i, j] = 1
            if (c_chasing[j] >= m) and (i >= t_a[j]) and (i < t_e[j]):
                sold = 0
                for k in range(m):
                    if math.pow(0.95, k) * v[i, j] >= p_chasing[i, j]:
                        sold = k + 1
                buy_chasing[i, j] = sold
                c_chasing[j] -= sold
                profit_chasing[i, j] = sold * p_chasing[i, j]

    profit_all = profit.sum(axis=(1, 2))
    bird = float(profit_chasing.sum())

    print("seed", args.seed)
    print("T", T)
    print("epsilon", epsilon)
    print("Revenue of different pricing strategies:")
    print("expert pricing:", float(profit_all[1]))
    print("decision table pricing:", float(profit_all[2]))
    print("dynamic pricing:", float(profit_all[3]))
    print("PyTorch RL pricing:", float(profit_all[4]))
    print("BIRD:", bird)


if __name__ == "__main__":
    main()
