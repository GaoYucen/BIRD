import argparse
import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RL-DDPG"))

from DP import AirPrice
import RL
from chasing_pytorch import TorchActor, decision_table, expert


BASE_STRATEGIES = ["Expert", "Decision Table", "DP", "DDPG"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@lru_cache(maxsize=8)
def _dp_models(inventory_with_buffer):
    base_price = np.array([2000, 2500, 3000, 3500, 4000], dtype=float)
    return tuple(
        AirPrice(
            real_min_demand_level=base_price[i] * 0.5,
            real_max_demand_level=base_price[i] * 1.5,
            max_days=14,
            num_tickets=float(inventory_with_buffer),
        )
        for i in range(5)
    )


def generate_base_trajectories(actor, seed=10, T=5000, inventory=200):
    """Simulate every base strategy on the same full-information buyer sequence.

    Crucially, inventory_before[gamma, t] stores s_t^gamma, which Algorithm 1
    requires DChasing to compare against BIRD's current inventory.
    """
    seed_all(seed)
    T = int(T)
    N_total = 5
    m = 5
    Gamma = 5

    initial_inventory = np.full(N_total, inventory + m, dtype=float)
    t_a = np.arange(0, T / 2, T / 10)
    t_e = np.arange(T / 5, T / 10 * 7, T / 10)
    T_end = min(T, int(T / 10 * 6) + 1)

    active_count = np.zeros(T, dtype=int)
    for i in range(T):
        active_count[i] = int(np.sum((t_a <= i) & (i < t_e)))
    N_active = int(active_count.max())

    p = np.zeros((Gamma, T, N_total), dtype=float)
    buy = np.zeros((Gamma, T, N_total), dtype=float)
    profit = np.zeros((Gamma, T, N_total), dtype=float)
    inventory_before = np.zeros((Gamma, T, N_total), dtype=float)

    base_price = np.array([2000, 2500, 3000, 3500, 4000], dtype=float)
    v = np.zeros((T, N_total), dtype=float)
    for i in range(T_end):
        for j in range(N_total):
            v[i, j] = random.random() * base_price[j] + base_price[j] * 0.5

    expert_base = base_price.copy()
    expert_range = base_price / 40
    expert_upper = base_price * 1.5
    expert_lower = base_price * 0.5

    dp_models = _dp_models(int(inventory + m))
    fix_rate_dp = T / 5 / 14
    round_rl = 14
    fix_rate_rl = np.array([(t_e[i] - t_a[i]) / round_rl for i in range(N_total)]).astype(int)
    c_rl = np.zeros(128, dtype=float)
    input_array = np.zeros((5, 3), dtype=float)

    for gamma in range(1, Gamma):
        c_remaining = initial_inventory.copy()
        volume = np.zeros(N_total)
        volume_round = np.zeros(N_total)

        for i in range(T):
            inventory_before[gamma, i] = c_remaining
            if i >= T_end:
                continue

            for j in range(N_total):
                active = (i >= t_a[j]) and (i < t_e[j])

                if gamma == 1:
                    expert_base[j] = expert(expert_base[j], expert_range[j])
                    expert_base[j] = min(max(expert_base[j], expert_lower[j]), expert_upper[j])
                    p[gamma, i, j] = expert_base[j]

                elif gamma == 2:
                    if i == t_a[j] or i == t_a[j] + 1:
                        p[gamma, i, j] = 2000 + 500 * j
                    elif active:
                        p[gamma, i, j] = decision_table(
                            volume[j],
                            volume_round[j],
                            initial_inventory[j],
                            i - t_a[j],
                            T / 5,
                            p[gamma, i - 1, j],
                        )

                elif gamma == 3 and active:
                    days_left = max(1, 14 - int((i - t_a[j]) / fix_rate_dp))
                    p[gamma, i, j] = dp_models[j].get_price(days_left, c_remaining[j])

                elif gamma == 4 and active:
                    if i == t_a[j]:
                        p[gamma, i, j] = base_price[j]
                    elif (i - t_a[j]) % max(1, fix_rate_rl[j]) == 0:
                        start_index = max(0, i - fix_rate_rl[j])
                        volume_rl = buy[gamma, start_index:i, j].sum()
                        remaining_round = max(
                            0,
                            round_rl - int((i - t_a[j]) / max(1, fix_rate_rl[j])),
                        )
                        p[gamma, i, j] = RL.get_price(
                            volume_rl,
                            remaining_round,
                            initial_inventory[j],
                            p[gamma, i - 1, j],
                            base_price[j] * 1.5,
                            base_price[j] * 0.5,
                            round_rl,
                        )
                        input_array[:, 0] = p[gamma, i - 1, j]
                        input_array[:, 1] = volume_rl
                        input_array[:, 2] = remaining_round
                        p[gamma, i, j] += actor.action(input_array, c_rl)
                    else:
                        p[gamma, i, j] = p[gamma, i - 1, j]

                if active and c_remaining[j] >= m:
                    sold = 0
                    for k in range(m):
                        if math.pow(0.95, k) * v[i, j] >= p[gamma, i, j]:
                            sold = k + 1
                    buy[gamma, i, j] = sold
                    c_remaining[j] -= sold
                    profit[gamma, i, j] = sold * p[gamma, i, j]
                    if gamma == 2:
                        volume_round[j] = volume[j]
                        volume[j] += sold

    return {
        "T": T,
        "m": m,
        "N_total": N_total,
        "N_active": N_active,
        "initial_inventory": initial_inventory,
        "t_a": t_a,
        "t_e": t_e,
        "p": p,
        "buy": buy,
        "profit": profit,
        "inventory_before": inventory_before,
        "valuation": v,
        "base_price": base_price,
    }


class PersistentFTPL:
    """Full-information FTPL for rewards in [0,1].

    Kalai-Vempala FPL perturbs cumulative expert totals and follows the
    perturbed leader. Reusing one perturbation vector across rounds preserves
    the marginal FPL distribution and avoids artificial high-frequency
    switching, matching the switching-cost role used by BIRD's reduction.
    """

    def __init__(self, num_experts, horizon, seed, eta_multiplier=1.0):
        self.num_experts = int(num_experts)
        self.horizon = max(1, int(horizon))
        self.eta = float(eta_multiplier) * math.sqrt(
            math.log(max(2, self.num_experts)) / self.horizon
        )
        rng = np.random.default_rng(seed)
        self.perturb = rng.exponential(scale=1.0 / max(self.eta, 1e-12), size=self.num_experts)
        self.cumulative = np.zeros(self.num_experts, dtype=float)

    def select(self):
        return int(np.argmax(self.cumulative + self.perturb))

    def update(self, reward_vector):
        self.cumulative += np.asarray(reward_vector, dtype=float)


class FollowTheLeader:
    def __init__(self, num_experts):
        self.cumulative = np.zeros(int(num_experts), dtype=float)

    def select(self):
        return int(np.argmax(self.cumulative))

    def update(self, reward_vector):
        self.cumulative += np.asarray(reward_vector, dtype=float)


def run_theory_bird(
    actor,
    seed=10,
    T=5000,
    inventory=200,
    selector="ftpl",
    eta_multiplier=1.0,
    epsilon_override=None,
    base=None,
):
    if base is None:
        base = generate_base_trajectories(actor, seed=seed, T=T, inventory=inventory)

    T = base["T"]
    m = base["m"]
    N_total = base["N_total"]
    N_active = base["N_active"]
    initial_inventory = base["initial_inventory"]
    t_a = base["t_a"]
    t_e = base["t_e"]
    p = base["p"]
    profit = base["profit"]
    inventory_before = base["inventory_before"]
    v = base["valuation"]
    base_price = base["base_price"]

    C = float(initial_inventory.max())

    # The synthetic experiment is category-separable and each buyer can buy at
    # most m units per active category, so it is the independent-goods + K_j
    # case of Theorem 3: epsilon = sqrt(C*N/|B|).
    epsilon_theory = math.sqrt(C * N_active / T)
    epsilon = epsilon_theory if epsilon_override is None else float(epsilon_override)
    epsilon = float(np.clip(epsilon, 0.0, 1.0))

    if selector == "ftpl":
        alg = PersistentFTPL(
            4,
            T,
            seed=seed + 1_000_003,
            eta_multiplier=eta_multiplier,
        )
        eta = alg.eta
    elif selector == "ftl":
        alg = FollowTheLeader(4)
        eta = 0.0
    else:
        raise ValueError(f"unknown selector={selector}")

    # A fixed bound preserves the reward objective while mapping R_j(gamma)
    # into [0,1], as required by the OLSC reduction.
    reward_bound = float(base_price.max() * 1.5 * m * N_active)
    max_observed_base_reward = float(profit[1:5].sum(axis=2).max())
    if max_observed_base_reward > reward_bound + 1e-6:
        raise RuntimeError(
            f"reward bound violated: observed={max_observed_base_reward} bound={reward_bound}"
        )

    chase_rng = np.random.default_rng(seed + 2_000_003)
    c_bird = initial_inventory.copy()
    bird_profit = np.zeros((T, N_total), dtype=float)
    bird_buy = np.zeros((T, N_total), dtype=float)
    target_history = np.zeros(T, dtype=int)
    distance_history = np.zeros(T, dtype=float)
    random_nosell_steps = 0
    mismatch_nosell_category_steps = 0
    switches = 0
    prev_target = None

    for i in range(T):
        target_local = alg.select()
        target_gamma = target_local + 1
        target_history[i] = target_gamma

        if prev_target is not None and target_gamma != prev_target:
            switches += 1
        prev_target = target_gamma

        target_state = inventory_before[target_gamma, i]
        distance_history[i] = float(np.maximum(target_state - c_bird, 0.0).sum())

        epsilon_nosell = bool(chase_rng.random() <= epsilon)
        if epsilon_nosell:
            random_nosell_steps += 1

        for j in range(N_total):
            active = (i >= t_a[j]) and (i < t_e[j])
            if not active or c_bird[j] < m:
                continue

            # Algorithm 1's all-1 normalized price is a no-purchase action.
            # In the raw experimental price scale, implement its semantics
            # directly instead of using the numeric value 1.
            if epsilon_nosell:
                continue

            if c_bird[j] < target_state[j]:
                mismatch_nosell_category_steps += 1
                continue

            price = p[target_gamma, i, j]
            sold = 0
            for k in range(m):
                if math.pow(0.95, k) * v[i, j] >= price:
                    sold = k + 1
            bird_buy[i, j] = sold
            c_bird[j] -= sold
            bird_profit[i, j] = sold * price

        # Algorithm 2, lines 10-12: every buyer reveals full-information
        # rewards for every base strategy, even if DChasing chose no-purchase.
        reward_vector = profit[1:5, i, :].sum(axis=1) / reward_bound
        alg.update(reward_vector)

    base_revenue = {
        BASE_STRATEGIES[g - 1]: float(profit[g].sum())
        for g in range(1, 5)
    }
    bird_revenue = float(bird_profit.sum())
    fixed_best = max(base_revenue, key=base_revenue.get)
    best_revenue = base_revenue[fixed_best]
    loss_pct = (best_revenue - bird_revenue) / best_revenue * 100.0

    target_counts = {
        BASE_STRATEGIES[g - 1]: int(np.sum(target_history == g))
        for g in range(1, 5)
    }

    return {
        "seed": int(seed),
        "T": int(T),
        "inventory": int(inventory),
        "C": C,
        "N_active": N_active,
        "epsilon_theory": epsilon_theory,
        "epsilon": epsilon,
        "selector": selector,
        "eta_multiplier": float(eta_multiplier),
        "eta": float(eta),
        "reward_bound": reward_bound,
        "max_observed_base_reward": max_observed_base_reward,
        "base_revenue": base_revenue,
        "BIRD": bird_revenue,
        "fixed_best": fixed_best,
        "best_fixed_revenue": best_revenue,
        "revenue_loss_pct": float(loss_pct),
        "switches": int(switches),
        "random_nosell_steps": int(random_nosell_steps),
        "mismatch_nosell_category_steps": int(mismatch_nosell_category_steps),
        "max_chasing_distance": float(distance_history.max()),
        "final_chasing_distance": float(distance_history[-1]),
        "target_counts": target_counts,
        "remaining_inventory": [float(x) for x in c_bird],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default="artifacts/ddpg_actor_critic.pt")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--inventory", type=int, default=200)
    ap.add_argument("--selector", choices=["ftpl", "ftl"], default="ftpl")
    ap.add_argument("--eta-multiplier", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=None)
    args = ap.parse_args()

    actor = TorchActor(args.actor)
    result = run_theory_bird(
        actor,
        seed=args.seed,
        T=args.T,
        inventory=args.inventory,
        selector=args.selector,
        eta_multiplier=args.eta_multiplier,
        epsilon_override=args.epsilon,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
