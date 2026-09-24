import argparse
import json
import math
from pathlib import Path

import numpy as np

from chasing_pytorch import TorchActor
from chasing_theory import BASE_STRATEGIES, generate_base_trajectories


class FTLSelector:
    def __init__(self, k):
        self.cum_reward = np.zeros(int(k), dtype=float)

    def select(self):
        return int(np.argmax(self.cum_reward))

    def update(self, reward):
        self.cum_reward += np.asarray(reward, dtype=float)


class KVFPLSelector:
    """Finite-expert FPL with a switching-aware parameter from the KV bound.

    For one-hot decisions over K experts:
      D = max ||d-d'||_1 = 2,
      A <= K for cost vectors in [0,1]^K,
      R <= 1.
    Adding a per-switch upper bound Delta to the KV FPL analysis gives the
    conservative upper bound

      eta * A*T*(R+Delta) + D/eta.

    We choose the minimizer eta=sqrt(D/(A*T*(R+Delta))).

    Kalai--Vempala show that using one perturbation vector for all rounds has
    the same per-round expectation as resampling. We use that coupled form,
    which is also the relevant low-switch interpretation for OLSC.
    """

    def __init__(self, k, horizon, delta, seed):
        self.k = int(k)
        self.horizon = int(horizon)
        self.delta = float(delta)
        self.D = 2.0
        self.A = float(self.k)
        self.R = 1.0
        self.eta = math.sqrt(
            self.D / (self.A * self.horizon * (self.R + self.delta))
        )
        rng = np.random.default_rng(seed)
        self.perturb = rng.uniform(0.0, 1.0 / self.eta, size=self.k)
        self.cum_cost = np.zeros(self.k, dtype=float)

    def select(self):
        return int(np.argmin(self.cum_cost + self.perturb))

    def update(self, reward):
        reward = np.asarray(reward, dtype=float)
        self.cum_cost += 1.0 - reward


class OracleFixedSelector:
    """Hindsight best fixed strategy; diagnostic only, never an online baseline."""

    def __init__(self, best_local):
        self.best_local = int(best_local)

    def select(self):
        return self.best_local

    def update(self, reward):
        pass


def _sell_units(valuation, price, m):
    sold = 0
    for k in range(m):
        if math.pow(0.95, k) * valuation >= price:
            sold = k + 1
    return sold


def run_bird_restart(
    actor,
    seed=0,
    T=5000,
    inventory=200,
    selector="kv-fpl",
    epsilon_override=None,
):
    base = generate_base_trajectories(
        actor, seed=seed, T=T, inventory=inventory
    )
    T = int(base["T"])
    m = int(base["m"])
    N_total = int(base["N_total"])
    N_active = int(base["N_active"])
    initial_inventory = base["initial_inventory"].copy()
    t_a = base["t_a"]
    t_e = base["t_e"]
    prices = base["p"]
    profits = base["profit"]
    valuation = base["valuation"]
    base_price = base["base_price"]

    C = float(initial_inventory.max())
    epsilon_theory = math.sqrt(C * N_active / T)
    epsilon = epsilon_theory if epsilon_override is None else float(epsilon_override)
    epsilon = float(np.clip(epsilon, 0.0, 1.0))

    # Exact expression from Theorem 3's proof at the chosen epsilon:
    # sigma <= epsilon*T + C*N/epsilon = 2*sqrt(C*N*T).
    sigma = epsilon_theory * T + C * N_active / epsilon_theory
    delta = sigma

    reward_bound = float(base_price.max() * 1.5 * m * N_active)
    max_observed = float(profits[1:5].sum(axis=2).max())
    if max_observed > reward_bound + 1e-9:
        raise RuntimeError(
            f"reward bound violated: observed={max_observed}, bound={reward_bound}"
        )

    base_revenue_vec = profits[1:5].sum(axis=(1, 2))
    best_local = int(np.argmax(base_revenue_vec))

    if selector == "kv-fpl":
        alg = KVFPLSelector(4, T, delta, seed + 3_000_003)
        selector_eta = alg.eta
    elif selector == "ftl":
        alg = FTLSelector(4)
        selector_eta = None
    elif selector == "oracle-fixed":
        alg = OracleFixedSelector(best_local)
        selector_eta = None
    else:
        raise ValueError(selector)

    rng = np.random.default_rng(seed + 4_000_003)
    bird_inventory = initial_inventory.copy()

    # DChasing target state. Algorithm 2 restarts this state from BIRD's
    # current state whenever the selector changes target strategy.
    target_inventory = bird_inventory.copy()
    target_gamma = None
    previous_gamma = None

    bird_profit = np.zeros((T, N_total), dtype=float)
    target_history = np.zeros(T, dtype=int)
    switch_count = 0
    restart_count = 0
    epsilon_nosell_steps = 0
    missing_steps = 0
    max_distance = 0.0

    for i in range(T):
        local = alg.select()
        gamma = local + 1
        target_history[i] = gamma

        if target_gamma is None or gamma != target_gamma:
            if target_gamma is not None:
                switch_count += 1
            restart_count += 1
            target_gamma = gamma
            # Algorithm 2, lines 4-6: invoke DChasing from scratch,
            # initialized at our current state s_j.
            target_inventory = bird_inventory.copy()

        distance = float(np.maximum(target_inventory - bird_inventory, 0.0).sum())
        max_distance = max(max_distance, distance)

        epsilon_nosell = bool(rng.random() <= epsilon)
        if epsilon_nosell:
            epsilon_nosell_steps += 1

        # Prices of each base strategy are the full-information action sequence
        # generated for this buyer sequence. Restart affects the target state,
        # which is the inventory state used by DChasing.
        target_price_vec = prices[gamma, i]

        for j in range(N_total):
            active = (i >= t_a[j]) and (i < t_e[j])
            if not active:
                continue

            target_before = target_inventory[j]

            # BIRD action according to Algorithm 1.
            if bird_inventory[j] >= m and not epsilon_nosell:
                if bird_inventory[j] >= target_before:
                    q_bird = _sell_units(
                        valuation[i, j], target_price_vec[j], m
                    )
                    q_bird = min(q_bird, int(bird_inventory[j]))
                    bird_inventory[j] -= q_bird
                    bird_profit[i, j] = q_bird * target_price_vec[j]
                else:
                    # Missing step: normalized all-1 price means no purchase.
                    missing_steps += 1

            # Evolve the restarted target strategy state under its own action.
            # This is separate from BIRD's transition and is required to keep
            # the comparison state s_j^gamma for the active chasing episode.
            if target_inventory[j] >= m:
                q_target = _sell_units(
                    valuation[i, j], target_price_vec[j], m
                )
                q_target = min(q_target, int(target_inventory[j]))
                target_inventory[j] -= q_target

        # Full-information feedback after every buyer, including no-sale rounds.
        reward_vector = profits[1:5, i, :].sum(axis=1) / reward_bound
        alg.update(reward_vector)
        previous_gamma = gamma

    base_revenue = {
        BASE_STRATEGIES[k]: float(base_revenue_vec[k])
        for k in range(4)
    }
    bird_revenue = float(bird_profit.sum())
    best_name = BASE_STRATEGIES[best_local]
    best_revenue = float(base_revenue_vec[best_local])
    loss_pct = (best_revenue - bird_revenue) / best_revenue * 100.0

    target_counts = {
        BASE_STRATEGIES[g - 1]: int(np.sum(target_history == g))
        for g in range(1, 5)
    }

    return {
        "seed": int(seed),
        "selector": selector,
        "T": T,
        "inventory": int(inventory),
        "N_active": N_active,
        "C": C,
        "epsilon": epsilon,
        "epsilon_theory": epsilon_theory,
        "sigma": sigma,
        "delta": delta,
        "selector_eta": selector_eta,
        "reward_bound": reward_bound,
        "max_observed_base_reward": max_observed,
        "base_revenue": base_revenue,
        "best_fixed": best_name,
        "best_fixed_revenue": best_revenue,
        "BIRD": bird_revenue,
        "revenue_loss_pct": float(loss_pct),
        "switches": int(switch_count),
        "restarts": int(restart_count),
        "epsilon_nosell_steps": int(epsilon_nosell_steps),
        "missing_steps": int(missing_steps),
        "max_chasing_distance": float(max_distance),
        "target_counts": target_counts,
        "remaining_inventory": [float(x) for x in bird_inventory],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default="artifacts/ddpg_actor_critic.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selector", choices=["kv-fpl", "ftl", "oracle-fixed"], default="kv-fpl")
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--inventory", type=int, default=200)
    ap.add_argument("--epsilon", type=float, default=None)
    args = ap.parse_args()

    actor = TorchActor(args.actor)
    result = run_bird_restart(
        actor,
        seed=args.seed,
        T=args.T,
        inventory=args.inventory,
        selector=args.selector,
        epsilon_override=args.epsilon,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
