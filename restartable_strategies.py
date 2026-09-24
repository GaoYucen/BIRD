import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pricing_dp import AirPrice
import pricing_rule as RL


BASE_NAMES = ["Expert", "Decision Table", "DP", "DDPG"]


@dataclass
class PairedContext:
    seed: int
    T: int
    inventory: int
    m: int
    N_total: int
    N_active: int
    initial_inventory: np.ndarray
    t_a: np.ndarray
    t_e: np.ndarray
    base_price: np.ndarray
    valuations: np.ndarray
    expert_noise: np.ndarray
    epsilon_mask: np.ndarray
    reward_bound: float
    epsilon: float
    epsilon_theory: float
    sigma: float


def make_context(seed=0, T=5000, inventory=200, epsilon_override=None):
    T = int(T)
    m = 5
    N_total = 5
    initial_inventory = np.full(N_total, inventory + m, dtype=float)
    t_a = np.arange(0, T / 2, T / 10)
    t_e = np.arange(T / 5, T / 10 * 7, T / 10)
    base_price = np.asarray([2000, 2500, 3000, 3500, 4000], dtype=float)

    active_count = np.asarray(
        [np.sum((t_a <= i) & (i < t_e)) for i in range(T)],
        dtype=int,
    )
    N_active = int(active_count.max())

    # Independent named RNG streams make paired comparisons stable even if a
    # later implementation adds unrelated random draws.
    val_rng = np.random.default_rng(seed + 101_003)
    expert_rng = np.random.default_rng(seed + 202_003)
    epsilon_rng = np.random.default_rng(seed + 303_003)

    valuations = np.zeros((T, N_total), dtype=float)
    T_end = min(T, int(T / 10 * 6) + 1)
    for i in range(T_end):
        valuations[i] = (
            val_rng.random(N_total) * base_price + 0.5 * base_price
        )

    # Original expert() is equivalent to signed Uniform[0,1] magnitude.
    expert_noise = expert_rng.uniform(-1.0, 1.0, size=(T, N_total))

    C = float(initial_inventory.max())
    epsilon_theory = float(np.clip(math.sqrt(C * N_active / T), 0.0, 1.0))
    epsilon = epsilon_theory if epsilon_override is None else float(epsilon_override)
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    epsilon_mask = epsilon_rng.random(T) <= epsilon

    # Theorem 3, independent-goods + K_j case.
    epsilon_for_bound = max(epsilon, 1e-12)
    sigma = float(epsilon_for_bound * T + C * N_active / epsilon_for_bound)

    # The theoretical expert reward R_j is in [0,1). Map one experimental
    # buyer reward into that scale using a deterministic valid upper bound.
    reward_bound = float(base_price.max() * 1.5 * m * N_active)

    return PairedContext(
        seed=int(seed),
        T=T,
        inventory=int(inventory),
        m=m,
        N_total=N_total,
        N_active=N_active,
        initial_inventory=initial_inventory,
        t_a=t_a,
        t_e=t_e,
        base_price=base_price,
        valuations=valuations,
        expert_noise=expert_noise,
        epsilon_mask=epsilon_mask,
        reward_bound=reward_bound,
        epsilon=epsilon,
        epsilon_theory=epsilon_theory,
        sigma=sigma,
    )


def active(ctx, t, j):
    return bool(t >= ctx.t_a[j] and t < ctx.t_e[j])


def sell_units(ctx, t, j, price, inventory):
    if not active(ctx, t, j) or inventory < ctx.m:
        return 0
    sold = 0
    v = ctx.valuations[t, j]
    for k in range(ctx.m):
        if math.pow(0.95, k) * v >= price:
            sold = k + 1
    return min(sold, int(inventory))


class RestartableStrategy:
    name = "base"

    def __init__(self, ctx, actor=None):
        self.ctx = ctx
        self.actor = actor
        self.inventory = ctx.initial_inventory.copy()
        self.last_price = ctx.base_price.copy()

    def reset(self, t, inventory, last_price=None):
        self.inventory = np.asarray(inventory, dtype=float).copy()
        if last_price is None:
            self.last_price = self.ctx.base_price.copy()
        else:
            self.last_price = np.asarray(last_price, dtype=float).copy()

    def prices(self, t):
        raise NotImplementedError

    def update(self, t, sold, price_vector):
        self.inventory -= np.asarray(sold, dtype=float)
        self.inventory = np.maximum(self.inventory, 0.0)
        self.last_price = np.asarray(price_vector, dtype=float).copy()


class ExpertStrategy(RestartableStrategy):
    name = "Expert"

    def reset(self, t, inventory, last_price=None):
        super().reset(t, inventory, last_price)
        self.price = self.last_price.copy()
        self.range = self.ctx.base_price / 40.0
        self.lower = self.ctx.base_price * 0.5
        self.upper = self.ctx.base_price * 1.5

    def prices(self, t):
        # Match the legacy expert random walk, but use pre-generated paired
        # noise so restarting another method does not change the random stream.
        self.price = np.clip(
            self.price + self.range * self.ctx.expert_noise[t],
            self.lower,
            self.upper,
        )
        return self.price.copy()

    def update(self, t, sold, price_vector):
        self.inventory -= np.asarray(sold, dtype=float)
        self.inventory = np.maximum(self.inventory, 0.0)
        self.last_price = np.asarray(price_vector, dtype=float).copy()


class DecisionTableStrategy(RestartableStrategy):
    name = "Decision Table"

    def reset(self, t, inventory, last_price=None):
        super().reset(t, inventory, last_price)
        self.capacity = self.ctx.initial_inventory.copy()
        # Reconstruct cumulative utilization from the current inventory state.
        self.volume = np.maximum(
            self.ctx.initial_inventory - self.inventory, 0.0
        )
        self.volume_before_last = self.volume.copy()
        self.price = self.last_price.copy()

    @staticmethod
    def _next_price(volume, volume_before_last, C, local_i, local_T, price):
        utilize = volume / max(C, 1e-12) * 100.0
        utilize_round = (volume - volume_before_last) / max(C, 1e-12) * 100.0
        fix_rate_up = local_T / 3.0

        if local_i < int(fix_rate_up):
            if utilize_round < 50 <= utilize:
                price += 30
            elif utilize_round < 70 <= utilize:
                price += 20
            elif utilize_round < 80 <= utilize:
                price += 40
        elif local_i < int(fix_rate_up * 2):
            if utilize_round < 60 <= utilize:
                price += 50
            elif utilize_round < 70 <= utilize:
                price += 30
            elif utilize_round < 80 <= utilize:
                price += 60
        else:
            if utilize_round < 50 <= utilize:
                price += 30
            elif utilize_round < 60 <= utilize:
                price += 40
            elif utilize_round < 70 <= utilize:
                price += 50
            elif utilize_round < 80 <= utilize:
                price += 60

        fix_rate_down = local_T / 28.0
        if local_i < int(local_T / 2):
            for z in range(14):
                if local_i == int(fix_rate_down * (z + 1)) and utilize < 10:
                    price -= 90
        else:
            for z in range(14, 28):
                if local_i == int(fix_rate_down * (z + 1)) and utilize < 20:
                    price -= 92
        return price

    def prices(self, t):
        for j in range(self.ctx.N_total):
            if not active(self.ctx, t, j):
                continue
            if t == int(self.ctx.t_a[j]):
                self.price[j] = self.ctx.base_price[j]
            else:
                local_i = int(t - self.ctx.t_a[j])
                self.price[j] = self._next_price(
                    self.volume[j],
                    self.volume_before_last[j],
                    self.capacity[j],
                    local_i,
                    self.ctx.T / 5.0,
                    self.price[j],
                )
        return self.price.copy()

    def update(self, t, sold, price_vector):
        sold = np.asarray(sold, dtype=float)
        self.volume_before_last = self.volume.copy()
        self.volume += sold
        super().update(t, sold, price_vector)


@lru_cache(maxsize=32)
def _cached_dp_models(inventory_tuple):
    base_price = np.asarray([2000, 2500, 3000, 3500, 4000], dtype=float)
    return tuple(
        AirPrice(
            real_min_demand_level=base_price[j] * 0.5,
            real_max_demand_level=base_price[j] * 1.5,
            max_days=14,
            num_tickets=float(inventory_tuple[j]),
        )
        for j in range(5)
    )


class DPStrategy(RestartableStrategy):
    name = "DP"

    def __init__(self, ctx, actor=None):
        super().__init__(ctx, actor)
        self.models = _cached_dp_models(
            tuple(float(x) for x in ctx.initial_inventory)
        )

    def reset(self, t, inventory, last_price=None):
        super().reset(t, inventory, last_price)
        self.price = self.last_price.copy()

    def prices(self, t):
        fix_rate = self.ctx.T / 5.0 / 14.0
        for j in range(self.ctx.N_total):
            if active(self.ctx, t, j):
                days_left = max(
                    1, 14 - int((t - self.ctx.t_a[j]) / fix_rate)
                )
                self.price[j] = self.models[j].get_price(
                    days_left, self.inventory[j]
                )
        return self.price.copy()


class DDPGStrategy(RestartableStrategy):
    name = "DDPG"

    def reset(self, t, inventory, last_price=None):
        super().reset(t, inventory, last_price)
        self.price = self.last_price.copy()
        self.round_rl = 14
        self.fix_rate = np.asarray(
            [
                max(
                    1,
                    int(
                        (self.ctx.t_e[j] - self.ctx.t_a[j])
                        / self.round_rl
                    ),
                )
                for j in range(self.ctx.N_total)
            ],
            dtype=int,
        )
        self.episode_capacity = self.ctx.initial_inventory.copy()
        self.volume_since_adjust = np.zeros(self.ctx.N_total, dtype=float)
        self.c_rl = np.zeros(128, dtype=float)
        self.input_array = np.zeros((5, 3), dtype=float)

    def prices(self, t):
        for j in range(self.ctx.N_total):
            if not active(self.ctx, t, j):
                continue
            if t == int(self.ctx.t_a[j]):
                self.price[j] = self.ctx.base_price[j]
                continue

            local = int(t - self.ctx.t_a[j])
            if local % self.fix_rate[j] == 0:
                remaining_round = max(
                    0, self.round_rl - int(local / self.fix_rate[j])
                )
                p0 = RL.get_price(
                    self.volume_since_adjust[j],
                    remaining_round,
                    self.episode_capacity[j],
                    self.price[j],
                    self.ctx.base_price[j] * 1.5,
                    self.ctx.base_price[j] * 0.5,
                    self.round_rl,
                )
                self.input_array[:, 0] = self.price[j]
                self.input_array[:, 1] = self.volume_since_adjust[j]
                self.input_array[:, 2] = remaining_round
                if self.actor is not None:
                    p0 += self.actor.action(self.input_array, self.c_rl)
                self.price[j] = p0
                self.volume_since_adjust[j] = 0.0
        return self.price.copy()

    def update(self, t, sold, price_vector):
        sold = np.asarray(sold, dtype=float)
        self.volume_since_adjust += sold
        super().update(t, sold, price_vector)


def make_strategy(name, ctx, actor=None):
    if name == "Expert":
        return ExpertStrategy(ctx, actor)
    if name == "Decision Table":
        return DecisionTableStrategy(ctx, actor)
    if name == "DP":
        return DPStrategy(ctx, actor)
    if name == "DDPG":
        return DDPGStrategy(ctx, actor)
    raise ValueError(name)


def simulate_base_strategy(name, ctx, actor=None):
    strategy = make_strategy(name, ctx, actor)
    strategy.reset(0, ctx.initial_inventory, ctx.base_price)
    prices = np.zeros((ctx.T, ctx.N_total), dtype=float)
    sold = np.zeros((ctx.T, ctx.N_total), dtype=float)
    profit = np.zeros((ctx.T, ctx.N_total), dtype=float)
    inventory_before = np.zeros((ctx.T, ctx.N_total), dtype=float)
    sold_out_index = np.full(ctx.N_total, np.nan, dtype=float)

    for t in range(ctx.T):
        inventory_before[t] = strategy.inventory
        p = strategy.prices(t)
        prices[t] = p
        q = np.zeros(ctx.N_total, dtype=float)
        for j in range(ctx.N_total):
            q[j] = sell_units(
                ctx, t, j, p[j], strategy.inventory[j]
            )
            profit[t, j] = q[j] * p[j]
        sold[t] = q
        strategy.update(t, q, p)
        for j in range(ctx.N_total):
            if np.isnan(sold_out_index[j]) and strategy.inventory[j] < ctx.m:
                sold_out_index[j] = float(t)

    return {
        "name": name,
        "prices": prices,
        "sold": sold,
        "profit": profit,
        "inventory_before": inventory_before,
        "sold_out_index": sold_out_index,
        "category_revenue": profit.sum(axis=0),
        "revenue": float(profit.sum()),
    }


def generate_base_feedback(ctx, actor=None):
    trajectories = {
        name: simulate_base_strategy(name, ctx, actor)
        for name in BASE_NAMES
    }
    rewards = np.stack(
        [
            trajectories[name]["profit"].sum(axis=1)
            / ctx.reward_bound
            for name in BASE_NAMES
        ],
        axis=1,
    )
    if rewards.max() > 1.0 + 1e-8:
        raise RuntimeError(
            f"normalized reward bound violated: max={rewards.max()}"
        )
    return trajectories, rewards
