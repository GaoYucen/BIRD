import math

import numpy as np


class BaseSelector:
    def __init__(self, k):
        self.k = int(k)
        self.switches = 0
        self.last = None

    def _record(self, choice):
        choice = int(choice)
        if self.last is not None and choice != self.last:
            self.switches += 1
        self.last = choice
        return choice

    def select(self):
        raise NotImplementedError

    def update(self, reward):
        raise NotImplementedError


class FTLSelector(BaseSelector):
    def __init__(self, k):
        super().__init__(k)
        self.cum_cost = np.zeros(self.k, dtype=float)

    def select(self):
        return self._record(np.argmin(self.cum_cost))

    def update(self, reward):
        reward = np.asarray(reward, dtype=float)
        self.cum_cost += 1.0 - reward


class UniformFPLSelector(BaseSelector):
    """Kalai--Vempala additive FPL using fresh uniform cube perturbations."""

    def __init__(self, k, epsilon_sel, seed):
        super().__init__(k)
        self.epsilon_sel = float(epsilon_sel)
        self.spacing = 1.0 / max(self.epsilon_sel, 1e-15)
        self.rng = np.random.default_rng(seed)
        self.cum_cost = np.zeros(self.k, dtype=float)

    def select(self):
        p = self.rng.uniform(0.0, self.spacing, size=self.k)
        return self._record(np.argmin(self.cum_cost + p))

    def update(self, reward):
        reward = np.asarray(reward, dtype=float)
        self.cum_cost += 1.0 - reward


class FLLSelector(BaseSelector):
    """Exact Follow-the-Lazy-Leader grid coupling from Kalai--Vempala.

    Once choose a random grid offset p in [0,1/epsilon)^K. At time t, the
    perturbed cumulative cost g_{t-1} is the unique grid point in

        S_{1:t-1} + [0, 1/epsilon)^K.

    Componentwise this is
        p + ceil((S-p)/spacing) * spacing.

    Its marginal decision distribution equals uniform-cube FPL, while the grid
    point changes only when a cumulative-cost coordinate crosses a grid cell.
    """

    def __init__(self, k, epsilon_sel, seed):
        super().__init__(k)
        self.epsilon_sel = float(epsilon_sel)
        self.spacing = 1.0 / max(self.epsilon_sel, 1e-15)
        rng = np.random.default_rng(seed)
        self.offset = rng.uniform(0.0, self.spacing, size=self.k)
        self.cum_cost = np.zeros(self.k, dtype=float)
        self.last_grid = None
        self.grid_updates = 0

    def _grid_point(self):
        z = np.ceil((self.cum_cost - self.offset) / self.spacing)
        return self.offset + z * self.spacing

    def select(self):
        g = self._grid_point()
        if self.last_grid is None or not np.array_equal(g, self.last_grid):
            self.grid_updates += 1
            self.last_grid = g.copy()
        return self._record(np.argmin(g))

    def update(self, reward):
        reward = np.asarray(reward, dtype=float)
        self.cum_cost += 1.0 - reward


class OracleFixedSelector(BaseSelector):
    def __init__(self, best_local, k):
        super().__init__(k)
        self.best_local = int(best_local)

    def select(self):
        return self._record(self.best_local)

    def update(self, reward):
        return None


def theoretical_selector_epsilon(k, horizon, delta, multiplier=1.0):
    """Minimize the additive FLL+switching upper bound.

    KV Theorem 1.1(a): epsilon*R*A*T + D/epsilon.
    FLL switch probability <= epsilon*A per step, so with switching cost Delta:
        epsilon*A*T*(R+Delta) + D/epsilon.

    For one-hot expert decisions:
      D=2, R<=1, A<=K for dense loss vectors in [0,1]^K.
    """
    D = 2.0
    R = 1.0
    A = float(k)
    base = math.sqrt(D / (A * horizon * (R + float(delta))))
    return float(multiplier) * base



class FPLStarSelector(BaseSelector):
    """Multiplicative/expert FPL* with symmetric exponential perturbations."""

    def __init__(self, k, epsilon_sel, seed):
        super().__init__(k)
        self.epsilon_sel = float(epsilon_sel)
        self.rng = np.random.default_rng(seed)
        self.cum_cost = np.zeros(self.k, dtype=float)

    def _sample_p(self):
        mag = self.rng.exponential(
            scale=1.0 / max(self.epsilon_sel, 1e-15),
            size=self.k,
        )
        sign = self.rng.choice(np.asarray([-1.0, 1.0]), size=self.k)
        return sign * mag

    def select(self):
        p = self._sample_p()
        return self._record(np.argmin(self.cum_cost + p))

    def update(self, reward):
        reward = np.asarray(reward, dtype=float)
        self.cum_cost += 1.0 - reward


class FLLStarSelector(BaseSelector):
    """Exact FLL* coupling from Kalai--Vempala (2005), Section 3.1.

    p_t has density proportional to exp(-epsilon*||p||_1). After observing
    state/cost s_t:
      - with probability min(1, d(p_t-s_t)/d(p_t)), set p_{t+1}=p_t-s_t;
      - otherwise set p_{t+1}=-p_t.

    In the first case cumulative_cost + p is unchanged, so the oracle decision
    need not change. Marginally p_t keeps the same symmetric exponential law.
    """

    def __init__(self, k, epsilon_sel, seed):
        super().__init__(k)
        self.epsilon_sel = float(epsilon_sel)
        self.rng = np.random.default_rng(seed)
        self.cum_cost = np.zeros(self.k, dtype=float)
        mag = self.rng.exponential(
            scale=1.0 / max(self.epsilon_sel, 1e-15),
            size=self.k,
        )
        sign = self.rng.choice(np.asarray([-1.0, 1.0]), size=self.k)
        self.p = sign * mag
        self.lazy_keeps = 0
        self.lazy_resets = 0

    def select(self):
        return self._record(np.argmin(self.cum_cost + self.p))

    def update(self, reward):
        reward = np.asarray(reward, dtype=float)
        s = 1.0 - reward

        old_norm = float(np.abs(self.p).sum())
        candidate = self.p - s
        new_norm = float(np.abs(candidate).sum())
        log_ratio = -self.epsilon_sel * (new_norm - old_norm)
        prob = min(1.0, math.exp(min(0.0, log_ratio)))

        if self.rng.random() <= prob:
            self.p = candidate
            self.lazy_keeps += 1
        else:
            self.p = -self.p
            self.lazy_resets += 1

        self.cum_cost += s


def theoretical_star_epsilon(k, horizon, delta, multiplier=1.0):
    """Finite-horizon scale matching the expert FPL*/FLL* log(K) bound.

    A coarse switching-cost balance is
        epsilon*T*(1+Delta) + (1+ln K)/epsilon,
    giving epsilon = sqrt((1+ln K)/(T*(1+Delta))).
    Any fixed multiplier preserves the same asymptotic order and is exposed
    explicitly for finite-horizon sensitivity rather than hidden tuning.
    """
    base = math.sqrt(
        (1.0 + math.log(max(2, int(k))))
        / (float(horizon) * (1.0 + float(delta)))
    )
    return float(multiplier) * base
