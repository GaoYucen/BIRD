import pickle

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.optimize import curve_fit


class DemandEnvironment:
    """Modern scalar-safe version of the legacy SimEnv.Environment."""

    def __init__(self, path, distribution="gamma"):
        self.path = path
        self.distribution = distribution
        self.df, self.start = self.read_data(path)
        self.price = self.df["PRICE"].unique()
        self.p = float(self.df["binom"].mean())
        self.popt, _ = curve_fit(
            self.log_func,
            np.asarray(self.df.loc[self.df["NUM"] != 0, "PRICE"], dtype=float),
            np.asarray(self.df.loc[self.df["NUM"] != 0, "NUM"], dtype=float),
            maxfev=10000,
        )
        self.beta = self.fit_beta()

    @staticmethod
    def read_data(path):
        df = pd.read_csv(path)
        df = df[(df["POL_CDE"] == "YIK") & (df["POD_CDE"] == "QZH")].head(300).copy()
        df["WBL_AUD_DTDL"] = pd.to_datetime(df["WBL_AUD_DTDL"])
        df.sort_values(by="WBL_AUD_DTDL", inplace=True)
        start = df["WBL_AUD_DTDL"].iloc[0]
        df["binom"] = (df["NUM"] != 0).astype(float)
        return df, start

    @staticmethod
    def log_func(x, a, b, c):
        with np.errstate(divide="ignore", invalid="ignore"):
            return a * np.log(b * x) + c

    @staticmethod
    def safe_log_func(x, a, b, c):
        z = b * x + 1e-7
        if z <= 0:
            return 1e-10
        y = a * np.log(z) + c
        return float(y) if np.isfinite(y) and y > 0 else 1e-10

    def neg_l_gamma(self, beta_like):
        beta = float(np.asarray(beta_like).reshape(-1)[0])
        if not np.isfinite(beta) or beta <= 1e-8:
            return 1e100
        total = 0.0
        for price in self.price:
            mean_sale = self.safe_log_func(
                float(price), self.popt[0], self.popt[1], self.popt[2]
            )
            alpha = max(beta * mean_sale, 1e-8)
            df2 = self.df.loc[
                self.df["PRICE"] == price, ["WBL_AUD_DTDL", "NUM"]
            ].reset_index(drop=True)
            for i in range(len(df2)):
                num = max(float(df2.iloc[i]["NUM"]), 1e-8)
                gap = (df2.iloc[i]["WBL_AUD_DTDL"] - self.start) / np.timedelta64(1, "D")
                lp = stats.gamma.logpdf(num, alpha, 0.999, 1.0 / beta)
                if np.isfinite(lp):
                    total += float(lp) / (float(gap) + 1.0)
        return -total if np.isfinite(total) else 1e100

    def fit_beta(self):
        if self.distribution != "gamma":
            raise NotImplementedError("modern reproduction currently uses the paper's gamma simulator")
        result = optimize.fmin(
            func=self.neg_l_gamma,
            x0=np.asarray([8.0]),
            maxfun=500,
            disp=False,
        )
        beta = float(np.asarray(result).reshape(-1)[0])
        if not np.isfinite(beta) or beta <= 1e-8:
            beta = 8.0
        print(f"demand_beta={beta:.6f} demand_nonzero_p={self.p:.6f}")
        return beta

    def sale(self, cur_price):
        if int(np.random.binomial(1, self.p)) == 0:
            return 0.0
        mean_sale = self.safe_log_func(
            float(cur_price), self.popt[0], self.popt[1], self.popt[2]
        )
        alpha = max(self.beta * mean_sale, 1e-8)
        num = float(np.random.gamma(shape=alpha, scale=1.0 / self.beta) + 1.0)
        return float(min(round(num), 50.0))

    def ave_sale(self, cur_price, samples=50):
        return float(np.mean([self.sale(cur_price) for _ in range(samples)]))


class Simulation:
    """Modern scalar-safe version of sample_gen.Simulation.step()."""

    def __init__(self, price, path, wbl_path):
        self.PRICE = float(price)
        self.price = float(price)
        self.util = 0.0
        self.remaining_time = 70.0
        self.max_util = 200.0
        self.env = DemandEnvironment(path, "gamma")
        with open(wbl_path, "rb") as f:
            self.cnt_distribution = pickle.load(f)

    def get_sample(self):
        return np.asarray([self.price, self.util, self.remaining_time], dtype=np.float32)

    def step(self, action):
        self.price = max(self.price + float(action), 0.0)
        sale = self.env.ave_sale(self.price)
        sale = min(self.max_util - self.util, sale)
        self.util += sale
        self.remaining_time -= 1.0
        reward = self.price * sale

        done = bool(self.remaining_time <= 0 or self.util >= self.max_util)
        sample = np.asarray(
            [self.price, self.util, self.remaining_time], dtype=np.float32
        )
        if done:
            self.reset()
        return float(reward), sample, int(done)

    def reset(self):
        self.util = 0.0
        self.price = self.PRICE
        self.remaining_time = 70.0
