import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import optimize,stats
from math import e
import datetime
from random import choice


class Environment:
    def __init__(self, path, distribution):
        '''

        :param path:
        :param distribution: gamma, log or chi2
        '''
        self.path = path
        self.distribution = distribution
        self.df, self.start = self.read_data(path)
        self.price = self.df['PRICE'].unique()
        self.p = self.binom()
        self.beta_est = self.fit()

    def read_data(self, path):
        df = pd.read_csv(path)
        df = df[df['POL_CDE'] == 'YIK']
        df = df[df['POD_CDE'] == 'QZH']
        df = df.head(300)
        df['WBL_AUD_DTDL'] = pd.to_datetime(df['WBL_AUD_DTDL'])
        df.sort_values(by='WBL_AUD_DTDL', inplace=True)  # 按照数据生成时间排序
        start = df['WBL_AUD_DTDL'][0]

        # 伯努利分布
        df.loc[df['NUM'] == 0, 'binom'] = 0
        df.loc[df['NUM'] != 0, 'binom'] = 1

        return df, start

    def binom(self):
        p = self.df['binom'].sum() / len(self.df['binom'])
        return p

    def fit(self):
        df2 = self.df.loc[self.df['NUM'] != 0, :]
        # 拟合需求价格函数（对数形式）
        # 定义x、y散点坐标
        x = df2['PRICE']
        x = np.array(x)
        num = df2['NUM']
        y = np.array(num)

        # 非线性最小二乘法拟合
        self.popt, self.pcov = curve_fit(self.log_func, x, y)
        '''
        Gamma distribution
        '''
        if self.distribution == 'gamma':
            beta_est = optimize.fmin(func=self.neg_L_gamma, x0=8, maxfun=100)
            sum = 0
            for i in self.price:
                sum += float(beta_est * self.log_func(i, self.popt[0], self.popt[1], self.popt[2]))
            alpha_est = sum / len(self.price)
        elif self.distribution == 'log':
            beta_est = optimize.fmin(func=self.neg_L_lognorm, x0=2, maxfun=100)
            sum = 0
            for i in self.price:
                sum += float(
                    np.log(self.log_func(i, self.popt[0], self.popt[1], self.popt[2]) * e ** (-beta_est[0] ** 2 / 2)))
            miu_est = sum / len(self.price)
        elif self.distribution == 'chi2':
            sum = 0
            for i in self.price:
                sum += float(np.round(self.log_func(i, self.popt[0], self.popt[1], self.popt[2])))
            k_total = sum / len(self.price)
            beta_est = optimize.fmin(func=self.neg_L_chi2, x0=k_total, maxfun=100)

        return beta_est

    def ave_sale(self, cur_price):
        sale_list = []
        for i in range(50):
            sale_list.append(self.sale(cur_price))
        return np.mean(np.array(sale_list))

    def sale(self, cur_price):
        test = np.random.binomial(1, self.p, 1)
        if test == 0:
            return 0
        #print('-------------popt----------', self.popt)
        alpha_est1 = float(self.beta_est * self.safe_log_func(cur_price, self.popt[0], self.popt[1], self.popt[2]))
        num = np.round(np.random.gamma(shape=alpha_est1, scale=1 / self.beta_est) + 1)  # 从gamma分布中取一个数
        #while num > 50:  # num大于50则重新抽取
            #num = np.round(np.random.gamma(shape=alpha_est1, scale=1 / self.beta_est) + 1)  # 从gamma分布中取一个数
        # print('a1:',alpha_est1,'a0:',alpha_est)
        if num > 50:
            num = 50
        return num

    def ave_sale_nonzero(self, cur_price):
        sale_list = []
        for i in range(50):
            sale_list.append(self.sale_nonzero(cur_price))
        return np.mean(np.array(sale_list))

    def sale_nonzero(self, cur_price): # output a nonzero sale
        alpha_est1 = float(self.beta_est * self.safe_log_func(cur_price, self.popt[0], self.popt[1], self.popt[2]))
        num = np.round(np.random.gamma(shape=alpha_est1, scale=1 / self.beta_est) + 1)  # 从gamma分布中取一个数
        #while num > 50:  # num大于50则重新抽取
            #num = np.round(np.random.gamma(shape=alpha_est1, scale=1 / self.beta_est) + 1)  # 从gamma分布中取一个数
        # print('a1:',alpha_est1,'a0:',alpha_est)
        if num > 50:
            num = 50
        return num

    def log_func(self, x, a, b, c):
        return a * np.log(b * x) + c

    def safe_log_func(self, x, a, b, c):
        y = a * np.log(b * x + 1e-7) + c
        if y > 0:
            return y
        else:
            return 1e-10

    def rscore(self, y, yval):
        SST = 0
        SSE = 0
        for i in range(len(y)):
            SST += (y[i] - y.mean()) ** 2
            SSE += (y[i] - yval[i]) ** 2
        R_2 = 1 - SSE / SST

        return R_2

    def neg_L_gamma(self, beta):
        m = 0
        for p in range(len(self.price)):
            alpha = float(beta * self.safe_log_func(self.price[p], self.popt[0], self.popt[1], self.popt[2]))
            df2 = self.df.loc[self.df['PRICE'] == self.price[p], ['WBL_AUD_DTDL', 'NUM']].reset_index(drop=True)
            num = df2['NUM']
            for i in range(len(num)):
                gap = (df2.iloc[i, 0] - self.start) / np.timedelta64(1, 'D')  # 该条数据距开始时间的差值
                m += stats.gamma.logpdf(num[i], alpha, 0.999, 1 / beta) * (1 / (gap + 1))  # 根据时间赋权重
        return float(-m)

    def neg_L_lognorm(self, s):
        m = 0
        for p in range(len(self.price)):
            miu = float(np.log(self.safe_log_func(self.price[p], self.popt[0], self.popt[1], self.popt[2]) * e ** (-s ** 2 / 2)))
            df2 = self.df.loc[self.df['PRICE'] == self.price[p], ['WBL_AUD_DTDL', 'NUM']].reset_index(drop=True)
            num = df2['NUM']
            for i in range(len(num)):
                gap = (df2.iloc[i, 0] - self.start) / np.timedelta64(1, 'D')
                m += np.log(stats.lognorm.pdf(num[i], s, miu, 1)) * (1 / (gap + 1))
        return float(-m)

    def neg_L_chi2(self, k):
        m = 0
        for p in range(len(self.price)):
            k = float(round(self.log_func(self.price[p], self.popt[0], self.popt[1], self.popt[2])))
            df2 = self.df.loc[self.df['PRICE'] == self.price[p], ['WBL_AUD_DTDL', 'NUM']].reset_index(drop=True)
            num = df2['NUM']
            for i in range(len(num)):
                gap = (df2.iloc[i, 0] - self.start) / np.timedelta64(1, 'D')
                m += np.log(stats.chi2.pdf(num[i], k, 0, 1)) * (1 / (gap + 1))
        return float(-m)

