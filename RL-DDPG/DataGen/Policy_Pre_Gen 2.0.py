# -*- coding: utf-8 -*-
import argparse
import random
import pandas as pd
import numpy as np
from numpy import deg2rad, polyfit, poly1d
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 常数
SALE_TIME = 14
DAILY_SALE_TIME = 10
INTERVAL = 2
INTERVAL_NUM = DAILY_SALE_TIME / INTERVAL
PRICE_DROP_FREQ = 1

def input_data():
    df = pd.read_excel("../data/PolicyTable.xlsx")
    df = df[['起始天数', '结束天数', '涨/跌', '阈值（放舱利用率）%', '20GP(运价调整额)']]
    df.columns = ['开始天数', '结束天数', '涨价/降价', '舱位利用率', '20GP']
    df = df.sort_values(by=['开始天数', '涨价/降价', '舱位利用率'], ascending=[True, False, True])
    return df

def process_data(df_tmp):
    # 调价累计，统计由于涨价的机制多次触发形成的总和
    df_tmp["调价累计"] = 0
    accum_counter = 0
    for ind, row in df_tmp.iterrows():
        if (row["涨价/降价"] in ["跌", "跌价","降","降价"]):
            df_tmp.loc[ind, "调价累计"] = (-1) * row["20GP"]
            accum_counter = 0
        else:
            accum_counter += row["20GP"]
            df_tmp.loc[ind, ["调价累计"]] = accum_counter

    df_tmp["20GP/间隔"] = 0

    # 基于涨价频率，给出实际的每间隔的降价
    for ind, row in df_tmp.iterrows():
        if (row["涨价/降价"] in ["涨", "涨价"]):
            days = row["结束天数"] - row["开始天数"]
            df_tmp.loc[ind, ["20GP/间隔"]] = row["调价累计"] / INTERVAL_NUM / days

    # 基于降价频率，给出实际的每间隔的降价
    for ind, row in df_tmp.iterrows():
        if (row["涨价/降价"] in ["跌", "跌价","降","降价"]):
            df_tmp.loc[ind, ["20GP/间隔"]] = row["调价累计"] / INTERVAL_NUM / PRICE_DROP_FREQ
        
    df_tmp["指征天数"] = 0

    # 指征天数
    for ind, row in df_tmp.iterrows():
        if (row["涨价/降价"] in ["涨", "涨价"]):
            df_tmp.loc[ind, ["指征天数"]] = row["结束天数"]
        else:
            df_tmp.loc[ind, ["指征天数"]] = (row["开始天数"] + row["结束天数"]) / 2
            
    df_tmp["售卖速率阈值"] = 0

    # 基于舱位利用率阈值和阶段售卖天数，计算售卖速率阈值（舱位利用率提升速率）
    for ind, row in df_tmp.iterrows():
        df_tmp.loc[ind, ["售卖速率阈值"]] = row["舱位利用率"] / INTERVAL_NUM / row["指征天数"]
    
    return df_tmp
    
def raising_polyfit(df_tmp):
    # 切割每个定价阶段的调价表格并开始拟合
    f_list = []
    counter = 0
    while (counter < len(df_tmp)):
        # 根据定价阶段切割表格
        start, end = counter, counter
        begin_date = list(df_tmp.loc[start, ["开始天数"]])[0]
        next_begin_date = list(df_tmp.loc[end, ["开始天数"]])[0]
        while ((begin_date == next_begin_date) and (end < len(df_tmp))):
            end += 1
            if (end != len(df_tmp)): 
                next_begin_date = list(df_tmp.loc[end, ["开始天数"]])[0]
            else:
                next_begin_date = 14
                break
        counter = end + 1
        if (start != 0): start -= 1
        df_to_fit = df_tmp[["20GP/间隔", "售卖速率阈值"]][start:end].round(2)
        
        # 拟合生成连续定价策略
        x = df_to_fit["售卖速率阈值"].to_list()
        y = df_to_fit["20GP/间隔"].to_list()
        
        # 删除降价
        pop_i = -1
        for i in range(len(y)):
            if (y[i] < 0):
                pop_i = i
        if (pop_i != -1):
            x.pop(pop_i)
            y.pop(pop_i)

        f = poly1d(polyfit(x,y,1))
        f_list.append(f)
    return f_list

def drop_polyfit(df_tmp):
    # 取出降价数据
    mask = (df_tmp["涨价/降价"] == "跌") | (df_tmp["涨价/降价"] == "跌价") | (df_tmp["涨价/降价"] == "降") | (df_tmp["涨价/降价"] == "降价")
    df_neg = df_tmp[mask].copy()
    df_neg["时段"] = df_neg["指征天数"] * INTERVAL_NUM

    # 二元线性拟合
    x = np.array(df_neg.loc[:, ["时段", "售卖速率阈值"]].values)
    y = np.array(df_neg.loc[:,"20GP/间隔"].values)
    cft = linear_model.LinearRegression()
    if (len(x) < 3):
        np.append(x, 0)
        np.append(y, 0)
    cft.fit(x, y)
    return cft

def price_raising_3(df_tmp, f_list):
    # 涨价策略 3.0，拟合涨价参数
    start_day = list(set(list(df_tmp["开始天数"]) + list(df_tmp['结束天数'])))
    start_day.sort()
    day = [(start_day[i] + start_day[i+1]) * 5 / 2 for i in range(len(start_day)-1)]
    coef_fit = {}
    coef_1_list = [f_list[i][1] for i in range(len(f_list))]
    coef_fit["coef1"] = poly1d(polyfit(day,coef_1_list,1))
    coef_0_list = [f_list[i][0] for i in range(len(f_list))]
    coef_fit["coef0"] = poly1d(polyfit(day,coef_0_list,1))
    return coef_fit

def raising_strategy(interval, util, coef_fit):
    # 涨价策略函数
    rate = util / interval
    coef_1 = coef_fit["coef1"](interval)
    coef_0 = coef_fit["coef0"](interval)
    f_ = poly1d([coef_1, coef_0])
    return f_(rate)
    
def drop_strategy(interval, util, cft):
    # 降价策略函数
    rate = util / interval
    return cft.predict(np.array([interval, rate]).reshape(-1,2))[0]

def continuous_pricing_strategy(interval, util, raising_para, drop_function):
    # 涨价降价函数拼接
    r = raising_strategy(interval, util, raising_para)
    d = drop_strategy(interval, util, drop_function)
    if (r > 0.6):
        return r
    else:
        return d
    
def drop_strategy_visualization(df_tmp, cft):
    # 可视化降价策略
    fig = plt.figure()
    ax = Axes3D(fig)
    x_=np.linspace(0, SALE_TIME*INTERVAL_NUM, 200)
    y_=np.linspace(0, 5, 100)

    X, Y = np.meshgrid(x_, y_)

    Z = np.zeros((len(y_), len(x_)))
    for i in range(len(x_)):
        for j in range(len(y_)):
            Z[j][i] = cft.predict(np.array([x_[i], y_[j]]).reshape(-1,2))

    plt.xlabel('Interval')
    plt.ylabel('Saling Rate')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    mask = (df_tmp["涨价/降价"] == "跌") | (df_tmp["涨价/降价"] == "跌价") | (df_tmp["涨价/降价"] == "降") | (df_tmp["涨价/降价"] == "降价")
    df_neg = df_tmp[mask].copy()
    df_neg["时段"] = df_neg["指征天数"] * INTERVAL_NUM
    x = np.array(df_neg.loc[:, ["时段", "售卖速率阈值"]].values)
    y = np.array(df_neg.loc[:,"20GP/间隔"].values)
    ax.scatter(x[:, 0], x[:, 1], y, c='r', marker='+', s=70)
    plt.show()
    
    
def strategy_visualization(df_tmp, raising_para, drop_function):
    # 可视化完整策略
    fig = plt.figure()
    ax = Axes3D(fig)
    x_=np.arange(1, SALE_TIME*INTERVAL_NUM, 1)
    y_=np.linspace(0, 3, 100)
    X, Y = np.meshgrid(x_, y_)#网格的创建，这个是关键
    Z = np.zeros((len(y_), len(x_)))
    Z_ = np.zeros((len(y_), len(x_)))
    for i in range(len(x_)):
        for j in range(len(y_)):
            Z[j][i] = continuous_pricing_strategy(x_[i], y_[j]*x_[i], raising_para, drop_function)
    plt.xlabel('Interval')
    plt.ylabel('Rate')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.scatter(df_tmp["指征天数"]*5, df_tmp["售卖速率阈值"], df_tmp["20GP/间隔"], c='r', marker='+', s=70)
    plt.show()

def sample_dataset(N, df_tmp, raising_para, drop_function):
    # 开始生成随机训练集
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

    time_steps = 15
    total_intervals = max(df_tmp['结束天数'])
    x_list = []
    y_list = []

    for _ in range(N):
        RT = random.randint(0, total_intervals-1)
        Util = random.random()
        a = continuous_pricing_strategy(total_intervals - RT, Util*100, raising_para, drop_function)
        # print("Remaining Time: {:d} \t \t Util: {:.2f} \t action {:.2f}".format(RT, Util, a))
        x = np.zeros((time_steps, 2))
        x[-1, :] = [Util, RT]
        x_list.append(x)
        y_list.append(a)

    x_numpy = np.array(x_list)
    y_numpy = np.array(y_list)
    print("The empircal range of the action value is between {:.2f} and {:.2f}.".format(np.min(y_numpy),np.max(y_numpy)))
    print("A sample of the training data")
    print(x_numpy[0])
    print("Label: {:.2f} \n".format(y_numpy[0]))
    print("Shape of x: {}".format(x_numpy.shape))
    print("Shape of y: {}".format(y_numpy.shape))

    np.save("../data/x_pretrain.npy", x_numpy)
    np.save("../data/y_pretrain.npy", y_numpy)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', '-n', help='生成数据集大小，默认为 10000', default=10000)
    parser.add_argument('--visual', '-v', help='是否可视化，默认为 False', default=False)
    args = parser.parse_args()
    try:
        N = int(args.number)
    except Exception as e:
        print(e)
    
    df = input_data()
    df_tmp = process_data(df)
    raising_function_list = raising_polyfit(df_tmp)
    raising_para = price_raising_3(df_tmp, raising_function_list)
    drop_function = drop_polyfit(df_tmp)
    if (args.visual != False):
        drop_strategy_visualization(df_tmp, drop_function)
        strategy_visualization(df_tmp, raising_para, drop_function)
    sample_dataset(N, df_tmp, raising_para, drop_function)

if __name__ == "__main__":
    main()