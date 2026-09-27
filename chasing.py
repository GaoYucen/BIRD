#%%
import tensorflow as tf
import numpy as np
import random
import math
from DP import AirPrice
import sys
sys.path.append('RL-DDPG/')
import RL
sys.path.append('RL-DDPG/DDPG/')
from DDPG import DDPG
import config

#%% pricing function
def expert(base, expert_range): #expert pricing
    if random.random() < 0.5:
        price = base+expert_range*random.random()
    else:
        price = base-expert_range*random.random()
    return price

def decision_table(volume,volume_round,C,i,T,price): # decision table pricing
    utilize = volume/C*100
    utilize_round = (volume-volume_round)/C*100
    # increase price
    fix_rate_up = T/3
    if i < int(fix_rate_up):
        if (utilize_round < 50) and (utilize >= 50):
            price = price+30
        elif (utilize_round<70) and (utilize >= 70):
            price = price+20
        elif (utilize_round < 80) and (utilize >= 80):
            price = price+40
    elif (i >= int(fix_rate_up)) and (i < int(fix_rate_up*2)):
        if (utilize_round < 60) and (utilize >= 60):
            price = price+50
        elif (utilize_round<70) and (utilize >= 70):
            price = price+30
        elif (utilize_round < 80) and (utilize >= 80):
            price = price+60
    else:
        if (utilize_round < 50) and (utilize >= 50):
            price = price+30
        elif (utilize_round < 60) and (utilize >= 60):
            price = price+40
        elif (utilize_round<70) and (utilize >= 70):
            price = price+50
        elif (utilize_round < 80) and (utilize >= 80):
            price = price+60

    # decrease price
    fix_rate_down = T/28
    if i < int(T/2):
        for j in range(0,14):
            if i == int(fix_rate_down*(j+1)):
                if (utilize<10):
                    price = price - 90
    else:
        for j in range(14,28):
            if i == int(fix_rate_down*(j+1)):
                if (utilize<20):
                    price = price - 92

    return price

def get_price_RL(input_array, c): # RL pricing
    action = ddpg.action(input_array, c)
    return(action[0][0])

#%% parameter
T = 5000
N = 5 # Categories of containers
c = np.zeros(N)
m = 5 # when remaining inventory >= m, buyers can purchase goods
t_a = np.arange(0, T/2, T/10) # start sell time matrix
t_e = np.arange(T/5, T/10*7, T/10) # end sell time matrix
C = c[0] # max inventory
W = 2 # max A_t
T_end = int(T/10*6)+1

#%% define price、buy、profit & c matrix
Gamma = 5 # Base strategy
p = np.zeros(Gamma*T*N).reshape(Gamma,T,N) # price
buy = np.zeros(Gamma*T*N).reshape(Gamma,T,N) # buyer
profit = np.zeros(Gamma*T*N).reshape(Gamma,T,N) # profit
c_Gamma = np.zeros(Gamma*N).reshape(Gamma,N) # remaing inventory of gamma
t_Gamma = np.zeros(Gamma*N).reshape(Gamma,N) # the buyer index when the category of containers is sold out

#%% define base_price
base_price = np.array([2000, 2500, 3000, 3500, 4000])

#%% generate valuation
v = np.zeros(T*5).reshape(T,5)
for i in range(0,T_end):
    for j in range(0,5):
        v[i,j] = random.random()*base_price[j]+base_price[j]*0.5

#%% determine inventory
for i in range(0,N):
    c[i] = 200
for i in range(0,5):
    c[i] = c[i] + m # add m
for gamma in range(1, Gamma): #initial inventory
    c_Gamma[gamma] = c

#%% define expert & expert continuous price
# gamma
# 1: expert continuous price
# 2: decision table price
# 3: DP price
# 4: DDPG price
# expert pricing parameter
base = base_price[:]
range_expert = base_price/40
expert_upper_bound = base_price * 1.5
expert_lower_bound = base_price * 0.5

# DP instance
DP = []
for i in range(0,Gamma):
    DP.append(AirPrice(real_min_demand_level = base_price[i]*0.5, real_max_demand_level=base_price[i]*1.5, max_days=14, num_tickets=c[i]))
fix_rate_DP = T/5/14
round_RL = 14 # adjustment round of dynamic pricing
fix_rate_RL = np.zeros(N)
for i in range(0,N):
    fix_rate_RL[i] = int((t_e[i] - t_a[i])/round_RL)
fix_rate_RL = fix_rate_RL.astype(int)

# RL pricing instance
tf.reset_default_graph()
c_RL = np.zeros(128, dtype=float)

ddpg = DDPG(state_dim=3, time_steps=5, c_dim=128, action_dim=1, action_range=40)

ddpg.load_weights('RL-DDPG/DDPG/weights_real_train/')

input_array = np.zeros(15).reshape(5, 3)

#%% sales procedure of strategy set
for gamma in range(1,Gamma):
    for i in range(0,T_end):
        for j in range(0,5):
            if gamma == 1: # expert pricing
                base[j] = expert(base[j], range_expert[j])
                if base[j] >= expert_upper_bound[j]:
                    base[j] = expert_upper_bound[j]
                elif base[j] <= expert_lower_bound[j]:
                    base[j] = expert_lower_bound[j]
                p[gamma, i, j] = base[j]
            if gamma == 2: # decision_table pricing
                volume = np.zeros(N)
                volume_round = np.zeros(N) # until last round
                if i == t_a[j]: # initial price
                    p[gamma,i,j] = 2000 + 500*j
                    if (c_Gamma[gamma,j] >= m) and (i >= t_a[j]) and (i < t_e[j]):
                        for k in range(0,m):
                            if math.pow(0.95,k)*v[i,j] >= p[gamma,i,j]:
                                buy[gamma,i,j] = k+1
                        c_Gamma[gamma,j] = c_Gamma[gamma,j] - buy[gamma,i,j]
                        profit[gamma,i,j] = buy[gamma,i,j]*p[gamma,i,j]
                        volume[j] = volume[j] + buy[gamma,i,j]
                elif i == t_a[j]+1:
                    p[gamma,i,j] = 2000 + 500*j
                    if (c_Gamma[gamma,j] >= m) and (i >= t_a[j]) and (i < t_e[j]):
                        for k in range(0,m):
                            if math.pow(0.95,k)*v[i,j] >= p[gamma,i,j]:
                                buy[gamma,i,j] = k+1
                        c_Gamma[gamma,j] = c_Gamma[gamma,j] - buy[gamma,i,j]
                        profit[gamma,i,j] = buy[gamma,i,j]*p[gamma,i,j]
                        volume[j] = volume[j] + buy[gamma,i,j]
                        volume_round[j] = volume_round[j] + buy[gamma,i-1,j]
                else:
                    if (i >= t_a[j]) and (i < t_e[j]):
                        p[gamma,i,j] = decision_table(volume[j],volume_round[j],c[j],i-t_a[j],T/5,p[gamma,i-1,j])
                    if (c_Gamma[gamma,j] >= m) and (i >= t_a[j]) and (i < t_e[j]):
                        for k in range(0,m):
                            if math.pow(0.95,k)*v[i,j] >= p[gamma,i,j]:
                                buy[gamma,i,j] = k+1
                        c_Gamma[gamma,j] = c_Gamma[gamma,j] - buy[gamma,i,j]
                        profit[gamma,i,j] = buy[gamma,i,j]*p[gamma,i,j]
                        volume = volume + buy[gamma,i,j]
                        volume_round = volume_round+buy[gamma,i-1,j]
                    elif (t_Gamma[gamma, j] == 0) and (i >= t_a[j]) and (i < t_e[j]):
                        t_Gamma[gamma, j] = i
            else:
                if (gamma == 3) and (i >= t_a[j]) and (i < t_e[j]):  # DP price
                    p[gamma, i, j] = DP[j].get_price(14 - int((i-t_a[j]) / fix_rate_DP), c_Gamma[gamma, j])
                if gamma == 4: # RL price
                    if (i >= t_a[j]) and (i < t_e[j]):
                        if (i == t_a[j]):
                            p[gamma,i,j] = base_price[j]
                        elif (i-t_a[j]) % fix_rate_RL[j] == 0:
                            volume_RL = 0
                            start_index_RL = i - fix_rate_RL[j]
                            for k in range(0,fix_rate_RL[j]):
                                volume_RL = volume_RL+buy[gamma,start_index_RL+k,j]
                            p[gamma,i,j] = RL.get_price(volume_RL, max(0,round_RL-int((i-t_a[j])/fix_rate_RL[j])), c[j], p[gamma,i-1,j], base_price[j]*1.5, base_price[j]*0.5, round_RL)
                            for e in range(0,5):
                                input_array[e][0] = p[gamma,i-1,j]
                                input_array[e][1] = volume_RL
                                input_array[e][2] = max(0,round_RL-int((i-t_a[j])/fix_rate_RL[j]))
                            p[gamma,i,j] += get_price_RL(input_array, c_RL)
                        else:
                            p[gamma,i,j] = p[gamma,i-1,j]
                # DP and RL selling
                if (c_Gamma[gamma,j] >= m) and (i >= t_a[j]) and (i < t_e[j]):
                    for k in range(0,m):
                        if math.pow(0.95,k)*v[i,j] >= p[gamma,i,j]:
                            buy[gamma,i,j] = k+1
                    c_Gamma[gamma,j] = c_Gamma[gamma,j] - buy[gamma,i,j]
                    profit[gamma,i,j] = buy[gamma,i,j]*p[gamma,i,j]
                elif (t_Gamma[gamma, j] == 0) and (i >= t_a[j]) and (i < t_e[j]):
                    t_Gamma[gamma, j] = i

#%% define parameters for BIRD
# BIRD parameter
epsilon = pow(C*W/T,0.5)
p_chasing = np.zeros(T*5).reshape(T,5)
c_chasing = c
buy_chasing = np.zeros(T*5).reshape(T,5)
profit_chasing = np.zeros(T*5).reshape(T,5)
t_chasing = np.zeros(N)

# OLSC parameter
R_gamma = np.zeros(Gamma) # reward last round
D = 2
profit_gamma = np.zeros(Gamma) # target strategy
profit_add_gamma = np.zeros(Gamma) # target strategy profit increment

#%% BIRD pricing and selling (assuimng not sell for buyer 1)
for i in range(1,T):
    if random.random() <= epsilon:
        for j in range(0,5):
            p_chasing[i,j] = 1
    else:
        # Select target strategy
        R_gamma = np.zeros(Gamma)
        for gamma in range(1,Gamma):
            for j in range(0,N):
                profit_gamma[gamma] = profit_gamma[gamma]+profit[gamma,i-1,j]
                R_gamma[gamma] = R_gamma[gamma]+profit[gamma,i-1,j]
        R = max(0.05, np.max(R_gamma))

        profit_add_gamma = np.zeros(Gamma)
        for gamma in range(1,Gamma):
            for j in range(0,N):
                profit_add_gamma[gamma] = profit_add_gamma[gamma]+p[gamma,i,j]*m/pow(D/(R*R*T),0.5)

        select_gamma = np.argmax(profit_gamma+profit_add_gamma)

        #chasing
        for j in range(0,5):
            if c_chasing[j] >= c_Gamma[select_gamma,j]:
                p_chasing[i,j] = p[select_gamma,i,j]
            else:
                p_chasing[i,j] = 1
            if (c_chasing[j] >= m) and (i >= t_a[j]) and (i < t_e[j]):
                for k in range(0,m):
                    if math.pow(0.95,k)*v[i,j] >= p_chasing[i,j]:
                        buy_chasing[i,j] = k+1
                c_chasing[j] = c_chasing[j] - buy_chasing[i,j]
                profit_chasing[i,j] = buy_chasing[i,j]*p_chasing[i,j]
            elif (t_chasing[j] == 0) and (i >= t_a[j]) and (i < t_e[j]):
                t_chasing[j] = i


#%% Profit and remaining inventory for gamma
profit_item = np.zeros(Gamma*N).reshape(Gamma,N)
profit_all = np.zeros(Gamma)
for gamma in range(1,Gamma):
    for i in range(0,N):
        for j in range (0,T_end):
            profit_item[gamma,i] = profit_item[gamma,i] + profit[gamma,j,i]
        profit_all[gamma] = profit_all[gamma]+profit_item[gamma,i]

#%% profit and remaining inventory for BIRD
profit_item_chasing = np.zeros(N)
profit_all_chasing = 0
for i in range(0,N):
    for j in range (0,T_end):
        profit_item_chasing[i] = profit_item_chasing[i] + profit_chasing[j,i]
    profit_all_chasing = profit_all_chasing+profit_item_chasing[i]

#%%
print('Revenue of different pricing strategies:', end='\n')
print('expert pricing: ', profit_all[1])
print('decision table pricing: ', profit_all[2])
print('dynamic pricing: ', profit_all[3])
print('RL pricing: ', profit_all[4])
print('BIRD: ', profit_all_chasing)

