'''
Use real data to train the DeepDLP model

LSTM-c
c->DeepDLP
'''

'''
Note: Use target network to output the policy
'''


import tensorflow as tf
import random
import pandas as pd
import numpy as np
import csv
import config
from DDPG import DDPG
from sample_gen import Simulation

'''
LSTM-DDPG part
'''

def read_data(file_path):
    '''Read history data from file
    
    Parameters
    ----------
    file_path: string
        File path where we can find the history data

    Return
    ------
    df: Pandas DataFrame
    '''
    df = pd.read_csv(file_path)
    return df

def read_real_data(path, DDPG):
    c = np.zeros(config.C_DIM, dtype=float)
    df = pd.read_excel(path, index_col = [0])
    lines = df.shape[0]
    for i in range(lines - 6):
        time_series = []
        for j in range(5):
            time_series.append([df.iloc[i+j][1], df.iloc[i+j][4], df.iloc[i+j][5]])
            if df.iloc[i+j][7] == 1:
                break
        if j < 4:
            continue
        action = df.iloc[i+6][1] - df.iloc[i+5][1]
        reward = df.iloc[i+6][3] * df.iloc[i+5][1]
        new_time_series = time_series[1:]
        new_time_series.append([df.iloc[i+6][1], df.iloc[i+6][4], df.iloc[i+6][5]])
        done = df.iloc[i+6][7]
        time_series_np = np.reshape(time_series, (5,3))
        new_time_series_np = np.reshape(new_time_series, (5,3))
        DDPG.perceive(time_series_np, c, action, reward, new_time_series_np, c, done)

def preprocess(df):
    return df[df['None' == 1]]

if __name__ == "__main__":
    random.seed(10)

    '''
    Hyperparameters
    '''
    c_dim = config.C_DIM
    state_dim = config.STATE_DIM
    time_steps = config.TIME_STEPS
    action_dim = config.ACTION_DIM
    action_range = config.ACTION_RANGE
    init_price = config.INIT_PRICE


    c = np.zeros(c_dim, dtype=float)

    ddpg = DDPG(state_dim=state_dim, time_steps=time_steps, c_dim=c_dim, action_dim=action_dim, action_range=action_range)
    tf.reset_default_graph()
    ddpg.load_pretrained_weights(config.PRE_WEIGHT_PATH)
    sim = Simulation(init_price, config.SIM_SALE_PATH, config.SIM_COUNT_PATH) # initial price is 1500
    time_series = np.zeros((time_steps-1,state_dim), dtype=float)
    init_sample = np.reshape(sim.get_sample(), (1,state_dim))
    time_series = np.concatenate([time_series, init_sample], axis = 0)
    '''
    Train
    '''
    real_path = config.REAL_DATA_PATH
    read_real_data(real_path, ddpg)

    for i in range(config.TRAIN_EPOCH):#10000
        origin_time_series = time_series
        action = (random.random() - 0.5) * 160
        reward, sample, done = sim.step(action)
        if done:
            time_series = np.zeros((time_steps, state_dim), dtype=float)
        time_series = time_series[1:]
        sample = np.reshape(sample, (1, state_dim))
        time_series = np.concatenate([time_series, sample], axis = 0)

        if i % 1000 == 0: print('Training epoch: {}'.format(i))
        ddpg.perceive(origin_time_series, c, action, reward, time_series, c, done)

    '''
    Test
    '''
    total_reward = 0
    train_num = 0
    price_list =[]
    best_reward = 0
    best_price = []
    for i in range(config.TEST_EPOCH):#1000
        origin_time_series = time_series
        action = ddpg.action(origin_time_series, c)
        reward, sample, done = sim.step(action)
        #print('reward: {}, sample: {}, done: {}'.format(reward, sample, done))
        total_reward += reward
        #price_list.append(sample)
        if isinstance(origin_time_series[-1][0], np.ndarray):
            price_list.append([origin_time_series[-1][0][0][0], origin_time_series[-1][1], origin_time_series[-1][2]])
        else:
            price_list.append(origin_time_series[-1][:])
        # print('state: {}, action: {}, reward: {}'.format(origin_time_series[-1], action, reward))
        if done:
            train_num += 1
            time_series = np.zeros((time_steps,state_dim), dtype=float)
            if (train_num % 1 == 0):
                print()
                print('Trained Samples: {}, total reward: {}, price_strategy: {}\n'.format(train_num, total_reward[0][0], price_list))
                print('Critic-Network Output:')
                for price in range(0, 5000, 1000):
                    print('\n---------------- Price: {} ----------------'.format(price))
                    for action in range(-40, 41, 20):
                        qvalue = ddpg.critic_network.q_value(np.reshape(np.array([[price, 1000, 35] for i in range(5)]), (1, 5, 3)), np.zeros((1, 128)), np.reshape(np.array([action]), (1, 1)))
                        print('Price: {},  Action: {}, Q-value: {}'.format(price, action, qvalue))
            if total_reward > best_reward:
                best_reward = total_reward
                best_price = price_list
            total_reward = 0
            price_list = []
        time_series = time_series[1:]
        sample = np.reshape(sample, (1,3))
        time_series = np.concatenate([time_series, sample], axis = 0)
        # 给过度涨降价以惩罚
        ddpg.perceive(origin_time_series, c, action, reward * (1 - abs(action - 80)/100), time_series, c, done)
        #ddpg.perceive(origin_time_series, c, action, reward, time_series, c, done)
        #print("end of an epoch ", i)

    print(best_reward)
    print(best_price)
    file = open(config.RL_RESULT_PATH, 'w', newline='')
    csvwriter = csv.writer(file)
    csvwriter.writerow(['revenue', best_reward])
    for idx, price in enumerate(best_price):
        csvwriter.writerow(['price', idx, price[0]])
    file.close()
    ddpg.save_weights(config.RL_WEIGHT_PATH)






