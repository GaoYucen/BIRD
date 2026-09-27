import tensorflow as tf
import random
import pandas as pd
import numpy as np
from DDPG import DDPG
import argparse
import config

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

def process_sample(time_series, c, args):
    original_price = time_series[-1][0].astype(float)
    original_util = time_series[-1][1].astype(float)
    original_rt = time_series[-1][2].astype(float)

    new_price = float(args.price)
    new_sale = float(args.sale)
    total_sale = float(args.total_sale)
    new_rt = original_rt - 1
    action = new_price - original_price
    reward = new_price * new_sale
    new_util = original_util - new_sale/total_sale*100
    new_time_series = time_series[1:]
    sample = np.reshape([new_price, new_util, new_rt], (1, 3))
    new_time_series = np.concatenate([new_time_series, sample], axis=0)
    return new_time_series, action, reward

if __name__ == "__main__":

    #TODO
    #df = read_data('some path')
    #samples = preprocess(df)
    random.seed(10)

    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--path', '-p', help='前5步state输入文件路径，必要参数')
    parser.add_argument('--train', '-t', help='是否训练模型，必要参数')
    parser.add_argument('--price', help='调价')
    parser.add_argument('--sale', help='销量')
    parser.add_argument('--total_sale', help='总舱位')
    args = parser.parse_args()
    tf.reset_default_graph()

    c = np.zeros(128, dtype=float)

    ddpg = DDPG(state_dim=3, time_steps=5, c_dim=128, action_dim=1, action_range=40)

    ddpg.load_weights(config.RL_WEIGHT_PATH)

    time_series = np.array(read_data(args.path))
    #print(time_series)
    action = ddpg.action(time_series, c)
    if args.train == 'y' or args.train == 'yes':
        new_time_series, real_action, reward = process_sample(time_series, c, args)
        ddpg.perceive(time_series, c, real_action, reward, new_time_series, c, 0)
        print('Training...')
        ddpg.train()
        ddpg.save_weights(config.RL_WEIGHT_PATH)

    print('建议调价为：', action[0][0])