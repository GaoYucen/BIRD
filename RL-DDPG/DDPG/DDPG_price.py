import tensorflow as tf
import pandas as pd
import numpy as np
from DDPG import DDPG
import config

#%%
# def read_data(file_path):
#     df = pd.read_csv(file_path)
#     return df
#
# time_series = np.array(read_data('../data/test_data.csv'))
# print(time_series)

def get_price_RL(input_array, c):
    action = ddpg.action(input_array, c)
    return(action[0][0])

#%%
tf.reset_default_graph()

c = np.zeros(128, dtype=float)

ddpg = DDPG(state_dim=3, time_steps=5, c_dim=128, action_dim=1, action_range=40)

ddpg.load_weights(config.RL_WEIGHT_PATH)

#%%test
input_array = np.zeros(15).reshape(5, 3)
price = np.array([2000, 2100, 2200, 2300, 2400])
sales = np.array([40, 42, 44, 46, 48])
rt = np.array([59, 58, 57, 56, 55])
input_array[:, 0] = price
input_array[:, 1] = sales
input_array[:, 2] = rt
print(input_array)

#%%
p = get_price_RL(input_array, c)[0][0]
print(p)

