import random
import pandas as pd
import numpy as np
from sample_gen import Simulation
from Decision_Env import PolicyTable
import csv

def run_episode():
    return 0

if __name__ == '__main__':
    random.seed(10)
    sim = Simulation(2500, '../data/2hours_price_setting_env.csv', '../data/num_count.pkl')
    policytable = PolicyTable('../data/PolicyTable.xlsx')
    file = open('policy_result.csv', 'w', newline ='')
    csvwriter = csv.writer(file)


    best_reward = 0
    best_price = []
    for i in range(100):
        done = 0
        sim.reset()
        total_reward = 0
        price_list = []
        while not done:
            sample = sim.get_sample()
            action = policytable.get_price(sample[1]/2, 70 - sample[2])
            reward, sample, done = sim.slow_step(action)
            price_list.append([sample[0], reward / sample[0], 70 - sample[2]])
            total_reward += reward
            #print('reward: {}, sample: {}, done: {}'.format(reward, sample, done))
        policytable.reset()
        if total_reward > best_reward:
            best_reward = total_reward
            best_price = [price for price in price_list]
        print('episode {}: reward {}, price: {}'.format(i, total_reward, price_list))
    csvwriter.writerow(['revenue', best_reward])
    for price in best_price:
        csvwriter.writerow(['price', price[0], price[2]])