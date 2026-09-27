# -*- coding: utf-8 -*-
import argparse
import random
import pandas as pd
import copy
import numpy as np


class PolicyTable:
    '''
    Return pricing strategy according to policy table
    get_price(util, rt) -- return the price adjustment according to the utilization and remaining time
    reset() -- reset the price rising record for the new record
    '''
    def __init__(self, path):
        self.df = self.input_data(path)
        self.check_time = 0


    def input_data(self, path):
        df = pd.read_excel(path)
        df = df[['起始天数', '结束天数', '涨/跌', '阈值（放舱利用率）%', '20GP(运价调整额)']]
        df.columns = ['开始天数', '结束天数', '涨价/降价', '舱位利用率', '20GP']
        df = df.sort_values(by=['开始天数', '涨价/降价', '舱位利用率'], ascending=[True, False, True])
        df['触发'] = 0
        return df

    def get_price(self, util, rt):
        action = 0
        for ind, row in self.df.iterrows():
            if row['开始天数'] * 5 < rt and row['结束天数'] * 5 > rt \
                    and row['涨价/降价'] == '涨价' and row['舱位利用率'] < util and row['触发'] == 0:
                action +=  row['20GP']
                self.df.loc[ind, '触发'] = 1

        if action != 0:
            return action

        for ind, row in self.df.iterrows():
            #print('check row: util:{}, rt:{}, start:{}, end:{}, policy util:{}, policy:{}'.format(util, rt, row['开始天数'], row['结束天数'], row['舱位利用率'], row['20GP']))
            if row['开始天数'] * 5 < rt and row['结束天数'] * 5 > rt \
                    and row['涨价/降价'] == '跌价' and row['舱位利用率'] > util:
                if rt - self.check_time > 5:
                    action = -row['20GP']
                    self.check_time += 5

        return action

    def reset(self):
        self.df['触发'] = 0
        self.check_time = 0


if __name__ == '__main__':
    path = "../data/PolicyTable.xlsx"
    table = PolicyTable(path)
    print(table.get_price(60, 12))
    print(table.df)