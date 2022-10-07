import pandas as pd
import numpy as np

FILE_NAME = '../data/IC29_CHECK_DATA_CNTR_LEG.xlsx'
df_cntr_leg = pd.read_excel(FILE_NAME)

df_cntr_leg_yik = df_cntr_leg[df_cntr_leg['POL_CDE'] == 'YIK']
df_cntr_leg_yik_qzn = df_cntr_leg_yik[df_cntr_leg_yik['POD_CDE'] == 'QZH']

for uuid in df_cntr_leg_yik_qzn['WBL_CNTR_UUID']:
    df_cntr_leg_tmp = df_cntr_leg[df_cntr_leg['WBL_CNTR_UUID'] == uuid]
    print(df_cntr_leg_tmp.iloc[0, 3], end='')
    for index, row in df_cntr_leg_tmp.iterrows():
        print('-' + row['POD_CDE'], end='')
    print()

'''
1. 找到YIK-QZH的UUID
2. 取出所有该UUID的路径
3. 输出路径
'''