import pandas as pd

FILE_NAME = "./data/all_info_qugang_zhida_TF_2019.csv"
df = pd.read_csv(FILE_NAME, encoding = 'UTF-8')
# print(dataframe) #读取正确

'''
总体思路：
1. 按照流向进行分类
2. 将时间化为统一标准
3. 找到时间的开头和结尾，划分时间
4. 统计每个时间段的价格和销量
'''

'''
按照流向进行分类。起点流向缩写-POL_CDE，终点流向缩写-POD_CDE
'''
POL_POD = [['YIK', 'QZH'], ['TSN', 'NSH'], ['YIK', 'NSH'], ['NSH', 'YIK'], ['NSH', 'TSN']] # 五个港口分别处理

'''
时间化为统一标准，即年月日时分秒
2019/4/21  6:10:00 -> [2019, 4, 21, 6, 10, 00]
'''

df['MAIN_SVVD_SAILING_DATE'] = pd.to_datetime(df['MAIN_SVVD_SAILING_DATE'], format='%Y/%m/%d %H:%M:%S')
#df['MAIN_SVVD_SAILING_DATE'].dt.year = df['MAIN_SVVD_SAILING_DATE'].dt.year % 2
#print(df['MAIN_SVVD_SAILING_DATE'][0:1].dt.year)
print(df['LDD_SVVD'][104])
if df['LDD_SVVD'][0] in {}:
    print(1)

# pd.date_range()
# 比较时间
#

#for direction in range(5):
#    df_dir = df[(df['POL_CDE'] == POL_POD[direction][0]) & (df['POD_CDE'] == POL_POD[direction][1])]

