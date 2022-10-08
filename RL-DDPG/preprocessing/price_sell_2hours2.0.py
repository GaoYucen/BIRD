import pandas as pd
from datetime import timedelta

FILE_NAME = "../data/all_info_qugang_zhida_TF_2019.csv"
df = pd.read_csv(FILE_NAME, encoding = 'UTF-8')
# print(dataframe) #读取正确

'''
总体思路：
1. 按照航次进行分类（当前集装箱仅考虑20GP）
2. 将时间化为统一标准
3. 找到时间的开头和结尾，划分时间
4. 统计每个时间段的最高价格和总销量
'''

'''
按照航次进行分类
'''

'''
时间化为统一标准，即年月日时分秒
2019/4/21  6:10:00 -> [2019, 4, 21, 6, 10, 00]
'''


df['WBL_AUD_DT'] = pd.to_datetime(df['WBL_AUD_DT'], format='%Y/%m/%d %H:%M:%S')
#print(df['MAIN_SVVD_SAILING_DATE'])
#df['MAIN_SVVD_SAILING_DATE'].dt.year


#按照各航次自动取到start与end
def data_start(time):
    tim = list(time.strftime('%Y/%m/%d %H:%M:%S'))
    tim[12] = str(int(tim[12]) - int(tim[12]) % 2)
    tim[14] = tim[15] = '0'
    tim = "".join(tim)
    return pd.to_datetime(tim, format='%Y/%m/%d %H:%M:%S')

#找函数可以写的简单点
def data_end(time):
    tim = list(time.strftime('%Y/%m/%d %H:%M:%S'))
    if ((int(tim[12]) % 2 == 0) & (tim[14] == '0') & (tim[15] == '0')):
        tim = "".join(tim)
        return pd.to_datetime(tim, format='%Y/%m/%d %H:%M:%S')
    else:
        tim = list(time.strftime('%Y/%m/%d %H:%M:%S'))
        bo = 0
        t = 2 + int(tim[12]) - int(tim[12]) % 2
        if (t > 9):
            tim[11] = str(int(tim[11]) + 1)
            t -= 10
        tim[12] = str(t)
        if ((tim[11] == '2') & (tim[12] == '4')):
            tim[11] = tim[12] = '0'
            bo = 1
        tim[14] = tim[15] = '0'
        tim = "".join(tim)
        return pd.to_datetime(tim, format='%Y/%m/%d %H:%M:%S') + timedelta(days = bo)


'''
time = list(df['WBL_AUD_DT'][0:1][0].strftime('%Y/%m/%d %H:%M:%S'))
time[12] = str(int(time[12]) - int(time[12]) % 2)
time[14] = time[15] = '0'
time = "".join(time)
print(pd.to_datetime(time, format='%Y/%m/%d %H:%M:%S'))
'''

#dff = pd.date_range(data_start(df['WBL_AUD_DT'][0:1]), data_end(df['WBL_AUD_DT'][num-2:num-1]), freq='2H')
#print(df['WBL_AUD_DT'][1], dff[0])


# 化为字符串比较时间 time.strftime('%Y/%m/%d %H:%M:%S')

#选取20GP集装箱,索引不连续，用.iloc[]读取
df2 = df[df['CNTR_TYPE'] == '20GP']

#按航次分类LDD_SVVD
#print(df2['LDD_SVVD'])

num2 = df2['LDD_SVVD'].size

ans = 0
Route = {}
for i in range(num2):
    if df2['LDD_SVVD'].iloc[i] in Route:
        Route[df2['LDD_SVVD'].iloc[i]] += 1
    else:
        Route[df2['LDD_SVVD'].iloc[i]] = 1
        ans += 1

route = list(Route)

ddf = []
for path in route:
    ddf.append(df2[(df2['LDD_SVVD'] == path)])


#ans个航线dff[k]
#第k个航线 定位时间段
number = []
price = []
LDD = []
POLCDE =[]
POLNME = []
PODCDE =[]
PODNME = []
DL = []
SADA = []
sum = 0

for k in range(ans):
    l = 0
    len = ddf[k]['WBL_AUD_DT'].size
    dff = pd.date_range(data_start(ddf[k]['WBL_AUD_DT'].iloc[0]), data_end(ddf[k]['WBL_AUD_DT'].iloc[len-1]), freq='2H')
    for i in range(dff.size):
        si = dff[i].strftime('%Y/%m/%d %H:%M:%S')
        sj = ddf[k]['WBL_AUD_DT'].iloc[l].strftime('%Y/%m/%d %H:%M:%S')
        maxamt = 0
        num = 0
        while ((sj < si) & (l + num < len - 1)):
            num += 1
            sj = ddf[k]['WBL_AUD_DT'].iloc[l+num].strftime('%Y/%m/%d %H:%M:%S')
            maxamt = max(ddf[k]['AMT'].iloc[l+num], maxamt)
        if num > 0 :
            l += num
            #print(si, num, maxamt, sj, ddf[k]['LDD_SVVD'].iloc[l])
            sum += 1
            number.append(num)
            price.append(maxamt)
            LDD.append(ddf[k]['LDD_SVVD'].iloc[l])
            POLCDE.append(ddf[k]['POL_CDE'].iloc[l])
            POLNME.append(ddf[k]['POL_NME'].iloc[l])
            PODCDE.append(ddf[k]['POD_CDE'].iloc[l])
            PODNME.append(ddf[k]['POD_NME'].iloc[l])
            DL.append(dff[i])
            SADA.append(ddf[k]['MAIN_SVVD_SAILING_DATE'].iloc[l])

list_res = []
for i in range(sum):
    list_res.append([LDD[i], POLCDE[i], POLNME[i], PODCDE[i], PODNME[i], DL[i], SADA[i], number[i], price[i]])
column_name = ['LDD_SVVD', 'POL_CDE', 'POL_NME', 'POD_CDE', 'POD_NME', 'WBL_AUD_DTDL', 'MAIN_SVVD_SAILING_DATE', 'NUM', 'PRICE']
csv_name = '2hours_price_setting.csv'
xml_df = pd.DataFrame(list_res , columns = column_name)
xml_df.to_csv('../data/2hours_price_setting.csv', index = None)











