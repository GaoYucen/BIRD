###### 说明
# 修订时间：2021-09-08
# data共有8列 ['MAIN_SVVD', 'price', 'time', 'volume', 'utilization', 'rt', 'date', 'label']
# 分别代表svvd，价格，时间段，时间段内销量，释放舱位利用率，剩余时间，日期，以及结束标签
# 其中时间段表示方法为，（2020-11-11 08:00:00, 2020-11-11 10:00:00]（半开半闭区间）用2020-11-11 10:00:00表示

# 存在问题：
# 1. CGO_TYPE_NME筛选普货的筛选条件需要进一步核对
# 2. 放舱方案及不同时段的释放舱位需要导入相应表格,现在使用的是0-4天24个舱位，5-8天56个舱位，9天后80个舱位


#%%导入库
import pandas as pd
import math

######   提取非矿电商箱量
#%%读取waybill到pandas
df_wbl=pd.read_excel('../data/YIK_QZH_COMPLETE_WBL.xlsx',sheet_name='Sheet1',engine='openpyxl') #pandas列名是单独计算的，不算在行索引内 xlrd2.0支持.xls文件，需要使用openpyxl引擎
df_wbl_epanasia_all = df_wbl[df_wbl['WBL_SOURCE']=='EPANASIA']
print(len(df_wbl_epanasia_all))

#%%读取FRT表格到pandas
df_frt=pd.read_excel('../data/YIK_QZH_COMPLETE_CNTR_FRT.xlsx',sheet_name='Sheet1',engine='openpyxl')
df_frt_oft=df_frt[df_frt['CHRG_CDE']=='OFT'] #提取海运费
print(len(df_frt_oft))

# %%提取启航时间/停止售卖时间全部数据
df_end_sale = pd.read_csv('../data/EXPECTED_END_SALE_DATE.csv')
print(len(df_end_sale))

#根据放舱方案找全部SVVD
#根据WBL表找全部SVVD，存储一个data_all Dataframe
#%% 构建data_all
data_all = pd.DataFrame()
col_name = ['MAIN_SVVD', 'price', 'time', 'volume', 'utilization', 'rt', 'date', 'label']
data_all=data_all.reindex(columns=col_name)              # DataFrame.reindex() 对原行/列索引重新构建索引值
print(data_all)

#%%找SVVD
SVVD_list = df_wbl_epanasia_all['MAIN_SVVD'].unique()
print(SVVD_list.shape[0])
i_svvd = 0
for i_svvd in range(SVVD_list.shape[0]):
#for i_svvd in range(5):

    #%%取一个SVVD的WBL
    df_svvd_wbl=df_wbl[df_wbl['MAIN_SVVD']==SVVD_list[i_svvd]] #20GP箱舱位分配为80
    #提取电商部分数据
    df_wbl_epanasia=df_svvd_wbl[df_svvd_wbl['WBL_SOURCE']=='EPANASIA']
    print(len(df_wbl_epanasia))

    #%%取非矿的epanasia的wbl
    df_wbl_epanasia_ord_1=df_wbl_epanasia[~df_wbl_epanasia['CGO_TYPE_NME'].str.contains('矿') ]
    df_wbl_epanasia_ord_2=df_wbl_epanasia_ord_1[~df_wbl_epanasia_ord_1['CGO_TYPE_NME'].str.contains('金属') ]
    df_wbl_epanasia_ord_3=df_wbl_epanasia_ord_2[~df_wbl_epanasia_ord_2['CGO_TYPE_NME'].str.contains('石油') ]
    df_wbl_epanasia_ord=df_wbl_epanasia_ord_3[~df_wbl_epanasia_ord_3['CGO_TYPE_NME'].str.contains('化工') ]
    print(len(df_wbl_epanasia_ord))

    #%%merge df_wbl_epanasia_ord和df_frt_oft以计算箱数量
    df_cntr_epanasia=pd.merge(df_wbl_epanasia,df_frt_oft,on='WBL_NUM')
    print(len(df_cntr_epanasia))
    df_cntr_epanasia_ord=pd.merge(df_wbl_epanasia_ord,df_frt_oft,on='WBL_NUM')
    print(len(df_cntr_epanasia_ord))

    if len(df_cntr_epanasia_ord) > 0:

        ###### 进行时间划分
        #%%转化运单时间为日期格式，排序并找到最早的成交时间
        df_wbl_epanasia_ord['WBL_AUD_DT']=pd.to_datetime(df_wbl_epanasia_ord['WBL_AUD_DT'])
        df_wbl_epanasia_ord=df_wbl_epanasia_ord.sort_values(by='WBL_AUD_DT') #排序
        print(df_wbl_epanasia_ord)

        #%%找到日期，构造10点时间
        df_start_time=df_wbl_epanasia_ord.iloc[0].at['WBL_AUD_DT'].strftime('%Y-%m-%d')
        df_start_time=df_start_time+' 10:00:00'
        df_start_time=pd.to_datetime(df_start_time)

        #%%单一航次的停止售卖时间
        df_end_sale_time = df_end_sale[df_end_sale['MAIN_SVVD'] == SVVD_list[i_svvd]]
        df_end_sale_time=df_end_sale_time[df_end_sale_time['FCIL_CDE'].str.contains('YIK')]
        df_end_sale_time=df_end_sale_time[df_end_sale_time['SCH_STATE'].str.contains('actual')]
        df_end_sale_time['BERTH_DEP_DT_LOC']=pd.to_datetime(df_end_sale_time['BERTH_DEP_DT_LOC'], format='%d/%m/%Y %H:%M')
        print(df_end_sale_time)
        df_end_sale_time=df_end_sale_time.sort_values(by='BERTH_DEP_DT_LOC')
        if df_end_sale_time['BERTH_DEP_DT_LOC'][0:1].empty==False:
             df_end_sale_time=df_end_sale_time['BERTH_DEP_DT_LOC'][0:1]
        else:
            df_end_sale_time = df_end_sale[df_end_sale['MAIN_SVVD'] == SVVD_list[i_svvd]]
            df_end_sale_time = df_end_sale_time[df_end_sale_time['FCIL_CDE'].str.contains('YIK')]
            df_end_sale_time = df_end_sale_time[df_end_sale_time['SCH_STATE'].str.contains('longterm')]
            df_end_sale_time['BERTH_DEP_DT_LOC'] = pd.to_datetime(df_end_sale_time['BERTH_DEP_DT_LOC'], format='%d/%m/%Y %H:%M')
            df_end_sale_time = df_end_sale_time.sort_values(by='BERTH_DEP_DT_LOC')
            df_end_sale_time = df_end_sale_time['BERTH_DEP_DT_LOC'][0:1]
        print(df_end_sale_time)

        #%%转换成str
        df_end_sale_time=df_end_sale_time.iloc[0].strftime('%Y-%m-%d %H:%M:%S')

        #%%计算终止售卖时间
        df_end_time = df_start_time+math.ceil((pd.to_datetime(df_end_sale_time)-df_start_time)/pd.Timedelta(2, 'H'))*pd.Timedelta(2, 'H')
        #%%构造时间段
        data=pd.DataFrame()
        df_construct_time=df_start_time
        while df_construct_time + pd.Timedelta(1, 'D') < df_end_time:
            df_time_period = pd.date_range(df_construct_time, periods=5, freq='2H')
            new = pd.DataFrame({'time': df_time_period}, index=[0, 1, 2, 3, 4])
            data=data.append(new, ignore_index=True)
            df_construct_time += pd.Timedelta(1, 'D')
        while df_construct_time + pd.Timedelta(2, 'H') <= df_end_time:
            df_construct_time += pd.Timedelta(2, 'H')
            print(df_construct_time)
            new = pd.DataFrame({'time': df_construct_time}, index=[0])
            data=data.append(new, ignore_index=True)
        print(data)

        ###### 添加日期和结束标签
        #%%添加日期和结束标签
        # col_name=data.columns.tolist()                   # 将数据框的列名全部提取出来存放在列表里
        # col_name.insert(1,'date')                      # 在列索引为2的位置插入一列,列名为:city，刚插入时不会有值，整列都是NaN
        # col_name.insert(2, 'label')
        # print(col_name)
        # print(type(col_name))
        col_name = ['MAIN_SVVD', 'price', 'time', 'volume', 'utilization', 'rt', 'date', 'label']
        data=data.reindex(columns=col_name)              # DataFrame.reindex() 对原行/列索引重新构建索引值
        print(data)

        #%%按要求写入date和结束标签
        i = 0
        while i < data.shape[0]:
            data.loc[i, 'date'] = pd.to_datetime(data.loc[i, 'time'].strftime('%Y-%m-%d'))
            i += 1

        i = 0
        while i + 1 < data.shape[0]:
            data.loc[i, 'label'] = 0
            i += 1
        data.loc[i, 'label'] = 1

        print(data)




        ###### 计算剩余时间
        #%%wbl索引重置
        df_wbl_epanasia_ord = df_wbl_epanasia_ord.reset_index()
        print(df_wbl_epanasia_ord)

        #%%寻找最接近订单，以计算剩余时间
        i = 0
        j = 0
        for i in range(data.shape[0]):
            flag = 0
            while (j < df_wbl_epanasia_ord.shape[0]) and (data.loc[i, 'time'] - df_wbl_epanasia_ord.loc[j, 'WBL_AUD_DT'] < pd.Timedelta(2, 'H')) and (data.loc[i, 'time'] >= df_wbl_epanasia_ord.loc[j, 'WBL_AUD_DT']):
                flag = 1
                data.loc[i, 'rt'] = df_end_time - df_wbl_epanasia_ord.loc[j, 'WBL_AUD_DT']
                j += 1
            if flag == 0:
                data.loc[i, 'rt'] = df_end_time - data.loc[i, 'time']
        print(data)





        ###### 计算时间段销量
        #%%排序
        df_cntr_epanasia_ord['WBL_AUD_DT'] = pd.to_datetime(df_cntr_epanasia_ord['WBL_AUD_DT'])
        df_cntr_epanasia_ord = df_cntr_epanasia_ord.sort_values(by='WBL_AUD_DT')
        df_cntr_epanasia_ord = df_cntr_epanasia_ord.reset_index()
        print(df_cntr_epanasia_ord['WBL_AUD_DT'])

        #%%计算时间段销量
        i = 0
        j = 0
        for i in range(data.shape[0]):
            num = 0 #计数器
            while (j < df_cntr_epanasia_ord.shape[0]) and (df_cntr_epanasia_ord.loc[j, 'WBL_AUD_DT'] <= data.loc[i, 'time']) and (df_cntr_epanasia_ord.loc[j, 'WBL_AUD_DT'] > data.loc[i, 'time'] - pd.Timedelta(2, 'H')):
                num += 1
                j += 1
            data.loc[i, 'volume'] = num
        print(data['volume'])

        #验证
        # i = 0
        # num = 0
        # for i in range(data.shape[0]):
        #     num += data.loc[i, 'volume']
        # print(num)





        ###### 提取总舱位利用率，并计算实时舱位利用率     此处存在问题！！！！！！！！！！！！！！！！
        #%%
        all = 24 #需要读取 现为简单模拟 0-4天24，5-8天56，之后80
        i = 0
        num = 0
        for i in range(data.shape[0]):
            if i > 20:
                all = 56
            if i > 40:
                all = 80
            num += data.loc[i, 'volume']
            data.loc[i, 'utilization'] = num/all
        print(data['utilization'])







        ###### 筛选最近YIK-QZH订单，以计算价格
        #%%找到最初定价
        j = 0
        p = 0 #如果错误，则p一直是0
        flag = 0
        while not(flag) and (j < df_cntr_epanasia_ord.shape[0]):
            if df_cntr_epanasia_ord.loc[j, 'F_POL_CDE'] == 'YIK' and df_cntr_epanasia_ord.loc[j, 'L_POD_CDE'] == 'QZH':
                p = df_cntr_epanasia_ord.loc[j, 'TTL_AMT']
                flag = 1
            j += 1
        print(p)

        #%% 计算定价
        i = 0
        j = 0
        for i in range(data.shape[0]):
            flag = 0
            while (j < df_wbl_epanasia_ord.shape[0]) and (data.loc[i, 'time'] - df_wbl_epanasia_ord.loc[j, 'WBL_AUD_DT'] < pd.Timedelta(2, 'H')) and (data.loc[i, 'time'] >= df_wbl_epanasia_ord.loc[j, 'WBL_AUD_DT']) and df_cntr_epanasia_ord.loc[j, 'F_POL_CDE'] == 'YIK' and df_cntr_epanasia_ord.loc[j, 'L_POD_CDE'] == 'QZH':
                flag = 1
                p = df_cntr_epanasia_ord.loc[j, 'TTL_AMT']
                data.loc[i, 'price'] = p
                j += 1
            if flag == 0:
                data.loc[i, 'price'] = p
        print(data['price'])


        ###### 写入data_all
        #%%添加data的SVVD信息
        i = 0
        for i in range(data.shape[0]):
            data.loc[i, 'MAIN_SVVD'] = SVVD_list[i_svvd]
        print(data['MAIN_SVVD'])

        #%% 写入data_all
        data_all = data_all.append(data, ignore_index=True, sort=False)
        print(data_all)


###### 写入excel
#%%写入excel
print(data_all)
data_all.to_excel('YIK_QZH_training_data_source.xlsx')



#Comments 修订可能会用到的代码（暂存）
#%%单一航次的停止售卖时间 test
# df_end_sale_time = df_end_sale[df_end_sale['MAIN_SVVD'] == 'CF1-TMB-145 N']
# df_end_sale_time = df_end_sale_time[df_end_sale_time['FCIL_CDE'].str.contains('YIK')]
# df_end_sale_time = df_end_sale_time[df_end_sale_time['SCH_STATE'].str.contains('actual')]
# df_end_sale_time['BERTH_DEP_DT_LOC'] = pd.to_datetime(df_end_sale_time['BERTH_DEP_DT_LOC'], format='%d/%m/%Y %H:%M')
# print(df_end_sale_time)
# df_end_sale_time = df_end_sale_time.sort_values(by='BERTH_DEP_DT_LOC')
# if df_end_sale_time['BERTH_DEP_DT_LOC'][0:1].empty == False:
#     df_end_sale_time = df_end_sale_time['BERTH_DEP_DT_LOC'][0:1]
# else:
#     df_end_sale_time = df_end_sale[df_end_sale['MAIN_SVVD'] == 'CF1-TMB-145 N']
#     df_end_sale_time = df_end_sale_time[df_end_sale_time['FCIL_CDE'].str.contains('YIK')]
#     df_end_sale_time = df_end_sale_time[df_end_sale_time['SCH_STATE'].str.contains('longterm')]
#     df_end_sale_time['BERTH_DEP_DT_LOC'] = pd.to_datetime(df_end_sale_time['BERTH_DEP_DT_LOC'], format='%d/%m/%Y %H:%M')
#     df_end_sale_time = df_end_sale_time.sort_values(by='BERTH_DEP_DT_LOC')
#     df_end_sale_time = df_end_sale_time['BERTH_DEP_DT_LOC'][0:1]
# print(df_end_sale_time)

#%%打印值
#data=df.values
#print("获取到所有的值:\n{0}".format(data))