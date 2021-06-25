import argparse
import datetime as dt
import gc
import numpy as np
import pandas as pd
import pathlib
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

"""
    Input: `Cainiao_dataset`
        - 
    Process:
        - 缺失值處理
        - 新增事件及時間欄位方便後續生成序列使用
    Output:  
        - Output Path: `Cainiao_dataset/` `Cainiao_preprocessed`
        - 
"""

# Variables
rawdata_folderpath = 'Cainiao_dataset/'
output_folderpath = 'Cainiao_preprocessed/'
order_filepath = 'msom_order_data_1.csv'
logistic_filepath = 'msom_logistic_detail_1.csv'
logistic_cols = ['order_id','order_date','logistic_order_id','action','facility_id','facility_type','city_id',
                 'logistic_company_id','timestamp']
order_cols = ['day','order_id','item_det_info','pay_timestamp','buyer_id','promise_speed','if_cainiao','merchant_id',
              'Logistics_review_score']
parser = argparse.ArgumentParser()
parser.add_argument('--produce_date', type=str, default=str(dt.date.today()).replace('-', ''))
parser.add_argument('--sent_day', type=int, default=10)
args = parser.parse_args()
produce_date = args.produce_date
sent_day = args.sent_day
file_name = f'{produce_date}-sentday_{sent_day}'


# Function



# Mains
# Load datasets
order = pd.read_csv(rawdata_folderpath + order_filepath, header=None)
order.columns = order_cols
order = order.drop(['day','item_det_info','buyer_id','merchant_id'], axis=1)
order = order[~order['Logistics_review_score'].isnull()]
# 一月份共 1400 多萬筆訂單，從評分 1-5 分各抽樣 3 萬筆訂單，減少運算時間
sample_oids = list()
for score in range(1, 6):
    review_oids = np.random.choice(a=order[order['Logistics_review_score']==score].order_id.unique(), size=30000, replace=False)
    sample_oids.append(review_oids)
sample_oids = np.concatenate(sample_oids)
order = order[order['order_id'].isin(sample_oids)]
print('Shape of order: ', order.shape)
print(order.head())

logistic = pd.read_csv(rawdata_folderpath + logistic_filepath, header=None)
logistic.columns = logistic_cols
logistic = pd.merge(logistic, order, on='order_id', how='left')
logistic = logistic[~logistic['Logistics_review_score'].isnull()]
logistic = logistic.drop_duplicates()
print('Shape of logistic: ', logistic.shape)
print(logistic.head())
logistic.to_csv(rawdata_folderpath + f'order_logistic_log-{produce_date}.csv', index=False)

del order
gc.collect()

# 計算每張訂單的運送天數
sent_times = list()
for oid in tqdm(logistic.order_id.unique()):
    tmp = logistic[logistic['order_id']==oid].sort_values(['timestamp'])
    try:
        end_time = tmp[tmp['action']=='SIGNED'].timestamp.values[-1]
    except:
        pass
    start_time = tmp.pay_timestamp.values[0]
    duration = dt.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S') - dt.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    sent_times.append([oid, start_time, end_time, duration.days])

df_sent_times = pd.DataFrame(sent_times, columns=['order_id', 'start_time', 'end_time', 'sent_duration'])
print(df_sent_times[df_sent_times['sent_duration']<0])
df_sent_times = df_sent_times[df_sent_times['sent_duration']>=0]
df_sent_times['sent_duration'] = np.where(df_sent_times['sent_duration']>=10, '10+', df_sent_times['sent_duration'])
df_sent_times['sent_duration'] = df_sent_times['sent_duration'].astype(str)
print(df_sent_times.groupby(['sent_duration']).size().reset_index().rename(columns={0: 'count'}))
df_sent_times.to_csv(output_folderpath + 'cainiao-sent_times.csv', index=False)
# 篩選出運送天數 >10 天的訂單
sent_duration_10dayup = df_sent_times[df_sent_times['sent_duration']=='10+'].order_id.unique(); print(sent_duration_10dayup.shape)
logistic = logistic[logistic['order_id'].isin(sent_duration_10dayup)]

# 依據滿意度評分各隨機抽樣 1000 筆訂單
sample_oids = list()
for score in [1, 5]:
    review_oids = np.random.choice(a=logistic[logistic['Logistics_review_score']==score].order_id.unique(), size=1000, replace=False)
    sample_oids.append(review_oids)
sample_oids = np.concatenate(sample_oids)
logistic = logistic[logistic['order_id'].isin(sample_oids)]
logistic = pd.merge(logistic, df_sent_times, on='order_id', how='left')
logistic.to_csv(output_folderpath + f'Cainiao-sampledData_reviewscore-{file_name}.csv', index=False)

# 計算物流狀態時長佔比，作為序列長度，將每筆訂單轉換為物流狀態序列
# start: ORDERED
# end: SIGNED
power = 100
actions_collect = []
reviews_collect = []
for oid in tqdm(sample_oids):

    tmp = logistic[logistic['order_id'] == oid].sort_values('timestamp')
    order_time = tmp.pay_timestamp.values[0]
    reviewscore = tmp.Logistics_review_score.values[0]
    actions = ['ORDERED'] + tmp.action.values.tolist()
    timestamps = np.array(
        [dt.datetime.strptime(order_time, '%Y-%m-%d %H:%M:%S')] + [dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in
                                                                   tmp.timestamp.values])
    gaps = timestamps[1:] - timestamps[:-1]

    gaps = [x.total_seconds() for x in gaps]
    gaps_percent = np.array([x / sum(gaps) for x in gaps])
    gaps = [int(round(x * int(power))) for x in gaps_percent]

    try:
        end_time = tmp[tmp['action']=='SIGNED'].timestamp.values[-1]
    except:
        print(f'oid: {oid} lack of SIGNED action.')

    try:
        start_time = tmp[tmp['action']=='CONSIGN'].timestamp.values[0]
    except:
        print(f'oid: {oid} lack of CONSIGN action.')

    actions_list = [oid]
    for i in range(0, len(gaps)):
        for num in range(0, gaps[i] + 1):
            actions_list.append(actions[i])

    actions_list.append(actions[-1])
    actions_collect.append(actions_list)
    reviews_collect.append(f'reviewscore_{reviewscore}')


action_cols = ['action_'+str(x) for x in range(0, pd.DataFrame(actions_collect).shape[1]-1)]
action_cols = ['order_id'] + action_cols
order_logistic_states = pd.DataFrame(actions_collect, columns=action_cols)
order_logistic_states['review_score'] = reviews_collect
order_logistic_states.to_csv(output_folderpath + f'order_logistic_states-{produce_date}-sentday_{sent_day}.csv', index=False)
print(order_logistic_states.groupby(['review_score']).size())
print(order_logistic_states.shape)
print(order_logistic_states.head())

