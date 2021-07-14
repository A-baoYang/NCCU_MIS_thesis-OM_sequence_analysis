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
    Input: 
        - Input Path: `Cainiao_dataset/`
        - 訂單資料: `msom_order_data_1.csv`
        - 物流進程日誌: `msom_logistic_detail_1.csv`
    Process:
        - 訂單運送天數計算
        - 訂單抽樣
        - 訂單狀態序列矩陣生成
    Output:  
        - Output Path: `Cainiao_dataset/` `Cainiao_preprocessed/`
        - 訂單狀態序列矩陣: `order_logistic_states-{produce_date}-sentday_{sent_day}.csv`
"""


# Variables
dataset_folderpath = 'Cainiao_dataset/'
preprocessed_folderpath = 'Cainiao_preprocessed/'
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
def compute_sent_days(df):

    sent_times = list()
    for oid in tqdm(df.order_id.unique()):
        tmp = df[df['order_id'] == oid].sort_values(['timestamp'])
        try:
            end_time = tmp[tmp['action'] == 'SIGNED'].timestamp.values[-1]
        except:
            pass
        start_time = tmp.pay_timestamp.values[0]
        duration = dt.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S') - dt.datetime.strptime(start_time,
                                                                                              '%Y-%m-%d %H:%M:%S')
        sent_times.append([oid, start_time, end_time, duration.days])

    df_sent_times = pd.DataFrame(sent_times, columns=['order_id', 'start_time', 'end_time', 'sent_duration'])
    print(df_sent_times[df_sent_times['sent_duration'] < 0])
    df_sent_times = df_sent_times[df_sent_times['sent_duration'] >= 0]
    df_sent_times['sent_duration'] = np.where(df_sent_times['sent_duration'] >= 10, '10+',
                                              df_sent_times['sent_duration'])
    df_sent_times['sent_duration'] = df_sent_times['sent_duration'].astype(str)

    return df_sent_times


def generate_logistic_state_sequence(df, order_id_list):

    actions_collect = list()
    reviews_collect = list()
    for oid in tqdm(order_id_list):

        tmp = df[df['order_id'] == oid].sort_values('timestamp')
        # 付款時間為訂單成立時間 (ORDERED)
        order_time = tmp.pay_timestamp.values[0]
        reviewscore = tmp.Logistics_review_score.values[0]
        actions = ['ORDERED'] + tmp.action.values.tolist()
        # 計算狀態間的間隔秒數，以總秒數長度計算佔比，最後用 100 份分給各狀態長度，無條件進位，因此會有總長度大於 100 的情況
        timestamps = np.array(
            [dt.datetime.strptime(order_time, '%Y-%m-%d %H:%M:%S')] + [dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in tmp.timestamp.values])
        gaps = timestamps[1:] - timestamps[:-1]
        gaps = [x.total_seconds() for x in gaps]
        gaps_percent = np.array([x / sum(gaps) for x in gaps])
        gaps = [int(round(x * 100)) for x in gaps_percent]

        # 將每位用戶的狀態拼接起來
        actions_list = [oid]
        # 該位用戶第幾個狀態、要重複的次數
        for i in range(0, len(gaps)):
            actions_list += [actions[i]] * gaps[i]

        # 最後一個動作統一只出現一次
        actions_list.append(actions[-1])

        actions_collect.append(actions_list)
        reviews_collect.append(f'reviewscore_{reviewscore}')

    return actions_collect, reviews_collect



# Mains
# 若本地端沒有該資料夾則則創建
path = pathlib.Path(preprocessed_folderpath)
path.mkdir(parents=True, exist_ok=True)

# 載入訂單及物流進程資料集
order = pd.read_csv(dataset_folderpath + order_filepath, header=None)
order.columns = order_cols
order = order.drop(['day','item_det_info','buyer_id','merchant_id'], axis=1)
order = order[~order['Logistics_review_score'].isnull()]
logistic = pd.read_csv(dataset_folderpath + logistic_filepath, header=None)
logistic.columns = logistic_cols

# 一月份共 1400 多萬筆訂單，從評分 1-5 分各抽樣 3 萬筆訂單，減少運算時間
sample_oids = list()
for score in range(1, 6):
    review_oids = np.random.choice(a=order[order['Logistics_review_score']==score].order_id.unique(), size=30000, replace=False)
    sample_oids.append(review_oids)
sample_oids = np.concatenate(sample_oids)
order = order[order['order_id'].isin(sample_oids)]
print('Shape of order: ', order.shape)
print(order.head())

# 合併訂單和其進程紀錄
logistic = pd.merge(logistic, order, on='order_id', how='left')
logistic = logistic[~logistic['Logistics_review_score'].isnull()]
logistic = logistic.drop_duplicates()
print('Shape of logistic: ', logistic.shape)
print(logistic.head())
# 儲存備用
logistic.to_csv(dataset_folderpath + f'order_logistic_log-{produce_date}.csv', index=False)

# 節省空間將不用的訂單資料表刪除
del order
gc.collect()


# 計算每張訂單的運送天數
df_sent_times = compute_sent_days(df=logistic)
print(df_sent_times.groupby(['sent_duration']).size().reset_index().rename(columns={0: 'count'}))
df_sent_times.to_csv(preprocessed_folderpath + 'cainiao-sent_times.csv', index=False)

# 篩選出運送天數 >10 天的訂單
sent_duration_10dayup = df_sent_times[df_sent_times['sent_duration']=='10+'].order_id.unique()
logistic = logistic[logistic['order_id'].isin(sent_duration_10dayup)]

# 依據滿意度評分各隨機抽樣 1000 筆訂單
sample_oids = list()
for score in [1, 5]:
    review_oids = np.random.choice(a=logistic[logistic['Logistics_review_score']==score].order_id.unique(),
                                   size=1000, replace=False)
    sample_oids.append(review_oids)
sample_oids = np.concatenate(sample_oids)
logistic = logistic[logistic['order_id'].isin(sample_oids)]

# 將抽樣後資料連同運送天數一起儲存備用
logistic = pd.merge(logistic, df_sent_times, on='order_id', how='left')
logistic.to_csv(preprocessed_folderpath + f'Cainiao-sampledData_reviewscore-{file_name}.csv', index=False)

# 計算物流狀態時長佔比，作為序列長度，將每筆訂單轉換為物流狀態序列
actions_collect, reviews_collect = generate_logistic_state_sequence(df=logistic, order_id_list=sample_oids)

# 將物流狀態序列存為 .csv
action_cols = ['action_'+str(x) for x in range(0, pd.DataFrame(actions_collect).shape[1]-1)]
action_cols = ['order_id'] + action_cols
order_logistic_states = pd.DataFrame(actions_collect, columns=action_cols)
order_logistic_states['review_score'] = reviews_collect
order_logistic_states.to_csv(preprocessed_folderpath + f'order_logistic_states-{produce_date}-sentday_{sent_day}.csv', index=False)
print(order_logistic_states.groupby(['review_score']).size())
print(order_logistic_states.shape)
print(order_logistic_states.head())

