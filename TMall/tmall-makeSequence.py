import argparse
from collections import Counter
import datetime as dt
import json
import numpy as np
import pandas as pd
import pathlib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

"""
    Input:
        - Input Path: `Tmall_dataset/`
        - `TMall_for_user_states_define_transformed.pkl`
    Process:
        - 序列狀態定義
        - 從日誌格式轉換為狀態序列
    Output:
        - Output Path: `TMall_preprocessed/`
        - 用戶狀態序列矩陣: `TMall_user_state_sequence_table_{product_date}_{version}.csv`
"""

# Variable
input_folderpath = 'TMall_dataset/'
folderpath = 'TMall_preprocessed/'
tmall_log_filepath = 'TMall_for_user_states_define_transformed.pkl'
parser = argparse.ArgumentParser()
parser.add_argument('--produce_date', type=str, default=str(dt.date.today()).replace('-', ''))
parser.add_argument('--start_day', type=int, default=0)
parser.add_argument('--end_day', type=int, default=184)
parser.add_argument('--label', type=str, default='none')
args = parser.parse_args()
produce_date = args.produce_date
start_day = args.start_day
end_day = args.end_day
label = args.label
version = f'V3.2-duration_{start_day}_{end_day}-label_{label}'
output_filepath = f'TMall_user_state_sequence_table_{produce_date}_{version}.csv'


# Function
def user_daily_behavior_todict(user_list, sampled_df, start_day, end_day):
    """
    Generate dictionary to record user daily behavior
    :param user_list:
    :param sampled_df:
    :param start_day:
    :param end_day:
    :return:
    """

    accum_by_timestamp = {}

    for uid in tqdm(user_list):
        accum_by_timestamp[str(uid)] = {}

        tmp_df = sampled_df[sampled_df['user_id'] == uid].groupby(['day_stamp'])[
            'click', 'add_to_cart', 'add_to_favorite', 'purchase'].sum().reset_index()
        tmp_df['add_to_favor_or_cart'] = tmp_df['add_to_favorite'] + tmp_df['add_to_cart']
        tmp_df = tmp_df.drop(['add_to_favorite', 'add_to_cart'], axis=1)
        tmp_df['unique_cat'] = sampled_df[sampled_df['user_id'] == uid].groupby(['day_stamp'])['cat_id'].nunique()

        for i in range(start_day, end_day + 1):
            if i not in tmp_df.day_stamp.values:
                tmp_df = pd.concat([tmp_df, pd.DataFrame([[i, 0, 0, 0, 0]], columns=tmp_df.columns)], ignore_index=True)

        tmp_df = tmp_df.sort_values('day_stamp').reset_index().drop(['index'], axis=1)

        for c in tmp_df.columns:
            accum_by_timestamp[str(uid)][c] = tmp_df[c].values.tolist()

    return accum_by_timestamp


def user_daily_state_tolist(user_list):
    collect = []
    for uid in tqdm(user_list):
        tmp_df = pd.DataFrame(tmall_user_count_stats[uid])

        cond_list = [
            (tmp_df['click'] + tmp_df['purchase'] + tmp_df['add_to_favor_or_cart'] + tmp_df['unique_cat'] == 0),
            (tmp_df['click'] == 0) & (tmp_df['purchase'] > 0),
            (tmp_df['click'] == 0) & (tmp_df['add_to_favor_or_cart'] > 0),
            (tmp_df['click'] > 0) & (tmp_df['purchase'] > 0),
            (tmp_df['click'] > 0) & (tmp_df['add_to_favor_or_cart'] > 0),
            (tmp_df['click'] > 0) & (tmp_df['add_to_favor_or_cart'] + tmp_df['purchase'] == 0)
        ]
        choice_list = [
            'no_browse', 'directly_purchase', 'directly_add_to_consider', 'browse_to_purchase',
            'browse_to_add_to_consider', 'browse'
        ]
        tmp_df['user_state'] = np.select(condlist=cond_list, choicelist=choice_list, default='')

        uss = tmp_df.user_state.values.tolist()
        uss.insert(0, uid)
        collect.append(uss)

    return collect



# Main
# 若本地端沒有該資料夾則則創建
path = pathlib.Path(folderpath)
path.mkdir(parents=True, exist_ok=True)

# 載入資料集
df = pd.read_pickle(input_folderpath + tmall_log_filepath)
# 排除 11/12 的少量流量，統一將雙 11 當天作為最後一天
df = df[df['time_stamp']<=1111]
print('Shape of tmall web log dataset: ', df.shape)
print(df.head())
print('Show proportion of user features: ')
for col in ['age_range', 'gender']:
    print(df.groupby([col]).size())

# 依據年齡層抽樣用戶
user_with_agerange = df[['user_id', 'age_range']].drop_duplicates()
agerange_proportion = user_with_agerange.groupby(['age_range']).size().reset_index().rename(columns={0: 'count'})
agerange_proportion['percentage'] = agerange_proportion['count'] / agerange_proportion['count'].sum()
user_with_agerange = pd.merge(user_with_agerange, agerange_proportion[['age_range', 'percentage']], on='age_range', how='left')
sampled_user_with_agerange = user_with_agerange.sample(frac=0.01, weights='percentage')
print(sampled_user_with_agerange.shape)
print(sampled_user_with_agerange.head())
user_list = sampled_user_with_agerange.user_id.unique()
sampled_df = df[df['user_id'].isin(user_list)].sort_values('day_stamp').reset_index()
sampled_df = sampled_df.drop(['index'], axis=1)
# 儲存抽樣後用戶備用
sampled_df.to_csv(folderpath + f'Tmall-sampledData_agerange-{produce_date}_{version}.csv', index=False)

# 僅篩選抽樣用戶行為紀錄使用
sampled_df = sampled_df[(sampled_df['day_stamp'] >= start_day) & (sampled_df['day_stamp'] <= end_day)]

# 以字典儲存抽樣用戶的每日各項行為次數
accum_by_timestamp = user_daily_behavior_todict(user_list=user_list, sampled_df=sampled_df, start_day=start_day, end_day=end_day)
with open(folderpath + 'TMall_user_counts_{}_{}.json'.format(produce_date, version), 'w') as fp:
    json.dump(accum_by_timestamp, fp)

# 測試能否順利載入
with open(folderpath + 'TMall_user_counts_{}_{}.json'.format(produce_date, version), 'r') as f:
    tmall_user_count_stats = json.load(f)
print(tmall_user_count_stats.keys())

# 將字典中所有抽樣用戶每日各項行為次數，轉換為狀態序列矩陣
user_daily_states = user_daily_state_tolist(user_list=list(tmall_user_count_stats.keys()))

# 儲存用戶狀態序列為 .csv
print(produce_date, version)
columns = ['user_id'] + ['day_' + str(x) for x in range(start_day, end_day + 1)]
user_states_table = pd.DataFrame(user_daily_states, columns=columns)
user_states_table.to_csv(folderpath + output_filepath, index=False)
print(user_states_table.head())

# 檢視用戶狀態分布比例
states_count_dict = dict(Counter(user_states_table.iloc[:, 1:user_states_table.shape[1]].values.reshape(-1).tolist()))
states_count_df = pd.DataFrame([list(states_count_dict.keys()), list(states_count_dict.values())]).T
states_count_df.columns = ['action', 'count']
states_count_df.to_csv(folderpath + f'TMall_user_state-count_{produce_date}_{version}.csv', index=False)
print(states_count_df)

