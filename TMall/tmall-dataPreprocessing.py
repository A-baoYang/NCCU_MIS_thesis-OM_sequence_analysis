import datetime as dt
import numpy as np
import pandas as pd
import pathlib
import warnings
warnings.filterwarnings('ignore')

"""
    Input:
        - data_format1/user_info_format1.csv
        - data_format1/user_log_format1.csv
    Process:
        - 缺失值處理
        - 新增事件及時間欄位方便後續生成序列使用
    Output:  
        - Output Path: `Tmall_dataset/`
        - TMall_for_user_states_define_transformed.pkl
"""


# Variables
rawdata_folderpath = 'TMall_dataset/'
userinfo_filepath = 'user_info_format1.csv'
userlog_filepath = 'user_log_format1.csv'


# Function
def generate_time_features(df, time_col):
    df['day'] = df[time_col].apply(lambda x: int(str(x)[-2:]))
    df['month'] = df[time_col].apply(lambda x: int(str(x)[:-2]))
    df['dayOfWeek'] = df[time_col].apply(
        lambda x: dt.datetime(2019, int(str(x)[:-2]), int(str(x)[-2:])).weekday() + 1)
    df['day_stamp'] = df[time_col].apply(
        lambda x: dt.datetime(2019, int(str(x)[:-2]), int(str(x)[-2:])).timetuple().tm_yday - dt.datetime(2019, 5, 11).timetuple().tm_yday)
    df['weekOfYear'] = df[time_col].apply(
        lambda x: dt.datetime.strftime(dt.datetime(2019, int(str(x)[:-2]), int(str(x)[-2:])), '%U'))

    return df


# Main

# Load datasets
fm1_user = pd.read_csv(rawdata_folderpath + 'data_format1/' + userinfo_filepath)
fm1_userLog = pd.read_csv(rawdata_folderpath + 'data_format1/' + userlog_filepath)
fm1_userLog.rename(columns={'seller_id': 'merchant_id'}, inplace=True)
df = pd.merge(fm1_userLog, fm1_user, on=['user_id'], how='left')
df.isnull().sum()  # brand_id, age_range, gender 有缺值

# Missing values preprocessing
for col in ['brand_id', 'age_range', 'gender']:
    df['brand_id'] = df[col].fillna(df[col].mode())
df.isnull().sum()  # 確認無缺值

# Features transformation
action_list = ['click', 'add_to_cart', 'purchase', 'add_to_favorite']
dummy_actionType = pd.get_dummies(df['action_type'])
dummy_actionType.columns = action_list
df = pd.concat([df, dummy_actionType], axis=1)
df.drop(['action_type'], axis=1, inplace=True)
df = generate_time_features(df=df, time_col='time_stamp')

# Save the preprocessed dataset
df.to_pickle(rawdata_folderpath + 'TMall_for_user_states_define_transformed.pkl')

