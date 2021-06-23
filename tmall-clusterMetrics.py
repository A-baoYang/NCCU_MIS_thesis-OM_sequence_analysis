from collections import defaultdict, Counter
import datetime as dt
import gc
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import plotly.express as px
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


"""
"""

# Variables
input_folderpath = 'TMall_preprocessed/'
output_folderpath = 'TMall_output/'
parser = argparse.ArgumentParser()
parser.add_argument('--produce_date', type=str, default=str(dt.date.today()).replace('-', ''))
parser.add_argument('--start_day', type=int, default=0)
parser.add_argument('--end_day', type=int, default=184)
parser.add_argument('--label', type=str, default='none')
parser.add_argument('--check_best_centers', type=bool, default=False)
parser.add_argument('--cluster_method', type=str, default='hcut')
parser.add_argument('--center', type=int, default=2)
parser.add_argument('--OM_version', type=str, default='OM')
parser.add_argument('--sm_method', type=str, default='TRATE')
parser.add_argument('--indel_method', type=str, default='auto')
args = parser.parse_args()
produce_date = args.produce_date
start_day = args.start_day
end_day = args.end_day
label = args.label
check_best_centers = args.check_best_centers
cluster_method = args.cluster_method
center = args.center
OM_version = args.OM_version
sm_method = args.sm_method
indel_method = args.indel_method
file_name = f'{produce_date}_V3.2-duration_{start_day}_{end_day}-label_{label}'


# Function
def silhouette_plot(file_name, OM_version, sm_method, indel_method, cluster_method, max_cluster=8):

    dis_matrix = pd.read_csv(f'dissimilarity_matrix-{file_name}-seqdist_{OM_version}-sm_{sm_method}-indel_{indel_method}.csv')
    n_clusters = list(range(2, max_cluster + 1))
    silhouette_avg_list = list()
    for n_cluster in n_clusters:
        if cluster_method == 'kmeans':
            # KMeans
            clf = KMeans(n_clusters=n_cluster, random_state=np.random.randint(100))
        elif cluster_method == 'hcut':
            # Hierarchical Clustering
            clf = AgglomerativeClustering(n_clusters=n_cluster)
        else:
            print('Need to update the function with new clustering algorithms.')
            break

        cluster_labels = clf.fit_predict(dis_matrix)
        silhouette_avg = silhouette_score(dis_matrix, cluster_labels)
        silhouette_avg_list.append(silhouette_avg)
        print('For n_clusters = ', n_cluster,
              'The average silhouette_score is: ', silhouette_avg)

    plt.figure(figsize=(10, 6))
    plt.grid()
    sns.lineplot(n_clusters, silhouette_avg_list, color='green')
    plt.title(f'Average Silhouette Score Plot ({file_name}-seqdist_{OM_version}-sm_{sm_method}-indel_{indel_method}-method_{cluster_method})', size=18)
    plt.xlabel('n_clusters', size=12)
    plt.ylabel('average silhouette score', size=12)
    plt.savefig(output_folderpath + f'silhouette_plot-{file_name}-seqdist_{OM_version}-sm_{sm_method}-indel_{indel_method}-method_{cluster_method}.png',
                bbox_inches='tight')


def cluster_result_transition(clusteredSeqs, action_counts_clust, euclidean_distance_clust, ):
    action_counts_clust['user_id'] = clusteredSeqs['user_id']
    action_counts_clust = action_counts_clust[['user_id', f'{cluster_method}_cluster']].rename(columns={f'{cluster_method}_cluster': 'action_counts_cluster'})
    euclidean_distance_clust['user_id'] = clusteredSeqs['user_id']
    euclidean_distance_clust = euclidean_distance_clust[['user_id', f'{cluster_method}_cluster']].rename(columns={f'{cluster_method}_cluster': 'euclidean_distance_cluster'})
    compare_seqdists_clusts = pd.concat([clusteredSeqs, action_counts_clust[['action_counts_cluster']]], axis=1)
    compare_seqdists_clusts = pd.concat([compare_seqdists_clusts, euclidean_distance_clust[['euclidean_distance_cluster']]], axis=1)
    print(compare_seqdists_clusts)

    # 與事件次數比較
    compare_om_actionCounts = compare_seqdists_clusts[[f'{cluster_method}_cluster', 'action_counts_cluster']]
    compare_om_actionCounts = [[k[0], k[1], v] for k, v in dict(Counter([(i, j) for i, j in compare_om_actionCounts.values])).items()]
    action_counts_cluster = pd.DataFrame(compare_om_actionCounts, columns=[f'{cluster_method}_cluster', 'action_counts_cluster', 'cover_num'])
    action_counts_cluster = action_counts_cluster.pivot_table(index='action_counts_cluster', columns=f'{cluster_method}_cluster', values='cover_num')
    action_counts_cluster = action_counts_cluster.fillna(0)
    action_counts_cluster = action_counts_cluster.astype(int)
    print(action_counts_cluster)
    action_counts_cluster_perc = action_counts_cluster.copy()
    action_counts_cluster_perc = round(action_counts_cluster_perc / action_counts_cluster_perc.sum(), 3)
    # 讓 0 的部分維整數，其餘為小數到第三位
    action_counts_cluster_perc = action_counts_cluster_perc.replace(0.0, '0')
    print(action_counts_cluster_perc)

    # 與事件次數向量之歐式距離比較
    compare_om_euclideanDist = compare_seqdists_clusts[[f'{cluster_method}_cluster','euclidean_distance_cluster']]
    compare_om_euclideanDist = [[k[0], k[1], v] for k, v in dict(Counter([(i, j) for i, j in compare_om_euclideanDist.values])).items()]
    euclidean_distance_cluster = pd.DataFrame(compare_om_euclideanDist, columns=[f'{cluster_method}_cluster','euclidean_distance_cluster','cover_num'])
    euclidean_distance_cluster = euclidean_distance_cluster.pivot_table(index='euclidean_distance_cluster', columns=f'{cluster_method}_cluster', values='cover_num')
    euclidean_distance_cluster = euclidean_distance_cluster.fillna(0)
    euclidean_distance_cluster = euclidean_distance_cluster.astype(int)
    print(euclidean_distance_cluster)
    euclidean_distance_cluster_perc = euclidean_distance_cluster.copy()
    euclidean_distance_cluster_perc = round(euclidean_distance_cluster_perc / euclidean_distance_cluster_perc.sum(), 3)
    # 讓 0 的部分維整數，其餘為小數到第三位
    euclidean_distance_cluster_perc = euclidean_distance_cluster_perc.replace(0.0, '0')
    print(euclidean_distance_cluster_perc)

    return action_counts_cluster_perc, euclidean_distance_cluster_perc


# 計算每人看過不重複品類數 當作特徵 cat_id, merchant_id, brand_id
def compute_user_metrics(df, clustered_df):
    df_by_user = pd.DataFrame({
        'gender': df.groupby(['user_id'])['gender'].mean(),
        'age_range': df.groupby(['user_id'])['age_range'].mean(),
        'daysVisited': df.groupby(['user_id'])['time_stamp'].nunique(),
        'clicks': df.groupby(['user_id'])['click'].sum(),
        'add_to_carts': df.groupby(['user_id'])['add_to_cart'].sum(),
        'add_to_favorites': df.groupby(['user_id'])['add_to_favorite'].sum(),
        'purchases': df.groupby(['user_id'])['purchase'].sum(),
        'unique_cat_clicks': df.groupby(['user_id'])['cat_id'].nunique(),
        'unique_merchant_clicks': df.groupby(['user_id'])['merchant_id'].nunique(),
        'unique_brand_clicks': df.groupby(['user_id'])['brand_id'].nunique(),
        'unique_cat_addToCarts': df[df['add_to_cart'] > 0].groupby(['user_id'])['cat_id'].nunique(),
        'unique_merchant_addToCarts': df[df['add_to_cart'] > 0].groupby(['user_id'])['merchant_id'].nunique(),
        'unique_brand_addToCarts': df[df['add_to_cart'] > 0].groupby(['user_id'])['brand_id'].nunique(),
        'unique_cat_addToFavorites': df[df['add_to_favorite'] > 0].groupby(['user_id'])['cat_id'].nunique(),
        'unique_merchant_addToFavorites': df[df['add_to_favorite'] > 0].groupby(['user_id'])['merchant_id'].nunique(),
        'unique_brand_addToFavorites': df[df['add_to_favorite'] > 0].groupby(['user_id'])['brand_id'].nunique(),
        'unique_cat_purchases': df[df['purchase'] > 0].groupby(['user_id'])['cat_id'].nunique(),
        'unique_merchant_purchases': df[df['purchase'] > 0].groupby(['user_id'])['merchant_id'].nunique(),
        'unique_brand_purchases': df[df['purchase'] > 0].groupby(['user_id'])['brand_id'].nunique(),
        'purchasesOn1111': df[df['time_stamp'] == 1111].groupby(['user_id'])['purchase'].sum(),
        'purchasesBefore1111': df[df['time_stamp'] < 1111].groupby(['user_id'])['purchase'].sum()
    })
    df_by_user = df_by_user.reset_index()
    df_by_user['clicks'] = df_by_user['clicks'] + df_by_user['add_to_carts'] + df_by_user['add_to_favorites'] + \
                           df_by_user['purchases']
    df_by_user['cvr'] = df_by_user['purchases'] / df_by_user['clicks']
    df_by_user['isRepeatBuyer'] = np.where(df_by_user['purchases'] > 1, 1, 0)
    print(df_by_user.isnull().sum())
    df_by_user = df_by_user.fillna(0)
    df_by_user = pd.merge(df_by_user, clustered_df, on='user_id', how='left')
    df_by_user.columns = ['user_id', '性別', '年齡層', '每人進站天數', '每人總點擊數', '每人總購物車數', '每人總願望清單數',
                          '每人總購買數', '每人點擊不重複品類數', '每人點擊不重複商家數', '每人點擊不重複品牌數',
                          '每人購物車不重複品類數', '每人購物車不重複商家數', '每人購物車不重複品牌數', '每人願望清單不重複品類數',
                          '每人願望清單不重複商家數', '每人願望清單不重複品牌數', '每人購買不重複品類數', '每人購買不重複商家數',
                          '每人購買不重複品牌數', '雙11當天購買數', '雙11前購買數', '轉換率', '是否為回購者', 'om_cluster']

    return df_by_user


def compute_origin_metrics(df_by_user):
    df_by_user_origin = df_by_user.drop(['om_cluster'], axis=1)
    origin_metrics = pd.concat([df_by_user_origin.iloc[:, 3:].mean().reset_index().rename(columns={0: 'average'}),
                                 df_by_user_origin.iloc[:, 3:].std().reset_index().rename(columns={0: 'standard deviation'})[['standard deviation']],
                                 df_by_user_origin.iloc[:, 3:].median().reset_index().rename(columns={0: 'median'})[['median']]], axis=1)

    return origin_metrics


def compute_cluster_metrics(df_by_user):
    cluster_metrics = df_by_user.groupby(['om_cluster']).mean().iloc[:, 8:20].T
    cluster_metrics.columns = [col+'(mean)' for col in cluster_metrics.columns]
    cluster_metrics_2 = df_by_user.groupby(['om_cluster']).std().iloc[:, 8:20].T
    cluster_metrics_2.columns = [col+'(std)' for col in cluster_metrics_2.columns]
    cluster_metrics_3 = df_by_user.groupby(['om_cluster']).median().iloc[:, 8:20].T
    cluster_metrics_3.columns = [col+'(median)' for col in cluster_metrics_3.columns]
    cluster_metrics = pd.concat([cluster_metrics, cluster_metrics_2, cluster_metrics_3], axis=1)
    cluster_metrics = cluster_metrics.apply(lambda x: round(x, 3))

    return cluster_metrics



# Main
# Create folder for store output datasets if the folder hasn't been built
path = pathlib.Path(output_folderpath)
path.mkdir(parents=True, exist_ok=True)

if check_best_centers:

    cluster_methods = ['hcut', 'kmeans']
    OM_versions = ['OM', 'OMloc', 'OMslen', 'OMspell', 'OMstran']

    print('Start checking for best cluster center number...')
    for cluster_method in cluster_methods:
        for OM_version in OM_versions:
            silhouette_plot(file_name=file_name, OM_version=OM_version, sm_method=sm_method, indel_method=indel_method,
                            cluster_method=cluster_method, max_cluster=8)

else:
    print('Start generating cluster metrics...')

    clusteredSeqs = pd.read_csv(output_folderpath + f'clustered-{file_name}-seqdist_{OM_version}-sm_{sm_method}-indel_{indel_method}-method_{cluster_method}-center_{center}.csv')
    clusteredSeqs = clusteredSeqs[['user_id', f'{cluster_method}_cluster']]
    print(clusteredSeqs.shape)
    print(clusteredSeqs.head())
    sampledWebLog = pd.read_csv(input_folderpath + f'Tmall-sampledData_agerange-{file_name}-seqdist_{OM_version}.csv')
    print(sampledWebLog.shape)
    print(sampledWebLog.head())

    # 計算分群結果移轉情形
    action_counts_clust = pd.read_csv(output_folderpath + f'clustered-{file_name}-seqdist_action_counts-method_{cluster_method}-center_{center}.csv')
    euclidean_distance_clust = pd.read_csv(output_folderpath + f'clustered-{file_name}-seqdist_euclidean_distance-method_{cluster_method}-center_{center}.csv')
    action_counts_cluster_perc, euclidean_distance_cluster_perc = cluster_result_transition(clusteredSeqs, action_counts_clust, euclidean_distance_clust)
    action_counts_cluster_perc.to_csv(output_folderpath + f'cluster_transition-action_counts-{file_name}-seqdist_{OM_version}-sm_{sm_method}-indel_{indel_method}-method_{cluster_method}-center_{center}.csv', index=False)
    euclidean_distance_cluster_perc.to_csv(output_folderpath + f'cluster_transition-euclidean_distance-{file_name}-seqdist_{OM_version}-sm_{sm_method}-indel_{indel_method}-method_{cluster_method}-center_{center}.csv', index=False)

    # 計算每位用戶的指標
    df_by_user = compute_user_metrics(df=sampledWebLog, clustered_df=clusteredSeqs)
    print(df_by_user)
    # 計算分群前的總體指標平均、標準差與中位數
    origin_metrics = compute_origin_metrics(df_by_user)
    origin_metrics.to_csv(output_folderpath + f'origin_metrics-{file_name}.csv', index=False)
    # 計算分群前的性別佔比
    gender_perc = df_by_user.groupby(['性別'])['user_id'].count().reset_index().rename(columns={'user_id': 'proportion'})
    gender_perc['proportion'] = round(gender_perc['proportion'] / gender_perc['proportion'].sum() * 100, 3)
    gender_perc.to_csv(output_folderpath + f'origin_metrics_gender-{file_name}.csv', index=False)
    print(gender_perc)
    fig = px.pie(data_frame=gender_perc, values='proportion', names='性別',
                     title=f'性別佔比', width=500, height=500)
    fig.show()
    # 計算分群前的年齡層佔比
    age_perc = df_by_user.groupby(['年齡層'])['user_id'].count().reset_index().rename(columns={'user_id': 'proportion'})
    age_perc['proportion'] = round(age_perc['proportion'] / age_perc['proportion'].sum() * 100, 3)
    age_perc.to_csv(output_folderpath + f'origin_metrics_age-{file_name}.csv', index=False)
    print(age_perc)
    fig = px.bar(data_frame=age_perc, x='年齡層', y='proportion', barmode='overlay',
                 title='年齡層佔比', width=700, height=500)
    fig.show()

    # ---
    # 再根據各群算出各指標的平均、標準差與中位數
    cluster_metrics = compute_cluster_metrics(df_by_user)
    print(cluster_metrics)
    cluster_metrics.to_csv(output_folderpath + f'cluster_metrics-{file_name}-seqdist_{OM_version}-sm_{sm_method}-indel_{indel_method}-method_{cluster_method}-center_{center}.csv',
                           index=False)
    # 計算各群性別佔比表
    gender_perc = df_by_user.groupby(['om_cluster', '性別'])['user_id'].count().groupby(level=0).apply(lambda x: x / x.sum() * 100).reset_index()
    gender_perc_pivot = gender_perc.pivot(index='性別', columns='om_cluster', values='user_id').apply(lambda x: round(x, 3))
    gender_perc_pivot.to_csv(output_folderpath + f'cluster_metrics_gender-{file_name}-seqdist_{OM_version}-sm_{sm_method}-indel_{indel_method}-method_{cluster_method}-center_{center}.csv',
                             index=False)
    print(gender_perc_pivot)
    # 計算各群年齡層佔比表
    age_perc = df_by_user.groupby(['om_cluster', '年齡層'])['user_id'].count().groupby(level=0).apply(lambda x: x / x.sum() * 100).reset_index()
    age_perc_pivot = age_perc.pivot(index='年齡層', columns='om_cluster', values='user_id').apply(lambda x: round(x, 3))
    age_perc_pivot.to_csv(output_folderpath + f'cluster_metrics_age-{file_name}-seqdist_{OM_version}-sm_{sm_method}-indel-{indel_method}-method_{cluster_method}-center_{center}.csv',
                          index=False)
    print(age_perc_pivot)

