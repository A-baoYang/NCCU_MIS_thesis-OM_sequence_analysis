from collections import Counter
import datetime as dt
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


"""
"""

# Variables
input_folderpath = 'Cainiao_preprocessed/'
output_folderpath = 'Cainiao_output/'
parser = argparse.ArgumentParser()
parser.add_argument('--produce_date', type=str, default=str(dt.date.today()).replace('-', ''))
parser.add_argument('--check_best_centers', type=bool, default=False)
parser.add_argument('--cluster_method', type=str, default='hcut')
parser.add_argument('--center', type=int, default=2)
parser.add_argument('--OM_version', type=str, default='OM')
parser.add_argument('--sm_method', type=str, default='TRATE')
parser.add_argument('--indel_method', type=str, default='auto')
parser.add_argument('--sent_day', type=int, default=10)
args = parser.parse_args()
produce_date = args.produce_date
sent_day = args.sent_day
check_best_centers = args.check_best_centers
cluster_method = args.cluster_method
center = args.center
OM_version = args.OM_version
sm_method = args.sm_method
indel_method = args.indel_method
file_name = f'{produce_date}-sentday_{sent_day}'


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


def generate_time_features(df, assign_cols, generate_cols):

    df = df.drop(['start_time'], axis=1)
    
    for cid in range(len(assign_cols)):
        df[f'{generate_cols[cid]}_hour'] = pd.to_datetime(df[assign_cols[cid]]).dt.hour
        df[f'{generate_cols[cid]}_dayOfWeek'] = pd.to_datetime(df[assign_cols[cid]]).dt.weekday + 1
        df[f'{generate_cols[cid]}_month'] = pd.to_datetime(df[assign_cols[cid]]).dt.month
        df[f'{generate_cols[cid]}_day'] = pd.to_datetime(df[assign_cols[cid]]).dt.day + 31 * (df[f'{generate_cols[cid]}_month'] - 1)
        print(df.groupby([f'{generate_cols[cid]}_month']).size())

        cond_list = [
            (df[f'{generate_cols[cid]}_hour'] >= 1) & (df[f'{generate_cols[cid]}_hour'] <= 7),
            (df[f'{generate_cols[cid]}_hour'] >= 8) & (df[f'{generate_cols[cid]}_hour'] <= 13),
            (df[f'{generate_cols[cid]}_hour'] >= 14) & (df[f'{generate_cols[cid]}_hour'] <= 19),
            (df[f'{generate_cols[cid]}_hour'] >= 20) & (df[f'{generate_cols[cid]}_hour'] <= 23) | (df[f'{generate_cols[cid]}_hour'] == 0)
        ]
        choice_list = [0, 1, 2, 3]
        df[f'{generate_cols[cid]}_hour_range'] = np.select(condlist=cond_list, choicelist=choice_list, default=-1)
        print(df.groupby([f'{generate_cols[cid]}_hour_range']).size())

        if assign_cols[cid] == 'pay_timestamp':
            cond_list = [
                (df[f'{generate_cols[cid]}_day'] >= 1) & (df[f'{generate_cols[cid]}_day'] <= 10),
                (df[f'{generate_cols[cid]}_day'] >= 11) & (df[f'{generate_cols[cid]}_day'] <= 20),
                (df[f'{generate_cols[cid]}_day'] >= 21) & (df[f'{generate_cols[cid]}_day'] <= 31)
            ]
            choice_list = [0, 1, 2]
            df[f'{generate_cols[cid]}_day_range'] = np.select(condlist=cond_list, choicelist=choice_list, default=-1)
            print(df.groupby([f'{generate_cols[cid]}_day_range']).size())
        df = df.drop([assign_cols[cid]], axis=1)

    return df


def compute_origin_metrics(df):
    df = df.drop(['om_cluster'], axis=1)
    origin_metrics = pd.concat([df.mean().reset_index().iloc[1:, :].rename(columns={0: 'mean'}),
                                df.std().reset_index().iloc[1:, :].rename(columns={0: 'std'})[['std']],
                                df.median().reset_index().iloc[1:, :].rename(columns={0: 'median'})[['median']]], axis=1)
    origin_metrics.index = ['是否承諾送達時限', '是否從菜鳥物流倉儲出貨', '物流滿意度評分', '訂單成立時間(小時)', '訂單成立時間(星期幾)',
                            '訂單成立時間(日)', '訂單成立時間(月)', '訂單成立時間(時段)', '訂單成立時間(日期區間)',
                            '送達時間(小時)', '送達時間(星期幾)', '送達時間(月)', '送達時間(日)', '送達時間(時段)']
    return origin_metrics


def compute_cluster_metrics(df):
    cluster_metrics = df.groupby(['om_cluster']).mean().iloc[:, 1:df.shape[1]].T
    cluster_metrics.columns = ['Cluster1(mean)','Cluster2(mean)']
    cluster_metrics_2 = df.groupby(['om_cluster']).std().iloc[:, 1:df.shape[1]].T
    cluster_metrics_2.columns = ['Cluster1(std)','Cluster2(std)']
    cluster_metrics_3 = df.groupby(['om_cluster']).median().iloc[:, 1:df.shape[1]].T
    cluster_metrics_3.columns = ['Cluster1(median)','Cluster2(median)']
    cluster_metrics = pd.concat([cluster_metrics, cluster_metrics_2, cluster_metrics_3], axis=1)
    cluster_metrics = cluster_metrics[['Cluster1(mean)', 'Cluster1(std)', 'Cluster1(median)', 'Cluster2(mean)', 'Cluster2(std)', 'Cluster2(median)']]
    cluster_metrics.index = ['是否承諾送達時限', '是否從菜鳥物流倉儲出貨', '物流滿意度評分', '訂單成立時間(小時)', '訂單成立時間(星期幾)',
                             '訂單成立時間(日)', '訂單成立時間(月)', '訂單成立時間(時段)', '訂單成立時間(日期區間)',
                             '送達時間(小時)', '送達時間(星期幾)', '送達時間(月)', '送達時間(日)', '送達時間(時段)']
    cluster_metrics = cluster_metrics.apply(lambda x: round(x, 3))

    return cluster_metrics


def compute_kruskal_wallis_test(df, metric, cluster_num):
    origin = df[metric].values
    for cid in range(1, cluster_num + 1):
        comp_ = df[df['om_cluster'] == f'Cluster{cid}'][metric].values
        print(f'Kruskal-Wallis Significance Test: {metric}\nOrigin  v.s.  Cluster{cid}\n{stats.kruskal(origin, comp_)}')



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
    clusteredSeqs = clusteredSeqs[['order_id', f'{cluster_method}_cluster']]
    print(clusteredSeqs.shape)
    print(clusteredSeqs.head())
    sampledOrderLog = pd.read_csv(input_folderpath + f'Cainiao-sampledData_reviewscore-{file_name}.csv')
    sampledOrderLog = sampledOrderLog.drop(['order_date', 'logistic_order_id', 'action', 'facility_id', 'facility_type',
                                            'city_id', 'logistic_company_id', 'timestamp', 'sent_duration'], axis=1)
    sampledOrderLog = sampledOrderLog[sampledOrderLog['order_id'].isin(clusteredSeqs.order_id.unique())]
    sampledOrderLog['promise_speed'] = sampledOrderLog['promise_speed'].fillna(0.0)
    sampledOrderLog['end_time'] = pd.to_datetime(sampledOrderLog['end_time'])
    sampledOrderLog['start_time'] = pd.to_datetime(sampledOrderLog['start_time'])

    assign_cols = ['pay_timestamp', 'end_time']
    generate_cols = ['pay', 'receive']
    sampledOrderLog = generate_time_features(df=sampledOrderLog, assign_cols=assign_cols, generate_cols=generate_cols)
    print(sampledOrderLog.shape)
    print(sampledOrderLog.head())

    # 計算分群前的總體指標平均、標準差與中位數
    sampledOrderLog = pd.merge(sampledOrderLog, clusteredSeqs, on='order_id', how='left')
    sampledOrderLog = sampledOrderLog.rename(columns={f'{cluster_method}_cluster': 'om_cluster'})
    print(sampledOrderLog)
    origin_metrics = compute_origin_metrics(sampledOrderLog)
    origin_metrics.to_csv(output_folderpath + f'origin_metrics-{file_name}.csv', index=False)

    # ---
    # 再根據各群算出各指標的平均、標準差與中位數
    cluster_metrics = compute_cluster_metrics(sampledOrderLog)
    print(cluster_metrics)
    cluster_metrics.to_csv(output_folderpath + f'cluster_metrics-{file_name}-seqdist_{OM_version}-sm_{sm_method}-indel_{indel_method}-method_{cluster_method}-center_{center}.csv',
                           index=False)

    # 計算兩樣本特徵分布顯著性
    for metric in ['Logistics_review_score', 'pay_day', 'receice_day']:
        compute_kruskal_wallis_test(df=sampledOrderLog, metric=metric, cluster_num=2)

