from collections import Counter
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


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
    plt.savefig(output_folderpath + f'silhouette_plot-{file_name}-seqdist_{OM_version}-sm_{sm_method}-indel_{indel_method}-method_{cluster_method}.png', bbox_inches='tight')


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
    print(f'Kruskal-Wallis Significance Test: {metric}\n')
    origin_metric = df[metric].values
    # 各群和分群前的原始樣本比
    for cid in range(1, cluster_num + 1):
        cid_metric = df[df['om_cluster'] == f'Cluster{cid}'][metric].values
        print(f'Origin  v.s.  Cluster{cid}\n{stats.kruskal(origin_metric, cid_metric)}\n')

        # 各群和其他群樣本比
        for cid_comp in range(cid + 1, cluster_num + 1):
            cid_comp_metric = df[df['om_cluster'] == f'Cluster{cid_comp}'][metric].values
            print(f'Cluster{cid}  v.s.  Cluster{cid_comp}\n{stats.kruskal(cid_metric, cid_comp_metric)}')


