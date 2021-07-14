from collections import Counter, OrderedDict
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


def user_daily_state_tolist(user_action_dict):
    collect = []
    user_list = list(user_action_dict.keys())
    for uid in tqdm(user_list):
        tmp_df = pd.DataFrame(user_action_dict[uid])

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


def count_action_num(row):
    action_types = ['browse', 'directly_add_to_consider', 'no_browse', 'browse_to_add_to_consider', 'browse_to_purchase', 'directly_purchase']
    dict_tmp = dict(Counter(row))
    for k in action_types:
        if k not in dict_tmp.keys():
            dict_tmp[k] = 0
            
    dict_tmp = dict(OrderedDict(sorted(dict_tmp.items())))
    return dict_tmp


def silhouette_plot(file_name, OM_version, sm_method, indel_method, cluster_method, max_cluster=8):
    """
    遍歷各個群數，比較側影係數，對標出最佳建議群數

    :param file_name: OM 相異度矩陣檔名
    :param OM_version: 使用的 OM 版本名稱
    :param sm_method: 使用的置換成本計算策略
    :param indel_method: 使用的增刪成本計算策略
    :param cluster_method: 使用的分群算法
    :param max_cluster: 要測試的最大群數範圍
    :return: Average Silhouette Score Plot: 分群群數側影係數關係圖
    """
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


def cluster_result_transition(clusteredSeqs, action_counts_clust, cluster_method):
    """

    :param clusteredSeqs:
    :param action_counts_clust:
    :param cluster_method:
    :return:
    """

    # 將用戶在「事件次數統計」和「所使用的 OM 變化型」的分群標籤並列
    action_counts_clust['user_id'] = clusteredSeqs['user_id']
    action_counts_clust = action_counts_clust[['user_id', f'{cluster_method}_cluster']].rename(columns={f'{cluster_method}_cluster': 'action_counts_cluster'})
    compare_seqdists_clusts = pd.concat([clusteredSeqs, action_counts_clust[['action_counts_cluster']]], axis=1)
    print(compare_seqdists_clusts)

    # 將上述表，計算當群轉移到其他群的個數和比例後，轉換成列聯表
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

    return action_counts_cluster_perc


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


