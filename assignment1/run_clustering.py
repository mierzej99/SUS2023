import argparse
import time

import load_data
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import os
from tqdm import tqdm
from scipy import signal
import numpy as np


def arguments():
    """
    prepering program argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--file_path')
    return parser.parse_args()


def assign_files_to_clusters(file_names, labels):
    return [[x, y] for x, y in zip(file_names, labels)]


def text_file_output(result, labels):
    with open('result.txt', 'w') as f:
        for label in list(set(labels)):
            for x in result:
                if x[1] == label:
                    f.write(os.path.basename(x[0]) + ' ')
            f.write('\n')


def html_file_output(result, labels):
    with open('result.html', 'w') as f:
        f.write('''<html>
        <head>
        <title>results</title>
        </head> 
        <body>
        <hr>
        ''')
        for label in list(set(labels)):
            for x in result:
                if x[1] == label:
                    f.write(f"<img src=\"{x[0]}\">\n")
            f.write('<hr>\n')
        f.write('''</body>
        </html>''')


def dict_result(data, labels):
    cluster_dict = {k: [] for k in set(labels)}
    for i, label in enumerate(labels):
        cluster_dict[label].append(data[i])
    return cluster_dict


def cross_corr_score(data, labels):
    datas = dict_result(data, labels)
    n = len(data[0])

    ovr_cross_corr = np.empty(len(datas))
    for label in datas.keys():
        in_cluster_cross_corr = np.empty(len(datas[label]) - 1)
        for i, x in enumerate(datas[label][:-1]):
            y1 = datas[label][i]
            y2 = datas[label][i + 1]
            in_cluster_cross_corr[i] = max(signal.correlate(y2, y1, mode='same') / np.sqrt(
                signal.correlate(y1, y1, mode='same')[int(n / 2)] * signal.correlate(y2, y2, mode='same')[
                    int(n / 2)]))
        ovr_cross_corr[label] = in_cluster_cross_corr.mean()
    return ovr_cross_corr.mean()


def dbscan(data):
    eps = [0.5 * x for x in range(1, 11)]
    min_smaple = [x for x in range(2, 7)]
    metric = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
    ps = [x for x in range(1, 6)]

    score = -1

    for e in tqdm(eps):
        for ms in min_smaple:
            for met in metric:
                for p in ps:
                    db = DBSCAN(eps=e, min_samples=ms, metric=met, p=p, n_jobs=-1).fit(data)
                    if len(set(db.labels_)) > 1 and cross_corr_score(data, db.labels_) > score:
                        eb, msb, metb, pb = e, ms, met, p

    db = DBSCAN(eps=eb, min_samples=msb, metric=metb, p=pb, n_jobs=-1).fit(data)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'dbscan: eps={eb}, min_samples={msb}, metric={metb}, p={pb}, cross_corr_score={cross_corr_score(data, db.labels_)}, silhouette_score={silhouette_score(data, db.labels_)}\n')
    return db


def kmeans(data):
    n_clusters = [x for x in range(2, 60)]
    max_iter = [100 * x for x in range(1, 5)]
    tol = [10 ** x for x in range(-5, 1)]

    score = -1

    for clust in tqdm(n_clusters):
        for mi in max_iter:
            for t in tol:
                kmeans = KMeans(n_clusters=clust, max_iter=mi, tol=t, n_init='auto').fit(data)
                if len(set(kmeans.labels_)) > 1 and cross_corr_score(data, kmeans.labels_) > score:
                    clustb, mib, tb = clust, mi, t

    kmeans = KMeans(n_clusters=clustb, max_iter=mib, tol=tb, n_init='auto').fit(data)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'kmeans: n_clusters={clustb}, max_iter={mib}, tol={tb}, cross_corr_score={cross_corr_score(data, kmeans.labels_)}, silhouette_score={silhouette_score(data, kmeans.labels_)}\n')
    return kmeans


def agg_n(data):
    n_clusters = [x for x in range(2, 60)]
    linkage = ['ward', 'complete', 'average', 'single']

    score = -1

    for clust in tqdm(n_clusters):
        for li in linkage:
            agglo = AgglomerativeClustering(n_clusters=clust, linkage=li).fit(data)
            if len(set(agglo.labels_)) > 1 and cross_corr_score(data, agglo.labels_) > score:
                clustb, lib = clust, li

    agglo = AgglomerativeClustering(n_clusters=clustb, linkage=lib).fit(data)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'agglo_n: n_clusters={clustb}, linkage={lib}, cross_corr_score={cross_corr_score(data, agglo.labels_)}, silhouette_score={silhouette_score(data, agglo.labels_)}\n')
    return agglo


def agg_guess(data):
    linkage = ['ward']  # ['ward', 'complete', 'average', 'single']
    distance_threshold = [10 ** x for x in range(-3, 3)]

    score = -1

    for thre in tqdm(distance_threshold):
        for li in linkage:
            agglo = AgglomerativeClustering(n_clusters=None, distance_threshold=thre, linkage=li).fit(data)
            print(agglo.labels_)
            if len(set(agglo.labels_)) > 1 and cross_corr_score(data, agglo.labels_) > score:
                threb, lib = thre, li

    agglo = AgglomerativeClustering(n_clusters=None, thre=threb, linkage=lib).fit(data)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'agglo_guess: distance_threshold={threb}, linkage={lib}, cross_corr_score={cross_corr_score(data, agglo.labels_)}, silhouette_score={silhouette_score(data, agglo.labels_)}\n')
    return agglo


def gausian_mm(data):
    n_components = [x for x in range(2, 60, 2)]
    covariance_type = ['full', 'tied', 'diag', 'spherical']
    tol = [10 ** x for x in range(-5, -3)]
    reg_covar = [10 ** x for x in range(-7, -5)]
    max_iter = [100 * x for x in range(1, 3)]
    init_params = ['kmeans', 'k-means++', 'random', 'random_from_data']

    score = -1

    for n_comp in tqdm(n_components):
        for cov in covariance_type:
            for t in tol:
                for reg in reg_covar:
                    for mi in max_iter:
                        for ini in init_params:
                            gmm = GaussianMixture(n_components=n_comp, covariance_type=cov, tol=t, reg_covar=reg,
                                                  max_iter=mi, init_params=ini).fit(data)
                            labels = gmm.predict(data)
                            if len(set(labels)) > 1 and cross_corr_score(data, labels) > score:
                                n_compb, covb, tb, regb, mib, inib = n_comp, cov, t, reg, mi, ini

    gmm = GaussianMixture(n_components=n_compb, covariance_type=covb, tol=tb, reg_covar=regb, max_iter=mib,
                          init_params=inib).fit(data)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'gausian_mm: n_components={n_compb}, covariance_type={covb}, tol={tb}, reg_covar={regb}, max_iter={mib}, init_params={inib}, cross_corr_score={cross_corr_score(data, gmm.labels_)}, silhouette_score={silhouette_score(data, gmm.labels_)}\n')
    return gmm


def main():
    args = arguments()
    data, file_names = load_data.load_images(args.file_path)
    clusterings = []
    clusterings.append(dbscan(data))
    clusterings.append(kmeans(data))
    clusterings.append(agg_n(data))
    # clusterings.append(agg_guess(data))
    clusterings.append(gausian_mm(data))

    # for clustering in clusterings:
    #    result = assign_files_to_clusters(file_names, clustering.labels_)
    # text_file_output(result, km.labels_)
    # html_file_output(result, km.labels_)


if __name__ == '__main__':
    main()
