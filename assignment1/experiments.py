from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from scipy import signal
import numpy as np

def dict_result(data, labels):
    cluster_dict = {k: [] for k in set(labels)}
    for i, label in enumerate(labels):
        cluster_dict[label].append(data[i])
    return cluster_dict


def weighted_cross_corr_score(data, labels):
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
        ovr_cross_corr[label] = in_cluster_cross_corr.mean()*len(datas[label]) if len(in_cluster_cross_corr) != 0 else None

    return ovr_cross_corr.mean()/len(data)


def dbscan(data):
    eps = [0.5 * x for x in range(1, 11)]
    min_smaple = [x for x in range(2, 6)]
    metric = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
    ps = [x for x in range(1, 6)]

    score = -1

    for e in tqdm(eps):
        for ms in min_smaple:
            for met in metric:
                for p in ps:
                    db = DBSCAN(eps=e, min_samples=ms, metric=met, p=p, n_jobs=-1).fit(data)
                    ccs = weighted_cross_corr_score(data, db.labels_)
                    if len(set(db.labels_)) > 1 and ccs > score:
                        score = ccs
                        eb, msb, metb, pb = e, ms, met, p

    db = DBSCAN(eps=eb, min_samples=msb, metric=metb, p=pb, n_jobs=-1).fit(data)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'dbscan: eps={eb}, min_samples={msb}, metric={metb}, p={pb}, weighted_cross_corr_score={weighted_cross_corr_score(data, db.labels_)}, silhouette_score={silhouette_score(data, db.labels_)}\n')
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
                ccs = weighted_cross_corr_score(data, kmeans.labels_)
                if len(set(kmeans.labels_)) > 1 and ccs > score:
                    score = ccs
                    clustb, mib, tb = clust, mi, t

    kmeans = KMeans(n_clusters=clustb, max_iter=mib, tol=tb, n_init='auto').fit(data)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'kmeans: n_clusters={clustb}, max_iter={mib}, tol={tb}, weighted_cross_corr_score={weighted_cross_corr_score(data, kmeans.labels_)}, silhouette_score={silhouette_score(data, kmeans.labels_)}\n')
    return kmeans


def agg_n(data):
    n_clusters = [x for x in range(2, 60)]
    linkage = ['ward', 'complete', 'average', 'single']

    score = -1

    for clust in tqdm(n_clusters):
        for li in linkage:
            agglo = AgglomerativeClustering(n_clusters=clust, linkage=li).fit(data)
            ccs = weighted_cross_corr_score(data, agglo.labels_)
            if len(set(agglo.labels_)) > 1 and ccs > score:
                score = ccs
                clustb, lib = clust, li

    agglo = AgglomerativeClustering(n_clusters=clustb, linkage=lib).fit(data)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'agglo_n: n_clusters={clustb}, linkage={lib}, weighted_cross_corr_score={weighted_cross_corr_score(data, agglo.labels_)}, silhouette_score={silhouette_score(data, agglo.labels_)}\n')
    return agglo


def agg_guess(data):
    linkage = ['ward']  # ['ward', 'complete', 'average', 'single']
    distance_threshold = [10 ** x for x in range(-3, 3)]

    score = -1

    for thre in tqdm(distance_threshold):
        for li in linkage:
            agglo = AgglomerativeClustering(n_clusters=None, distance_threshold=thre, linkage=li).fit(data)
            print(agglo.labels_)
            ccs = weighted_cross_corr_score(data, agglo.labels_)
            if len(set(agglo.labels_)) > 1 and ccs > score:
                score = ccs
                threb, lib = thre, li

    agglo = AgglomerativeClustering(n_clusters=None, thre=threb, linkage=lib).fit(data)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'agglo_guess: distance_threshold={threb}, linkage={lib}, weighted_cross_corr_score={weighted_cross_corr_score(data, agglo.labels_)}, silhouette_score={silhouette_score(data, agglo.labels_)}\n')
    return agglo


def gausian_mm(data):
    n_components = [x for x in range(2, 60, 3)]
    covariance_type = ['full', 'tied', 'diag', 'spherical']
    tol = [10 ** x for x in range(-5, -3)]
    reg_covar = [10 ** x for x in range(-7, -5)]
    max_iter = [100 * x for x in range(1, 3)]
    init_params = ['kmeans', 'k-means++', 'random', 'random_from_data']

    score = -1

    for n_comp in tqdm(n_components):
        for cov in covariance_type[:1]:
            for t in tol[:1]:
                for reg in reg_covar[:1]:
                    for mi in max_iter[:1]:
                        for ini in init_params[:1]:
                            gmm = GaussianMixture(n_components=n_comp, covariance_type=cov, tol=t, reg_covar=reg,
                                                  max_iter=mi, init_params=ini).fit(data)
                            labels = gmm.predict(data)
                            ccs = weighted_cross_corr_score(data, labels)
                            if len(set(labels)) > 1 and ccs > score:
                                score = ccs
                                n_compb, covb, tb, regb, mib, inib = n_comp, cov, t, reg, mi, ini

    gmm = GaussianMixture(n_components=n_compb, covariance_type=covb, tol=tb, reg_covar=regb, max_iter=mib,
                          init_params=inib).fit(data)
    labels = gmm.predict(data)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'gausian_mm: n_components={n_compb}, covariance_type={covb}, tol={tb}, reg_covar={regb}, max_iter={mib}, init_params={inib}, weighted_cross_corr_score={weighted_cross_corr_score(data, labels)}, silhouette_score={silhouette_score(data, labels)}\n')
    return gmm
