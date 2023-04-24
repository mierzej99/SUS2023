from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def cross_corr_metric(a, b):
    """
    Metric that I used for calculating distances between data points.
    """
    norm_a = np.linalg.norm(a)
    a = a / norm_a
    norm_b = np.linalg.norm(b)
    b = b / norm_b
    c = np.correlate(a, b, mode='same')
    return max(0, 1 - max(c))


def compute_row(i, data):
    """
    Help function for efficient calculations purpose.
    """
    cross_corr_matrix_row = np.zeros(len(data))
    x = data[i]
    for j, y in enumerate(data):
        cross_corr_matrix_row[j] = cross_corr_metric(x, y)
    return cross_corr_matrix_row


def cross_corr_matrix(data):
    """
    Function for calculating distance matrix between all data points.
    """
    cross_corr_matrix_r = np.empty((len(data), len(data)))
    with Pool(cpu_count()) as p:
        results = [p.apply_async(compute_row, args=(i, data)) for i in range(len(data))]
        for i, result in enumerate(results):
            row = result.get()
            cross_corr_matrix_r[i] = row
    return cross_corr_matrix_r


def dbscan(data, distance_matrix):
    """
    Experimenting with different hyperparameters for dbscan and returning best clustering.
    """
    eps = [0.001 * x for x in range(25, 75)]
    min_smaple = [x for x in range(2, 7)]

    score = -1

    for e in tqdm(eps):
        for ms in min_smaple:
            db = DBSCAN(eps=e, min_samples=ms, metric='precomputed', n_jobs=-1).fit(distance_matrix)
            ccs = silhouette_score(data, db.labels_) if len(set(db.labels_)) > 1 and len(set(db.labels_)) < len(
                data) - 1 else -1
            if ccs - score > 0.05:
                score = ccs
                eb, msb = e, ms

    db = DBSCAN(eps=eb, min_samples=msb, metric='precomputed', n_jobs=-1).fit(distance_matrix)
    return db


####################################################################################################
"""
I don't use below functions in my final solution so i will not comment them. I used them for experiments
"""
####################################################################################################


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
            in_cluster_cross_corr[i] = max(np.correlate(y2, y1, mode='same') / np.sqrt(
                np.correlate(y1, y1, mode='same')[int(n / 2)] * np.correlate(y2, y2, mode='same')[
                    int(n / 2)]))
        ovr_cross_corr[label] = in_cluster_cross_corr.mean() * len(datas[label]) if len(
            in_cluster_cross_corr) != 0 else None

    return ovr_cross_corr.mean() / len(data)


def dbscan_classic(data):
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
                    ccs = silhouette_score(data, db.labels_) if len(set(db.labels_)) > 1 else -1
                    if ccs > score:
                        score = ccs
                        eb, msb, metb, pb = e, ms, met, p

    db = DBSCAN(eps=eb, min_samples=msb, metric=metb, p=pb, n_jobs=-1).fit(data)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'dbscan: eps={eb}, min_samples={msb}, metric={metb}, p={pb}, weighted_cross_corr_score={weighted_cross_corr_score(data, db.labels_)}, silhouette_score={silhouette_score(data, db.labels_)}\n')
    return db


def dbscan_elbow(data, distance_matrix):
    eps = [0.001 * x for x in range(25, 75)]
    min_smaple = [x for x in range(2, 7)]

    score = -1
    dbs = []
    for e in tqdm(eps):
        for ms in min_smaple:
            db = DBSCAN(eps=e, min_samples=ms, metric='precomputed', n_jobs=-1).fit(distance_matrix)
            ccs = silhouette_score(data, db.labels_) if len(set(db.labels_)) > 1 and len(set(db.labels_)) < len(
                data) - 1 else -1
            dbs.append([ccs, db])

    dbs = sorted(dbs, key=lambda x: x[0])
    print([x[0] for x in dbs])
    for i, db in enumerate(dbs[:-1]):
        if dbs[i + 1][0] - dbs[i][0] > 0.002:
            result = dbs[i + 1][1]
    return result


def kmeans(data):
    n_clusters = [x for x in range(50, 65)]
    max_iter = [100 * x for x in range(1, 5)]
    tol = [10 ** x for x in range(-5, 1)]

    score = -1

    for clust in tqdm(n_clusters):
        for mi in max_iter:
            for t in tol:
                kmeans = KMeans(n_clusters=clust, max_iter=mi, tol=t, n_init='auto').fit(data)
                ccs = silhouette_score(data, kmeans.labels_) if len(set(kmeans.labels_)) > 1 else -1
                if ccs > score:
                    score = ccs
                    clustb, mib, tb = clust, mi, t

    kmeans = KMeans(n_clusters=clustb, max_iter=mib, tol=tb, n_init='auto').fit(data)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'kmeans: n_clusters={clustb}, max_iter={mib}, tol={tb}, weighted_cross_corr_score={weighted_cross_corr_score(data, kmeans.labels_)}, silhouette_score={silhouette_score(data, kmeans.labels_)}\n')
    return kmeans


def agg_n(data, distance_matrix):
    n_clusters = [x for x in range(50, 65)]
    linkage = ['complete', 'average', 'single']

    score = -1

    for clust in tqdm(n_clusters):
        for li in linkage:
            agglo = AgglomerativeClustering(n_clusters=clust, metric='precomputed', linkage=li).fit(distance_matrix)
            ccs = silhouette_score(data, agglo.labels_) if len(set(agglo.labels_)) > 1 else -1
            if ccs > score:
                score = ccs
                clustb, lib = clust, li

    agglo = AgglomerativeClustering(n_clusters=clustb, linkage=lib, metric='precomputed').fit(distance_matrix)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'agglo_n: n_clusters={clustb}, linkage={lib}, metric=precomputed, weighted_cross_corr_score={weighted_cross_corr_score(data, agglo.labels_)}, silhouette_score={silhouette_score(data, agglo.labels_)}\n')
    return agglo


def agg_guess(data, distance_matrix):
    linkage = ['complete', 'average', 'single']
    distance_threshold = [0.001 * x for x in range(1, 250)]

    score = -1

    for thre in tqdm(distance_threshold):
        for li in linkage:
            agglo = AgglomerativeClustering(n_clusters=None, distance_threshold=thre, linkage=li,
                                            metric='precomputed').fit(distance_matrix)
            ccs = silhouette_score(data, agglo.labels_) if len(set(agglo.labels_)) > 1 and len(
                set(agglo.labels_)) < len(data) - 1 else -1
            if ccs > score:
                score = ccs
                threb, lib = thre, li

    agglo = AgglomerativeClustering(n_clusters=None, distance_threshold=threb, linkage=lib, metric='precomputed').fit(
        distance_matrix)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'agglo_guess: distance_threshold={threb}, linkage={lib}, weighted_cross_corr_score={weighted_cross_corr_score(data, agglo.labels_)}, silhouette_score={silhouette_score(data, agglo.labels_)}\n')
    return agglo


def sc(data, distance_matrix):
    affinity = ['precomputed_nearest_neighbors', 'precomputed']
    n_clusters = [x for x in range(50, 66)]
    assign_labels = ['kmeans', 'discretize', 'cluster_qr']
    degree = list(range(1, 5))
    n_neighbors = list(range(3, 15))

    score = -1

    for aff in tqdm(affinity):
        for clust in n_clusters:
            for al in assign_labels:
                for de in degree:
                    for nn in n_neighbors:
                        spec = SpectralClustering(n_clusters=clust, affinity=aff, assign_labels=al, degree=de,
                                                  n_jobs=-1, n_neighbors=nn).fit(distance_matrix)
                        ccs = silhouette_score(data, spec.labels_) if len(set(spec.labels_)) > 1 else -1
                        if ccs > score:
                            score = ccs
                            affb, clustb, alb, deb, nnb = aff, clust, al, de, nn
        distance_matrix = -(distance_matrix - 1)

    if affb == 'precomputed_nearest_neighbors':
        distance_matrix = -(distance_matrix - 1)
    spec = SpectralClustering(n_clusters=clustb, affinity=affb, assign_labels=alb, degree=deb, n_jobs=-1,
                              n_neighbors=nnb).fit(distance_matrix)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'spectral: n_clusters={clustb}, metric={affb}, assign_labels={alb}, degree={deb}, n_neighbors={nnb}, weighted_cross_corr_score={weighted_cross_corr_score(data, spec.labels_)}, silhouette_score={silhouette_score(data, spec.labels_)}\n')
    return spec
