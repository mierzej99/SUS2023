import argparse
import load_data
from sklearn.cluster import KMeans, DBSCAN
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


def cross_corr_score(data, labels):
    print(data, labels)
    datas = [np.append(x, labels[i]) for i, x in enumerate(data)]
    print(datas)
    """
    n = len(data[0])
    for label in list(set(labels)):
        for i, x in enumerate(data[:-1]):
            if labels[i] == label:
                y1 = data[i]
                y2 = data[i+1]
                corr = signal.correlate(y2, y1, mode='same') / np.sqrt(
                    signal.correlate(y1, y1, mode='same')[int(n / 2)] * signal.correlate(y2, y2, mode='same')[
                        int(n / 2)])"""


def dbscan(data):
    eps = [0.5 * x for x in range(1, 11)]
    min_smaple = [x for x in range(2, 7)]
    metric = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
    ps = [x for x in range(1,6)]

    score = -1

    for e in tqdm(eps):
        for ms in min_smaple:
            for met in metric:
                for p in ps:
                    db = DBSCAN(eps=e, min_samples=ms, metric=met, p=p, n_jobs=-1).fit(data)
                    if len(set(db.labels_)) > 1 and silhouette_score(data, db.labels_) > score:
                        eb, msb, metb, pb = e, ms, met, p

    db = DBSCAN(eps=eb, min_samples=msb, metric=metb, p=pb, n_jobs=-1).fit(data)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'dbscan: eps={eb}, min_smaples={msb}, metric={metb}, p={pb}, silhouette_score={silhouette_score(data, db.labels_)}\n')
    return db


def kmeans(data):
    n_clusters = [x for x in range(2, 60)]
    max_iter = [100*x for x in range(1,5)]
    tol = [10**x for x in range(-5, 1)]

    score = -1

    for clust in tqdm(n_clusters):
        for mi in max_iter:
            for t in tol:
                kmeans = KMeans(n_clusters=clust, max_iter=mi, tol=t, n_init='auto').fit(data)
                if len(set(kmeans.labels_)) > 1 and silhouette_score(data, kmeans.labels_) > score:
                    clustb, mib, tb = clust, mi, t,

    kmeans = KMeans(n_clusters=clustb, max_iter=mib, tol=tb, n_init='auto').fit(data)
    with open('best_params.txt', 'a') as f:
        f.write(
            f'kmeans: n_clusters={clustb}, max_iter={mib}, tol={tb}, silhouette_score={silhouette_score(data, kmeans.labels_)}\n')
    return kmeans


def main():
    args = arguments()
    data, file_names = load_data.load_images(args.file_path)
    db = dbscan(data)
    km = kmeans(data)
    result = assign_files_to_clusters(file_names, km.labels_)
    text_file_output(result, km.labels_)
    html_file_output(result, km.labels_)



if __name__ == '__main__':
    main()
