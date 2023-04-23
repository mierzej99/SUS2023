import argparse
import time
from tqdm import tqdm
import load_data, experiments
import os
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score


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


def html_file_output(result, labels, file_name):
    with open(file_name, 'w') as f:
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


def main():
    t = time.time()
    args = arguments()
    data, file_names = load_data.load_images(args.file_path)

    # shrinking data
    rng = np.random.default_rng(4)
    indexes = rng.choice(a=range(7601), size=5000, replace=False)
    data = data[indexes]
    file_names = [file_names[i] for i in indexes]

    distance_matrix = experiments.cross_corr_matrix(data)
    print(f'distance matrix calulated time in min:{(time.time() - t) / 60}')

    db_pre = experiments.dbscan(data, distance_matrix)
    
    print(f'db_pre score: {silhouette_score(data, db_pre.labels_)}')

    result_db_pre = assign_files_to_clusters(file_names, db_pre.labels_)

    #text_file_output(result, db.labels_)
    html_file_output(result_db_pre, db_pre.labels_, file_name='result_db_pre.html')

    print((time.time() - t)/60)


if __name__ == '__main__':
    main()
