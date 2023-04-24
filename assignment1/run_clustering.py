import argparse
import os
import time

import numpy as np

import experiments
import load_data


def arguments():
    """
    prepering program argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--file_path')
    return parser.parse_args()


def assign_files_to_clusters(file_names, labels):
    """
    assigning labels to file names
    """
    return [[x, y] for x, y in zip(file_names, labels)]


def text_file_output(result, labels):
    """
    generating text file output
    """
    with open('result.txt', 'w') as f:
        for label in list(set(labels)):
            for x in result:
                if x[1] == label:
                    f.write(os.path.basename(x[0]) + ' ')
            f.write('\n')


def html_file_output(result, labels, file_name):
    """
    generating html file output
    """
    with open(file_name, 'w') as f:
        f.write(f'''<html>
        <head>
        <title>{file_name}</title>
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

    # handling input parameters
    args = arguments()
    data, file_names = load_data.load_images(args.file_path)

    # calculating distance matrix
    distance_matrix = experiments.cross_corr_matrix(data)

    # clustering
    db = experiments.dbscan(data, distance_matrix)

    # assigning files to cluster for output purpose
    result = assign_files_to_clusters(file_names, db.labels_)

    # output results to txt and html
    text_file_output(result, db.labels_)
    html_file_output(result, db.labels_, 'result.html')

    print(f'Program was running: {(time.time() - t) / 60} minutes')


if __name__ == '__main__':
    main()
