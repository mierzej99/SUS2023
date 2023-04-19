import argparse
import load_data, experiments
import os


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


def main():
    args = arguments()
    data, file_names = load_data.load_images(args.file_path)

    experiments.kmeans(data)
    db = experiments.dbscan(data)
    experiments.agg_n(data)

    result = assign_files_to_clusters(file_names, db.labels_)
    text_file_output(result, db.labels_)
    html_file_output(result, db.labels_)


if __name__ == '__main__':
    main()
