AUTHOR:
Michał Mierzejewski

PURPOSE:
The purpose of the task is to write a program that clusters characters.

ASSUMPTIONS:
1. There are absolute photo paths in the input file.

OUTPUT:
After the program finishes, two files will appear in the folder:
1. "result.txt" - A file containing in each line   a space-separated   list of files containing images belonging to the given cluster.
2. "result.html" - File in html format that displays images belonging to individual clusters.

METHOD:
In addition to the necessary elements such as loading data and creating output files, my method consists of three main steps.
1. Changing images to vectors:
After image is loaded it is changed to grayscal. Then It is resized to 15X13 resolution. Both of these operations are intended to reduce dimensions.
Finaly every photo is falttend in row manner to make them easier to handle.
2.Metric calculation:
I calculate distance, using cross correlation metric, between all points and put the result in distance matrix. So the i-th row is distance between i-th photo and all photos.
This matrix is symetric and has zeros on diagonal.
How to calculate cross correlation metric between x and y:
    a) normalize x and y vectors
    b) calculate cross correlation (f.e using numpy.correalte())
    c) take maximum value from that cross correlation
    d) subtract that maximum value from 1
Intuition: If x and y are images of the same letter but shifted this maximum value should be close to 1 so metric-wise they are close to each other.
3.Clustering:
I use Density-based spatial clustering of applications with noise (DBSCAN) for clustering. I conduct grid serach for two hyper parameters - eps and mn_sample.
I use silhouette_score for clustering choice I start with score = -1:
When there is bigger difference that 0.05 in silhouette_score I make this model my model of choice.
Intuition: I choose the first decent model, I prefer that the eps be smaller than to maximize the silhouette_score. Then the model is better at distinguishing between similar classes.

HOW TO RUN:
python venv -m first_assignment

source first_assignment/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

python run_clustering.py --file_path <file with absolute paths to images>