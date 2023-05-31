AUTHOR:
Michał Mierzejewski

PURPOSE:
The purpose of this task is to write a program that assigns pairs of photos and encoding labels:
1 - if encoding is based on the photo
0 - if encoding is not based on the photo

ASSUMPTIONS:

    The data from the Knowledge Pit is located in the "data" folder.

METHODS:

    Data:
        I generate the training data by pairing the paths to the photos and encodings and assigning a label to each pair.
        The file "data_functions.py" contains two important functions:
            "list_of_paths" creates a list of paths to the files.
            "create_list_of_pairs_and_labels" outputs the pairs in the form of (<list of pairs [photo_path, encoding_path]>, <list of labels>).
            The parameter "number_of_not_correct_pairs" determines how many pairs with label 0 will be generated for each correct pair.
        I also transform the data using "data_transformation.py" to make it compatible with scikit-learn.
            The output of this script is multiple files where each pair of photo and encoding is resized, optionally converted to grayscale, flattened, and concatenated.

    Scikit-Learn baseline - "scikit_baselines.ipynb":
        This approach was used to generate a baseline for comparison with the CNN.
        The data transformed with "data_transformation.py" is used, and a grid search with cross-validation (GridSearchCV) is performed using RandomForest as the classifier.
        The best model found is then used to predict the labels of the test data.

    PyTorch CNN - "pytorch_experiments.ipynb":
        The file "pytorch_classes.py" contains the dataset class for loading data in a format suitable for PyTorch and the TwoImagesTwoConvNets class.
        The TwoImagesTwoConvNets architecture consists of three independent convolutional layers for both the photo and encoding inputs, followed by max pooling layers.
        The outputs of these subnetworks are concatenated and fed into fully connected layers.
        This approach did not outperform the scikit-learn baseline, possibly due to the limited time available for experimenting with larger network architectures and higer presence of 0-labled pairs (parameter "number_of_not_correct_pairs").
        Training was time-consuming without access to a GPU.