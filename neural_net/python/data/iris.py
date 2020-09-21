#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import wget
import pandas as pd
try:
    from sklearn import datasets, preprocessing
    from sklearn.model_selection import train_test_split
except ImportError:
    raise ImportError("sklearn not installed yet")


CURRENT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
URL_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/" +\
           "iris.data"
IRIS_DATASET = "iris.data"


def download_file(url_base, file_path, file_name):
    """
    method to download file from url_base
    """
    if not os.path.exists(os.path.join(file_path, file_name)):
        print("Downloading iris dataset")
        print(os.path.join(file_path, file_name))
        wget.download(url_base, os.path.join(file_path, file_name))
        print("Finish downloading")


def load_iris_from_scikitlearn():
    """
    load iris dataset built in scikitlearn
    """
    iris = datasets.load_iris()
    size = len(iris.data)
    X = iris.data[size - 1, :]
    return X


def _load_iris(file_path, file_name, normalize=True, norm="l2", test_size=0.2,
               random_state=42):
    """
    load iris dataset from machine learning repository
    """
    header_names = ["sepal_length", "sepal_width",
                    "petal_length", "petal_width", "species"]
    species_names = {"Iris-setosa": 0,
                     "Iris-versicolor": 1,
                     "Iris-virginica": 2}

    iris = pd.read_csv(os.path.join(file_path, file_name), header=None,
                       names=header_names)
    iris["species"] = iris["species"].map(species_names).astype(int)
    X = iris.iloc[:, 0:4].values
    y = iris.iloc[:, 4].values
    if normalize:
        X = preprocessing.normalize(X, norm=norm)
    X_train, X_test,\
        y_train, y_test = train_test_split(X, y, test_size=test_size,
                                           random_state=random_state)
    return X_train, X_test, y_train, y_test


def load_iris():
    X_train, X_test,\
        y_train, y_test = _load_iris(CURRENT_DIR_PATH, IRIS_DATASET)
    return X_train, X_test, y_train, y_test


def main():
    if sys.version_info[0] < 3:
        raise Exception("Please use python 3")
    download_file(URL_BASE, CURRENT_DIR_PATH, IRIS_DATASET)
    X_train, X_test,\
        y_train, y_test = load_iris()
    print(len(X_test))
    print(len(X_train))


if __name__ == '__main__':
    main()
