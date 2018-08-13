#!/usr/env python
# -*- coding: utf-8 -*-
from model import ClassificationModel
from config import Config
from data.iris import load_iris
import numpy as np
from numpy import array as array


# loading config
IRIS_CONFIG = Config()


def main():
    model = ClassificationModel(input_size=4, hidden_size=20, output_size=3)
    try:
        model.load_weights(IRIS_CONFIG.MODEL_PATH, IRIS_CONFIG.SAVED_WEIGHT)
    except FileNotFoundError:
        print("the weights you are trying to load do not exist")
    X_train, X_test, y_train, y_test = load_iris()
    print(X_test)
    print(y_test)
    acc = model.accuracy(X_test, y_test)
    print(acc)


if __name__ == '__main__':
    main()
