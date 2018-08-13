#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from collections import OrderedDict


class AbstractModel(object):
    """
    Abstract Model
    """
    def __init__(self, *args, **kwargs):
        self.params = {}
        self.layers = OrderedDict()
        self.last_layer = None

    def load_weights(self, weight_path, weight_file):
        """
        method to load saved weights
        """
        with open(os.path.join(weight_path, weight_file), "rb") as f:
            self.params = pickle.load(f)

    def save_weights(self, weight_path, weight_file):
        """
        method to save weights
        """
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)

        with open(os.path.join(weight_path, weight_file), "wb") as f:
            pickle.dump(self.params, f, pickle.HIGHEST_PROTOCOL)

    def predict(self, X):
        for layer in self.layers.values():
            X = layer.forward(X)
        return X

    def loss(self, X, y):
        y_predict = self.predict(X)
        return self.last_layer.forward(y_predict, y)

    def accuracy(self, X, y):
        y_predict = self.predict(X)
        y_predict = np.argmax(y_predict, axis=1)
        if y.ndim != 1:
            y = np.argmax(y, axis=1)
        accuracy = 1.0 * np.sum(y_predict == y) / X.shape[0]
        return accuracy

    def gradient(self, X, y):
        self.loss(X, y)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
