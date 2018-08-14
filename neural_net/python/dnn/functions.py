#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_loss(X, y, eps=1e-12):
    if X.ndim == 1:
        X = X.reshape(1, -1)
        y = y.reshape(1, -1)

    if X.size == y.size:
        y = y.argmax(axis=1)

    batch_size = X.shape[0]
    return -np.sum(np.log(X[np.arange(batch_size), y] + eps)) / batch_size


def softmax_with_loss(X, y):
    X = softmax(X)
    return cross_entropy_loss(X, y)
