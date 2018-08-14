#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from dnn.functions import softmax, cross_entropy_loss


class SoftmaxWithLoss(object):
    """
    Softmax With Loss Layer
    """
    def __init__(self):
        self.loss = None
        self.X = None
        self.y = None

    def forward(self, X, y):
        self.X = softmax(X)
        self.y = y
        self.loss = cross_entropy_loss(self.X, y)
        return self.loss

    def backward(self, dout):
        batch_size = self.X.shape[0]
        if self.X.size == self.y.size:
            dx = (self.X - self.y) / batch_size
        else:
            dx = self.X.copy()
            dx[np.arange(batch_size), self.y] -= 1
            dx = dx / batch_size
        return dx
