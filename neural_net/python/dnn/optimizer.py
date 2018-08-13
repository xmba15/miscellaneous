#!/usr/bin/env python
# -*- coding: utf-8 -*-


class SGD(object):
    """
    stochastic gradient descent
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    @property
    def lr(self):
        return self.__lr

    @lr.setter
    def lr(self, lr):
        self.__lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
        return params
