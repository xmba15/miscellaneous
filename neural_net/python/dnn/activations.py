#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dnn.functions as functions


class Sigmoid(object):
    """
    Sigmoid activation
    """
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = functions.sigmoid(x)
        # print (out.shape)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
