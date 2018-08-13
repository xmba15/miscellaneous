#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from dnn.affine_layer import AffineLayer
from dnn.activations import Sigmoid
from dnn.softmax_with_loss import SoftmaxWithLoss
from dnn.abstract_model import AbstractModel


class ClassificationModel(AbstractModel):
    """
    Classification Model
    """
    def __init__(self, input_size, hidden_size, output_size, w_init_std=0.01):
        super(ClassificationModel, self).__init__()
        self.params["W1"] = w_init_std * np.random.randn(input_size,
                                                         hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = w_init_std * np.random.randn(hidden_size,
                                                         output_size)
        self.params["b2"] = np.zeros(output_size)
        self.init_layer()

    def init_layer(self):
        self.layers["Affine1"] = AffineLayer(self.params["W1"],
                                             self.params["b1"])
        self.layers["Sigmoid1"] = Sigmoid()
        self.layers["Affine2"] = AffineLayer(self.params["W2"],
                                             self.params["b2"])
        self.last_layer = SoftmaxWithLoss()

    def load_weights(self, weight_path, weight_file):
        super(ClassificationModel, self).load_weights(weight_path, weight_file)
        self.init_layer()

    def gradient(self, X, y):
        super(ClassificationModel, self).gradient(X, y)

        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db
        return grads
