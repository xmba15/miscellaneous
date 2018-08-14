#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from data.iris import load_iris
from model import ClassificationModel
from dnn.optimizer import SGD
from config import Config


# loading config
IRIS_CONFIG = Config()


def train(load_data, model, optimizer, iters_num, batch_size=5, lr=0.01):
    X_train, X_test, y_train, y_test = load_data()
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    train_size = X_train.shape[0]
    iter_per_epoch = int(max(train_size / batch_size, 1))

    max_test_acc = 0.0
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        X_batch = X_train[batch_mask]
        y_batch = y_train[batch_mask]

        grads = model.gradient(X_batch, y_batch)
        model.params = optimizer.update(model.params, grads)
        model.init_layer()
        if i % iter_per_epoch == 0:
            loss = model.loss(X_batch, y_batch)
            train_loss_list.append(loss)

            train_acc = model.accuracy(X_train, y_train)
            test_acc = model.accuracy(X_test, y_test)
            print("iter: %d, train_acc: %f, test_acc: %f" %(i, train_acc,
                                                            test_acc))
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            if test_acc > max_test_acc:
                # print(test_acc)
                max_test_acc = test_acc
                model.save_weights(IRIS_CONFIG.MODEL_PATH,
                                   IRIS_CONFIG.SAVED_WEIGHT)

    IRIS_CONFIG.dump_logs(train_loss_list, IRIS_CONFIG.LOG_PATH, "loss.pkl")
    IRIS_CONFIG.dump_logs(train_acc_list, IRIS_CONFIG.LOG_PATH, "train_acc.pkl")
    IRIS_CONFIG.dump_logs(test_acc_list, IRIS_CONFIG.LOG_PATH, "test_acc.pkl")


def main():
    model = ClassificationModel(input_size=4, hidden_size=5, output_size=3)
    optimizer = SGD()
    optimizer.lr = 0.01
    iters_num = 50000
    batch_size = IRIS_CONFIG.BATCH_SIZE
    train(load_iris, model, optimizer, iters_num, batch_size)


if __name__ == '__main__':
    main()
