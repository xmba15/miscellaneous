#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from config import Config


IRIS_CONFIG = Config()


def visualize_loss():
    loss_log = IRIS_CONFIG.load_logs(IRIS_CONFIG.LOG_PATH, "loss.pkl")
    y = np.array(loss_log)
    x = np.arange(0, len(loss_log))
    plt.xlim(0, len(loss_log))
    plt.plot(x, y, label="loss")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(IRIS_CONFIG.LOG_PATH, "loss"))
    plt.close()


def visualize_acc():
    train_acc = IRIS_CONFIG.load_logs(IRIS_CONFIG.LOG_PATH, "train_acc.pkl")
    test_acc = IRIS_CONFIG.load_logs(IRIS_CONFIG.LOG_PATH, "test_acc.pkl")
    plt.ylim(0, 1.0)
    x = np.arange(0, len(train_acc))
    plt.xlim(0, len(train_acc))
    plt.plot(x, train_acc, label="train", markevery=500)
    plt.plot(x, test_acc, label="test", markevery=500)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(IRIS_CONFIG.LOG_PATH, "acc"))
    plt.close()


def main():
    visualize_loss()
    visualize_acc()


if __name__ == '__main__':
    main()
