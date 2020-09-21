#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle


_DIRECTORY_ROOT = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
_MODEL_PATH = os.path.join(_DIRECTORY_ROOT, "models")
_LOG_PATH = os.path.join(_DIRECTORY_ROOT, "logs")
_IRIS_LABELS = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


class Config(object):
    """
    return constant parameters
    """
    MODEL_PATH = _MODEL_PATH
    LOG_PATH = _LOG_PATH
    LEARNING_RATE = 0.01
    BATCH_SIZE = 20
    SAVED_WEIGHT = "weights.pkl"

    def __init__(self, *args):
        print("Loading configs...")

    def dump_logs(self, logs, log_path, log_file):
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        with open(os.path.join(log_path, log_file), "wb") as f:
            pickle.dump(logs, f, pickle.HIGHEST_PROTOCOL)

    def load_logs(self, log_path, log_file):
        with open(os.path.join(log_path, log_file), "rb") as f:
            logs = pickle.load(f)
        return logs
