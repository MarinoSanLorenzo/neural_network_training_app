import os
import tensorflow as tf
import numpy as np

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="poisson",
        min_delta=0.01,
        patience=3,
        verbose=3,
        baseline=1,
        restore_best_weights=True,
    )
]

hyperparameter_space = {
    "nb_hidden_layers": [1, 2, 3, 4],
    "units": [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100],
    "dropout_rate": list(np.arange(0.01, 0.51, 0.01)),
    "activation": ["sigmoid", "relu", "tanh", "softmax", None],
    "use_bias": [True],
    "kernel_initializer": ["glorot_uniform"],
    "optimizer": ["adam"],
    "callbacks": [callbacks],
    "batch_size": [128, 256],
}


class Constants:
    PATH_TO_DATA = "PATH_TO_DATA"
    NB_CLAIMS = "NB_CLAIMS"
    CLAIM_AMOUNT = "CLAIM_AMOUNT"
    CLAIM_FREQUENCY = "CLAIM_FREQUENCY"
    EXPOSURE_NAME = "EXPOSURE_NAME"
    VARIABLES_TO_EXCLUDE = "VARIABLES_TO_EXCLUDE"
    DATASET_FREQ_NAME = "DATASET_FREQ_NAME"
    DATASET_SEV_NAME = "DATASET_SEV_NAME"
    TEST_SIZE = "TEST_SIZE"
    VAL_SIZE_FROM_TRAIN_SIZE = "VAL_SIZE_FROM_TRAIN_SIZE"
    RANDOM_STATE = "RANDOM_STATE"
    N_MAX_EXPERIMENTS = "N_MAX_EXPERIMENTS"
    MAX_OPTIMIZATION_TIME = "MAX_OPTIMIZATION_TIME"



params = {
    Constants.PATH_TO_DATA: "./data",
    Constants.NB_CLAIMS: "ClaimNb",
    Constants.CLAIM_AMOUNT: "ClaimAmount",
    Constants.EXPOSURE_NAME: "Exposure",
    Constants.VARIABLES_TO_EXCLUDE: ["PolicyID"],
    Constants.CLAIM_FREQUENCY: "claim_frequency",
    Constants.DATASET_FREQ_NAME: "dataset_freq.pkl",
    Constants.DATASET_SEV_NAME: "dataset_sev.pkl",
    Constants.TEST_SIZE: 0.2,
    Constants.VAL_SIZE_FROM_TRAIN_SIZE: 0.1,
    Constants.RANDOM_STATE: 42,
    Constants.N_MAX_EXPERIMENTS: 250,
    Constants.MAX_OPTIMIZATION_TIME: 1_800,
}

