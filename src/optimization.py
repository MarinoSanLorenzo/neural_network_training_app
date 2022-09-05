from typing import *
import pandas as pd
import numpy as np
import time
import keras
from collections import defaultdict
import random
import math
from keras.models import Sequential
from keras import Input  # for instantiating a keras tensor
from keras.layers import (
    Dense,
    multiply,
    Dropout,
)  # for creating regular densely-connected NN layers.
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_poisson_deviance
import streamlit as st
from pprint import pprint

@st.cache
def run_optimization_neural_network(
    x_train: np.array,
    y_train: np.array,
    x_val: np.array,
    y_val: np.array,
    x_test: np.array,
    y_test: np.array,
    exp_test: np.array,
    hyperparameter_space: Dict[str, Iterable[Any]],
    n_max_experiments: int = 5,
    max_optimization_time: int = 60,
) -> Tuple[pd.DataFrame, keras.engine.sequential.Sequential]:
    results = defaultdict(list)
    nb_experiment = 0
    start = time.time()
    elapsed_time = time.time() - start
    layer_hyperparameters = [
        "activation",
        "dropout_rate",
        "units",
        "use_bias",
        "kernel_initializer",
    ]
    best_deviance_score, best_model = math.inf, None
    while nb_experiment < n_max_experiments and elapsed_time < max_optimization_time:
        print(f'{"-" * 50} {nb_experiment}th EXPERIMENT {"-" * 50}')
        selected_hyperparams = {}
        for param_name, v in hyperparameter_space.items():
            if all(
                [param_name != layer_param for layer_param in layer_hyperparameters]
            ):
                selected_hyperparams[param_name] = random.choice(v)
        else:
            for n_layer in range(
                selected_hyperparams.get("nb_hidden_layers")
            ):  # randomly set hyperparam for each layer of the NN
                selected_hyperparams[f"layer_param_{n_layer}"] = {
                    k: random.choice(v)
                    for k, v in hyperparameter_space.items()
                    if k in layer_hyperparameters
                }
        pprint(selected_hyperparams)
        model = fit_feed_forward_neural_network(
            x_train, y_train, x_val, y_val, params=selected_hyperparams
        )
        y_pred = model.predict(x_test)
        poisson_dev = mean_poisson_deviance(
            y_test, y_pred[:, 0], sample_weight=exp_test
        )
        results["selected_hyperparams"].append(selected_hyperparams)
        results["poisson_dev"].append(poisson_dev)
        if poisson_dev < best_deviance_score:
            best_deviance_score = poisson_dev
            best_model = model
        nb_experiment += 1
        elapsed_time = time.time() - start
    else:
        results_df = pd.DataFrame.from_dict(results)
        results_df.sort_values(by="poisson_dev", ascending=True, inplace=True)
        results_df.reset_index(inplace=True, drop=True)
    return results_df, best_model


def fit_feed_forward_neural_network(
    x_train: np.array,
    y_train: np.array,
    x_val: np.array,
    y_val: np.array,
    params: Dict[str, Any],
) -> keras.engine.sequential.Sequential:
    #     x_train = params.get('x_train')
    #     y_train = params.get('y_train')
    nb_hidden_layers = params.get("nb_hidden_layers")
    optimizer = params.get("optimizer")
    batch_size = params.get("batch_size")
    dropout_rate = params.get("dropout_rate")
    callbacks = params.get("callbacks")

    model = Sequential()
    model.add(Input(shape=(None, x_train.shape[1])))
    for n_layer in range(nb_hidden_layers):
        layer_params = params.get(f"layer_param_{n_layer}")
        dropout_rate = layer_params.get("dropout_rate")
        del layer_params["dropout_rate"]
        model.add(Dense(**layer_params))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="exponential"))
    model.compile(
        optimizer=optimizer,  # default='rmsprop', an algorithm to be used in backpropagation
        loss=tf.keras.losses.poisson,
        # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
        metrics=tf.keras.metrics.Poisson(),
        # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
        loss_weights=None,
        # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
        weighted_metrics=None,
        # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
        run_eagerly=None,
        # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
        steps_per_execution=None
        # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
    )
    ##### Step 5 - Fit keras model on the dataset
    model.fit(
        x_train,  # input data
        y_train,  # target data
        batch_size=batch_size,
        # Number of samples per gradient update. If unspecified, batch_size will default to 32.
        epochs=50,
        # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
        verbose="auto",
        # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
        callbacks=callbacks,  # default=None, list of callbacks to apply during training. See tf.keras.callbacks
        validation_split=0.2,
        # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
        validation_data=(x_val, y_val),
        # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch.
        shuffle=True,
        # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
        class_weight=None,
        # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
        sample_weight=None,
        # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
        initial_epoch=0,
        # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
        steps_per_epoch=None,
        # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
        validation_steps=None,
        # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
        validation_batch_size=None,
        # Integer or None, default=None, Number of samples per validation batch. If unspecified, will default to batch_size.
        validation_freq=1,
        # default=1, Only relevant if validation data is provided. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs.
        max_queue_size=10,
        # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
        workers=1,
        # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
        use_multiprocessing=False,
        # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False.
    )
    return model
