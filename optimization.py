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
        optimizer=optimizer,
        loss=tf.keras.losses.poisson,
        metrics=tf.keras.metrics.Poisson(),
    )
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=50,
        verbose="auto",
        callbacks=callbacks,
        validation_split=0.2,
        validation_data=(x_val, y_val),
        shuffle=True,
        initial_epoch=0,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    )
    return model
