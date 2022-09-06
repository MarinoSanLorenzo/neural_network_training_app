import os
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from streamlit_disqus import st_disqus
import time
from constants import Constants, params, hyperparameter_space

from sklearn.model_selection import train_test_split
from src.optimization import run_optimization_neural_network

def main()-> None:

    ##########################################################################################
    #################################### BACK END ###########################################
    ##########################################################################################

    TEST_SIZE = params.get(Constants.TEST_SIZE)
    VAL_SIZE = params.get(Constants.VAL_SIZE_FROM_TRAIN_SIZE)
    RANDOM_STATE = params.get(Constants.RANDOM_STATE)
    PATH_TO_DATA = params.get(Constants.PATH_TO_DATA)
    nb_claims_name, claim_amount_name = params.get(Constants.NB_CLAIMS), params.get(
        Constants.CLAIM_AMOUNT)
    claim_frequency_name = params.get(Constants.CLAIM_FREQUENCY)
    exposure_name = params.get(Constants.EXPOSURE_NAME)

    df_freq = pd.read_pickle(os.path.join(PATH_TO_DATA, params.get(Constants.DATASET_FREQ_NAME)))
    df_sev = pd.read_pickle(os.path.join(PATH_TO_DATA, params.get(Constants.DATASET_SEV_NAME)))

    nb_claims, claim_amount = df_freq[nb_claims_name], df_freq[claim_amount_name]
    X = df_freq.drop(columns=[claim_amount_name, claim_frequency_name])

    x_train_val, x_test, y_train_val, y_test = train_test_split(X, nb_claims,
                                                                test_size=TEST_SIZE,
                                                                random_state=RANDOM_STATE,
                                                                stratify=X[nb_claims_name])

    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
                                                      test_size=VAL_SIZE,
                                                      random_state=RANDOM_STATE,
                                                      stratify=x_train_val[nb_claims_name])

    exp_train, exp_val, exp_test = x_train[exposure_name], x_val[exposure_name], x_test[exposure_name]

    x_train = x_train.drop(columns=[nb_claims_name])
    x_val = x_val.drop(columns=[nb_claims_name])
    x_test = x_test.drop(columns=[nb_claims_name])

    ##########################################################################################
    #################################### FRONT END ###########################################
    ##########################################################################################

    st.title("Training Neural Network to predict claim frequencies")
    # """
    # Demo application written in the context of an Actuarial Data Science training certification
    # """
    st.write("Example of pre-processed dataset:")
    st.write(df_freq.head(10))


    st.sidebar.title("Optimization parameters")
    n_max_experiments = st.sidebar.slider('Number of maximum experiments', min_value=1, max_value=100, step=5)  # 👈 this is a widget
    max_optimization_time = st.sidebar.slider('Number of maximum optimization time ',  min_value=60, max_value=3600, step=60)  # 👈 this is a widget
    run_optimization = st.sidebar.button('RUN OPTIMIZATION!')

    if run_optimization:
        results, best_model = run_optimization_neural_network(x_train, y_train, x_val, y_val, x_test, y_test,
                                        exp_test, hyperparameter_space,
                                        n_max_experiments=n_max_experiments,
                                        max_optimization_time=max_optimization_time,
                                        )
        AgGrid(results.loc[:,"poisson_dev"])
        best_hyperparameters = results.head(1).selected_hyperparams.squeeze()
        st.write(f'Best hyperparameters:\n{best_hyperparameters}')

    st.session_state['n_max_experiments'] = n_max_experiments
    st.session_state['max_optimization_time'] = max_optimization_time
    st.session_state['run_optimization'] = run_optimization
    st.sidebar.write(st.session_state)
    st_disqus("streamlit-disqus-demo")

if __name__ == '__main__':
    main()