import os
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn_export import Export

from util.util import setup_logger

__defaultmodel = MLPRegressor(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.9,
                              beta_2=0.999, early_stopping=False, epsilon=1e-08,
                              hidden_layer_sizes=(200, 300, 400, 300, 200), learning_rate='adaptive',
                              learning_rate_init=0.0001, max_iter=5000, momentum=0.5,
                              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
                              random_state=None, shuffle=True, solver='adam', tol=0.0001,
                              validation_fraction=0.1, verbose=False, warm_start=False)

current_datetime = datetime.now()
ann_logger = setup_logger('ann_training',
                          "log/ann_training_{}_{}_{}_{}.log".format(current_datetime.year, current_datetime.month,
                                                                    current_datetime.day,
                                                                    current_datetime.hour))


def create_ANN_list(df_list, target_list, model=__defaultmodel, grid_search=False):
    ann_logger.info("Started create_ANN_list method")
    if grid_search:
        for (testDataFrame, target) in zip(df_list, target_list):
            __grid_search_ANN(testDataFrame, target)
    else:
        for (testDataFrame, target) in zip(df_list, target_list):
            __train_ANN(testDataFrame, target, model)
    ann_logger.info("Done create_ANN_list method")
    return


def __train_ANN(features, target, model):
    x_train, x_test, y_train, y_test = train_test_split(features, target)
    ann_logger.info("Starting training for the {}".format(target.name))
    model.fit(x_train, y_train)
    ann_logger.info("Finished  training for {}".format(target.name))
    ann_logger.info("Saving model for  {}".format(target.name))
    __save_model(model, "{}".format(target.name))
    ann_logger.debug("Model score of {} : {}".format(target.name, model.score(x_test, y_test)))
    return


def __grid_search_ANN(features, target):
    x_train, x_test, y_train, y_test = train_test_split(features, target)
    ann_logger.info("Starting training for the {}".format(target.name))

    parameter_space = {
        'hidden_layer_sizes': [(8, 16, 32, 64)],
        'activation': ['tanh', 'relu', 'identity'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.003, 0.001, 0.0003, 0.0001],
        'learning_rate_init': [0.03, 0.01, 0.003, 0.001, 0.0001],
    }

    clf = GridSearchCV(MLPRegressor(max_iter=2000, learning_rate="adaptive"), parameter_space, n_jobs=-1, verbose=2)
    clf.fit(x_train, y_train)
    ann_logger.debug(clf.best_estimator_)
    ann_logger.debug(clf.best_params_)
    ann_logger.debug(clf.best_score_)
    ann_logger.info("Finished  training for {}".format(target.name))
    ann_logger.info("Saving model for  {}".format(target.name))
    model = clf.best_estimator_
    __save_model(model, "{}".format(target.name))
    ann_logger.debug("Model score of {} : {}".format(target.name, model.score(x_test, y_test)))
    return


def __save_model(model, name):
    if not os.path.exists('model/ann_models'):
        os.makedirs('model/ann_models')
    loc = "model/ann_models/{}".format(name)
    with open(loc, "wb") as fp:
        pickle.dump(model, fp)
    if not os.path.exists('model/ann_models/json'):
        os.makedirs('model/ann_models/json')
    export = Export(model)
    export.to_json(directory="model/ann_models/json", filename="{}.json".format(name))
    ann_logger.info("Model pickling for  {} complete!".format(name))
