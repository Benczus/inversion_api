import os
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from util.util import setup_logger

current_datetime = datetime.now()
ann_logger = setup_logger('ann_training',
                      "log/ann_training_{}_{}_{}_{}.log".format(current_datetime.year, current_datetime.month,
                                                            current_datetime.day,
                                                            current_datetime.hour))


def create_ANN_list(df_list, target_list):
    ann_logger.info("Started create_ANN_list method")
    ANN_List = []
    for (testDataFrame, target) in zip(df_list, target_list):
        x_train, x_test, y_train, y_test = train_test_split(testDataFrame, target)
        ann_logger.info("Starting training for the {}".format(target.name))
        model = MLPRegressor(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.9,
                             beta_2=0.999, early_stopping=False, epsilon=1e-08,
                             hidden_layer_sizes=(200, 300, 400, 300, 200), learning_rate='adaptive',
                             learning_rate_init=0.0001, max_iter=5000, momentum=0.5,
                             n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
                             random_state=None, shuffle=True, solver='adam', tol=0.0001,
                             validation_fraction=0.1, verbose=False, warm_start=False)
        model.fit(x_train, y_train)
        ann_logger.info("Finished  training for {}".format(target.name))
        ann_logger.info("Saving model for  {}".format(target.name))
        if not os.path.exists('models/ann_models'):
             os.makedirs('models/ann_models')
        loc="model/ann_models/{}".format(target.name)
        with open(loc, "wb") as fp:
             pickle.dump(model, fp)
        ann_logger.info("Model pickling for  {} complete!".format(target.name))
        ann_logger.debug("Model score of {} : {}".format(target.name, model.score(x_test, y_test)))
        ANN_List.append(model)
    ann_logger.info("Done create_ANN_list method")
    return ANN_List


