from sklearn.neural_network import MLPRegressor


def create_ANN_list(df_list, target_list):
    ANN_List=[]
    for (testDataFrame, target) in zip(df_list,target_list):
        x_train, x_test, y_train, y_test= train_test_split(testDataFrame, target)
        model= MLPRegressor(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.9,
               beta_2=0.999, early_stopping=False, epsilon=1e-08,
               hidden_layer_sizes=(200, 300, 400, 300, 200), learning_rate='adaptive',
               learning_rate_init=0.0001, max_iter=5000, momentum=0.5,
               n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
               random_state=None, shuffle=True, solver='adam', tol=0.0001,
               validation_fraction=0.1, verbose=False, warm_start=False)
        model.fit(x_train, y_train)
        ANN_List.append(model)
        print(model.score(x_test, y_test))
    return ANN_List