"""
Tests for dessia_common.modeling file

"""
import time
import numpy as npy
from sklearn import linear_model, tree, ensemble, svm, neural_network
from dessia_common.models import all_cars_no_feat
from dessia_common.datatools.dataset import Dataset
from dessia_common.datatools.modeling import StandardScaler, IdentityScaler, LinearRegression, SVR, SVC, MLPRegressor,\
    DecisionTreeRegressor, DecisionTreeClassifier, RandomForestRegressor, RandomForestClassifier, MLPClassifier

# Load Data and put it in a Dataset (matrix is automatically computed)
dataset_example = Dataset(all_cars_no_feat)

# Test scalers
idty_scaler = IdentityScaler().fit(dataset_example.matrix)
idty_matrix = idty_scaler.transform(dataset_example.matrix)
idty_scaler, idty_matrix = IdentityScaler().fit_transform(dataset_example.matrix)

inputs = dataset_example.sub_matrix(['displacement', 'horsepower', 'model', 'acceleration', 'cylinders'])
double_outputs = dataset_example.sub_matrix(['mpg', 'weight'])
std_scaler = StandardScaler().fit(inputs)
std_inputs = std_scaler.transform(inputs)
std_scaler, std_inputs = StandardScaler().fit_transform(inputs)
labelled_outputs = [npy.random.randint(4) for _ in double_outputs]
mono_outputs = [output[0] for output in double_outputs]

# Hyperparameters
ridge_hyperparams = {'alpha': 0.1, 'tol': 0.00001, 'fit_intercept': True}
dt_hyperparams = {'max_depth': None}
rf_hyperparams = {'n_estimators': 40, 'max_depth': None}
svm_hyperparams = {'C': 0.1, 'kernel': 'rbf'}
mlp_hyperparams = {'hidden_layer_sizes': (100, 100, 100, 100, 100), 'alpha': 100, 'max_iter': 1000, 'solver': 'adam',
                   'activation': 'identity', 'tol': 1.}
hyperparameters = {'linear_regressor': ridge_hyperparams,
                   'dt_regressor': dt_hyperparams, 'dt_classifier': dt_hyperparams,
                   'rf_regressor': rf_hyperparams, 'rf_classifier': rf_hyperparams,
                   'svm_regressor': svm_hyperparams, 'svm_classifier': svm_hyperparams,
                   'mlp_regressor': mlp_hyperparams, 'mlp_classifier': mlp_hyperparams}

# Sklearn models
skl_models = {}
skl_models['linear_regressor'] = linear_model.Ridge(**ridge_hyperparams)
skl_models['dt_regressor'] = tree.DecisionTreeRegressor(**dt_hyperparams)
skl_models['dt_classifier'] = tree.DecisionTreeClassifier(**dt_hyperparams)
skl_models['rf_regressor'] = ensemble.RandomForestRegressor(**rf_hyperparams)
skl_models['rf_classifier'] = ensemble.RandomForestClassifier(**rf_hyperparams)
skl_models['svm_regressor'] = svm.SVR(**svm_hyperparams)
skl_models['svm_classifier'] = svm.SVC(**svm_hyperparams)
skl_models['mlp_regressor'] = neural_network.MLPRegressor(**mlp_hyperparams)
skl_models['mlp_classifier'] = neural_network.MLPClassifier(**mlp_hyperparams)

# Fit sklearn models
for key, model in skl_models.items():
    if 'regressor' in key:
        if 'svm' in key:
            model.fit(std_inputs, mono_outputs)
            continue
        model.fit(std_inputs, double_outputs)
        continue
    model.fit(std_inputs, labelled_outputs)

# Dessia models
dessia_classes = {'linear_regressor': LinearRegression, 'dt_regressor': DecisionTreeRegressor,
                 'dt_classifier': DecisionTreeClassifier, 'rf_regressor': RandomForestRegressor,
                 'rf_classifier': RandomForestClassifier, 'svm_regressor': SVR, 'svm_classifier': SVC,
                 'mlp_regressor': MLPRegressor, 'mlp_classifier': MLPClassifier}

# Assert regenerated sklearn models from dessia models make the same predictions as sklearn models from sklearn.fit
dessia_models = {}
for key, model in skl_models.items():
    dessia_models[key] = dessia_classes[key]._instantiate_dessia(model)
    assert(npy.all(dessia_models[key].predict(std_inputs[50:100]) == model.predict(std_inputs[50:100])))

# Test dessia models methods
dessia_models = {}
for key, model in skl_models.items():
    if 'regressor' in key:
        if 'svm' in key:
            local_outputs = mono_outputs
        else:
            local_outputs = double_outputs
    else:
        local_outputs = labelled_outputs

    params = hyperparameters[key]
    dessia_models[key] = dessia_classes[key].fit_predict(std_inputs, local_outputs, std_inputs[50:100], **params)
    dessia_models[key] = dessia_classes[key].fit(std_inputs, local_outputs, **params)
    dessia_models[key].score(std_inputs, local_outputs)



# # Dessia models
# linear_regression = LinearRegression().fit(std_inputs, double_outputs, alpha = 0.1)
# rf_regressor = RandomForestRegressor(n_estimators=20)
# rf_classifier = RandomForestClassifier(n_estimators=20)



# predicted_data = linear_regression.predict(std_inputs[50:100])
# linear_regression, pred_dessia = LinearRegression().fit_predict(std_inputs, raw_outputs, std_inputs[50:100], alpha = 0.1)
# assert(npy.all(pred_dessia == predicted_data))

# # Tree, DecisionTree, RandomForest
# rf_regressor = ensemble.RandomForestRegressor(n_estimators=20)
# rf_regressor.fit(std_inputs, raw_outputs)

# pred_skl_tree = rf_regressor.estimators_[12].tree_.predict(npy.array(std_inputs[50:100], dtype=npy.float32))
# pred_dessia_tree = DecisionTreeRegressor._instantiate_dessia(rf_regressor.estimators_[12])
# assert(npy.all(pred_dessia_tree.predict(std_inputs[50:100]) == pred_skl_tree[:,:,0]))


# skl_dectree = rf_regressor.estimators_[12]
# new_tree = tree.DecisionTreeRegressor()
# new_tree.tree_ = skl_dectree.tree_
# new_tree.n_outputs_ = skl_dectree.tree_.n_outputs
# assert(npy.all(new_tree.predict(npy.array(std_inputs[50:100], dtype=npy.float32)) == skl_dectree.predict(npy.array(std_inputs[50:100], dtype=npy.float32))))

# dessia_tree = DecisionTreeRegressor.fit(std_inputs, raw_outputs)
# test = dessia_tree._instantiate_skl()
# assert(npy.all(dessia_tree.predict(npy.array(std_inputs[50:100], dtype=npy.float32)) == test.predict(npy.array(std_inputs[50:100], dtype=npy.float32))))

# pp=DecisionTreeRegressor._instantiate_dessia(test)
# assert(npy.all(pp.predict(npy.array(std_inputs[50:100], dtype=npy.float32)) == test.predict(npy.array(std_inputs[50:100], dtype=npy.float32))))

# labelled_outputs = [npy.random.randint(4) for _ in raw_outputs]
# dessia_tree = DecisionTreeClassifier.fit(std_inputs, labelled_outputs)

# dessia_forest = RandomForestRegressor.fit(std_inputs, raw_outputs)
# dessia_forest.predict(std_inputs[50:100])

# dessia_forest = RandomForestClassifier.fit(std_inputs, labelled_outputs)
# dessia_forest.predict(std_inputs[50:100])

# outputs = [output[0] for output in raw_outputs]
# dessia_svr = SVR.fit(std_inputs, outputs, kernel='rbf')
# dessia_svc = SVC.fit(std_inputs, labelled_outputs, kernel='rbf')
# skl_svr = svm.SVR(kernel='rbf')
# skl_svr.fit(std_inputs, outputs)

# dessia_svr.predict(std_inputs[50:55])
# dessia_svc.predict(std_inputs[50:55])
# skl_svr.predict(std_inputs[50:55])


# dessia_mlp = MLPRegressor.fit(std_inputs, outputs, hidden_layer_sizes = (100, 100, 100, 100, 100),
#                               alpha=100, max_iter = 1000, activation = 'identity', solver='adam', tol=1)
# skl_mlp = neural_network.MLPRegressor(hidden_layer_sizes = (100, 100, 100, 100, 100), alpha=100, max_iter = 1000,
#                                       activation = 'identity', solver='adam', tol=1)
# skl_mlp.fit(std_inputs, outputs)

# dessia_mlp.predict(std_inputs[50:55])
# skl_mlp.predict(std_inputs[50:55])

# test_dessia_mlp = MLPRegressor._instantiate_dessia(skl_mlp)
# test_dessia_mlp.predict(std_inputs[50:55])


# dessia_mlp = MLPClassifier.fit(std_inputs, labelled_outputs, hidden_layer_sizes = (100, 100, 100, 100, 100),
#                               alpha=100, max_iter = 1000, activation = 'identity', solver='adam', tol=1)
# skl_mlp = neural_network.MLPClassifier(hidden_layer_sizes = (100, 100, 100, 100, 100), alpha=100, max_iter = 1000,
#                                       activation = 'identity', solver='adam', tol=1)
# skl_mlp.fit(std_inputs, labelled_outputs)

# dessia_mlp.predict(std_inputs[50:55])
# skl_mlp.predict(std_inputs[50:55])

# test_dessia_mlp = MLPClassifier._instantiate_dessia(skl_mlp)
# test_dessia_mlp.predict(std_inputs[50:55])


# t = time.time()
# dessia_mlp._check_platform()
# r2 = dessia_mlp.score(std_inputs[50:200], labelled_outputs[50:200])
# print("MLP : ", time.time() - t, "R2 = ", r2)

# t = time.time()
# dessia_svc._check_platform()
# r2 = dessia_svc.score(std_inputs[50:200], labelled_outputs[50:200])
# print("SVC : ", time.time() - t, "R2 = ", r2)

# t = time.time()
# dessia_svr._check_platform()
# r2 = dessia_svr.score(std_inputs[50:200], outputs[50:200])
# print("SVR : ", time.time() - t, "R2 = ", r2)

# t = time.time()
# dessia_forest._check_platform()
# r2 = dessia_forest.score(std_inputs[50:200], labelled_outputs[50:200])
# print("RF : ", time.time() - t, "R2 = ", r2)

# t = time.time()
# linear_model._check_platform()
# r2 = linear_model.score(std_inputs[50:200], raw_outputs[50:200])
# print("Linear : ", time.time() - t, "R2 = ", r2)

