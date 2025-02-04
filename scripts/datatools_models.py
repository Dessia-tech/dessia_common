"""
Tests for dessia_common.datatools.models file
"""
import time
import numpy as npy
from sklearn import linear_model, tree, ensemble, svm, neural_network

from dessia_common.models import all_cars_no_feat
from dessia_common.datatools import learning_models as models
from dessia_common.datatools.dataset import Dataset

max_time_check_platform = 30.

# TODO review the way data are generated
# Load Data and put it in a Dataset (matrix is automatically computed)
dataset_example = Dataset(all_cars_no_feat)
inputs = dataset_example.sub_matrix(['displacement', 'horsepower', 'model', 'acceleration', 'cylinders'])
double_outputs = dataset_example.sub_matrix(['mpg', 'weight'])
labelled_outputs = [npy.random.randint(4) for _ in double_outputs]
doubled_labelled_outputs = [[npy.random.randint(4),npy.random.randint(4)] for _ in double_outputs]
mono_outputs = [output[0] for output in double_outputs]

# Train test split
inputs_train, inputs_test, outputs_train, outputs_test = models.train_test_split(inputs, double_outputs, ratio=0.7)
assert(len(inputs_train) + len(inputs_test) == len(inputs))
assert(len(outputs_train) + len(outputs_test) == len(double_outputs))
assert(len(inputs_train) == len(outputs_train))
assert(len(inputs_test) == len(outputs_test))

# Test scalers
idty_scaler = models.IdentityScaler().fit(dataset_example.matrix)
idty_matrix = idty_scaler.transform(dataset_example.matrix)
idty_scaler, idty_matrix = models.IdentityScaler().fit_transform(dataset_example.matrix)

std_scaler = models.StandardScaler().fit(inputs)
std_inputs = std_scaler.transform(inputs)
std_multi_inputs = std_scaler.transform_matrices(inputs, inputs, inputs)
multi_inputs = std_scaler.inverse_transform_matrices(*std_multi_inputs)
std_scaler, std_inputs = models.StandardScaler().fit_transform(inputs)

for i in range(2):
    assert(all(value in std_multi_inputs[i][255] for value in std_multi_inputs[i + 1][255]))
    assert(all(value in multi_inputs[i][255] for value in multi_inputs[i + 1][255]))

# Hyperparameters
ridge_hyperparams = {'alpha': 0.1, 'tol': 0.00001, 'fit_intercept': True}
linearreg_hyperparams = {'fit_intercept': True, 'positive': False}
dt_hyperparams = {'max_depth': None}
rf_hyperparams = {'n_estimators': 40, 'max_depth': None}
svm_hyperparams = {'C': 0.1, 'kernel': 'rbf'}
mlp_hyperparams = {'hidden_layer_sizes': (100, 100, 100, 100, 100), 'alpha': 100, 'max_iter': 1000, 'solver': 'adam',
                   'activation': 'identity', 'tol': 1.}

hyperparameters = {'ridge_regressor': ridge_hyperparams, 'linearreg_regressor': linearreg_hyperparams,
                   'dt_regressor': dt_hyperparams, 'dt_classifier': dt_hyperparams,
                   'dt_classifier_doubled': dt_hyperparams, 'rf_regressor': rf_hyperparams,
                   'rf_classifier': rf_hyperparams,
                   'rf_classifier_doubled': rf_hyperparams,
                   'svm_regressor': svm_hyperparams, 'svm_classifier': svm_hyperparams,
                   #'svm_classifier_doubled': svm_hyperparams,
                   'mlp_regressor': mlp_hyperparams, 'mlp_classifier': mlp_hyperparams}
                   # 'mlp_classifier_doubled': mlp_hyperparams}


# Sklearn models
skl_models = {'ridge_regressor': linear_model.Ridge(**ridge_hyperparams),
              'linearreg_regressor': linear_model.LinearRegression(**linearreg_hyperparams),
              'dt_regressor': tree.DecisionTreeRegressor(**dt_hyperparams),
              'dt_classifier': tree.DecisionTreeClassifier(**dt_hyperparams),
              'dt_classifier_doubled': tree.DecisionTreeClassifier(**dt_hyperparams),
              'rf_regressor': ensemble.RandomForestRegressor(**rf_hyperparams),
              'rf_classifier': ensemble.RandomForestClassifier(**rf_hyperparams),
              'rf_classifier_doubled': ensemble.RandomForestClassifier(**rf_hyperparams),
              'svm_regressor': svm.SVR(**svm_hyperparams),
              'svm_classifier': svm.SVC(**svm_hyperparams),
              # 'svm_classifier_doubled': svm.SVC(**svm_hyperparams),
              'mlp_regressor': neural_network.MLPRegressor(**mlp_hyperparams),
              'mlp_classifier': neural_network.MLPClassifier(**mlp_hyperparams)}
              # 'mlp_classifier_doubled': neural_network.MLPClassifier(**mlp_hyperparams)}

# Fit sklearn models
for key, model in skl_models.items():
    if 'regressor' in key:
        if 'svm' in key:
            model.fit(std_inputs[:-10], mono_outputs[:-10])
            continue
        model.fit(std_inputs[:-10], double_outputs[:-10])
        continue
    if 'doubled' in key:
        model.fit(std_inputs[:-10], doubled_labelled_outputs[:-10])
        continue
    model.fit(std_inputs[:-10], labelled_outputs[:-10])

# Dessia models
dessia_classes = {'ridge_regressor': models.Ridge, 'linearreg_regressor': models.LinearRegression,
                  'dt_regressor': models.DecisionTreeRegressor, 'dt_classifier': models.DecisionTreeClassifier,
                  'dt_classifier_doubled': models.DecisionTreeClassifier,
                  'rf_regressor': models.RandomForestRegressor, 'rf_classifier': models.RandomForestClassifier,
                  'rf_classifier_doubled': models.RandomForestClassifier,
                  'svm_regressor': models.SupportVectorRegressor, 'svm_classifier': models.SupportVectorClassifier,
                  'mlp_regressor': models.MLPRegressor, 'mlp_classifier': models.MLPClassifier}
                   #, 'mlp_classifier_doubled': models.MLPClassifier}


# Assert regenerated sklearn models from dessia models make the same predictions as sklearn models from sklearn.fit
dessia_models = {}
for key, model in skl_models.items():
    dessia_models[key] = dessia_classes[key]._instantiate_dessia(model, hyperparameters[key])
    assert(npy.all(dessia_models[key].predict(std_inputs[-10:]) == model.predict(std_inputs[-10:])))


# Test dessia models methods
dessia_models = {}
for key, model in skl_models.items():
    print(key)
    if 'regressor' in key:
        if 'svm' in key:
            local_outputs = mono_outputs
        else:
            local_outputs = double_outputs
    else:
        if 'doubled' in key:
            local_outputs = doubled_labelled_outputs
        else:
            local_outputs = labelled_outputs

    parameters = hyperparameters[key]
    dessia_models[key], preds = dessia_classes[key].fit_predict(std_inputs[:-10], local_outputs[:-10], std_inputs[-10:],
                                                                **parameters)
    dessia_models[key] = dessia_classes[key].fit(std_inputs[:-10], local_outputs[:-10], **parameters)
    try:
        assert(isinstance(dessia_models[key].score(std_inputs[-10:], local_outputs[-10:]), float))
    except ValueError as e:
        assert(e.args[0] == 'multiclass-multioutput is not supported' and
               isinstance(dessia_models[key], (models.DecisionTreeClassifier, models.MLPClassifier,
                                               models.SupportVectorClassifier, models.RandomForestClassifier)))
    t=time.time()
    dessia_models[key]._check_platform()
    assert(time.time() - t <= max_time_check_platform)


# Tests errors and base objects
base_models = [models.Scaler(None), models.Model(None), models.LinearModel(None), models.RandomForest(None),
               models.SupportVectorMachine(None), models.MultiLayerPerceptron(None)]
model = models.Model(None)

for base_model in base_models:
    try:
        base_model._skl_class()
        raise ValueError(f"_skl_class() should not work for {type(base_model)} object.")
    except NotImplementedError as e:
        assert isinstance(e, NotImplementedError)

try:
    model._instantiate_skl()
    raise ValueError("_instantiate_skl() should not work for Model object.")
except NotImplementedError as e:
    assert isinstance(e, NotImplementedError)

try:
    model._instantiate_dessia(None, None)
    raise ValueError("_instantiate_dessia() should not work for Model object.")
except NotImplementedError as e:
    assert isinstance(e, NotImplementedError)

try:
    models.train_test_split(inputs, [[1,2,3], [1,2,3]], ratio=0.7)
    raise ValueError("train_test_split function should not work with matrices of different sizes.")
except ValueError as e:
    assert isinstance(e, ValueError)
