"""
Tests for dessia_common.modeling file

"""
import numpy as npy
from sklearn import linear_model, tree, ensemble, svm, neural_network

from dessia_common.models import all_cars_no_feat
from dessia_common.datatools.dataset import Dataset
from dessia_common.datatools.modeling import StandardScaler, IdentityScaler, Ridge, SupportVectorRegressor,\
    SupportVectorClassifier, MLPRegressor, DecisionTreeRegressor, DecisionTreeClassifier, RandomForestRegressor, \
        RandomForestClassifier, MLPClassifier, Scaler, Model, Tree, RandomForest, SupportVectorMachine, \
            MultiLayerPerceptron, LinearRegression


# Load Data and put it in a Dataset (matrix is automatically computed)
dataset_example = Dataset(all_cars_no_feat)
inputs = dataset_example.sub_matrix(['displacement', 'horsepower', 'model', 'acceleration', 'cylinders'])
double_outputs = dataset_example.sub_matrix(['mpg', 'weight'])
labelled_outputs = [npy.random.randint(4) for _ in double_outputs]
doubled_labelled_outputs = [[npy.random.randint(4),npy.random.randint(4)] for _ in double_outputs]
mono_outputs = [output[0] for output in double_outputs]


# Test scalers
idty_scaler = IdentityScaler().fit(dataset_example.matrix)
idty_matrix = idty_scaler.transform(dataset_example.matrix)
idty_scaler, idty_matrix = IdentityScaler().fit_transform(dataset_example.matrix)

std_scaler = StandardScaler().fit(inputs)
std_inputs = std_scaler.transform(inputs)
std_scaler, std_inputs = StandardScaler().fit_transform(inputs)


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
                   'dt_classifier_doubled': dt_hyperparams,
                   'rf_regressor': rf_hyperparams, 'rf_classifier': rf_hyperparams,
                   'svm_regressor': svm_hyperparams, 'svm_classifier': svm_hyperparams,
                   'mlp_regressor': mlp_hyperparams, 'mlp_classifier': mlp_hyperparams}


# Sklearn models
skl_models = {}
skl_models['ridge_regressor'] = linear_model.Ridge(**ridge_hyperparams)
skl_models['linearreg_regressor'] = linear_model.LinearRegression(**linearreg_hyperparams)
skl_models['dt_regressor'] = tree.DecisionTreeRegressor(**dt_hyperparams)
skl_models['dt_classifier'] = tree.DecisionTreeClassifier(**dt_hyperparams)
skl_models['dt_classifier_doubled'] = tree.DecisionTreeClassifier(**dt_hyperparams)
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
    if 'doubled' in key:
        model.fit(std_inputs, doubled_labelled_outputs)
        continue
    model.fit(std_inputs, labelled_outputs)

# Dessia models
dessia_classes = {'ridge_regressor': Ridge, 'linearreg_regressor': LinearRegression,
                  'dt_regressor': DecisionTreeRegressor, 'dt_classifier': DecisionTreeClassifier,
                  'dt_classifier_doubled': DecisionTreeClassifier,
                  'rf_regressor': RandomForestRegressor, 'rf_classifier': RandomForestClassifier,
                  'svm_regressor': SupportVectorRegressor, 'svm_classifier': SupportVectorClassifier,
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
        if 'doubled' in key:
            local_outputs = doubled_labelled_outputs
        else:
            local_outputs = labelled_outputs

    params = hyperparameters[key]
    dessia_models[key] = dessia_classes[key].fit_predict(std_inputs, local_outputs, std_inputs[50:100], **params)
    dessia_models[key] = dessia_classes[key].fit(std_inputs, local_outputs, **params)
    try:
        assert(isinstance(dessia_models[key].score(std_inputs, local_outputs), float))
    except ValueError as e:
        assert(e.args[0] == 'multiclass-multioutput is not supported' and
               isinstance(dessia_models[key], DecisionTreeClassifier))
    dessia_models[key]._check_platform()


# Tests errors and base objects
base_scaler = Scaler()
base_model = Model()
base_tree = Tree()
base_rf = RandomForest()
base_svm = SupportVectorMachine()
base_mlp = MultiLayerPerceptron()

try:
    base_scaler._skl_class()
    raise ValueError("_skl_class() should not work for Scaler object.")
except Exception as e:
    assert (e.args[0] == 'Method _skl_class not implemented for Scaler. Please use children.')

try:
    base_model._skl_class()
    raise ValueError("_skl_class() should not work for Model object.")
except Exception as e:
    assert (e.args[0] == 'Method _skl_class not implemented for Model.')

try:
    base_rf._skl_class()
    raise ValueError("_skl_class() should not work for RandomForest object.")
except Exception as e:
    assert (e.args[0] == 'Method _skl_class not implemented for RandomForest. Please use RandomForestClassifier '\
            'or RandomForestRegressor.')

try:
    base_svm._skl_class()
    raise ValueError("_skl_class() should not work for SVM object.")
except Exception as e:
    assert (e.args[0] == 'Method _skl_class not implemented for SupportVectorMachine. Please use '\
                              'SupportVectorClassifier or SupportVectorRegressor.')

try:
    base_mlp._skl_class()
    raise ValueError("_skl_class() should not work for MultiLayerPerceptron object.")
except Exception as e:
    assert (e.args[0] == 'Method _skl_class not implemented for MultiLayerPerceptron. Please use MLPRegressor '\
            'or MLPClassifier.')

try:
    base_model._instantiate_skl()
    raise ValueError("_instantiate_skl() should not work for Model object.")
except Exception as e:
    assert (e.args[0] == 'Method _instantiate_skl not implemented for Model.')

try:
    base_model._instantiate_dessia(None)
    raise ValueError("_instantiate_dessia() should not work for Model object.")
except Exception as e:
    assert (e.args[0] == 'Method _instantiate_dessia not implemented for Model.')


dt_clf = tree.DecisionTreeClassifier(**dt_hyperparams)
dt_clf.fit(std_inputs, labelled_outputs)
dt_dessia = DecisionTreeClassifier.fit_predict(std_inputs, labelled_outputs, std_inputs[50:100])

