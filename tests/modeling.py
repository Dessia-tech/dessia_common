"""
Tests for dessia_common.modeling file

"""

import numpy as npy
from sklearn import tree, ensemble
from dessia_common.core import DessiaObject
from dessia_common.models import all_cars_no_feat
from dessia_common.datatools.dataset import Dataset
from dessia_common.datatools.modeling import DessiaScaler, StandardScaler, IdentityScaler, LinearRegression, DessiaTree,\
    DecisionTreeRegressor, DecisionTreeClassifier


# Load Data and put it in a Dataset (matrix is automatically computed)
dataset_example = Dataset(all_cars_no_feat)

# Test scalers
idty_scaler = IdentityScaler().fit(dataset_example.matrix)
idty_matrix = idty_scaler.transform(dataset_example.matrix)
idty_scaler, idty_matrix = IdentityScaler().fit_transform(dataset_example.matrix)

inputs = dataset_example.sub_matrix(['displacement', 'horsepower', 'model', 'acceleration', 'cylinders'])
outputs = dataset_example.sub_matrix(['mpg', 'weight'])
std_scaler = StandardScaler().fit(inputs)
std_inputs = std_scaler.transform(inputs)
std_scaler, std_inputs = StandardScaler().fit_transform(inputs)

# Tests models
linear_model = LinearRegression().fit(std_inputs, outputs, alpha = 0.1)
predicted_data = linear_model.predict(std_inputs[50:100])
linear_model, pred_dessia = LinearRegression().fit_predict(std_inputs, outputs, std_inputs[50:100], alpha = 0.1)
assert(npy.all(pred_dessia == predicted_data))

# Tree, DecisionTree, RandomForest
rf_regressor = ensemble.RandomForestRegressor(n_estimators=20)
rf_regressor.fit(std_inputs, outputs)

pred_skl_tree = rf_regressor.estimators_[12].tree_.predict(npy.array(std_inputs[50:100], dtype=npy.float32))
pred_dessia_tree = DessiaTree._instantiate_dessia_model(rf_regressor.estimators_[12].tree_)
assert(npy.all(pred_dessia_tree.predict(npy.array(std_inputs[50:100], dtype=npy.float32)) == pred_skl_tree))


skl_dectree = rf_regressor.estimators_[12]
new_tree = tree.DecisionTreeRegressor()
new_tree.tree_ = skl_dectree.tree_
new_tree.n_outputs_ = skl_dectree.tree_.n_outputs
assert(npy.all(new_tree.predict(npy.array(std_inputs[50:100], dtype=npy.float32)) == skl_dectree.predict(npy.array(std_inputs[50:100], dtype=npy.float32))))

dessia_tree = DecisionTreeRegressor.fit(std_inputs, outputs)
test = dessia_tree._instantiate_skl_model()
assert(npy.all(dessia_tree.predict(npy.array(std_inputs[50:100], dtype=npy.float32)) == test.predict(npy.array(std_inputs[50:100], dtype=npy.float32))))

pp=DecisionTreeRegressor._instantiate_dessia_model(test)
assert(npy.all(pp.predict(npy.array(std_inputs[50:100], dtype=npy.float32)) == test.predict(npy.array(std_inputs[50:100], dtype=npy.float32))))

labelled_outputs = [[npy.random.randint(4)] for _ in outputs]
dessia_tree = DecisionTreeClassifier.fit(std_inputs, labelled_outputs)

