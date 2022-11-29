"""
Tests for dessia_common.modeling file

"""
import time
import numpy as npy
from sklearn import tree, ensemble, svm, neural_network
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
raw_outputs = dataset_example.sub_matrix(['mpg', 'weight'])
std_scaler = StandardScaler().fit(inputs)
std_inputs = std_scaler.transform(inputs)
std_scaler, std_inputs = StandardScaler().fit_transform(inputs)

# Tests models
linear_model = LinearRegression().fit(std_inputs, raw_outputs, alpha = 0.1)
predicted_data = linear_model.predict(std_inputs[50:100])
linear_model, pred_dessia = LinearRegression().fit_predict(std_inputs, raw_outputs, std_inputs[50:100], alpha = 0.1)
assert(npy.all(pred_dessia == predicted_data))

# Tree, DecisionTree, RandomForest
rf_regressor = ensemble.RandomForestRegressor(n_estimators=20)
rf_regressor.fit(std_inputs, raw_outputs)

pred_skl_tree = rf_regressor.estimators_[12].tree_.predict(npy.array(std_inputs[50:100], dtype=npy.float32))
pred_dessia_tree = DecisionTreeRegressor._instantiate_dessia(rf_regressor.estimators_[12])
assert(npy.all(pred_dessia_tree.predict(std_inputs[50:100]) == pred_skl_tree[:,:,0]))


skl_dectree = rf_regressor.estimators_[12]
new_tree = tree.DecisionTreeRegressor()
new_tree.tree_ = skl_dectree.tree_
new_tree.n_outputs_ = skl_dectree.tree_.n_outputs
assert(npy.all(new_tree.predict(npy.array(std_inputs[50:100], dtype=npy.float32)) == skl_dectree.predict(npy.array(std_inputs[50:100], dtype=npy.float32))))

dessia_tree = DecisionTreeRegressor.fit(std_inputs, raw_outputs)
test = dessia_tree._instantiate_skl()
assert(npy.all(dessia_tree.predict(npy.array(std_inputs[50:100], dtype=npy.float32)) == test.predict(npy.array(std_inputs[50:100], dtype=npy.float32))))

pp=DecisionTreeRegressor._instantiate_dessia(test)
assert(npy.all(pp.predict(npy.array(std_inputs[50:100], dtype=npy.float32)) == test.predict(npy.array(std_inputs[50:100], dtype=npy.float32))))

labelled_outputs = [npy.random.randint(4) for _ in raw_outputs]
dessia_tree = DecisionTreeClassifier.fit(std_inputs, labelled_outputs)

dessia_forest = RandomForestRegressor.fit(std_inputs, raw_outputs)
dessia_forest.predict(std_inputs[50:100])

dessia_forest = RandomForestClassifier.fit(std_inputs, labelled_outputs)
dessia_forest.predict(std_inputs[50:100])

outputs = [output[0] for output in raw_outputs]
dessia_svr = SVR.fit(std_inputs, outputs, kernel='rbf')
dessia_svc = SVC.fit(std_inputs, labelled_outputs, kernel='rbf')
skl_svr = svm.SVR(kernel='rbf')
skl_svr.fit(std_inputs, outputs)

dessia_svr.predict(std_inputs[50:55])
dessia_svc.predict(std_inputs[50:55])
skl_svr.predict(std_inputs[50:55])


dessia_mlp = MLPRegressor.fit(std_inputs, outputs, hidden_layer_sizes = (100, 100, 100, 100, 100),
                              alpha=100, max_iter = 1000, activation = 'identity', solver='adam', tol=1)
skl_mlp = neural_network.MLPRegressor(hidden_layer_sizes = (100, 100, 100, 100, 100), alpha=100, max_iter = 1000,
                                      activation = 'identity', solver='adam', tol=1)
skl_mlp.fit(std_inputs, outputs)

dessia_mlp.predict(std_inputs[50:55])
skl_mlp.predict(std_inputs[50:55])

test_dessia_mlp = MLPRegressor._instantiate_dessia(skl_mlp)
test_dessia_mlp.predict(std_inputs[50:55])


dessia_mlp = MLPClassifier.fit(std_inputs, labelled_outputs, hidden_layer_sizes = (100, 100, 100, 100, 100),
                              alpha=100, max_iter = 1000, activation = 'identity', solver='adam', tol=1)
skl_mlp = neural_network.MLPClassifier(hidden_layer_sizes = (100, 100, 100, 100, 100), alpha=100, max_iter = 1000,
                                      activation = 'identity', solver='adam', tol=1)
skl_mlp.fit(std_inputs, labelled_outputs)

dessia_mlp.predict(std_inputs[50:55])
skl_mlp.predict(std_inputs[50:55])

test_dessia_mlp = MLPClassifier._instantiate_dessia(skl_mlp)
test_dessia_mlp.predict(std_inputs[50:55])


t = time.time()
dessia_mlp._check_platform()
r2 = dessia_mlp.score(std_inputs[50:200], labelled_outputs[50:200])
print("MLP : ", time.time() - t, "R2 = ", r2)

t = time.time()
dessia_svc._check_platform()
r2 = dessia_svc.score(std_inputs[50:200], labelled_outputs[50:200])
print("SVC : ", time.time() - t, "R2 = ", r2)

t = time.time()
dessia_svr._check_platform()
r2 = dessia_svr.score(std_inputs[50:200], outputs[50:200])
print("SVR : ", time.time() - t, "R2 = ", r2)

t = time.time()
dessia_forest._check_platform()
r2 = dessia_forest.score(std_inputs[50:200], labelled_outputs[50:200])
print("RF : ", time.time() - t, "R2 = ", r2)

t = time.time()
linear_model._check_platform()
r2 = linear_model.score(std_inputs[50:200], raw_outputs[50:200])
print("Linear : ", time.time() - t, "R2 = ", r2)

