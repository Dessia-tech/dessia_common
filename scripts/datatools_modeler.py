"""
Tests for dessia_common.datatools.modeler file.
"""
from dessia_common.models import all_cars_no_feat
from dessia_common.datatools.dataset import Dataset
from dessia_common.datatools import learning_models as models
from dessia_common.datatools.modeler import Modeler, ModelValidation, CrossValidation

# ======================================================================================================================
#                                                   Load Data
# ======================================================================================================================
# Load data and put it in Datasets
training_data, testing_data = Dataset(all_cars_no_feat).train_test_split(0.8, False) # True is better but it is for test
inputs = ['displacement', 'horsepower', 'acceleration']

# Set outputs to predict for regression
outputs_reg = ['weight', 'mpg']

# Set outputs to predict for classification
outputs_clf = ['cylinders', 'model']

# Extract training matrices
input_train = training_data.sub_matrix(inputs)
output_train_reg = training_data.sub_matrix(outputs_reg)

# Extract testing matrices
input_test = testing_data.sub_matrix(inputs)
output_test_reg = testing_data.sub_matrix(outputs_reg)


# ======================================================================================================================
#                                                   Train and test
# ======================================================================================================================
# Initialize machine learning modeling regressors to fit in next operations and use to predict (2 examples)
ridge = models.Ridge.init_for_modeler(alpha=0.01, fit_intercept=True, tol=0.01) # linear regression with regularization
random_forest = models.RandomForestClassifier.init_for_modeler(n_estimators=100, criterion='squared_error') # classifier
mlp = models.MLPRegressor.init_for_modeler(hidden_layer_sizes=(50, 50, 50), activation='relu', max_iter=500) # NN
svr = models.SupportVectorRegressor.init_for_modeler(C=0.1, kernel='rbf')

# Train / Fit models (scaled output is not advised)
ridge_mdlr = Modeler.fit_matrix(input_train, output_train_reg, ridge, True, False, "ridge_modeler")
rf_mdlr = Modeler.fit_dataset(training_data, inputs, outputs_clf, random_forest, True, False, "rf_modeler")
mlp_mdlr = Modeler.fit_dataset(training_data, inputs, outputs_reg, mlp, True, True, "mlp_modeler")
svr_mdlr = Modeler.fit_dataset(training_data, inputs, [outputs_reg[1]], svr, True, True, "svr_modeler")

# Get score of a trained model (use test data)
ridge_score = ridge_mdlr.score_matrix(input_test, output_test_reg)
# rf_score = rf_mdlr.score_dataset(testing_data, inputs, outputs_clf) # score not available with multioutput
mlp_scrore = mlp_mdlr.score_dataset(testing_data, inputs, outputs_reg)

# Fit and score in a row
ridge_mdlr, ridge_score = Modeler.fit_score_matrix(input_train, output_train_reg, ridge, True, False, 250, "ridge")
mlp_mdlr, mlp_scrore = Modeler.fit_score_dataset(training_data, inputs, outputs_reg, mlp, True, True, 0.8, "mlp")

# Predict with models
ridge_predictions = ridge_mdlr.predict_matrix(input_test)
rf_predictions = rf_mdlr.predict_dataset(testing_data, inputs, outputs_clf)
mlp_predictions = mlp_mdlr.predict_dataset(testing_data, inputs, outputs_reg)

# Fit and predict in one operation (to use if hyperparameters are known to give a model with a good score)
ridge_mdlr, ridge_predictions = Modeler.fit_predict_matrix(input_train, output_train_reg, input_test, ridge, True,
                                                           False, "ridge_modeler")
rf_mdlr, rf_predictions = Modeler.fit_predict_dataset(training_data, testing_data, inputs, outputs_clf, random_forest,
                                                      True, False, "rf_modeler")
mlp_mdlr, mlp_predictions = Modeler.fit_predict_dataset(training_data, testing_data, inputs, outputs_reg, mlp, True,
                                                        True, "mlp_modeler")
svr_mdlr, svr_predictions = Modeler.fit_predict_dataset(training_data, testing_data, inputs, [outputs_reg[1]], svr,
                                                        True, False, "svr_modeler")

# ======================================================================================================================
#                                                   Validate models
# ======================================================================================================================
# Mono validation (not advised)
model_validation = ModelValidation.from_dataset(mlp_mdlr, training_data, inputs, outputs_reg, 0.8, "validation")
model_validation_2 = ModelValidation.from_matrix(mlp_mdlr, input_train, output_train_reg, inputs, outputs_reg, 0.8)
assert(abs(model_validation.score - model_validation_2.score) <= 0.2)

# Plots (and tests)
model_validation_clf = ModelValidation.from_dataset(rf_mdlr, training_data, inputs, [outputs_clf[0]], 0.8, "clf val")
validation_reg_plot = model_validation.plot_data()
validation_clf_plot = model_validation_clf.plot_data()
assert(type(validation_reg_plot[0]).__name__ == "MultiplePlots")
assert(type(validation_clf_plot[0]).__name__ == "Graph2D")
assert(len(validation_reg_plot[0].elements) == 335)
assert(len(validation_clf_plot[0].graphs[0].elements) == 259)

# Cross Validation (advised)
cross_validation = CrossValidation.from_dataset(rf_mdlr, training_data, inputs, [outputs_clf[0]], 10, 0.75)
cross_validation_2 = CrossValidation.from_matrix(mlp_mdlr, input_train, output_train_reg, inputs, outputs_reg, 5, 0.75)
cross_val_plot = cross_validation.plot_data()
assert(len(cross_val_plot) == 2)
assert(len(cross_val_plot[1].plots) == 10)
assert(len(cross_val_plot[1].elements) == 1)
assert(len(cross_val_plot[1].plots[1].graphs[1].elements) == 81)


# ======================================================================================================================
#                                               Additionnal Tests
# ======================================================================================================================
linear_regression = models.LinearRegression.init_for_modeler()
decision_tree_regression = models.DecisionTreeRegressor.init_for_modeler()
randomforest_reg = models.RandomForestRegressor.init_for_modeler()
supportvector_clf = models.SupportVectorClassifier.init_for_modeler()
mlp_clf = models.MLPClassifier.init_for_modeler(hidden_layer_sizes=(50, 50, 50))

try:
    svr_mdlr = Modeler.fit_dataset(training_data, inputs, outputs_reg, svr, True, True, "svr_modeler")
except Exception as e:
    assert isinstance(e, NotImplementedError)

rf_predictions._check_platform()
