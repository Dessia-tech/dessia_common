"""
Tests for dessia_common.datatools.modeler file.
"""
import json
from dessia_common.models import all_cars_no_feat
from dessia_common.datatools.dataset import Dataset
from dessia_common.datatools import learning_models as models
from dessia_common.datatools.modeler import ModeledDataset, Modeler, ModelValidation, CrossValidation

# ======================================================================================================================
#                                                   Load Data
# ======================================================================================================================
# Load data and put it in Datasets
training_data, testing_data = Dataset(all_cars_no_feat).train_test_split(0.8, False) # True is better but it is for test
inputs = ['displacement', 'horsepower', 'acceleration']

# Set outputs to predict for regression
outputs_reg = ['weight', 'mpg']

# Set outputs to predict for classification
outputs_clf = ['cylinders']

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
rf_score = rf_mdlr.score_dataset(testing_data, inputs, outputs_clf)
mlp_scrore = mlp_mdlr.score_dataset(testing_data, inputs, outputs_reg)

# Fit and score in a row
ridge_mdlr, ridge_score = Modeler.fit_score_matrix(input_train, output_train_reg, ridge, True, False, 250, "ridge")
rf_mdlr, rf_score = Modeler.fit_score_dataset(training_data, inputs, outputs_clf, random_forest, True, False, 0.8, "rf")
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
model_validation_clf = ModelValidation.from_dataset(rf_mdlr,training_data, inputs, outputs_clf, 0.8, "clf validation")
model_validation.plot_data()
model_validation_clf.plot_data()
assert(abs(model_validation.score - model_validation_2.score) <= 0.2)


# Train machine learning models and predict new data in one operation
# ridge_modeler, ridge_pred = Modeler.fit_predict_dataset(modeling_data, data_to_predict, inputs, outputs_reg,
#                                                         ridge, True, False, "ridge_modeler")
# mlp_modeler, mlp_pred = Modeler.fit_predict_dataset(modeling_data, data_to_predict, inputs, outputs_reg,
#                                                     mlp, True, False, "mlp_modeler")

# Build a ModeledDataset of predicted data

# a
# # Initialize machine learning modeling classifiers to fit in next operations and use to predict (2 examples)
# random_forest = models.RandomForestClassifier.init_for_modeler(n_estimators=10, criterion='gini', max_depth=None)


# # Load data and put it in a Dataset (matrix is automatically computed)
# dataset_for_fit = Dataset(all_cars_no_feat)[:-100]
# dataset_to_pred = Dataset(all_cars_no_feat)[-100:]
# input_names_reg = ['displacement', 'horsepower', 'acceleration']#, 'model']#, 'cylinders'
# output_names_reg = ['weight', 'mpg']
# output_names_reg_solo = ['weight']
# input_names_clf = ['displacement', 'horsepower', 'acceleration', 'mpg', 'weight']
# output_names_clf = ['cylinders', 'model']
# output_names_clf_solo = ['cylinders']

# Tests
linear_regression = models.LinearRegression.init_for_modeler()
decision_tree_regression = models.DecisionTreeRegressor.init_for_modeler()
randomforest_reg = models.RandomForestRegressor.init_for_modeler()
supportvector_clf = models.SupportVectorClassifier.init_for_modeler()
mlp_clf = models.MLPClassifier.init_for_modeler(hidden_layer_sizes=(50, 50, 50))
try:
    svr_mdlr = Modeler.fit_dataset(training_data, inputs, outputs_reg, svr, True, True, "svr_modeler")
except Exception as e:
    assert isinstance(e, NotImplementedError)


# # Train models and predict data for regressions
# for output_names in [output_names_reg_solo, output_names_reg]:
#     # Ri_mdlr, Ri_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_reg, output_names,
#     #                                                ridge, True, True, "ridgeer")
#     # LR_mdlr, LR_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_reg, output_names,
#     #                                                linear_regression, True, True, "Linear_regression_modeler")
#     # DR_mdlr, DR_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_reg, output_names,
#     #                                                dectree_reg, True, True, "DTRegressor_modeler")
#     RR_mdlr, RR_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_reg, output_names,
#                                                     randomforest_reg, True, True, "RFRegressor_modeler")
#     MR_mdlr, MR_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_reg, output_names,
#                                                    mlp_reg, True, True, "MLPRegressor_modeler")

# # SVR_mdlr, SVR_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_reg,
# #                                                  output_names_reg_solo, supportvector_reg, True, True, "SVRegressor_modeler")

# # # Train  models and predict data for classifications
# # DC_mdlr, DC_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_clf,
# #                                                output_names_clf_solo, dectree_clf, True, False, "DTClassifier_modeler")
# # RC_mdlr, RC_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_clf,
# #                                                output_names_clf_solo, randomforest_clf, True, False, "RFClassifier_modeler")
# # DC_mdlr, DC_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_clf, output_names_clf,
# #                                                dectree_clf, True, False, "DTClassifier_modeler")
# # RC_mdlr, RC_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_clf, output_names_clf,
# #                                                randomforest_clf, True, False, "RFClassifier_modeler")
# # SVC_mdlr, SVC_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_clf,
# #                                                  output_names_clf_solo, supportvector_clf, True, False, "SVClassifier_modeler")
# MC_mdlr, MC_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_clf,
#                                                output_names_clf_solo, mlp_clf, True, False, "MLPClassifier_modeler")

# # TODO: make impossible scaling for classifier (set to False in any case)
# # modelers = [Ri_mdlr, LR_mdlr, DR_mdlr, DC_mdlr, RR_mdlr, RC_mdlr, SVR_mdlr, SVC_mdlr, MR_mdlr, MC_mdlr]
# modelers = [MR_mdlr, MC_mdlr]

# # Run cross_validation for all models instantiated in a Modeler
# CV_MR = CrossValidation.from_dataset(MR_mdlr, dataset_for_fit, input_names_reg, output_names_reg, 4, 0.8)

# # Plot cross validations
# plot_cv = CV_MR.plot_data()
# assert(len(plot_cv) == 2)
# assert(len(plot_cv[1].plots) == 8)
# assert(type(plot_cv[1].plots[1]).__name__ == 'Graph2D')

# # modeled_dataset = ModeledDataset.from_predicted_dataset(RR_mdlr, dataset_to_pred, input_names_reg, output_names_reg)
# modeled_dataset = ModeledDataset.from_predicted_dataset(RR_mdlr, dataset_to_pred, input_names_reg, output_names_reg)

# modeled_dataset, modeler, cross_validation = ModeledDataset.fit_validate_predict(dataset_for_fit, dataset_to_pred,
#                                                                                  dectree_reg, input_names_reg, output_names,
#                                                                                  True, False, 10, 0.8,
#                                                                                  "test_fit_validate_predict")
# modeled_dataset.plot()
# cross_validation.plot()

# modeled_dict = modeled_dataset.to_dict()
# json_dict = json.dumps(modeled_dict)
# decoded_json = json.loads(json_dict)
# deserialized_object = modeled_dataset.dict_to_object(decoded_json)

# assert(all(string in modeled_dataset.__str__() for string in ["0.141", "90.0", "Horsepower"]))

########################################################################################################################

# # Visuals to check test and train data are correctly separated in cross validations and modeler stuff
# for mdlr, cv in zip(mdlrs, cvs):
#     if 'lassifier' in type(mdlr.model).__name__:
#         input_names = input_names_clf
#         if 'MLP' in type(mdlr.model).__name__:
#             output_names = output_names_clf_solo
#         else:
#             output_names = output_names_clf
#         idx = 0
#     else:
#         input_names = input_names_reg
#         output_names = output_names_reg
#         idx = 1

#     out_train = [x[idx] for x in dataset_for_fit.sub_matrix(output_names)]
#     pred_train = [x[idx] for x in mdlr.predict_dataset(dataset_for_fit, input_names)]

#     out_test = [x[idx] for x in dataset_to_pred.sub_matrix(output_names)]
#     pred_test = [x[idx] for x in mdlr.predict_dataset(dataset_to_pred, input_names)]

#     plt.figure()
#     plt.plot(out_test, pred_test, color='r', linestyle='None', marker='x')
#     plt.plot(out_train, pred_train, color='b', linestyle='None', marker='x')
#     points = [minimums(cv.model_validations[0].data._concatenate_outputs())[idx],
#               maximums(cv.model_validations[0].data._concatenate_outputs())[idx]]
#     plt.plot(points,points, color = 'k')

# ======================================================================================================================
#                                            F R O M   M A T R I X
# ======================================================================================================================
# inputs = dataset_example.sub_matrix(input_names)
# double_outputs = dataset_example.sub_matrix(output_names)
# inputs_train, inputs_test, outputs_train, outputs_test = models.train_test_split(inputs, double_outputs, ratio=0.7)

# pp=modeler._plot_data_list(inputs_train, outputs_train, modeler.predict_matrix(inputs_train), input_names, output_names)

# modeler.plot(ref_inputs=inputs_train, ref_outputs=outputs_train, val_inputs=inputs_test, val_outputs=outputs_test,
#              input_names=input_names, output_names=output_names)
# modeler.score(inputs_train, outputs_train)

# test_mdlr = Modeler.from_dataset_fit_validate(dataset_example, input_names, output_names, models_class, hyperparameters,
#                                               True, True, 0.8, True, 'pouet')

# test = Modeler.cross_validation(dataset=dataset_example, input_names=input_names, output_names=output_names, class_=models_class,
#                                 hyperparameters=hyperparameters, input_is_scaled=True, output_is_scaled=True, nb_tests=5,
#                                 ratio=0.8, name='')

# test = Modeler(None,None,None).plot(dataset=dataset_example, input_names=input_names, output_names=output_names, class_=models_class,
#                                 hyperparameters=hyperparameters, input_is_scaled=True, output_is_scaled=True, nb_tests=5,
#                                 ratio=0.8, name='')


# print(test_mdlr[1])
