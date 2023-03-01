"""
Tests for dessia_common.datatools.modeler file.
"""
import json

import matplotlib.pyplot as plt

from dessia_common.utils import helpers
from dessia_common.models import all_cars_no_feat
from dessia_common.datatools.dataset import Dataset
from dessia_common.datatools import learning_models as models
from dessia_common.datatools.modeler import Modeler, CrossValidation, ModeledDataset

# ======================================================================================================================
#                                            F R O M   D A T A S E T
# ======================================================================================================================

# Load data and put it in a Dataset (matrix is automatically computed)
dataset_for_fit = Dataset(all_cars_no_feat)[:-100]
dataset_to_pred = Dataset(all_cars_no_feat)[-100:]
input_names_reg = ['displacement', 'horsepower', 'acceleration']#, 'model']#, 'cylinders'
output_names_reg = ['weight'] #'mpg']
output_names_reg_solo = ['weight']
input_names_clf = ['displacement', 'horsepower', 'acceleration', 'mpg', 'weight']
output_names_clf = ['cylinders', 'model']
output_names_clf_solo = ['cylinders']

# Load class of models with their hyperparameters
Ri_model = models.Ridge.init_for_modeler(alpha=0.01, fit_intercept=True, tol=0.01)
LR_model = models.LinearRegression.init_for_modeler(fit_intercept=True, positive=False)
DR_model = models.DecisionTreeRegressor.init_for_modeler(criterion='squared_error', max_depth=None)
DC_model = models.DecisionTreeClassifier.init_for_modeler(criterion='gini', max_depth=None)
RR_model = models.RandomForestRegressor.init_for_modeler(n_estimators=10, criterion='squared_error', max_depth=None)
RC_model = models.RandomForestClassifier.init_for_modeler(n_estimators=10, criterion='gini', max_depth=None)
MR_model = models.MLPRegressor.init_for_modeler(hidden_layer_sizes=(50, 50, 50), activation='relu', max_iter=500)
MC_model = models.MLPClassifier.init_for_modeler(hidden_layer_sizes=(50, 50, 50), activation='relu', max_iter=500)

# Train models and predict data
for output_names in [output_names_reg_solo, output_names_reg]:
    # Ri_mdlr, Ri_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_reg, output_names,
    #                                                Ri_model, True, True, "ridge_modeler")
    # LR_mdlr, LR_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_reg, output_names,
    #                                                LR_model, True, True, "linear_regression_modeler")
    # DR_mdlr, DR_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_reg, output_names,
    #                                                DR_model, True, True, "DTRegressor_modeler")
    RR_mdlr, RR_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_reg, output_names,
                                                   RR_model, True, True, "RFRegressor_modeler")
    # MR_mdlr, MR_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_reg, output_names,
    #                                                MR_model, True, True, "MLPRegressor_modeler")

# for output_names in [output_names_clf, output_names_clf_solo]:
# DC_mdlr, DC_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_clf, output_names_clf,
#                                                DC_model, True, False, "DTClassifier_modeler")
# RC_mdlr, RC_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_clf, output_names_clf,
#                                                RC_model, True, False, "RFClassifier_modeler")
# MC_mdlr, MC_pred = Modeler.fit_predict_dataset(dataset_for_fit, dataset_to_pred, input_names_clf, output_names_clf_solo,
#                                                MC_model, True, False, "MLPClassifier_modeler")

# TODO: make impossible scaling for classifier (set to False in any case)
# mdlrs = [Ri_mdlr, LR_mdlr, DR_mdlr, DC_mdlr, RR_mdlr, RC_mdlr, MR_mdlr, MC_mdlr]

# Run cross_validation for all models instantiated in a Modeler
# CV_Ri = CrossValidation.from_dataset(Ri_mdlr, dataset_for_fit, input_names_reg, output_names_reg, 10, 0.8)
# CV_LR = CrossValidation.from_dataset(LR_mdlr, dataset_for_fit, input_names_reg, output_names_reg, 9, 0.8)
# CV_DR = CrossValidation.from_dataset(DR_mdlr, dataset_for_fit, input_names_reg, output_names_reg, 8, 0.8)
# CV_DC = CrossValidation.from_dataset(DC_mdlr, dataset_for_fit, input_names_clf, output_names_clf, 7, 0.8)
CV_RR = CrossValidation.from_dataset(RR_mdlr, dataset_for_fit, input_names_reg, output_names_reg, 20, 0.8)
# CV_RC = CrossValidation.from_dataset(RC_mdlr, dataset_for_fit, input_names_clf, output_names_clf, 5, 0.8)
# CV_MR = CrossValidation.from_dataset(MR_mdlr, dataset_for_fit, input_names_reg, output_names_reg, 4, 0.8)
# CV_MC = CrossValidation.from_dataset(MC_mdlr, dataset_for_fit, input_names_clf, output_names_clf_solo, 3, 0.8)

# Plot cross validations
# CV_Ri.plot()
# CV_LR.plot()
# CV_DR.plot()
# CV_DC.plot()
# CV_RR.plot()
# CV_RC.plot()
# CV_MR.plot()
# CV_MC.plot()
# cvs = [CV_Ri, CV_LR, CV_DR, CV_DC, CV_RR, CV_RC, CV_MR, CV_MC]

modeled_dataset = ModeledDataset.from_predicted_dataset(RR_mdlr, dataset_to_pred, input_names_reg, output_names_reg)
# modeled_dataset.plot()

modeled_dataset, modeler, cross_validation = ModeledDataset.fit_validate_predict(dataset_for_fit, dataset_to_pred,
                                                                                 DR_model, input_names_reg, output_names,
                                                                                 True, False, 10, 0.8,
                                                                                 "test_fit_validate_predict")
modeled_dataset.plot()
cross_validation.plot()

modeled_dict = modeled_dataset.to_dict()
json_dict = json.dumps(modeled_dict)
decoded_json = json.loads(json_dict)
deserialized_object = modeled_dataset.dict_to_object(decoded_json)

assert(all(string in modeled_dataset.__str__() for string in ["0.141", "90.0", "Horsepower"]))

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
#     points = [helpers.minimums(cv.model_validations[0].data._concatenate_outputs())[idx],
#               helpers.maximums(cv.model_validations[0].data._concatenate_outputs())[idx]]
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
