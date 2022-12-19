"""
Tests for dessia_common.datatools.modeler file.
"""
from dessia_common.models import all_cars_no_feat
from dessia_common.datatools.dataset import Dataset
import dessia_common.datatools.models as models
from dessia_common.datatools.modeler import Modeler, ModelValidation, CrossValidation, ValidationData

# Load Data and put it in a Dataset (matrix is automatically computed)
dataset_example = Dataset(all_cars_no_feat)
input_names_reg = ['displacement', 'horsepower', 'model', 'acceleration', 'cylinders']
output_names_reg = ['mpg', 'weight']
output_names_reg_solo = ['weight']
input_names_clf = ['displacement', 'horsepower', 'model', 'acceleration', 'mpg', 'weight']
output_names_clf = ['cylinders']

# ======================================================================================================================
#                                            F R O M   D A T A S E T
# ======================================================================================================================
# Load class of models with their hyperparameters
Ri_class, Ri_hyperparams = models.Ridge.init_for_modeler(alpha=0.01, fit_intercept=True, tol=0.01)
LR_class, LR_hyperparams = models.LinearRegression.init_for_modeler(fit_intercept=True, positive=False)
DR_class, DR_hyperparams = models.DecisionTreeRegressor.init_for_modeler(criterion='squared_error', max_depth=None)
DC_class, DC_hyperparams = models.DecisionTreeClassifier.init_for_modeler(criterion='gini', max_depth=None)
RR_class, RR_hyperparams = models.RandomForestRegressor.init_for_modeler(n_estimators=10, criterion='squared_error', max_depth=None)
RC_class, RC_hyperparams = models.RandomForestClassifier.init_for_modeler(n_estimators=10, criterion='gini', max_depth=None)
MR_class, MR_hyperparams = models.MLPRegressor.init_for_modeler(hidden_layer_sizes=(20, 20, 20), activation='relu', max_iter=500)
MC_class, MC_hyperparams = models.MLPClassifier.init_for_modeler(hidden_layer_sizes=(20, 20, 20), activation='relu', max_iter=500)

# Run cross_validation for all models instantiated in a Modeler
Ri_mdlr, CV_Ri = Modeler.cross_validation(dataset_example, input_names_reg, output_names_reg_solo, Ri_class, Ri_hyperparams,
                                          True, True, 5, 0.8, "ridge_modeler")
LR_mdlr, CV_LR = Modeler.cross_validation(dataset_example, input_names_reg, output_names_reg_solo, LR_class, LR_hyperparams,
                                          True, True, 5, 0.8, "linearregression_modeler")
DR_mdlr, CV_DR = Modeler.cross_validation(dataset_example, input_names_reg, output_names_reg_solo, DR_class, DR_hyperparams,
                                          True, True, 3, 0.8, "DTRegressor_modeler")
DC_mdlr, CV_DC = Modeler.cross_validation(dataset_example, input_names_clf, output_names_clf, DC_class, DC_hyperparams,
                                          True, False, 3, 0.8, "DTClassifier_modeler")
RR_mdlr, CV_RR = Modeler.cross_validation(dataset_example, input_names_reg, output_names_reg_solo, RR_class, RR_hyperparams,
                                          True, True, 3, 0.8, "RFRegressor_modeler")
RC_mdlr, CV_RC = Modeler.cross_validation(dataset_example, input_names_clf, output_names_clf, RC_class, RC_hyperparams,
                                          True, False, 3, 0.8, "RF_classifier_modeler")
MR_mdlr, CV_MR = Modeler.cross_validation(dataset_example, input_names_reg, output_names_reg_solo, MR_class, MR_hyperparams,
                                          True, True, 3, 0.8, "MLPRegressor_modeler")
MC_mdlr, CV_MC = Modeler.cross_validation(dataset_example, input_names_clf, output_names_clf, MC_class, MC_hyperparams,
                                          True, False, 3, 0.8, "MLPClassifier_modeler")

CV_Ri.plot()
CV_LR.plot()
CV_DR.plot()
CV_DC.plot()
CV_RR.plot()
CV_RC.plot()
CV_MR.plot()
CV_MC.plot()

modeler = Modeler.fit_dataset(dataset_example, input_names, output_names, models_class, hyperparameters, True, True,
                              "test_modeler")

# validation_data = ValidationData.from_dataset(dataset_example, input_names, output_names)
test = CrossValidation.from_dataset(modeler, dataset_example, input_names, output_names, 5, 0.8, "validation_test")
test.plot()


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
