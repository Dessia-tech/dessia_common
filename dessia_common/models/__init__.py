"""
__init__ method for models module
"""
from .cars import all_cars_no_feat, all_cars_wi_feat
from .rand_data import rand_data_small, rand_data_middl, rand_data_large
from .power_test import simulation1, simulation_list, system1
from .workflows.wokflow_exports import workflow_export, workflow_export_state
from .workflows.forms_workflow import workflow_, workflow_state
