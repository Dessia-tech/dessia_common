#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:54:34 2019

@author: jezequel
"""
from powerpack import cells, power_profile, electrical
from powerpack.optimization import electrical as eo
from dessia_common import workflow as wf
from dessia_api_client import Client
import dessia_common as dc


# =============================================================================
# Vehicle specs
# =============================================================================
specs = {'SCx' : 1,
         'Crr' : 0.01,
         'mass' : 1200,
         'powertrain_efficiency' : 0.7,
         'charge_efficiency' : 0.5}

# =============================================================================
# Electrical Specs
# =============================================================================

limits_voltage_module = {'charge': {'minimum': 1, 'maximum': 100},
                         'discharge': {'minimum': 1, 'maximum': 120}}
limits_current_module = {'charge': {'minimum': 0, 'maximum': 100},
                         'discharge': {'minimum': -100, 'maximum': 0}}
limits_voltage_battery = {'charge': {'minimum': 0, 'maximum': 500},
                          'discharge': {'minimum': 0, 'maximum': 500}}
limits_current_battery = {'charge': {'minimum': 0, 'maximum': 1000},
                          'discharge': {'minimum': -1000, 'maximum': 0}}


power_profile_1, time_1 = power_profile.WltpProfile(specs, (1, ))
t_wltp1 = electrical.Evolution(list(time_1))
p_wltp1 = electrical.Evolution(list(power_profile_1))

power_profile_2, time_2 = power_profile.WltpProfile(specs, (2, ))
t_wltp2 = electrical.Evolution(list(time_2))
p_wltp2 = electrical.Evolution(list(power_profile_2))

power_profile_3, time_3 = power_profile.WltpProfile(specs, (3, ))
t_wltp3 = electrical.Evolution(list(time_3))
p_wltp3 = electrical.Evolution(list(power_profile_3))
t_load = electrical.Evolution(list(range(10)))
p_load = electrical.Evolution([1e5]*10)
t_end = electrical.Evolution(list(range(10)))
p_end = electrical.Evolution([-2e4]*10)

ce_end = electrical.CombinationEvolution(evolution1=[t_end],
                                   evolution2=[p_end],
                                   name='End Profile')
ce_wltp1 = electrical.CombinationEvolution(evolution1=[t_wltp1],
                                     evolution2=[p_wltp1],
                                     name='WLTP1 profile')
ce_wltp2 = electrical.CombinationEvolution(evolution1=[t_wltp2],
                                     evolution2=[p_wltp2],
                                     name='WLTP2 profile')
ce_wltp3 = electrical.CombinationEvolution(evolution1=[t_wltp3],
                                     evolution2=[p_wltp3],
                                     name='WLTP3 profile')
ce_load = electrical.CombinationEvolution(evolution1=[t_load],
                                    evolution2=[p_load],
                                    name='Load Profile')

load_bat = electrical.PowerProfile(soc_init=0.05*180000,
                             combination_evolutions=[ce_load],
                             loop=True,
                             soc_end=0.95*180000,
                             charger=True,
                             name='Load profile')
end_bat = electrical.PowerProfile(combination_evolutions=[ce_end],
                            loop=False,
                            power_accuracy=0.2,
                            soc_init=0.1*180000,
                            name='End profile')
wltp_bat = electrical.PowerProfile(combination_evolutions=[ce_wltp1, ce_wltp2, ce_wltp3],
                             loop=True,
                             power_accuracy=0.2,
                             soc_init=0.95*180000,
                             max_loop=1,
                             soc_end=0.1*180000,
                             use_selection=False,
                             name='WLTP profile')

comb_profile_wltp = electrical.CombinationPowerProfile([wltp_bat],
                                                 name='wltp_profil')
comb_profile_load = electrical.CombinationPowerProfile([load_bat],
                                                 name='load_profil')
comb_profile_end = electrical.CombinationPowerProfile([end_bat],
                                                name='end_soc_profil')

# =============================================================================
# Electrical Optimizer
# =============================================================================
input_values = {}
blocks = []
block_ebo = wf.InstanciateModel(eo.ElecBatteryOptimizer, name='EBO')
optimize_ebo = wf.ModelMethod(eo.ElecBatteryOptimizer, 'Optimize', 'Optimize EBO')
attribute_selection_ebo = wf.ModelAttribute('powerpack_electric_simulators', 'Attribute Selection EBO')

filters = [{'attribute' : 'bms.battery.number_module_parallel', 'operator' : 'gt', 'bound' : 2},
           {'attribute' : 'bms.battery.number_module_serie', 'operator' : 'lte', 'bound' : 10},
           {'attribute' : 'bms.number_cells', 'operator' : 'gte', 'bound' : 750},
           {'attribute' : 'bms.number_cells', 'operator' : 'lte', 'bound' : 800},
           {'attribute' : 'bms.battery.number_cells', 'operator' : 'gt', 'bound' : 700}]

filter_sort = wf.Filter(filters, 'Filters EBO')

blocks.extend([block_ebo, optimize_ebo, attribute_selection_ebo, filter_sort])
pipes = [wf.Pipe(block_ebo.outputs[0], optimize_ebo.inputs[0]),
         wf.Pipe(optimize_ebo.outputs[1], attribute_selection_ebo.inputs[0]),
         wf.Pipe(attribute_selection_ebo.outputs[0], filter_sort.inputs[0])]

workflow = wf.Workflow(blocks, pipes, filter_sort.outputs[0])
input_values = {0: cells.CELL1_2RC,
                1: limits_voltage_module,
                2: limits_current_module,
                3: limits_voltage_battery,
                4: limits_current_battery,
                5: [33, 34],
                6: [24, 25],
                7: [comb_profile_load,
                    comb_profile_wltp,
                    comb_profile_end],
                12: 5}
#workflow_run = workflow.run(input_values)
#d = workflow_run.to_dict()
#w = wf.WorkflowRun.dict_to_object(d)
#methods_jsonschemas = workflow._method_jsonschemas
#run_default_dict = models.get_jsonschema_default_dict(methods_jsonschemas['run'])
#serialized_run_default_dict = workflow.dict_to_arguments(run_default_dict, 'run')['input_variables_values']

## From serialized default dict to input_values
#sdd_to_iv = {}
#for variable, default_value in serialized_run_default_dict.items():
#    sdd_to_iv[variable] = input_values[variable]
#
#sdd_to_iv['10'] = 2
#run1 = workflow.run(sdd_to_iv)

## Setting values before dict_to_arguments
#run_dict = {}
#for i, input_ in enumerate(block_ebo.inputs):
#    if str(i) in run_default_dict:
#        run_dict[str(i)] = input_values[input_]
#
#run_dict['9'] = 2
#serialized_run_dict = workflow.dict_to_arguments(run_dict, 'run')
#
#run2 = workflow.run(serialized_run_dict['input_variables_values'])
#
## Requetes
#r = c.CreateObject(workflow)
#reference = r.json()
#job = c.SubmitJob(reference['object_class'], reference['id'], 'run', run_dict)
