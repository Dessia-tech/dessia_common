"""
Vectored_workflow tools

"""
import pkg_resources
from dessia_common.typings import ClassMethodType
from dessia_common.vectored_objects import Catalog, ParetoSettings
from dessia_common.core import DessiaFilter
import dessia_common.workflow as wf
import dessia_common.workflow.blocks as wfb
import dessia_common.typings as dct

choice_args = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model']  # Ordered

minimized_attributes = {'MPG': False, 'Horsepower': True, 'Weight': True, 'Acceleration': False}

aimpoint = {'MPG': 4, 'Weight': 1000}

filters = [DessiaFilter(attribute='MPG', operator='gte', bound=10),
           DessiaFilter(attribute='MPG', operator='lte', bound=35)]

# Blocks
import_csv = wfb.ClassMethod(ClassMethodType(Catalog, 'from_csv'))
instantiate_pareto = wfb.InstantiateModel(model_class=ParetoSettings, name='Pareto Settings')
set_pareto_catalog = wfb.SetModelAttribute('pareto_settings', name='Set Pareto in catalog')
filter_method = wfb.ModelMethod(dct.MethodType(Catalog, 'filter_'), name='Filters')
# filtered_catalog = wf.InstantiateModel(model_class=Catalog, name='Filtered Catalog')
display_ = wfb.Display(order=0, name="Display")
filtered = wfb.Display(order=1, name="Display of filtered object")

blocks = [import_csv, instantiate_pareto, set_pareto_catalog, filter_method, display_, filtered]


# Pipes
pipes = [
         wf.Pipe(import_csv.outputs[0], set_pareto_catalog.inputs[0]),
         wf.Pipe(instantiate_pareto.outputs[0], set_pareto_catalog.inputs[1]),
         # wf.Pipe(instantiate_pareto.outputs[0], instantiate_catalog.inputs[2]),
         wf.Pipe(set_pareto_catalog.outputs[0], display_.inputs[0]),
         wf.Pipe(set_pareto_catalog.outputs[0], filter_method.inputs[0]),
         # wf.Pipe(filter_method.outputs[0], filtered_catalog.inputs[0]),
         # wf.Pipe(import_csv.outputs[1], filtered_catalog.inputs[1]),
         wf.Pipe(filter_method.outputs[0], filtered.inputs[0]),
         # wf.Pipe(instantiate_pareto.outputs[0], filtered_catalog.inputs[2])
         ]

# Workflow
vectored_workflow = wf.Workflow(blocks=blocks, pipes=pipes, output=filter_method.outputs[0], name='Cars workflow')

# # Input values
cars_stream = pkg_resources.resource_stream(pkg_resources.Requirement('dessia_common'),
                                            'dessia_common/models/data/cars.csv')

input_values = {
    vectored_workflow.input_index(import_csv.inputs[0]): cars_stream,
    vectored_workflow.input_index(instantiate_pareto.inputs[0]): minimized_attributes,
    vectored_workflow.input_index(instantiate_pareto.inputs[1]): True,
    # vectored_workflow.input_index(instantiate_catalog.inputs[4]): choice_args,
    # vectored_workflow.input_index(instantiate_catalog.inputs[3]): [],
    # vectored_workflow.input_index(instantiate_catalog.inputs[5]): 'Cars',
    vectored_workflow.input_index(filter_method.inputs[1]): filters,
    # vectored_workflow.input_index(filtered_catalog.inputs[4]): choice_args,
    # vectored_workflow.input_index(filtered_catalog.inputs[3]): [],
    # vectored_workflow.input_index(filtered_catalog.inputs[5]): 'Filtered Cars',
}
workflow_run = vectored_workflow.run(input_values=input_values)

# Testing jsonschema blocks
for block in blocks:
    block.jsonschema()
