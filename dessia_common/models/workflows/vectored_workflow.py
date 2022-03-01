from dessia_common.vectored_objects import Catalog, ParetoSettings
from dessia_common.core import DessiaFilter
import dessia_common.workflow as wf
import dessia_common.typings as dct

choice_args = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model']  # Ordered

minimized_attributes = {'MPG': False, 'Horsepower': True, 'Weight': True, 'Acceleration': False}

aimpoint = {'MPG': 4, 'Weight': 1000}

filters = [DessiaFilter(attribute='MPG', operator='gte', bound=10),
           DessiaFilter(attribute='MPG', operator='lte', bound=35)]

# Blocks
import_csv = wf.Import(type_='csv')
instantiate_pareto = wf.InstantiateModel(model_class=ParetoSettings, name='Pareto Settings')
instantiate_catalog = wf.InstantiateModel(model_class=Catalog, name='Cars Instantiation')
filter_method = wf.ModelMethod(dct.MethodType(Catalog, 'filter_'), name='Filters')
filtered_catalog = wf.InstantiateModel(model_class=Catalog, name='Filtered Catalog')
display_ = wf.Display(order=0, name="Display")
filtered = wf.Display(order=1, name="Filtered")

blocks = [import_csv, instantiate_pareto, instantiate_catalog, filter_method, filtered_catalog, display_, filtered]

# Pipes
pipes = [wf.Pipe(import_csv.outputs[0], instantiate_catalog.inputs[0]),
         wf.Pipe(import_csv.outputs[1], instantiate_catalog.inputs[1]),
         wf.Pipe(instantiate_pareto.outputs[0], instantiate_catalog.inputs[2]),
         wf.Pipe(instantiate_catalog.outputs[0], display_.inputs[0]),
         wf.Pipe(instantiate_catalog.outputs[0], filter_method.inputs[0]),
         wf.Pipe(filter_method.outputs[0], filtered_catalog.inputs[0]),
         wf.Pipe(import_csv.outputs[1], filtered_catalog.inputs[1]),
         wf.Pipe(filtered_catalog.outputs[0], filtered.inputs[0]),
         wf.Pipe(instantiate_pareto.outputs[0], filtered_catalog.inputs[2])]

# Workflow
vectored_workflow = wf.Workflow(blocks=blocks, pipes=pipes, output=filter_method.outputs[0], name='Cars workflow')

# # Input values
input_values = {
    vectored_workflow.input_index(import_csv.inputs[0]): 'cars.csv',
    vectored_workflow.input_index(instantiate_pareto.inputs[0]): minimized_attributes,
    vectored_workflow.input_index(instantiate_pareto.inputs[1]): True,
    vectored_workflow.input_index(instantiate_catalog.inputs[4]): choice_args,
    vectored_workflow.input_index(instantiate_catalog.inputs[3]): [],
    vectored_workflow.input_index(instantiate_catalog.inputs[5]): 'Cars',
    vectored_workflow.input_index(filter_method.inputs[1]): filters,
    vectored_workflow.input_index(filtered_catalog.inputs[4]): choice_args,
    vectored_workflow.input_index(filtered_catalog.inputs[3]): [],
    vectored_workflow.input_index(filtered_catalog.inputs[5]): 'Filtered Cars',
}
workflow_run = vectored_workflow.run(input_values=input_values)
