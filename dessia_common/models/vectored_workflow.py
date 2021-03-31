from dessia_common.vectored_objects import Catalog, ParetoSettings
import dessia_common.workflow as wf
from dessia_api_client import Client

choice_args = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
               'Weight', 'Acceleration', 'Model']  # Ordered

minimized_attributes = {'MPG': False, 'Horsepower': True,
                        'Weight': True, 'Acceleration': False}

aimpoint = {'MPG': 4, 'Weight': 1000}

filters = [{'attribute': 'MPG', 'operator': 'gte', 'bound': 10},
           {'attribute': 'MPG', 'operator': 'lte', 'bound': 35}]

# Blocks
import_csv = wf.Import(type_='csv') 
instantiate_pareto = wf.InstanciateModel(model_class=ParetoSettings,
                                         name='Pareto Settings')
instantiate_catalog = wf.InstanciateModel(model_class=Catalog,
                                          name='Cars Instantiation')
filter_method = wf.ModelMethod(model_class=Catalog,
                               method_name='filter_',
                               name='Filters')
filtered_catalog = wf.InstanciateModel(model_class=Catalog,
                                       name='Filtered Catalog')
display = wf.Display(order=0, name="Display")
# display = wf.MultiPlot(choice_args, 1, name='Display')
filtered = wf.Display(order=1, name="Filtered")
# filtered = wf.MultiPlot(choice_args, 1, name='Filtered')
# objectives_method = wf.ModelMethod(model_class=Catalog,
#                                    method_name='find_best_objective',
#                                    name="Find best objectives")
# objectives_attributes = wf.ModelAttribute(attribute_name='objectives',
#                                           name='Objectives')

blocks = [import_csv, instantiate_pareto, instantiate_catalog, filter_method,
          filtered_catalog, display, filtered]
# , objectives_method, objectives_attributes]

# Pipes
pipes = [
    wf.Pipe(import_csv.outputs[0], instantiate_catalog.inputs[0]),
    wf.Pipe(import_csv.outputs[1], instantiate_catalog.inputs[1]),
    wf.Pipe(instantiate_pareto.outputs[0], instantiate_catalog.inputs[2]),
    wf.Pipe(instantiate_catalog.outputs[0], display.inputs[0]),
    # wf.Pipe(instantiate_catalog.outputs[0], objectives_method.inputs[0]),
    wf.Pipe(instantiate_catalog.outputs[0], filter_method.inputs[0]),
    wf.Pipe(filter_method.outputs[0], filtered_catalog.inputs[0]),
    wf.Pipe(import_csv.outputs[1], filtered_catalog.inputs[1]),
    wf.Pipe(filtered_catalog.outputs[0], filtered.inputs[0]),
    wf.Pipe(instantiate_pareto.outputs[0], filtered_catalog.inputs[2])
    # wf.Pipe(objectives_method.outputs[1], objectives_attributes.inputs[0])
]

# Workflow
workflow = wf.Workflow(blocks=blocks, pipes=pipes,
                       output=filter_method.outputs[0],
                       name='Cars workflow')
# workflow.plot_jointjs()

# # Input values
input_values = {
    workflow.index(import_csv.inputs[0]): 'cars.csv',
    workflow.index(instantiate_pareto.inputs[0]): minimized_attributes,
    workflow.index(instantiate_pareto.inputs[1]): True,
    workflow.index(instantiate_catalog.inputs[4]): choice_args,
    workflow.index(instantiate_catalog.inputs[3]): [],
    workflow.index(instantiate_catalog.inputs[5]): 'Cars',
    workflow.index(filter_method.inputs[1]): filters,
    workflow.index(filtered_catalog.inputs[4]): choice_args,
    workflow.index(filtered_catalog.inputs[3]): [],
    workflow.index(filtered_catalog.inputs[5]): 'Filtered Cars',
    # workflow.index(objectives_method.inputs[1]): aimpoint,
    # workflow.index(objectives_method.inputs[2]): minimized_attributes
}
workflow_run = workflow.run(input_values=input_values)

c = Client(api_url='https://api.platform-dev.dessia.tech')
r = c.create_object_from_python_object(workflow_run)