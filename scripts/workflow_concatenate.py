from dessia_common.vectored_objects import Catalog, ParetoSettings
import dessia_common.workflow as wf

choice_args = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
               'Weight', 'Acceleration', 'Model']  # Ordered

minimized_attributes = {'MPG': False, 'Horsepower': True,
                        'Weight': True, 'Acceleration': False}

aimpoint = {'MPG': 4, 'Weight': 1000}

filter_min = [{'attribute': 'MPG', 'operator': 'lte', 'bound': 20}]

# Blocks
import_csv = wf.Import(type_='csv')
instantiate_pareto = wf.InstanciateModel(model_class=ParetoSettings,
                                         name='Pareto Settings')
catalog1 = wf.InstanciateModel(model_class=Catalog,
                               name='Cars 1')
filter1 = wf.ModelMethod(model_class=Catalog,
                         method_name='filter_',
                         name='Filter 1')
filtered_catalog1 = wf.InstanciateModel(model_class=Catalog,
                                        name='Filtered Catalog 1')

filter_max = [{'attribute': 'MPG', 'operator': 'gte', 'bound': 30}]
catalog2 = wf.InstanciateModel(model_class=Catalog,
                               name='Car 2')
filter2 = wf.ModelMethod(model_class=Catalog,
                         method_name='filter_',
                         name='Filter 2')
filtered_catalog2 = wf.InstanciateModel(model_class=Catalog,
                                        name='Filtered Catalog 2')

catalogs = wf.Sequence(number_arguments=2, name='Catalogs')
concatenate = wf.ClassMethod(class_=Catalog, method_name='concatenate',
                             name='Concatenate')
display = wf.Display(order=0)

array_attribute = wf.ModelAttribute(attribute_name='array', name='Array')

blocks = [import_csv, instantiate_pareto,
          catalog1, filter1, filtered_catalog1,
          catalog2, filter2, filtered_catalog2,
          catalogs, concatenate, display, array_attribute]

# Pipes
pipes = [
    wf.Pipe(import_csv.outputs[0], catalog1.inputs[0]),
    wf.Pipe(import_csv.outputs[1], catalog1.inputs[1]),
    wf.Pipe(instantiate_pareto.outputs[0], catalog1.inputs[2]),
    wf.Pipe(import_csv.outputs[0], catalog2.inputs[0]),
    wf.Pipe(import_csv.outputs[1], catalog2.inputs[1]),
    wf.Pipe(instantiate_pareto.outputs[0], catalog2.inputs[2]),
    wf.Pipe(catalog1.outputs[0], filter1.inputs[0]),
    wf.Pipe(catalog2.outputs[0], filter2.inputs[0]),
    wf.Pipe(filter1.outputs[0], filtered_catalog1.inputs[0]),
    wf.Pipe(import_csv.outputs[1], filtered_catalog1.inputs[1]),
    wf.Pipe(instantiate_pareto.outputs[0], filtered_catalog1.inputs[2]),
    wf.Pipe(filtered_catalog1.outputs[0], catalogs.inputs[0]),
    wf.Pipe(filter2.outputs[0], filtered_catalog2.inputs[0]),
    wf.Pipe(import_csv.outputs[1], filtered_catalog2.inputs[1]),
    wf.Pipe(instantiate_pareto.outputs[0], filtered_catalog2.inputs[2]),
    wf.Pipe(filtered_catalog2.outputs[0], catalogs.inputs[1]),
    wf.Pipe(catalogs.outputs[0], concatenate.inputs[0]),
    wf.Pipe(import_csv.outputs[1], concatenate.inputs[3]),
    wf.Pipe(concatenate.outputs[0], display.inputs[0]),
    wf.Pipe(concatenate.outputs[0], array_attribute.inputs[0])
]

# Workflow
workflow = wf.Workflow(blocks=blocks, pipes=pipes,
                       output=catalogs.outputs[0],
                       name='Cars workflow')
# workflow.plot_jointjs()

# # Input values
input_values = {
    workflow.index(import_csv.inputs[0]): 'cars.csv',
    workflow.index(instantiate_pareto.inputs[0]): minimized_attributes,
    workflow.index(instantiate_pareto.inputs[1]): True,
    workflow.index(catalog1.inputs[4]): choice_args,
    workflow.index(catalog1.inputs[5]): 'Cars 1',
    workflow.index(filter1.inputs[1]): filter_min,
    workflow.index(filtered_catalog1.inputs[4]): choice_args,
    workflow.index(filtered_catalog1.inputs[5]): 'Filtered Cars 1',
    workflow.index(catalog2.inputs[4]): choice_args,
    workflow.index(catalog2.inputs[5]): 'Cars 2',
    workflow.index(filter2.inputs[1]): filter_max,
    workflow.index(filtered_catalog2.inputs[4]): choice_args,
    workflow.index(filtered_catalog2.inputs[5]): 'Filtered Cars 2',
}
workflow_run = workflow.run(input_values=input_values)
