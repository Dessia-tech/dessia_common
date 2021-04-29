from dessia_common.workflow import InstantiateModel, ModelMethod,\
    ModelAttribute, Pipe, Workflow, WorkflowBlock, ForEach, MultiPlot
from dessia_common.forms import Generator, Optimizer
from dessia_common import DessiaObject


# class Submodel(DessiaObject):
#     _generic_eq = True
#
#     def __init__(self, subvalue: int, name: str = ''):
#         self.subvalue = subvalue
#         self.name = name
#
#         DessiaObject.__init__(self, name=name)


# class Model(DessiaObject):
#     _generic_eq = True
#
#     def __init__(self, value: int, submodel: Submodel, name: str = ''):
#         self.value = value
#         self.submodel = submodel
#
#         DessiaObject.__init__(self, name=name)



# class Optimizer(DessiaObject):
#     def __init__(self, model_to_optimize: Model, name: str = ''):
#         self.model_to_optimize = model_to_optimize
#
#         DessiaObject.__init__(self, name=name)
#
#     def optimize(self, optimization_value: int = 3) -> None:
#         self.model_to_optimize.value += optimization_value


instanciate_generator = InstantiateModel(model_class=Generator,
                                         name='Instantiate Generator')
generator_generate = ModelMethod(model_class=Generator,
                                 method_name='generate',
                                 name='Generator Generate')
attribute_selection = ModelAttribute(attribute_name='models',
                                     name='Attribute Selection')

# Subworkflow of model optimization
instanciate_optimizer = InstantiateModel(model_class=Optimizer,
                                         name='Instantiate Optimizer')
optimization = ModelMethod(model_class=Optimizer, method_name='optimize',
                           name='Optimization')
model_fetcher = ModelAttribute(attribute_name='model_to_optimize',
                               name='Model Fetcher')

pipe1_opt = Pipe(input_variable=instanciate_optimizer.outputs[0],
                 output_variable=optimization.inputs[0])
pipe2_opt = Pipe(input_variable=optimization.outputs[1],
                 output_variable=model_fetcher.inputs[0])
optimization_blocks = [instanciate_optimizer, optimization, model_fetcher]
optimization_pipes = [pipe1_opt, pipe2_opt]
optimization_workflow = Workflow(blocks=optimization_blocks,
                                 pipes=optimization_pipes,
                                 output=model_fetcher.outputs[0],
                                 name='Optimization Workflow')

optimization_workflow_block = WorkflowBlock(workflow=optimization_workflow,
                                            name='Workflow Block')

parallel_optimization = ForEach(workflow_block=optimization_workflow_block,
                                iter_input_index=0, name='ForEach')

display_attributes = ['intarg', 'strarg', 'standalone_subobject/floatarg']
display = MultiPlot(attributes=display_attributes, name='Display')

pipe_1 = Pipe(input_variable=instanciate_generator.outputs[0],
              output_variable=generator_generate.inputs[0])
pipe_2 = Pipe(input_variable=generator_generate.outputs[1],
              output_variable=attribute_selection.inputs[0])
pipe_3 = Pipe(input_variable=attribute_selection.outputs[0],
              output_variable=parallel_optimization.inputs[0])
pipe_4 = Pipe(input_variable=parallel_optimization.outputs[0],
              output_variable=display.inputs[0])

blocks = [instanciate_generator, generator_generate,
          attribute_selection, parallel_optimization, display]
pipes = [pipe_1, pipe_2, pipe_3, pipe_4]
demo_workflow = Workflow(blocks=blocks, pipes=pipes,
                         output=parallel_optimization.outputs[0])

input_values = {0: 5}

demo_workflow_run = demo_workflow.run(input_values=input_values,
                                      verbose=True, name='Dev Objects')

# Assert to_dict, dict_to_object, hashes, eqs
# dict_ = demo_workflow_run.to_dict()
# object_ = WorkflowRun.dict_to_object(dict_=dict_)
#
# assert hash(demo_workflow_run) == hash(object_)

# Assert deserialization
# demo_workflow_dict = demo_workflow.to_dict()
# import json
# demo_workflow_json = json.dumps(demo_workflow_dict)
# dict_from_json = json.loads(demo_workflow_json)
# deserialized_demo_workflow = wf.Workflow.dict_to_object(dict_from_json)
# assert demo_workflow == deserialized_demo_workflow
