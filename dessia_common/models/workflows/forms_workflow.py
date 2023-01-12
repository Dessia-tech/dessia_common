"""
Tools for forms of workflows.
"""
from dessia_common.workflow.core import TypedVariable, TypedVariableWithDefaultValue, Pipe, Workflow
from dessia_common.workflow.blocks import InstantiateModel, ModelMethod, ModelAttribute, ForEach,\
    MultiPlot, Unpacker, WorkflowBlock
from dessia_common.forms import Generator, Optimizer
from dessia_common.typings import MethodType

instanciate_generator = InstantiateModel(model_class=Generator, name='Instantiate Generator')

generate_method = MethodType(class_=Generator, name='generate')
generator_generate = ModelMethod(method_type=generate_method, name='Generator Generate')
attribute_selection = ModelAttribute(attribute_name='models', name='Attribute Selection')

# Subworkflow of model optimization
instanciate_optimizer = InstantiateModel(model_class=Optimizer, name='Instantiate Optimizer')


generate_method = MethodType(class_=Optimizer, name='optimize')
optimization = ModelMethod(method_type=generate_method, name='Optimization')

model_fetcher = ModelAttribute(attribute_name='model_to_optimize', name='Model Fetcher')

pipe1_opt = Pipe(input_variable=instanciate_optimizer.outputs[0], output_variable=optimization.inputs[0])
pipe2_opt = Pipe(input_variable=optimization.outputs[1], output_variable=model_fetcher.inputs[0])
optimization_blocks = [instanciate_optimizer, optimization, model_fetcher]
optimization_pipes = [pipe1_opt, pipe2_opt]
optimization_workflow = Workflow(blocks=optimization_blocks, pipes=optimization_pipes,
                                 output=model_fetcher.outputs[0], name='Optimization Workflow')

optimization_workflow_block = WorkflowBlock(workflow=optimization_workflow, name='Workflow Block')

parallel_optimization = ForEach(workflow_block=optimization_workflow_block, iter_input_index=0, name='ForEach')

multiplot_attributes = ['intarg', 'strarg', 'standalone_subobject/floatarg']
multiplot = MultiPlot(attributes=multiplot_attributes, name='Multiplot')

unpacker = Unpacker(indices=[0], name="Unpacker")

int_variable = TypedVariable(type_=int, name="Some Integer")
name_variable = TypedVariableWithDefaultValue(type_=str, name="Shared Name", default_value="Shared Name")

pipe_int_1 = Pipe(input_variable=int_variable, output_variable=instanciate_generator.inputs[1])
pipe_name_1 = Pipe(input_variable=name_variable, output_variable=instanciate_generator.inputs[3])
pipe_name_2 = Pipe(input_variable=name_variable, output_variable=parallel_optimization.inputs[1])
pipe_gene = Pipe(input_variable=instanciate_generator.outputs[0], output_variable=generator_generate.inputs[0])
pipe_attr = Pipe(input_variable=generator_generate.outputs[1], output_variable=attribute_selection.inputs[0])
pipe_opti = Pipe(input_variable=attribute_selection.outputs[0], output_variable=parallel_optimization.inputs[0])
pipe_mult = Pipe(input_variable=parallel_optimization.outputs[0], output_variable=multiplot.inputs[0])
pipe_unpack = Pipe(input_variable=parallel_optimization.outputs[0], output_variable=unpacker.inputs[0])

blocks = [instanciate_generator, generator_generate, attribute_selection,
          parallel_optimization, multiplot, unpacker]
pipes = [pipe_int_1, pipe_name_1, pipe_name_2, pipe_gene, pipe_attr, pipe_opti, pipe_mult, pipe_unpack]
workflow_ = Workflow(blocks=blocks, pipes=pipes, output=parallel_optimization.outputs[0], name="Workflow with NBVs")
workflow_state = workflow_.start_run({})
workflow_state.name = "WorkflowState with NBVs"
