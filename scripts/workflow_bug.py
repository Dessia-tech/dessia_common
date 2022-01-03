import dessia_common.workflow as wf
from typing import List
import renault_custom as rc

class Cluster(rc.DessiaObject):
    _standalone_in_db = True
    _non_serializable_attributes = []
    _non_data_eq_attributes = []
    _non_data_hash_attributes = []
    _eq_is_data_eq = True

    def __init__(self, number_clusters: float = None,
                 name: str = ''):
        rc.DessiaObject.__init__(self, name=name)
        self.number_clusters = number_clusters

class Generator(rc.DessiaObject):
    _standalone_in_db = True
    _non_serializable_attributes = []
    _non_data_eq_attributes = []
    _non_data_hash_attributes = []
    _eq_is_data_eq = True
    _allowed_methods = ['selection_clusters']

    def __init__(self, number_clusters: float = None,
                 name: str = ''):
        rc.DessiaObject.__init__(self, name=name)
        self.number_clusters = number_clusters

    def selection_clusters(self, verbose: bool = True) -> List[Cluster]:
        outputs = []
        for i in range(10):
            outputs.append(Cluster(i))
        return outputs


block_generator = wf.InstantiateModel(Generator, name='Generator')
generate_method = wf.MethodType(class_=Generator, name='selection_clusters')
methode_generate = wf.ModelMethod(method_type=generate_method, name='selection_clusters')

blocks = [block_generator, methode_generate]
sub_workflow2 = wf.Workflow(blocks=blocks,
                            pipes=[],
                            output=methode_generate.outputs[0],
                            name='sub_workflow')
sub_workflow_block2 = wf.WorkflowBlock(workflow=sub_workflow2,
                                       name='sub_workflow')
