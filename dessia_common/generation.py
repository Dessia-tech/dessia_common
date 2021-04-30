
import dectree as dt
import dessia_common.core as dc


class Generator(dc.DessiaObject):
    """
    Common parts of generator
    """
    def __init__(self, name:str=''):
        dc.DessiaObject.__init__(self, name=name)
        
    def model_valid(self, model):
        raise NotImplementedError('the method must be overloaded by subclassing class')

    def model_from_vector(self, vector):
        raise NotImplementedError('the method must be overloaded by subclassing class')



class RegularDecisionTreeGenerator(Generator):
    """
    Abstract class, to be subclassed by real class
    This is still experimental and might be buggy
    """

    def __init__(self, number_possibilities:int, name:str=''):
        self.number_possibilities = number_possibilities
        self.leaves_depth = len(self.number_possibilities) -1
        self.tree = dt.RegularDecisionTree(number_possibilities)
        Generator.__init__(self, name=name)  

    def generate(self, **kwargs):
        while not self.tree.finished():
            model = self.model_from_vector(self.tree.current_node)
            valid = self.model_valid(model)
            self.tree.NextNode(valid)
            # TODO create a function in dectreee to know if a leaf
            if valid and self.tree.current_depth == self.leaves_depth:
                yield model



class DecisionTreeGenerator(Generator):
    """
    Abstract class, to be subclassed by real class
    This is still experimental and might be buggy
    """

    def __init__(self, name:str=''):
        self.tree = dt.DecisionTree()
        Generator.__init__(self, name=name)
