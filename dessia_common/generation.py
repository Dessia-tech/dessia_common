
import dectree as dt
import dessia_common.core as dc


class Generator(dc.DessiaObject):
    """
    Common parts of generator
    """
    def __init__(self, name:str=''):
        dc.DessiaObject.__init__(self, name=name)
        
    def model_valid(self, model):
        raise NotImplementedError('the method model_valid must be overloaded by subclassing class')

    def model_from_vector(self, vector):
        raise NotImplementedError('the method model_from_vector must be overloaded by subclassing class')



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
        while not self.tree.finished:
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

    _allowed_methods = ['generate']
    _non_serializable_attributes = ['tree']
    _non_data_eq_attributes = ['tree']

    def __init__(self, name:str=''):
        self.tree = dt.DecisionTree()
        Generator.__init__(self, name=name)

    def number_possibilities_from_model(self, model):
        raise NotImplementedError('the method number_possibilities_from_model must be overloaded by subclassing class')

    def current_node_possibilities(self, vector):
        model = self.mdoel_from_vector
        
    def generate(self, verbose=False):
        model = self.model_from_vector(self.tree.current_node)
        self.tree.SetCurrentNodeNumberPossibilities(self.number_possibilities_from_model(model))
        while not self.tree.finished:
            if verbose:
                print('current node: ', self.tree.current_node)
            model = self.model_from_vector(self.tree.current_node)
            valid = self.model_valid(model)
            if verbose:
                print('valid', valid)
            if  valid:
                number_possibilities = self.number_possibilities_from_model(model)
                if verbose:
                    print('number possibilities', number_possibilities)
                self.tree.SetCurrentNodeNumberPossibilities(number_possibilities)
                
            self.tree.NextNode(valid)
            # TODO create a function in dectreee to know if a leaf
            if valid and (number_possibilities == 0):
                yield model