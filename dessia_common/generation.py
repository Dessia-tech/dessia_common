
from typing import List
import dectree as dt
import dessia_common.core as dc


class Generator(dc.DessiaObject):
    """
    Common parts of generator
    """

    def __init__(self, name: str = ''):
        dc.DessiaObject.__init__(self, name=name)

    def is_model_valid(self, model) -> bool:
        raise NotImplementedError('the method is_model_valid must be overloaded by subclassing class')

    def number_possibilities_from_model(self, model):
        raise NotImplementedError('the method number_possibilities_from_model must be overloaded by subclassing class')


class TreeGenerator(Generator):
    """
    Common parts of generator
    """
    _allowed_methods = ['generate']
    _non_serializable_attributes = ['tree']
    _non_data_eq_attributes = ['tree']

    def __init__(self, tree, name: str = ''):
        self.tree = tree
        Generator.__init__(self, name=name)

    def model_from_vector(self, vector: List[int]):
        raise NotImplementedError('the method model_from_vector must be overloaded by subclassing class')

    def is_vector_valid(self, vector: List[int]) -> bool:
        return True

    def number_possibilities_from_vector(self, vector: List[int]):
        """
        This method is generic but can be overloaded to avoid model instanciation
        """
        model = self.model_from_vector(vector)
        return self.number_possibilities_from_model(model)


class DecisionTreeGenerator(TreeGenerator):
    """
    Abstract class, to be subclassed by real class
    This is still experimental and might be buggy
    """

    def __init__(self, name: str = ''):
        """
        :param name: The name of the generator
        """
        tree = dt.DecisionTree()
        TreeGenerator.__init__(self, tree, name=name)

    def generate(self, verbose: bool = False):
        model = self.model_from_vector(self.tree.current_node)

        self.tree.SetCurrentNodeNumberPossibilities(self.number_possibilities_from_model(model))
        while not self.tree.finished:
            if verbose:
                print('current node: ', self.tree.current_node)
            valid = self.is_vector_valid(self.tree.current_node)
            if valid:
                model = self.model_from_vector(self.tree.current_node)
                valid = self.is_model_valid(model)

            if verbose:
                print('node validity:', valid)
            if valid:
                number_possibilities = self.number_possibilities_from_model(model)
                if verbose:
                    print('number possibilities', number_possibilities)
                self.tree.SetCurrentNodeNumberPossibilities(number_possibilities)

            # TODO create a function in dectreee to know if a leaf
            if valid and (number_possibilities == 0):
                yield model

            self.tree.NextNode(valid)


class RegularDecisionTreeGenerator(TreeGenerator):
    """
    Abstract class, to be subclassed by real class
    This is still experimental and might be buggy
    """

    def __init__(self, number_possibilities: List[int], name: str = ''):
        self.number_possibilities = number_possibilities
        self.leaves_depth = len(self.number_possibilities) - 1
        tree = dt.RegularDecisionTree(number_possibilities)
        TreeGenerator.__init__(self, tree=tree, name=name)

    def generate(self, sorted_nodes: bool = False,
                 unique_nodes: bool = False,
                 verbose: bool = False):
        """

        Parameters
        ----------
        sorted_nodes : bool, optional
            DESCRIPTION. The default is False.
        unique_nodes : bool, optional
            DESCRIPTION. The default is False.
        verbose : bool, optional
            DESCRIPTION. The default is False.

        Yields
        ------
        model : TYPE
            DESCRIPTION.

        """
        if sorted_nodes:
            if unique_nodes:
                next_node_function = self.tree.NextSortedUniqueNode
            else:
                next_node_function = self.tree.NextSortedNode
        else:
            if unique_nodes:
                next_node_function = self.tree.NextUniqueNode
            else:
                next_node_function = self.tree.NextNode

        while not self.tree.finished:
            valid = self.is_vector_valid(self.tree.current_node)
            if verbose:
                print('current node: ', self.tree.current_node)
                print('node vector validity:', valid)
            if valid:
                model = self.model_from_vector(self.tree.current_node)
                valid = self.is_model_valid(model)
                if verbose:
                    print('node model validity:', valid)

                # TODO create a function in dectreee to know if a leaf
                if self.tree.current_depth == self.leaves_depth:
                    yield model

            next_node_function(valid)
