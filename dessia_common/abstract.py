""" Abstract module that defines a base DessiaObject in order to avoid circular imports. """


ABSTRACT_ERROR = NotImplementedError("CoreDessiaObject is an abstract class and should not be use directly.")


class CoreDessiaObject:
    """ Base DessiaObject for checking inheritance purpose. """

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = "#", id_method: bool = True, id_memo=None):
        """ Abstract to_dict method. """
        raise ABSTRACT_ERROR

    @classmethod
    def dict_to_object(cls, dict_, **kwargs):
        """ Abstract dict_to_object method. """
        raise ABSTRACT_ERROR

    def _data_diff(self, other_object: 'CoreDessiaObject'):
        """ Abstract _data_diff method. """
        raise ABSTRACT_ERROR

    @property
    def method_schemas(self):
        """ Abstract method_schemas method. """
        raise ABSTRACT_ERROR

    def _displays(self):
        """ Abstract _displays method. """
        raise ABSTRACT_ERROR
