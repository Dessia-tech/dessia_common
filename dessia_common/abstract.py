""" Abstract module that defines a base DessiaObject in order to avoid circular imports. """


class CoreDessiaObject:
    """ Base DessiaObject for checking inheritance purpose. """

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = "#", id_method: bool = True, id_memo=None):
        raise NotImplementedError("CoreDessiaObject is an abstract class and should not be use directly.")

    @classmethod
    def dict_to_object(cls, dict_=None, force_generic: bool = False, global_dict=None,
                       pointers_memo=None, path: str = '#'):
        raise NotImplementedError("CoreDessiaObject is an abstract class and should not be use directly.")

    def _data_diff(self, other_object: 'CoreDessiaObject'):
        raise NotImplementedError("CoreDessiaObject is an abstract class and should not be use directly.")
