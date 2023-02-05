""" __init__ method for datatools module """

import warnings
from typing import List
from dessia_common.core import DessiaObject
import dessia_common.datatools.dataset as DS
import dessia_common.datatools.cluster as DC

# Imports for retrocompatibility
# TODO: Remove it kindly in next releases.

class HeterogeneousList(DS.Dataset):
    def __init__(self, dessia_objects: List[DessiaObject] = None, name: str = ''):
        self.warning_string()
        DS.Dataset.__init__(self, dessia_objects=dessia_objects, name=name)

    def warning_string(self):
        string = "Class HeterogeneousList is not supported anymore and will be deleted in next releases (0.11.0).\n"
        string += "Please use the Dataset class, which is exactly the same.\n"
        string += "Dataset is imported with <from dessia_common.datatools.dataset import Dataset>.\n"
        warnings.warn(string, DeprecationWarning)


class CategorizedList(DC.ClusteredDataset):
    def __init__(self, dessia_objects: List[DessiaObject] = None, labels: List[int] = None, name: str = ''):
        self.warning_string()
        DC.ClusteredDataset.__init__(self, dessia_objects=dessia_objects, labels=labels, name=name)

    def warning_string(self):
        string = "Class CatagorizedList is not supported anymore and will be deleted in next releases (0.11.0).\n"
        string += "Please use the ClusteredDataset class, which is exactly the same.\n"
        string += "ClusteredDataset is imported with <from dessia_common.datatools.cluster import ClusteredDataset>.\n"
        warnings.warn(string, DeprecationWarning)
