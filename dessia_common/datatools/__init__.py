"""
__init__ method for datatools module
"""

# import warnings
# from dessia_common.datatools.dataset import Dataset
# from dessia_common.datatools.cluster import ClusteredDataset

# # Imports for retrocompatibility
# # TODO: Remove it kindly in next releases.

# class HeterogeneousList(Dataset):
#     def __init__(self, *kwargs):
#         self.warning_string()
#         Dataset.__init__(self, *kwargs)

#     def warning_string(self):
#         string = "Class HeterogeneousList is not supported anymore and will be deleted in next releases (0.11.0).\n"
#         string += "Please use the Dataset class, which is exactly the same.\n"
#         string += "Dataset is imported with <from dessia_common.datatools.dataset import Dataset>.\n"
#         warnings.warn(string, DeprecationWarning)

#         prefix = "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  WARNING  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n"
#         string = prefix + string
#         string += "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  WARNING  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
#         print(string)


# class CategorizedList(ClusteredDataset):
#     def __init__(self, *kwargs):
#         self.warning_string()
#         ClusteredDataset.__init__(self, *kwargs)

#     def warning_string(self):
#         string = "Class CatagorizedList is not supported anymore and will be deleted in next releases (0.11.0).\n"
#         string += "Please use the ClusteredDataset class, which is exactly the same.\n"
#        string += "ClusteredDataset is imported with <from dessia_common.datatools.cluster import ClusteredDataset>.\n"
#         warnings.warn(string, DeprecationWarning)

#         prefix = "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  WARNING  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n"
#         string = prefix + string
#         string += "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  WARNING  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
#         print(string)
